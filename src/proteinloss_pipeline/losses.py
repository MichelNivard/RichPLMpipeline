from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F

from .alphabet import MASK_TOKEN_ID, THREE_DI_PAD_ID
from .teacher import StructEncoder, encode_teacher_blocks

OBJECTIVES = ("mlm", "pssm", "dense_mi", "distance", "contact", "struct_latent", "3di")


def make_mlm_inputs(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, probability: float = 0.15
) -> tuple[torch.Tensor, torch.Tensor]:
    eligible = attention_mask.bool() & (input_ids >= 0) & (input_ids < 20)
    selected = (torch.rand_like(input_ids, dtype=torch.float32) < probability) & eligible
    # Avoid a zero-loss sequence in very small smoke batches.
    for batch_index in range(input_ids.shape[0]):
        if not selected[batch_index].any() and eligible[batch_index].any():
            first = torch.nonzero(eligible[batch_index], as_tuple=False)[0, 0]
            selected[batch_index, first] = True
    labels = torch.full_like(input_ids, -100)
    labels[selected] = input_ids[selected]
    masked = input_ids.clone()
    draw = torch.rand_like(input_ids, dtype=torch.float32)
    masked[selected & (draw < 0.8)] = MASK_TOKEN_ID
    random_mask = selected & (draw >= 0.8) & (draw < 0.9)
    random_values = torch.randint(0, 20, input_ids.shape, device=input_ids.device)
    masked[random_mask] = random_values[random_mask]
    return masked, labels


def _safe_pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.numel() < 2:
        return x.sum() * 0.0
    x, y = x.float() - x.float().mean(), y.float() - y.float().mean()
    return (x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-8)


def _pair_mask(attention: torch.Tensor, available: torch.Tensor, min_separation: int) -> torch.Tensor:
    length = attention.shape[1]
    index = torch.arange(length, device=attention.device)
    separated = (index[:, None] - index[None, :]).abs() >= min_separation
    return (
        attention[:, :, None].bool()
        & attention[:, None, :].bool()
        & separated.unsqueeze(0)
        & available[:, None, None].bool()
    )


def compute_objectives(
    model,
    batch: dict[str, Any],
    weights: dict[str, float],
    *,
    teacher: StructEncoder | None = None,
    mlm_probability: float = 0.15,
    min_pair_separation: int = 6,
    distance_correlation_weight: float = 1.0,
    contact_cutoff: float = 8.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    active = {name for name, weight in weights.items() if float(weight) != 0.0}
    input_ids, attention = batch["input_ids"], batch["attention_mask"].bool()
    unmasked_active = active - {"mlm"}
    output = model(input_ids, attention, objectives=unmasked_active)
    zero = output["pssm_logits"].sum() * 0.0
    raw: dict[str, torch.Tensor] = {name: zero for name in OBJECTIVES}
    metrics: dict[str, float] = {}

    pssm_available = batch["has_pssm"].bool()
    residue_pssm_mask = attention & pssm_available[:, None]
    if residue_pssm_mask.any():
        target_prob = batch["pssm_log_probs"].float().exp()
        log_prob = F.log_softmax(output["pssm_logits"].float(), dim=-1)
        pssm_per_residue = -(target_prob * log_prob).sum(-1)
        pssm_readout = pssm_per_residue[residue_pssm_mask].mean()
        metrics["pssm_loss"] = float(pssm_readout.detach())
        metrics["pssm_pearson"] = float(
            _safe_pearson(log_prob[residue_pssm_mask].flatten(), batch["pssm_log_probs"][residue_pssm_mask].flatten()).detach()
        )
        if "pssm" in active:
            raw["pssm"] = pssm_readout

    if "dense_mi" in active:
        mask = _pair_mask(attention, batch["has_mi"], min_pair_separation)
        if mask.any():
            prediction, target = output["dense_mi_pred"].float()[mask], batch["dense_mi_target"].float()[mask]
            raw["dense_mi"] = F.mse_loss(prediction, target)
            metrics["mi_pearson"] = float(_safe_pearson(prediction.detach(), target).detach())
            metrics["mi_rmse"] = float(torch.sqrt(raw["dense_mi"].detach()).item())

    distance_mask = _pair_mask(attention, batch["has_structure"], min_pair_separation)
    if "distance" in active and distance_mask.any():
        prediction = output["distance_pred"].float()[distance_mask]
        target = batch["distance_target"].float()[distance_mask]
        mse = F.mse_loss(prediction, target)
        correlation = _safe_pearson(prediction, target)
        raw["distance"] = mse + distance_correlation_weight * (1.0 - correlation)
        metrics.update(
            distance_mse=float(mse.detach()),
            distance_pearson=float(correlation.detach()),
            distance_mae_angstrom=float(
                (prediction.clamp(0.0, torch.log1p(torch.tensor(64.0, device=prediction.device))).expm1() - target.expm1())
                .abs()
                .mean()
                .detach()
            ),
        )

    if "contact" in active and distance_mask.any():
        logits = output["contact_pred"].float()[distance_mask]
        labels = (batch["distance_target"].float()[distance_mask].expm1() < contact_cutoff).float()
        positives = labels.sum()
        negative = labels.numel() - positives
        positive_weight = (negative / positives.clamp_min(1.0)).clamp(max=100.0)
        raw["contact"] = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=positive_weight)
        predicted = logits.sigmoid() >= 0.5
        metrics["contact_accuracy"] = float((predicted == labels.bool()).float().mean().detach())
        k = min(max(1, int(batch["has_structure"].sum().item())), logits.numel())
        metrics["contact_precision_top_l_proxy"] = float(labels[torch.topk(logits.detach(), k).indices].mean().detach())

    if "3di" in active:
        three_di_mask = attention & batch["has_3di"][:, None].bool() & (batch["three_di_ids"] != THREE_DI_PAD_ID)
        if three_di_mask.any():
            logits, labels = output["three_di_logits"].float()[three_di_mask], batch["three_di_ids"][three_di_mask].long()
            raw["3di"] = F.cross_entropy(logits, labels)
            metrics["three_di_accuracy"] = float((logits.argmax(-1) == labels).float().mean().detach())

    if "struct_latent" in active:
        if teacher is None:
            raise ValueError("struct_latent objective requires a frozen StructEncoder teacher")
        latent_mask = attention & batch["has_3di"][:, None].bool() & (batch["three_di_ids"] != teacher.pad_id)
        if latent_mask.any():
            target = encode_teacher_blocks(teacher, batch["three_di_ids"], latent_mask)
            prediction = output["struct_latent_pred"].float()
            raw["struct_latent"] = F.mse_loss(prediction[latent_mask], target[latent_mask])
            cosine = F.cosine_similarity(prediction[latent_mask], target[latent_mask], dim=-1).mean()
            metrics["latent_cosine"] = float(cosine.detach())
            decoded = teacher.decode(F.normalize(prediction[latent_mask], dim=-1))
            metrics["latent_decoded_3di_accuracy"] = float(
                (decoded.argmax(-1) == batch["three_di_ids"][latent_mask]).float().mean().detach()
            )

    if "mlm" in active:
        masked, labels = make_mlm_inputs(input_ids, attention, mlm_probability)
        masked_output = model(masked, attention, objectives=set(), return_hidden=False)["pssm_logits"].float()
        raw["mlm"] = F.cross_entropy(masked_output.reshape(-1, 20), labels.reshape(-1), ignore_index=-100)
        selected = labels != -100
        metrics["mlm_accuracy"] = float((masked_output.argmax(-1)[selected] == labels[selected]).float().mean().detach())

    total = zero
    for name in OBJECTIVES:
        weight = float(weights.get(name, 0.0))
        total = total + raw[name] * weight
        metrics[f"raw_{name}"] = float(raw[name].detach())
        metrics[f"weighted_{name}"] = float((raw[name] * weight).detach())
    metrics["weighted_total"] = float(total.detach())
    return total, metrics
