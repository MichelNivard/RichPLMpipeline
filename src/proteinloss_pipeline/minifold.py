from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .coordinates import CoordinateDataset
from .dataset import collate_records
from .metadata import atomic_json, sha256_file
from .model import load_checkpoint_model
from .training import resolve_device


class CaspNpzDataset(Dataset):
    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as data:
            self.records = []
            index = 0
            while f"input_ids_{index}" in data.files:
                input_ids = data[f"input_ids_{index}"].astype(np.int64)
                mask = data[f"attention_mask_{index}"].astype(np.bool_)
                coordinates = np.zeros((len(input_ids), 3), dtype=np.float32)
                raw_coords = data[f"coords_ca_{index}"].astype(np.float32)
                coordinates[: len(raw_coords)] = raw_coords
                self.records.append(
                    {
                        "example_id": str(data[f"example_id_{index}"]),
                        "input_ids": torch.from_numpy(input_ids),
                        "attention_mask": torch.from_numpy(mask),
                        "coords_ca": torch.from_numpy(coordinates),
                    }
                )
                index += 1

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        return {
            **record,
            "pssm_log_probs": torch.zeros((len(record["input_ids"]), 20)),
            "dense_mi_target": torch.zeros((len(record["input_ids"]), len(record["input_ids"]))),
            "distance_target": torch.zeros((len(record["input_ids"]), len(record["input_ids"]))),
            "three_di_ids": torch.full((len(record["input_ids"]),), 20),
            "has_pssm": torch.tensor(False),
            "has_mi": torch.tensor(False),
            "has_structure": torch.tensor(True),
            "has_3di": torch.tensor(False),
        }


class MiniFoldProbe(nn.Module):
    """Frozen-embedding MiniFold probe with low-rank pair and CA heads.

    This is the production probe's portable low-rank form: PLM embeddings are
    never updated; a residue trunk creates low-rank pair scores plus centered CA
    coordinates. Pair distance/contact and coordinate-distance losses provide
    the same diagnostic readouts without a dense pair-state tensor.
    """

    def __init__(self, plm_dim: int, single_dim: int = 128, pair_rank: int = 64, blocks: int = 2):
        super().__init__()
        self.adapter = nn.Sequential(nn.LayerNorm(plm_dim), nn.Linear(plm_dim, single_dim), nn.GELU())
        layer = nn.TransformerEncoderLayer(
            d_model=single_dim,
            nhead=4,
            dim_feedforward=single_dim * 4,
            activation="gelu",
            dropout=0.05,
            batch_first=True,
            norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(layer, blocks)
        self.distance_left = nn.Linear(single_dim, pair_rank)
        self.distance_right = nn.Linear(single_dim, pair_rank)
        self.contact_left = nn.Linear(single_dim, pair_rank)
        self.contact_right = nn.Linear(single_dim, pair_rank)
        self.coords = nn.Sequential(nn.LayerNorm(single_dim), nn.Linear(single_dim, single_dim), nn.GELU(), nn.Linear(single_dim, 3))
        self.pair_rank = pair_rank

    def _pair(self, hidden: torch.Tensor, left: nn.Linear, right: nn.Linear) -> torch.Tensor:
        values = torch.matmul(left(hidden), right(hidden).transpose(-1, -2)) / math.sqrt(self.pair_rank)
        return (values + values.transpose(-1, -2)) * 0.5

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        single = self.adapter(hidden)
        single = self.trunk(single, src_key_padding_mask=~mask.bool())
        coordinates = self.coords(single).float() * mask.unsqueeze(-1)
        center = coordinates.sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
        return {
            "distance_log": self._pair(single, self.distance_left, self.distance_right),
            "contact_logits": self._pair(single, self.contact_left, self.contact_right),
            "coords_ca": coordinates - center,
        }


def _pair_mask(mask: torch.Tensor, minimum: int = 6) -> torch.Tensor:
    length = mask.shape[1]
    index = torch.arange(length, device=mask.device)
    return mask[:, :, None] & mask[:, None, :] & ((index[:, None] - index[None, :]).abs() >= minimum)


def _loss(output: dict[str, torch.Tensor], coordinates: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    true_distance = torch.cdist(coordinates.float(), coordinates.float())
    predicted_coordinate_distance = torch.cdist(output["coords_ca"].float(), output["coords_ca"].float())
    pair_mask = _pair_mask(mask)
    target_log = torch.log1p(true_distance.clamp(max=64.0))
    distance = F.mse_loss(output["distance_log"][pair_mask], target_log[pair_mask])
    contact_labels = (true_distance[pair_mask] < 8.0).float()
    contact = F.binary_cross_entropy_with_logits(output["contact_logits"][pair_mask], contact_labels)
    geometry = F.smooth_l1_loss(predicted_coordinate_distance[pair_mask], true_distance[pair_mask])
    total = distance + 0.2 * contact + geometry
    return total, {"total": float(total.detach()), "distance": float(distance.detach()), "contact": float(contact.detach()), "coordinate_distance": float(geometry.detach())}


def _kabsch(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted = predicted - predicted.mean(0, keepdim=True)
    target = target - target.mean(0, keepdim=True)
    covariance = predicted.T @ target
    u, _s, vh = torch.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if torch.det(rotation) < 0:
        vh[-1] *= -1
        rotation = vh.T @ u.T
    return predicted @ rotation


@torch.no_grad()
def _evaluate(plm, probe, loader, device: torch.device, label: str) -> dict[str, Any]:
    probe.eval()
    rows = []
    for batch in loader:
        input_ids, mask, true = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["coords_ca"].to(device)
        hidden = plm(input_ids, mask, objectives=set())["hidden"]
        output = probe(hidden, mask)
        for index, example_id in enumerate(batch["example_id"]):
            valid = mask[index].bool()
            predicted = output["coords_ca"][index, valid].float()
            target = true[index, valid].float()
            aligned = _kabsch(predicted, target)
            error = (aligned - (target - target.mean(0, keepdim=True))).norm(dim=-1)
            true_distance = torch.cdist(target, target)
            predicted_distance = torch.cdist(predicted, predicted)
            compact_mask = torch.ones((1, len(target)), dtype=torch.bool, device=device)
            pair = _pair_mask(compact_mask)[0]
            contact = true_distance[pair] < 8.0
            compact_logits = output["contact_logits"][index][valid][:, valid]
            logits = compact_logits[pair]
            k = min(len(target), logits.numel())
            precision = float(contact[torch.topk(logits, k).indices].float().mean()) if k else 0.0
            rows.append(
                {
                    "dataset": label,
                    "example_id": example_id,
                    "length": len(target),
                    "rmsd": float(torch.sqrt((error * error).mean())),
                    "lddt_proxy": float((error < 4.0).float().mean()),
                    "tm_score_proxy": float((1.0 / (1.0 + (error / 4.0) ** 2)).mean()),
                    "distance_mae": float((predicted_distance[pair] - true_distance[pair]).abs().mean()) if pair.any() else 0.0,
                    "contact_precision_top_l": precision,
                }
            )
    probe.train()
    numeric = [key for key in rows[0] if key not in {"dataset", "example_id", "length"}] if rows else []
    return {"rows": rows, "summary": {"dataset": label, "examples": len(rows), **{key: float(np.mean([row[key] for row in rows])) for key in numeric}}}


def train_minifold_probe(
    checkpoint: str | Path,
    coordinate_root: str | Path,
    output_dir: str | Path,
    *,
    casp15_data: str | Path | None = None,
    device_name: str = "auto",
    steps: int = 2,
    batch_size: int = 1,
    single_dim: int = 64,
    pair_rank: int = 32,
    blocks: int = 1,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    plm, _payload = load_checkpoint_model(str(checkpoint), device)
    for parameter in plm.parameters():
        parameter.requires_grad_(False)
    plm_dim = int(plm.preset.d_model)
    probe = MiniFoldProbe(plm_dim, single_dim, pair_rank, blocks).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=2e-4)
    root = Path(coordinate_root)
    train_set = CoordinateDataset(root, "train")
    test_set = CoordinateDataset(root, "test")
    if not train_set or not test_set:
        raise ValueError("MiniFold requires non-empty structure-backed train and test splits")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_records)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_records)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    iterator = iter(train_loader)
    started = time.perf_counter()
    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        input_ids, mask, coordinates = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["coords_ca"].to(device)
        with torch.no_grad():
            hidden = plm(input_ids, mask, objectives=set())["hidden"]
        prediction = probe(hidden, mask)
        loss, metrics = _loss(prediction, coordinates, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append({"step": step, "elapsed_seconds": time.perf_counter() - started, **metrics})
    evaluations = [_evaluate(plm, probe, test_loader, device, "pdb_test")]
    if casp15_data:
        casp_path = Path(casp15_data)
        casp_set = CaspNpzDataset(casp_path) if casp_path.is_file() else CoordinateDataset(casp_path, "casp15")
        casp_loader = DataLoader(casp_set, batch_size=batch_size, collate_fn=collate_records)
        evaluations.append(_evaluate(plm, probe, casp_loader, device, "casp15_replication"))
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    all_rows = [row for evaluation in evaluations for row in evaluation["rows"]]
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    checkpoint_path = output / "final.pt"
    torch.save({"model_state_dict": probe.state_dict(), "config": {"plm_dim": plm_dim, "single_dim": single_dim, "pair_rank": pair_rank, "blocks": blocks}}, checkpoint_path)
    summary = {
        "optimizer_steps": steps,
        "trainable_parameters": sum(parameter.numel() for parameter in probe.parameters()),
        "plm_frozen": all(not parameter.requires_grad for parameter in plm.parameters()),
        "evaluations": [evaluation["summary"] for evaluation in evaluations],
        "head_checkpoint_sha256": sha256_file(checkpoint_path),
    }
    atomic_json(output / "summary.json", summary)
    return summary
