from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .alphabet import THREE_DI_VOCAB_SIZE, VOCAB_SIZE


@dataclass(frozen=True, slots=True)
class ModelPreset:
    name: str
    max_len: int
    d_model: int
    layers: int
    heads: int
    feedforward: int
    pair_rank: int
    dropout: float = 0.1
    struct_latent_dim: int = 128
    three_di_vocab_size: int = THREE_DI_VOCAB_SIZE
    pair_relpos_bias: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


MODEL_PRESETS = {
    "100m": ModelPreset("100m", 512, 768, 16, 12, 3072, 128),
    "300m": ModelPreset("300m", 512, 1024, 24, 16, 4096, 128),
}

# Full checkpoint-compatible state schema, including all current heads. The
# compatibility-only sparse pair and amino-acid-pair tensors are not trainable
# objectives in this portable interface but permit established checkpoints to
# load without architectural surgery.
EXPECTED_PARAMETER_COUNTS = {"100m": 115_669_163, "300m": 305_588_395}


class ProteinTransformer(nn.Module):
    def __init__(self, preset: ModelPreset):
        super().__init__()
        self.preset = preset
        self.max_len = preset.max_len
        self.pair_rank = preset.pair_rank
        self.struct_latent_dim = preset.struct_latent_dim
        self.three_di_vocab_size = preset.three_di_vocab_size
        self.pair_relpos_bias = preset.pair_relpos_bias
        self.token_emb = nn.Embedding(VOCAB_SIZE, preset.d_model)
        self.pos_emb = nn.Embedding(preset.max_len, preset.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=preset.d_model,
            nhead=preset.heads,
            dim_feedforward=preset.feedforward,
            dropout=preset.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=preset.layers)
        self.norm = nn.LayerNorm(preset.d_model)
        self.pssm_head = nn.Linear(preset.d_model, 20)
        self.left_head = nn.Linear(preset.d_model, preset.pair_rank)
        self.right_head = nn.Linear(preset.d_model, preset.pair_rank)
        self.contact_left_head = nn.Linear(preset.d_model, preset.pair_rank)
        self.contact_right_head = nn.Linear(preset.d_model, preset.pair_rank)
        self.dense_mi_left_head = nn.Linear(preset.d_model, preset.pair_rank)
        self.dense_mi_right_head = nn.Linear(preset.d_model, preset.pair_rank)
        self.distance_left_head = nn.Linear(preset.d_model, preset.pair_rank)
        self.distance_right_head = nn.Linear(preset.d_model, preset.pair_rank)
        self.distance_relpos_bias = nn.Embedding(preset.max_len, 1) if preset.pair_relpos_bias else None
        self.contact_relpos_bias = nn.Embedding(preset.max_len, 1) if preset.pair_relpos_bias else None
        if self.distance_relpos_bias is not None:
            nn.init.zeros_(self.distance_relpos_bias.weight)
        if self.contact_relpos_bias is not None:
            nn.init.zeros_(self.contact_relpos_bias.weight)
        self.struct_latent_head = nn.Sequential(
            nn.Linear(preset.d_model, preset.d_model),
            nn.GELU(),
            nn.LayerNorm(preset.d_model),
            nn.Linear(preset.d_model, preset.struct_latent_dim),
        )
        self.three_di_head = nn.Linear(preset.d_model, preset.three_di_vocab_size)
        self.aa_pair_emb = nn.Embedding(21 * 21, preset.d_model)

    def _pair(
        self,
        hidden: torch.Tensor,
        left_head: nn.Linear,
        right_head: nn.Linear,
        relative_bias: nn.Embedding | None = None,
    ) -> torch.Tensor:
        left, right = left_head(hidden), right_head(hidden)
        pair = torch.matmul(left, right.transpose(-1, -2)) / math.sqrt(self.pair_rank)
        pair = (pair + pair.transpose(-1, -2)) * 0.5
        length = hidden.shape[1]
        if relative_bias is not None:
            index = torch.arange(length, device=hidden.device)
            separation = (index[:, None] - index[None, :]).abs().clamp(max=self.max_len - 1)
            pair = pair + relative_bias(separation).squeeze(-1).to(pair.dtype).unsqueeze(0)
        diagonal = torch.eye(length, dtype=torch.bool, device=hidden.device)
        return pair.masked_fill(diagonal.unsqueeze(0), 0.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        objectives: frozenset[str] | set[str] = frozenset({"mlm"}),
        return_hidden: bool = True,
    ) -> dict[str, torch.Tensor]:
        length = input_ids.shape[1]
        if length > self.max_len:
            raise ValueError(f"input length {length} exceeds preset max_len {self.max_len}")
        position = torch.arange(length, device=input_ids.device).unsqueeze(0)
        hidden = self.token_emb(input_ids) + self.pos_emb(position)
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask.bool())
        hidden = self.norm(hidden)
        out: dict[str, torch.Tensor] = {"pssm_logits": self.pssm_head(hidden)}
        if return_hidden:
            out["hidden"] = hidden
        if "dense_mi" in objectives:
            out["dense_mi_pred"] = self._pair(hidden, self.dense_mi_left_head, self.dense_mi_right_head)
        if "distance" in objectives:
            out["distance_pred"] = self._pair(
                hidden, self.distance_left_head, self.distance_right_head, self.distance_relpos_bias
            )
        if "contact" in objectives:
            out["contact_pred"] = self._pair(
                hidden, self.contact_left_head, self.contact_right_head, self.contact_relpos_bias
            )
        if "struct_latent" in objectives:
            out["struct_latent_pred"] = self.struct_latent_head(hidden)
        if "3di" in objectives:
            out["three_di_logits"] = self.three_di_head(hidden)
        return out


def get_preset(name: str) -> ModelPreset:
    try:
        return MODEL_PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown model preset {name!r}; choose from {sorted(MODEL_PRESETS)}") from exc


def build_model(name: str, *, device: torch.device | str | None = None) -> ProteinTransformer:
    model = ProteinTransformer(get_preset(name))
    return model.to(device) if device is not None else model


def preset_from_mapping(mapping: dict) -> ModelPreset:
    return ModelPreset(
        name=str(mapping.get("preset", "custom")),
        max_len=int(mapping["max_len"]),
        d_model=int(mapping["d_model"]),
        layers=int(mapping["layers"]),
        heads=int(mapping["heads"]),
        feedforward=int(mapping.get("feedforward", 4 * int(mapping["d_model"]))),
        pair_rank=int(mapping["pair_rank"]),
        dropout=float(mapping.get("dropout", 0.1)),
        struct_latent_dim=int(mapping.get("struct_latent_dim", 128)),
        three_di_vocab_size=int(mapping.get("three_di_vocab_size", THREE_DI_VOCAB_SIZE)),
        pair_relpos_bias=bool(mapping.get("pair_relpos_bias", True)),
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def preset_parameter_count(name: str) -> int:
    with torch.device("meta"):
        model = build_model(name)
    return parameter_count(model)


def strip_compiled_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("_orig_mod."): value for key, value in state.items()}


def load_checkpoint_model(path: str, device: torch.device | str = "cpu") -> tuple[ProteinTransformer, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("resolved_config") or payload.get("config") or {}
    model_mapping = config.get("model") if isinstance(config.get("model"), dict) else None
    model_name = (model_mapping.get("preset") or model_mapping.get("name")) if model_mapping else None
    model_name = model_name or config.get("model_preset") or config.get("preset")
    if not model_name:
        d_model = int(config.get("d_model", 0))
        model_name = "300m" if d_model == 1024 else "100m"
    if model_mapping and str(model_name) not in MODEL_PRESETS:
        model = ProteinTransformer(preset_from_mapping(model_mapping))
    else:
        model = build_model(str(model_name))
    state = payload.get("model_state_dict") or payload.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no model state: {path}")
    missing, unexpected = model.load_state_dict(strip_compiled_prefix(state), strict=False)
    required_missing = [name for name in missing if not name.startswith("three_di_head")]
    if required_missing or unexpected:
        raise ValueError(f"incompatible checkpoint: missing={required_missing[:8]} unexpected={unexpected[:8]}")
    model.to(device).eval()
    return model, payload
