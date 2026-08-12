from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class StructEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        block_size: int,
        d_model: int,
        latent_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        normalize_latent: bool = True,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.block_size = block_size
        self.normalize_latent = normalize_latent
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.to_latent = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Linear(d_model, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Linear(d_model, vocab_size)
        )

    def encode(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        position = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.tok_emb(tokens) + self.pos_emb(position).unsqueeze(0)
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask.bool())
        latent = self.to_latent(self.norm(hidden))
        return F.normalize(latent, dim=-1) if self.normalize_latent else latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


@dataclass(frozen=True, slots=True)
class TeacherInfo:
    pad_id: int
    vocab_size: int
    latent_dim: int
    block_size: int
    tokens: tuple[str, ...]


def load_teacher(path: str | Path, device: torch.device | str = "cpu") -> tuple[StructEncoder, TeacherInfo]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    tokens = tuple(str(token) for token in payload.get("tokens", list("ACDEFGHIKLMNPQRSTVWY")))
    pad_id = int(payload.get("pad_id", len(tokens)))
    model = StructEncoder(
        vocab_size=len(tokens) + 3,
        pad_id=pad_id,
        block_size=int(config["block_size"]),
        d_model=int(config["d_model"]),
        latent_dim=int(config["latent_dim"]),
        layers=int(config.get("n_layer", config.get("layers", 12))),
        heads=int(config.get("n_head", config.get("heads", 8))),
        dropout=float(config.get("dropout", 0.1)),
        normalize_latent=bool(config.get("normalize_latent", True)),
    )
    state = payload.get("model") or payload.get("model_state_dict")
    model.load_state_dict({key.removeprefix("_orig_mod."): value for key, value in state.items()})
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, TeacherInfo(pad_id, len(tokens) + 3, int(config["latent_dim"]), int(config["block_size"]), tokens)


@torch.no_grad()
def encode_teacher_blocks(
    teacher: StructEncoder,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    batch, length = token_ids.shape
    latent_dim = int(teacher.to_latent[-1].out_features)
    output = torch.zeros((batch, length, latent_dim), dtype=torch.float32, device=token_ids.device)
    for start in range(0, length, teacher.block_size):
        width = min(teacher.block_size, length - start)
        tokens = torch.full((batch, teacher.block_size), teacher.pad_id, dtype=torch.long, device=token_ids.device)
        mask = torch.zeros((batch, teacher.block_size), dtype=torch.bool, device=token_ids.device)
        tokens[:, :width] = token_ids[:, start : start + width]
        mask[:, :width] = attention_mask[:, start : start + width]
        if mask.any():
            output[:, start : start + width] = teacher.encode(tokens, mask)[:, :width].float()
    return output


def create_smoke_teacher(path: str | Path, seed: int = 7) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    config = {
        "block_size": 32,
        "d_model": 32,
        "latent_dim": 128,
        "n_layer": 1,
        "n_head": 4,
        "dropout": 0.0,
        "normalize_latent": True,
    }
    tokens = list("ACDEFGHIKLMNPQRSTVWY")
    model = StructEncoder(len(tokens) + 3, len(tokens), 32, 32, 128, 1, 4, 0.0, True)
    torch.save({"model": model.state_dict(), "config": config, "tokens": tokens, "pad_id": 20}, destination)
    return destination
