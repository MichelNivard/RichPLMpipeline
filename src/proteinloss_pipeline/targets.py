from __future__ import annotations

import math

import numpy as np
import torch

from .alphabet import AA_TO_ID

GAP_STATE = 20
STATE_COUNT = 21
_ASCII_LUT = np.full(256, -1, dtype=np.int16)
for _aa, _index in AA_TO_ID.items():
    _ASCII_LUT[ord(_aa)] = _index
    _ASCII_LUT[ord(_aa.lower())] = _index
_ASCII_LUT[ord("-")] = GAP_STATE


def msa_to_states(msa: list[str], max_depth: int = 0) -> np.ndarray:
    rows = msa[:max_depth] if max_depth > 0 else msa
    if not rows:
        raise ValueError("MSA is empty")
    length = len(rows[0])
    if length == 0 or any(len(row) != length for row in rows):
        raise ValueError("MSA rows must have the same non-zero aligned length")
    encoded = np.frombuffer("".join(rows).encode("ascii", "replace"), dtype=np.uint8)
    return _ASCII_LUT[encoded].reshape(len(rows), length)


def msa_to_pssm(msa: list[str], pseudocount: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    states = msa_to_states(msa)
    amino_acids = np.arange(20, dtype=np.int16)
    counts = (states[:, :, None] == amino_acids).sum(axis=0, dtype=np.float64) + pseudocount
    probabilities = counts / counts.sum(axis=-1, keepdims=True)
    gaps = (states == GAP_STATE).sum(axis=0) / np.maximum((states >= 0).sum(axis=0), 1)
    return np.log(np.clip(probabilities, 1e-9, 1.0)).astype(np.float32), gaps.astype(np.float32)


def _resolve_mi_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@torch.no_grad()
def msa_to_mi_apc(
    msa: list[str],
    *,
    max_depth: int = 2048,
    pseudocount: float = 1e-3,
    device: str = "auto",
    block_size: int = 32,
) -> np.ndarray:
    """Exact unweighted 21-state MI using bounded position blocks.

    Dense MI exists only in this reduction function. Callers immediately
    compress it and must not retain the dense matrix in production records.
    The blocked implementation works with CPU PyTorch, CUDA, and ROCm.
    """

    states_np = msa_to_states(msa, max_depth=max_depth)
    target_device = _resolve_mi_device(device)
    states = torch.as_tensor(states_np, dtype=torch.long, device=target_device)
    valid = (states >= 0) & (states < STATE_COUNT)
    safe = states.clamp(0, STATE_COUNT - 1)
    one_hot = torch.nn.functional.one_hot(safe, STATE_COUNT).to(torch.float32) * valid.unsqueeze(-1)
    counts = one_hot.sum(dim=0)
    valid_count = valid.sum(dim=0).to(torch.float32)
    marginals = (counts + pseudocount) / (valid_count[:, None] + pseudocount * STATE_COUNT)
    length = states.shape[1]
    pair = torch.zeros((length, length), dtype=torch.float32, device=target_device)
    valid_f = valid.to(torch.float32)
    epsilon = 1e-12
    for i0 in range(0, length, block_size):
        i1 = min(length, i0 + block_size)
        left = one_hot[:, i0:i1]
        left_valid = valid_f[:, i0:i1]
        for j0 in range(i0, length, block_size):
            j1 = min(length, j0 + block_size)
            right = one_hot[:, j0:j1]
            joint_counts = torch.einsum("dia,djb->ijab", left, right)
            pair_valid = torch.matmul(left_valid.transpose(0, 1), valid_f[:, j0:j1])
            joint = (joint_counts + pseudocount) / (
                pair_valid[:, :, None, None] + pseudocount * STATE_COUNT * STATE_COUNT
            )
            independent = marginals[i0:i1, None, :, None] * marginals[None, j0:j1, None, :]
            values = (joint * torch.log(joint.clamp_min(epsilon) / independent.clamp_min(epsilon))).sum((-1, -2))
            pair[i0:i1, j0:j1] = values
            if j0 != i0:
                pair[j0:j1, i0:i1] = values.transpose(0, 1)
    pair.fill_diagonal_(0.0)
    row_mean = pair.mean(dim=1)
    global_mean = pair.mean()
    if torch.isfinite(global_mean) and global_mean.abs() > 1e-8:
        pair -= torch.outer(row_mean, row_mean) / global_mean
    pair = (pair + pair.transpose(0, 1)) * 0.5
    pair.fill_diagonal_(0.0)
    off_diagonal = pair[~torch.eye(length, dtype=torch.bool, device=target_device)]
    standard_deviation = off_diagonal.std(unbiased=False)
    if standard_deviation > 1e-6:
        pair = (pair - off_diagonal.mean()) / standard_deviation
        pair.fill_diagonal_(0.0)
    return pair.cpu().numpy().astype(np.float32)


def pad_core(
    sequence_ids: np.ndarray,
    pssm_log_probs: np.ndarray,
    gap_fraction: np.ndarray,
    max_len: int,
) -> dict[str, np.ndarray]:
    length = min(len(sequence_ids), max_len)
    input_ids = np.full(max_len, 21, dtype=np.int16)
    attention = np.zeros(max_len, dtype=np.bool_)
    pssm = np.full((max_len, 20), -math.log(20.0), dtype=np.float16)
    gaps = np.ones(max_len, dtype=np.float16)
    input_ids[:length] = sequence_ids[:length]
    attention[:length] = True
    pssm[:length] = pssm_log_probs[:length].astype(np.float16)
    gaps[:length] = gap_fraction[:length].astype(np.float16)
    return {"input_ids": input_ids, "attention_mask": attention, "pssm_log_probs": pssm, "gap_fraction": gaps}


def validate_pair_target(matrix: np.ndarray, atol: float = 2e-4) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"pair target must be square, got {matrix.shape}")
    if not np.allclose(matrix, matrix.T, atol=atol):
        raise ValueError("pair target is not symmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=atol):
        raise ValueError("pair target diagonal is not zero")
