from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np


@dataclass(slots=True)
class MICompressedTarget:
    length: int
    fft_top: np.ndarray
    fft_bottom: np.ndarray
    block32: np.ndarray
    diagline: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    fft_k: int = 4
    block_grid: int = 32
    eig_k: int = 3


def prepare_mi_matrix(mi: np.ndarray) -> np.ndarray:
    array = np.asarray(mi, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"MI matrix must be square, got {array.shape}")
    array = (array + array.T) * 0.5
    np.fill_diagonal(array, 0.0)
    return array


def block_edges(length: int, grid: int = 32) -> np.ndarray:
    return np.linspace(0, length, grid + 1, dtype=np.int32)


def block_index(length: int, grid: int = 32) -> np.ndarray:
    edges = block_edges(length, grid)
    bins = np.searchsorted(edges[1:-1], np.arange(length), side="right")
    return (bins[:, None] * grid + bins[None, :]).astype(np.int32)


def diagonal_index(length: int) -> np.ndarray:
    idx = np.arange(length, dtype=np.int32)
    return np.abs(idx[:, None] - idx[None, :])


def _fft_encode(array: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    coeff = np.fft.rfft2(array)
    top = coeff[:k, :k].astype(np.complex64)
    bottom = coeff[-(k - 1) :, :k].astype(np.complex64) if k > 1 else np.empty((0, k), np.complex64)
    return top, bottom


def _fft_decode(length: int, top: np.ndarray, bottom: np.ndarray, k: int) -> np.ndarray:
    coeff = np.zeros((length, length // 2 + 1), dtype=np.complex64)
    coeff[:k, :k] = top
    if k > 1:
        coeff[-(k - 1) :, :k] = bottom
    return np.fft.irfft2(coeff, s=(length, length)).astype(np.float32)


def _block_encode(array: np.ndarray, grid: int) -> np.ndarray:
    edges = block_edges(array.shape[0], grid)
    out = np.zeros((grid, grid), dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            block = array[edges[i] : edges[i + 1], edges[j] : edges[j + 1]]
            out[i, j] = float(block.mean()) if block.size else 0.0
    return out.astype(np.float16)


def _block_decode(values: np.ndarray, length: int, index: np.ndarray | None = None) -> np.ndarray:
    index = block_index(length, values.shape[0]) if index is None else index
    return values.astype(np.float32).reshape(-1)[index]


def _diag_encode(array: np.ndarray) -> np.ndarray:
    return np.asarray([np.diag(array, k=offset).mean() for offset in range(array.shape[0])], dtype=np.float16)


def _diag_decode(values: np.ndarray, index: np.ndarray | None = None) -> np.ndarray:
    index = diagonal_index(len(values)) if index is None else index
    return values.astype(np.float32)[index]


def encode_mi_target(mi: np.ndarray, fft_k: int = 4, block_grid: int = 32, eig_k: int = 3) -> MICompressedTarget:
    array = prepare_mi_matrix(mi)
    length = array.shape[0]
    fft_top, fft_bottom = _fft_encode(array, fft_k)
    fft = _fft_decode(length, fft_top, fft_bottom, fft_k)
    coarse = _block_encode(array - fft, block_grid)
    block = _block_decode(coarse, length)
    diag = _diag_encode(array - fft - block)
    residual = array - fft - block - _diag_decode(diag)
    vals, vecs = np.linalg.eigh(residual.astype(np.float64))
    order = np.argsort(np.abs(vals))[::-1][:eig_k]
    return MICompressedTarget(
        length=length,
        fft_top=fft_top,
        fft_bottom=fft_bottom,
        block32=coarse,
        diagline=diag,
        eigvals=vals[order].astype(np.float16),
        eigvecs=vecs[:, order].astype(np.float16),
        fft_k=fft_k,
        block_grid=block_grid,
        eig_k=eig_k,
    )


def decode_mi_target(
    target: MICompressedTarget,
    cached_block_index: np.ndarray | None = None,
    cached_diagonal_index: np.ndarray | None = None,
) -> np.ndarray:
    vecs = target.eigvecs.astype(np.float32)
    eigen = (vecs * target.eigvals.astype(np.float32)) @ vecs.T
    decoded = (
        _fft_decode(target.length, target.fft_top, target.fft_bottom, target.fft_k)
        + _block_decode(target.block32, target.length, cached_block_index)
        + _diag_decode(target.diagline, cached_diagonal_index)
        + eigen
    )
    decoded = (decoded + decoded.T) * 0.5
    np.fill_diagonal(decoded, 0.0)
    return decoded.astype(np.float32)


def target_to_arrays(target: MICompressedTarget) -> dict[str, np.ndarray]:
    return {
        "mi_length": np.asarray(target.length, dtype=np.int32),
        "mi_fft_top": target.fft_top,
        "mi_fft_bottom": target.fft_bottom,
        "mi_block32": target.block32,
        "mi_diagline": target.diagline,
        "mi_eigvals": target.eigvals,
        "mi_eigvecs": target.eigvecs,
    }


def target_from_arrays(data: dict[str, np.ndarray]) -> MICompressedTarget:
    return MICompressedTarget(
        length=int(data["mi_length"]),
        fft_top=data["mi_fft_top"],
        fft_bottom=data["mi_fft_bottom"],
        block32=data["mi_block32"],
        diagline=data["mi_diagline"],
        eigvals=data["mi_eigvals"],
        eigvecs=data["mi_eigvecs"],
    )


def encode_ca_coords(coordinates: np.ndarray) -> np.ndarray:
    array = np.asarray(coordinates, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"CA coordinates must have shape [L,3], got {array.shape}")
    return array.astype(np.float16)


def distance_log_target(coordinates_fp16: np.ndarray, max_distance: float = 64.0) -> np.ndarray:
    coordinates = np.asarray(coordinates_fp16, dtype=np.float32)
    difference = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.maximum(np.sum(difference * difference, axis=-1), 0.0))
    out = np.log1p(np.minimum(distances, max_distance)).astype(np.float32)
    out = (out + out.T) * 0.5
    np.fill_diagonal(out, 0.0)
    return out


def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    handle = BytesIO()
    np.savez_compressed(handle, **arrays)
    return handle.getvalue()


def arrays_from_npz_bytes(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(BytesIO(payload), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
