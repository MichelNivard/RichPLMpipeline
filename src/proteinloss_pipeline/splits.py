from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable

REQUIRED_SPLITS = ("train", "valid", "test")


def stable_fraction(value: str, seed: int = 7) -> float:
    digest = hashlib.sha256(f"proteinloss:{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def stable_split(
    accession: str,
    *,
    group_id: str | None = None,
    seed: int = 7,
    valid_fraction: float = 0.05,
    test_fraction: float = 0.05,
) -> str:
    if valid_fraction <= 0 or test_fraction <= 0 or valid_fraction + test_fraction >= 1:
        raise ValueError("valid_fraction and test_fraction must be positive and sum to less than 1")
    key = group_id or accession
    value = stable_fraction(key, seed)
    if value < test_fraction:
        return "test"
    if value < test_fraction + valid_fraction:
        return "valid"
    return "train"


def validate_split_counts(counts: dict[str, int] | Counter[str]) -> None:
    missing = [split for split in REQUIRED_SPLITS if int(counts.get(split, 0)) <= 0]
    if missing:
        raise ValueError(f"required data splits are empty: {', '.join(missing)}")


def split_counts(values: Iterable[str]) -> Counter[str]:
    counts: Counter[str] = Counter(values)
    validate_split_counts(counts)
    return counts
