from __future__ import annotations

import gzip
import string
from pathlib import Path
from typing import Iterator

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: i for i, aa in enumerate(AA)}
ID_TO_AA = dict(enumerate(AA))
UNK_TOKEN_ID = 20
PAD_TOKEN_ID = 21
MASK_TOKEN_ID = 22
VOCAB_SIZE = 23
THREE_DI_PAD_ID = 20
THREE_DI_VOCAB_SIZE = 23
_INSERTION_DELETE = str.maketrans("", "", string.ascii_lowercase + ".")


def query_to_ids(sequence: str) -> list[int]:
    return [AA_TO_ID.get(ch, UNK_TOKEN_ID) for ch in sequence.upper()]


def ids_to_query(ids: list[int]) -> str:
    return "".join(ID_TO_AA.get(int(token), "X") for token in ids)


def strip_a3m_insertions(sequence: str) -> str:
    return sequence.translate(_INSERTION_DELETE)


def normalize_msa_rows(sequences: list[str]) -> list[str]:
    if not sequences:
        return []
    stripped = [strip_a3m_insertions(seq).upper() for seq in sequences]
    length = len(stripped[0])
    return [seq for seq in stripped if length and len(seq) == length]


def iter_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else source.open
    header: str | None = None
    parts: list[str] = []
    with opener("rt", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header, parts = line[1:], []
            else:
                parts.append(line)
    if header is not None:
        yield header, "".join(parts)


def read_a3m(path: str | Path) -> list[str]:
    return [sequence for _header, sequence in iter_fasta(path)]
