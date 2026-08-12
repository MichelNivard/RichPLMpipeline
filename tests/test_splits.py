from collections import Counter

import pytest

from proteinloss_pipeline.splits import stable_split, validate_split_counts


def test_stable_group_split_prevents_family_leakage():
    assignments = {
        stable_split(f"accession-{index}", group_id="same-family", seed=7, valid_fraction=0.1, test_fraction=0.1)
        for index in range(100)
    }
    assert len(assignments) == 1
    assert stable_split("x", seed=3, valid_fraction=0.1, test_fraction=0.1) == stable_split(
        "x", seed=3, valid_fraction=0.1, test_fraction=0.1
    )


def test_empty_required_split_is_rejected():
    with pytest.raises(ValueError, match="test"):
        validate_split_counts(Counter(train=10, valid=1, test=0))
