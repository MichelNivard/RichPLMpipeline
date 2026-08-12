import json
from pathlib import Path

import pytest

from proteinloss_pipeline.data_build import build_smoke_dataset
from proteinloss_pipeline.coordinates import CoordinateDataset
from proteinloss_pipeline.dataset import PackedProteinDataset
from proteinloss_pipeline.manifest import ManifestReader


@pytest.mark.integration
def test_smoke_build_is_complete_mixed_and_clean(tmp_path):
    config = {
        "seed": 7,
        "paths": {
            "data_root": str(tmp_path / "data"),
            "run_root": str(tmp_path / "runs"),
            "scratch_root": str(tmp_path / "scratch"),
        },
        "data": {
            "backend": "smoke",
            "count": 9,
            "min_len": 12,
            "max_len": 20,
            "msa_depth": 6,
            "records_per_shard": 3,
            "valid_fraction": 0.2,
            "test_fraction": 0.2,
            "cleanup_scratch": True,
            "mixed_sequence_fraction": 0.2,
        },
    }
    summary = build_smoke_dataset(config, project_root=Path(__file__).parents[1])
    assert summary["written"] == 9
    assert summary["transient_a3m_removed"]
    assert summary["mixed_supervision"]["structure_backed"] > 0
    assert summary["mixed_supervision"]["sequence_only"] > 0
    reader = ManifestReader(tmp_path / "data" / "manifest.sqlite")
    assert all(reader.validate().values())
    reader.close()
    item = PackedProteinDataset(tmp_path / "data", tmp_path / "data" / "manifest.sqlite", "train")[0]
    assert item["input_ids"].shape == (20,)
    assert item["dense_mi_target"].shape == (20, 20)
    assert (tmp_path / "data" / "COMPLETED").is_file()
    coordinates = tmp_path / "data" / "validation" / "pdb_coordinates"
    assert CoordinateDataset(coordinates, "train")
    assert CoordinateDataset(coordinates, "test")
    assert not list((tmp_path / "scratch").glob("proteinloss-*"))
