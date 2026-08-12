import numpy as np

from proteinloss_pipeline.coordinates import CoordinateDataset, write_coordinate_fixture


def test_experimental_coordinate_manifest_uses_relative_shards(tmp_path):
    records = []
    for index, split in enumerate(("train", "valid", "test")):
        length = 8 + index
        records.append(
            {
                "example_id": f"test_{index}",
                "pdb_id": f"x{index:03d}",
                "group_id": f"family_{index}",
                "split": split,
                "sequence": "ACDEFGHI" + "K" * index,
                "coords_ca": np.arange(length * 3, dtype=np.float32).reshape(length, 3),
            }
        )
    root = tmp_path / "coordinates"
    write_coordinate_fixture(root, records, 16)
    row = CoordinateDataset(root, "test")[0]
    assert row["coords_ca"].shape == (16, 3)
    assert row["attention_mask"].sum().item() == 10
    import sqlite3
    connection = sqlite3.connect(root / "manifest.sqlite")
    shard = connection.execute("SELECT shard_path FROM examples LIMIT 1").fetchone()[0]
    connection.close()
    assert not shard.startswith("/")
