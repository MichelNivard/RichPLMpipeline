import shutil

from proteinloss_pipeline.codec import npz_bytes
from proteinloss_pipeline.manifest import ManifestReader, ManifestRow, ManifestWriter
from proteinloss_pipeline.shards import PackedShardWriter, read_record
from proteinloss_pipeline.splits import stable_split


def test_relative_manifest_survives_data_root_move(tmp_path):
    root = tmp_path / "first"
    root.mkdir()
    with PackedShardWriter(root, root / "shards", records_per_shard=3) as writer:
        refs = [writer.write(f"p{i}", npz_bytes({"x": __import__("numpy").asarray(i)})) for i in range(3)]
    manifest = ManifestWriter(root / "manifest.sqlite")
    for index, split in enumerate(("train", "valid", "test")):
        ref = refs[index]
        manifest.add(
            ManifestRow(
                f"p{index}", f"g{index}", split, ref.shard_path, ref.byte_offset, ref.byte_length,
                ref.record_sha256, 1, True, False, False, False, {"test": True}
            )
        )
    manifest.close()
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    reader = ManifestReader(moved / "manifest.sqlite")
    assert reader.validate() == {"train": 1, "valid": 1, "test": 1}
    for row in reader.iter_rows():
        assert not row.shard_path.startswith("/")
        assert read_record(moved, row.ref, verify=True)
    reader.close()


def test_manifest_rejects_absolute_paths(tmp_path):
    writer = ManifestWriter(tmp_path / "manifest.sqlite")
    try:
        writer.add(ManifestRow("x", "g", "train", "/tmp/x", 0, 1, "a", 1, True, False, False, False, {}))
    except ValueError as error:
        assert "relative" in str(error)
    else:
        raise AssertionError("absolute path was accepted")
