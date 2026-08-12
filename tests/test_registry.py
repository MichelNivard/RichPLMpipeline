import hashlib

import yaml

from proteinloss_pipeline.registry import preflight


def test_preflight_reports_a_configured_checksum_mismatch(tmp_path):
    source = tmp_path / "source.dat"
    source.write_bytes(b"proteinloss")
    registry = tmp_path / "sources.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "sources": {
                    "input": {
                        "path": str(source),
                        "kind": "file",
                        "sha256": "0" * 64,
                        "required_for": ["test"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    result = preflight(registry, capabilities={"test"})
    assert not result["ok"]
    assert "SHA-256 mismatch" in result["missing"][0]


def test_preflight_accepts_a_matching_checksum(tmp_path):
    source = tmp_path / "source.dat"
    source.write_bytes(b"proteinloss")
    registry = tmp_path / "sources.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "sources": {
                    "input": {
                        "path": str(source),
                        "kind": "file",
                        "sha256": hashlib.sha256(b"proteinloss").hexdigest(),
                        "required_for": ["test"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    assert preflight(registry, capabilities={"test"})["ok"]
