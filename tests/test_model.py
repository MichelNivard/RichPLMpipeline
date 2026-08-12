import torch

from proteinloss_pipeline.model import (
    EXPECTED_PARAMETER_COUNTS,
    ModelPreset,
    ProteinTransformer,
    load_checkpoint_model,
    preset_parameter_count,
)


def test_exact_parameter_counts_for_documented_presets():
    assert preset_parameter_count("100m") == EXPECTED_PARAMETER_COUNTS["100m"] == 115_669_163
    assert preset_parameter_count("300m") == EXPECTED_PARAMETER_COUNTS["300m"] == 305_588_395


def test_checkpoint_reload_roundtrip(tmp_path):
    preset = ModelPreset("test", 16, 32, 2, 4, 64, 8, dropout=0.0)
    model = ProteinTransformer(preset).eval()
    checkpoint = tmp_path / "tiny.pt"
    config = {"model": preset.to_dict()}
    torch.save({"model_state_dict": model.state_dict(), "resolved_config": config}, checkpoint)
    restored, _payload = load_checkpoint_model(str(checkpoint))
    inputs = torch.arange(16).remainder(20).unsqueeze(0)
    mask = torch.ones_like(inputs, dtype=torch.bool)
    with torch.no_grad():
        expected = model(inputs, mask, objectives={"dense_mi", "distance", "contact", "struct_latent", "3di"})
        actual = restored(inputs, mask, objectives={"dense_mi", "distance", "contact", "struct_latent", "3di"})
    for key in expected:
        assert torch.equal(expected[key], actual[key])
