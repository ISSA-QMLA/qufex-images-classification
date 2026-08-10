from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from qmla.config import load_config


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="machine-specific PyTorch is absent")


def test_classical_and_qufex_forward_and_gradient(tmp_path: Path) -> None:
    import torch

    from qmla.model import GalaxyClassifier

    default = Path(__file__).parents[1] / "configs" / "default.toml"
    text = default.read_text(encoding="utf-8")
    text = text.replace('project_root = ".."', f"project_root = {json.dumps(str(tmp_path))}")
    text = text.replace("image_size = 128", "image_size = 32")
    text = text.replace("encoder_channels = [16, 32, 64, 32]", "encoder_channels = [4, 4, 4, 8]")
    text = text.replace("post_quantum_channels = [16]", "post_quantum_channels = [4]")
    text = text.replace("classifier_hidden_neurons = [32]", "classifier_hidden_neurons = [4]")
    text = text.replace('device = "cuda"', 'device = "cpu"')
    config_path = tmp_path / "model.toml"
    config_path.write_text(text, encoding="utf-8")
    config = load_config(config_path)

    inputs = torch.randn(2, 3, 32, 32)
    for mode, expected_quantum_parameters in (("classical", 0), ("qufex", 4)):
        model = GalaxyClassifier(config, mode=mode)
        logits = model(inputs)
        assert logits.shape == (2, 3)
        assert model.quantum_parameter_count == expected_quantum_parameters
        loss = logits.square().mean()
        loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())

        checkpoint = tmp_path / f"{mode}.pt"
        torch.save(model.state_dict(), checkpoint)
        restored = GalaxyClassifier(config, mode=mode)
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        assert restored(inputs).shape == (2, 3)


def test_default_qubit_broadcast_runs_on_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    from qmla.model import QuFeXLayer

    layer = QuFeXLayer(backend="default.qubit", diff_method="backprop").cuda()
    inputs = torch.randn(2, 8, 2, 2, device="cuda", requires_grad=True)
    outputs = layer(inputs)

    assert outputs.shape == inputs.shape
    assert outputs.device == inputs.device
    outputs.square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert layer.theta.grad is not None and torch.isfinite(layer.theta.grad).all()

