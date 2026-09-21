from dataclasses import replace

import pytest
import torch

from qmla.model import CNNReplacement, GalaxyClassifier, QuFeXLayer, residual_for_filters


@pytest.mark.parametrize("mode", ["qufex", "cnn_replacement", "direct_cnn"])
@pytest.mark.parametrize("qubits,filters", [(8, 1), (4, 1), (4, 2)])
def test_forward_backward_all_variants(tiny_config, mode, qubits, filters):
    torch.set_num_threads(2)
    config = replace(tiny_config.for_model(mode), quantum=replace(tiny_config.quantum, qubits=qubits, filters=filters))
    model = GalaxyClassifier(config)
    logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 3)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert next(model.encoder.parameters()).grad.abs().sum() > 0
    assert model.quantum_parameter_count == (filters * 4 if mode == "qufex" else 0)


def test_qufex_grouping_and_residual_order():
    layer = QuFeXLayer()
    layer._circuits = (lambda inputs, _theta: inputs,)
    features = torch.arange(64.).reshape(1, 16, 2, 2)
    assert torch.equal(layer(features), features)
    layer4 = QuFeXLayer(qubits=4, filters=2)
    layer4._circuits = (lambda x, _: x, lambda x, _: x + 100)
    result = layer4(features)
    assert torch.equal(result[:, ::2], features)
    assert torch.equal(result[:, 1::2], features + 100)
    residual = residual_for_filters(features, 2)
    assert torch.equal(residual[:, ::2], features) and torch.equal(residual[:, 1::2], features)


@pytest.mark.parametrize("qubits,filters", [(8, 1), (4, 1), (4, 2)])
def test_replacement_shares_weights_without_crossing_groups(tiny_config, qubits, filters):
    layer = CNNReplacement(tiny_config.architectures.replacement, qubits, filters)
    x = torch.randn(2, 16, 2, 2)
    group_size = 2 if qubits == 8 else 1
    permutation = torch.arange(16 // group_size).flip(0)
    permuted = x.reshape(2, -1, group_size, 2, 2)[:, permutation].reshape_as(x)
    expected = layer(x).reshape(2, -1, group_size * filters, 2, 2)[:, permutation]
    assert torch.allclose(layer(permuted), expected.reshape(2, 16 * filters, 2, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("mode", ["qufex", "cnn_replacement", "direct_cnn"])
def test_cuda_autocast_backward(tiny_config, mode):
    model = GalaxyClassifier(tiny_config, mode=mode).cuda()
    with torch.autocast("cuda"):
        logits = model(torch.randn(2, 3, 32, 32, device="cuda"))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1], device="cuda"))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
