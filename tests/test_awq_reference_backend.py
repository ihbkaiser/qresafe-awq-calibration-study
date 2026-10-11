import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from common import _reference_awq_forward


class _FakeQuantizedLinear:
    out_features = 3
    bias = torch.tensor([0.5, -0.5, 1.0], dtype=torch.float16)
    _qresafe_reference_weight = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=torch.float16,
    )


def test_reference_awq_forward_matches_dense_linear_and_restores_shape():
    module = _FakeQuantizedLinear()
    inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)

    actual = _reference_awq_forward(module, inputs)
    expected = torch.nn.functional.linear(
        inputs,
        module._qresafe_reference_weight.transpose(0, 1).float(),
        module.bias.float(),
    )

    assert actual.shape == (1, 2, 3)
    assert actual.dtype == inputs.dtype
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
