"""Contract tests for the pinned RangeFormer Swin-T re-implementation."""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.models.rangeformer_adapter import (  # noqa: E402
    CONVENTIONAL_CHANNELS, build_rangeformer,
)


def test_rangeformer_uses_fixed_five_channel_input():
    assert CONVENTIONAL_CHANNELS == (8, 9, 10, 0, 1)
    model = build_rangeformer()
    assert not hasattr(model, "input_proj")


def test_rangeformer_output_and_parameter_budget():
    model = build_rangeformer().eval()
    with torch.no_grad():
        output = model(torch.randn(1, 16, 64, 512))
    assert output.shape == (1, 9, 64, 512)
    params_m = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    assert params_m == pytest.approx(38.2, abs=0.15)
