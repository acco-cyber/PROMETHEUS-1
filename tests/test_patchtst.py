from __future__ import annotations

import torch
from transformers import PatchTSTForPrediction

from sv1.models.patchtst import _model_config


def test_official_patchtst_forward_shape() -> None:
    spec = {
        "patch_length": 8,
        "patch_stride": 4,
        "layers": 1,
        "d_model": 16,
        "heads": 4,
        "ffn_dim": 32,
        "dropout": 0.0,
    }
    model = PatchTSTForPrediction(_model_config(32, 7, spec))
    output = model(past_values=torch.randn(3, 32, 1), future_values=torch.randn(3, 7, 1))
    assert output.prediction_outputs.shape == (3, 7, 1)
    assert torch.isfinite(output.loss)
