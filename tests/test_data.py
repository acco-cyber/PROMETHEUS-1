from __future__ import annotations

import numpy as np

from sv1.data.dataset import PreparedDataset, make_range_windows


def test_range_windows_keep_targets_inside_role() -> None:
    raw = np.arange(400, dtype=np.float32)
    data = PreparedDataset(
        name="synthetic",
        domain="test",
        frequency="daily",
        context=32,
        horizon=12,
        seasonal_period=7,
        values_z=(raw,),
        values_raw=(raw,),
        series_names=("one",),
        means=np.asarray([0.0], dtype=np.float32),
        scales=np.asarray([1.0], dtype=np.float32),
    )
    batch = make_range_windows(data, 0.6, 0.75, 7, "test_role")
    assert np.all(batch.origin >= np.ceil(0.6 * len(raw)))
    assert np.all(batch.origin + data.horizon <= np.floor(0.75 * len(raw)))
    assert batch.x.shape == (7, 32)
    assert batch.y.shape == (7, 12)


def test_context_may_predate_role_but_never_origin() -> None:
    raw = np.linspace(-2, 3, 200, dtype=np.float32)
    data = PreparedDataset(
        name="synthetic",
        domain="test",
        frequency="daily",
        context=50,
        horizon=10,
        seasonal_period=7,
        values_z=(raw,),
        values_raw=(raw,),
        series_names=("one",),
        means=np.asarray([0.0], dtype=np.float32),
        scales=np.asarray([1.0], dtype=np.float32),
    )
    batch = make_range_windows(data, 0.5, 0.7, 3, "test_role")
    for context, origin in zip(batch.x, batch.origin, strict=True):
        np.testing.assert_allclose(context, raw[origin - data.context : origin])
