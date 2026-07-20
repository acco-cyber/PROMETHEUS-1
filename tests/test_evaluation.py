from __future__ import annotations

import numpy as np
import pandas as pd

from sv1.data.dataset import WindowBatch
from sv1.evaluation import aggregate_metrics, hierarchical_bootstrap, window_metrics


def _batch() -> WindowBatch:
    generator = np.random.default_rng(17)
    x = generator.normal(size=(12, 24)).astype(np.float32)
    y = generator.normal(size=(12, 6)).astype(np.float32)
    return WindowBatch(
        dataset="toy",
        role="confirmation",
        x=x,
        y=y,
        y_raw=y,
        series=np.repeat(np.arange(3), 4),
        origin=np.tile(np.arange(4), 3),
        mean=np.zeros(12, dtype=np.float32),
        scale=np.ones(12, dtype=np.float32),
    )


def test_metrics_identify_perfect_revision() -> None:
    batch = _batch()
    forecasts = {"patchtst": np.zeros_like(batch.y), "candidate": batch.y.copy()}
    frame = window_metrics(batch, forecasts, 7, "test", "confirmation")
    aggregate = aggregate_metrics(frame)
    candidate = aggregate[aggregate["method"] == "candidate"].iloc[0]
    assert candidate["mse"] == 0
    assert candidate["relative_mse_reduction"] == 1


def test_hierarchical_bootstrap_is_reproducible() -> None:
    rows = []
    for dataset in ("a", "b"):
        for series in range(3):
            for origin in range(4):
                rows.append(
                    {
                        "dataset": dataset,
                        "series": series,
                        "origin": origin,
                        "method": "patchtst",
                        "mse": 1.0,
                    }
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "series": series,
                        "origin": origin,
                        "method": "candidate",
                        "mse": 0.8,
                    }
                )
    frame = pd.DataFrame(rows)
    first = hierarchical_bootstrap(frame, "candidate", replicates=100, seed=5)
    second = hierarchical_bootstrap(frame, "candidate", replicates=100, seed=5)
    np.testing.assert_array_equal(first.distribution, second.distribution)
    assert abs(first.estimate - 0.2) < 1e-12
