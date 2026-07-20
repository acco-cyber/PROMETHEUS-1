from __future__ import annotations

import numpy as np

from sv1.retrieval import (
    apply_router,
    context_keys,
    fit_router,
    retrieve_revision,
    select_threshold,
)


def test_shape_keys_are_affine_scale_invariant() -> None:
    x = np.stack([np.sin(np.linspace(0, 5, 64)), np.cos(np.linspace(0, 4, 64))]).astype(np.float32)
    original = context_keys(x)
    transformed = context_keys(7.5 * x + 193.0)
    np.testing.assert_allclose(original, transformed, atol=2e-5)


def test_exact_self_retrieval_and_exclusion() -> None:
    generator = np.random.default_rng(3)
    contexts = generator.normal(size=(12, 48)).astype(np.float32)
    values = generator.normal(size=(12, 8)).astype(np.float32)
    included = retrieve_revision(contexts, contexts, values, k=1, temperature=0.1)
    np.testing.assert_allclose(included.revision, values, atol=1e-5)
    excluded = retrieve_revision(
        contexts,
        contexts,
        values,
        k=1,
        temperature=0.1,
        exclusion=np.arange(len(contexts)),
    )
    assert np.all(excluded.indices[:, 0] != np.arange(len(contexts)))


def test_router_is_fit_without_target_features() -> None:
    generator = np.random.default_rng(8)
    features = generator.normal(size=(80, 7)).astype(np.float32)
    base = generator.normal(size=(80, 6)).astype(np.float32)
    correction = 0.1 * generator.normal(size=(80, 6)).astype(np.float32)
    truth = base + correction * (features[:, :1] > 0)
    revised = base + correction
    router = fit_router(features[:40], base[:40], revised[:40], truth[:40])
    threshold, sweep = select_threshold(
        router,
        features[40:60],
        base[40:60],
        revised[40:60],
        truth[40:60],
    )
    forecast, mask, utility = apply_router(router, features[60:], base[60:], revised[60:])
    assert np.isfinite(threshold)
    assert sweep.shape[1] == 3
    assert forecast.shape == truth[60:].shape
    assert mask.dtype == bool
    assert utility.shape == (20,)
