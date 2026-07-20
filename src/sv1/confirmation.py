from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .artifacts import load_prediction_artifact, save_prediction_artifact
from .backbones import build_confirmation_inputs
from .common import ROOT, atomic_json, canonical_hash, sha256_file, utc_now
from .config import StudyConfig, load_config
from .evaluation import (
    aggregate_metrics,
    hierarchical_bootstrap,
    horizon_metrics,
    macro_summary,
    paired_dataset_tests,
    probabilistic_metrics,
    window_metrics,
)
from .experiments import _bank, _pir_alpha, _window_mse
from .integrity import verify_confirmation_lock
from .retrieval import apply_router, raft_revision, randomized_revision, retrieve_revision


def _confirmation_forecasts(
    dataset: str,
    config: StudyConfig,
    router: Any,
    k: int,
    temperature: float,
    alpha: float,
) -> tuple[Any, dict[str, np.ndarray], dict[str, np.ndarray]]:
    path = ROOT / "artifacts" / "base" / "confirmation" / f"{dataset}.npz"
    batch, forecasts, extras = load_prediction_artifact(path, dataset, "confirmation")
    bins = int(config.payload["retrieval"]["key_bins"])
    safe_bank = _bank(dataset, "crossfit")
    safe = retrieve_revision(batch.x, safe_bank["x"], safe_bank["residual"], k, temperature, bins)
    safe_revised = forecasts["patchtst"] + safe.revision
    gated, mask, utility = apply_router(router, safe.features, forecasts["patchtst"], safe_revised)
    insample_bank = _bank(dataset, "insample")
    insample = retrieve_revision(
        batch.x, insample_bank["x"], insample_bank["residual"], k, temperature, bins
    )
    insample_revised = forecasts["patchtst"] + insample.revision
    insample_gated, _, _ = apply_router(
        router, insample.features, forecasts["patchtst"], insample_revised
    )
    raft = raft_revision(batch.x, insample_bank["x"], insample_bank["y"], k, temperature, bins).revision
    shuffled_values = insample_bank["residual"].copy()
    np.random.default_rng(44021).shuffle(shuffled_values, axis=0)
    shuffled = retrieve_revision(
        batch.x, insample_bank["x"], shuffled_values, k, temperature, bins
    ).revision
    random = randomized_revision(insample_bank["residual"], len(batch), k, 9981)
    base_error = _window_mse(batch.y, forecasts["patchtst"])
    safe_error = _window_mse(batch.y, safe_revised)
    oracle_mask = safe_error < base_error
    forecasts.update(
        {
            "sv1_xfit_router": gated,
            "xfit_no_router": safe_revised,
            "insample_router": insample_gated,
            "raft_adapter": raft,
            "pir_adapter": forecasts["patchtst"] + alpha * safe.revision,
            "shuffled_control": forecasts["patchtst"] + shuffled,
            "random_control": forecasts["patchtst"] + random,
            "oracle_selector": np.where(oracle_mask[:, None], safe_revised, forecasts["patchtst"]),
        }
    )
    extras.update(
        {
            "selected_mask": mask.astype(np.int8),
            "predicted_utility": utility,
            "retrieval_features": safe.features,
            "retrieval_similarities": safe.similarities,
            "retrieval_indices": safe.indices,
        }
    )
    return batch, forecasts, extras


def run_confirmation(config: StudyConfig | None = None) -> dict[str, Any]:
    config = config or load_config()
    lock = verify_confirmation_lock(config)
    completion_path = ROOT / "CONFIRMATION_COMPLETE.json"
    if completion_path.exists():
        return json.loads(completion_path.read_text())
    started_path = ROOT / "CONFIRMATION_STARTED.json"
    if not started_path.exists():
        atomic_json(
            started_path,
            {"started_at": utc_now(), "lock_hash": lock["lock_hash"], "status": "running"},
        )
    else:
        started = json.loads(started_path.read_text())
        if started["lock_hash"] != lock["lock_hash"]:
            raise RuntimeError("Confirmation recovery lock does not match")
    selection = lock["selection"]
    k = int(selection["k"])
    temperature = float(selection["temperature"])
    router = joblib.load(ROOT / "artifacts" / "development" / "selected_router.joblib")
    alpha = _pir_alpha(k, temperature, config)
    build_confirmation_inputs(config)
    metric_frames: list[pd.DataFrame] = []
    horizon_frames: list[pd.DataFrame] = []
    probabilistic_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    for dataset, spec in config.datasets.items():
        batch, forecasts, extras = _confirmation_forecasts(
            dataset, config, router, k, temperature, alpha
        )
        save_prediction_artifact(
            ROOT / "artifacts" / "confirmation" / "predictions" / f"{dataset}.npz",
            batch,
            forecasts,
            extras,
        )
        metric_frames.append(
            window_metrics(
                batch,
                forecasts,
                int(spec["seasonal_period"]),
                str(spec["domain"]),
                "confirmation",
            )
        )
        horizon_frames.append(horizon_metrics(batch, forecasts, str(spec["domain"])))
        probabilistic_frames.append(
            probabilistic_metrics(
                batch,
                "chronos",
                extras["chronos_quantiles"],
                [float(value) for value in config.payload["chronos"]["quantiles"]],
            )
        )
        base_mse = _window_mse(batch.y, forecasts["patchtst"])
        selected_mse = _window_mse(batch.y, forecasts["sv1_xfit_router"])
        diagnostic_frames.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "domain": spec["domain"],
                    "series": batch.series,
                    "origin": batch.origin,
                    "top_similarity": extras["retrieval_similarities"][:, 0],
                    "neighbour_gap": extras["retrieval_features"][:, 2],
                    "selected": extras["selected_mask"].astype(bool),
                    "predicted_utility": extras["predicted_utility"],
                    "realized_utility": base_mse - selected_mse,
                }
            )
        )
    tables = ROOT / "results" / "tables"
    frame = pd.concat(metric_frames, ignore_index=True)
    aggregate = aggregate_metrics(frame)
    macro = macro_summary(aggregate)
    frame.to_parquet(tables / "confirmation_window_metrics.parquet", index=False)
    aggregate.to_csv(tables / "confirmation_dataset_metrics.csv", index=False)
    macro.to_csv(tables / "confirmation_macro_metrics.csv", index=False)
    pd.concat(horizon_frames, ignore_index=True).to_csv(
        tables / "confirmation_horizon_metrics.csv", index=False
    )
    pd.concat(probabilistic_frames, ignore_index=True).to_csv(
        tables / "confirmation_probabilistic_metrics.csv", index=False
    )
    pd.concat(diagnostic_frames, ignore_index=True).to_parquet(
        tables / "confirmation_retrieval_diagnostics.parquet", index=False
    )
    methods = sorted(frame["method"].unique())
    paired_dataset_tests(frame, methods).to_csv(tables / "confirmation_paired_tests.csv", index=False)
    bootstrap_rows: list[dict[str, Any]] = []
    primary_distribution: np.ndarray | None = None
    for index, method in enumerate(methods):
        if method == "patchtst":
            continue
        result = hierarchical_bootstrap(
            frame,
            method,
            replicates=int(config.payload["evaluation"]["bootstrap_replicates"]),
            seed=int(config.payload["evaluation"]["bootstrap_seed"]) + index,
            block_length=int(config.payload["evaluation"]["temporal_block"]),
        )
        bootstrap_rows.append(
            {
                "method": method,
                "metric": "mse",
                "estimate": result.estimate,
                "lower_95": result.lower,
                "upper_95": result.upper,
                "probability_positive": result.probability_positive,
                "replicates": len(result.distribution),
            }
        )
        if method == "sv1_xfit_router":
            primary_distribution = result.distribution
    for offset, metric in enumerate(("mae", "mase"), start=100):
        result = hierarchical_bootstrap(
            frame,
            "sv1_xfit_router",
            replicates=int(config.payload["evaluation"]["bootstrap_replicates"]),
            seed=int(config.payload["evaluation"]["bootstrap_seed"]) + offset,
            block_length=int(config.payload["evaluation"]["temporal_block"]),
            metric=metric,
        )
        bootstrap_rows.append(
            {
                "method": "sv1_xfit_router",
                "metric": metric,
                "estimate": result.estimate,
                "lower_95": result.lower,
                "upper_95": result.upper,
                "probability_positive": result.probability_positive,
                "replicates": len(result.distribution),
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(tables / "confirmation_bootstrap_summary.csv", index=False)
    if primary_distribution is None:
        raise RuntimeError("Primary bootstrap was not computed")
    pd.DataFrame({"relative_mse_reduction": primary_distribution}).to_csv(
        tables / "primary_bootstrap_distribution.csv", index=False
    )
    primary = bootstrap[
        (bootstrap["method"] == "sv1_xfit_router") & (bootstrap["metric"] == "mse")
    ].iloc[0]
    robustness = bootstrap[
        (bootstrap["method"] == "sv1_xfit_router") & (bootstrap["metric"].isin(["mae", "mase"]))
    ]
    dataset_primary = aggregate[aggregate["method"].isin(["sv1_xfit_router", "patchtst"])].pivot(
        index="dataset", columns="method", values="mse"
    )
    dataset_wins = int((dataset_primary["sv1_xfit_router"] < dataset_primary["patchtst"]).sum())
    conditions = {
        "ci_excludes_zero_positive": bool(primary["lower_95"] > 0.0),
        "point_exceeds_sesoi": bool(
            primary["estimate"] >= float(config.payload["study"]["sesoi_relative_mse"])
        ),
        "at_least_six_dataset_wins": dataset_wins >= 6,
        "mae_and_mase_not_significantly_harmed": bool((robustness["upper_95"] >= 0.0).all()),
        "integrity_lock_verified": True,
    }
    if all(conditions.values()):
        label = "beneficial"
    elif primary["upper_95"] < 0:
        label = "harmful"
    elif primary["lower_95"] > 0 and primary["estimate"] < float(
        config.payload["study"]["sesoi_relative_mse"]
    ):
        label = "statistically_positive_but_negligible"
    elif primary["lower_95"] <= 0 <= primary["upper_95"]:
        label = "inconclusive"
    else:
        label = "mixed"
    decision = {
        "completed_at": utc_now(),
        "protocol_id": config.protocol_id,
        "lock_hash": lock["lock_hash"],
        "primary_method": "sv1_xfit_router",
        "baseline": "patchtst",
        "primary_estimate": float(primary["estimate"]),
        "primary_lower_95": float(primary["lower_95"]),
        "primary_upper_95": float(primary["upper_95"]),
        "sesoi": float(config.payload["study"]["sesoi_relative_mse"]),
        "dataset_wins": dataset_wins,
        "conditions": conditions,
        "decision": label,
        "bootstrap_summary_sha256": sha256_file(tables / "confirmation_bootstrap_summary.csv"),
        "confirmation_metrics_sha256": sha256_file(tables / "confirmation_dataset_metrics.csv"),
    }
    decision["decision_hash"] = canonical_hash(decision)
    atomic_json(ROOT / "CONFIRMATION_DECISION.json", decision)
    complete = {
        "completed_at": utc_now(),
        "lock_hash": lock["lock_hash"],
        "decision_hash": decision["decision_hash"],
        "decision": label,
    }
    complete["completion_hash"] = canonical_hash(complete)
    atomic_json(completion_path, complete)
    return decision
