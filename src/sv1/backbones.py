from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import load_prediction_artifact, save_prediction_artifact
from .common import ROOT, atomic_json, ensure_layout, sha256_file, utc_now
from .config import StudyConfig, load_config
from .data.dataset import load_prepared, make_range_windows, make_role_windows
from .models.chronos import load_chronos, predict_chronos
from .models.classical import auto_theta, drift, last_value, seasonal_naive
from .models.patchtst import predict_ensemble, train_patchtst


def _model_paths(dataset: str, config: StudyConfig) -> list[Path]:
    directory = ROOT / "artifacts" / "models" / dataset
    return [directory / f"final_seed_{seed}.pt" for seed in config.payload["study"]["seeds"]]


def _train_final_models(dataset: str, config: StudyConfig) -> list[Path]:
    data = load_prepared(dataset, config)
    train = make_role_windows(data, "backbone_fit", config)
    validation = make_role_windows(data, "early_stop", config)
    paths = _model_paths(dataset, config)
    for seed, path in zip(config.payload["study"]["seeds"], paths, strict=True):
        if not path.exists():
            train_patchtst(
                train=train,
                validation=validation,
                spec=config.payload["patchtst"],
                seed=int(seed),
                output=path,
            )
    return paths


def _build_residual_banks(dataset: str, config: StudyConfig, final_paths: list[Path]) -> None:
    directory = ROOT / "artifacts" / "banks"
    crossfit_path = directory / f"{dataset}_crossfit.npz"
    insample_path = directory / f"{dataset}_insample.npz"
    data = load_prepared(dataset, config)
    if not insample_path.exists():
        fit = make_role_windows(data, "backbone_fit", config)
        prediction, seconds = predict_ensemble(
            final_paths, fit, int(config.payload["patchtst"]["batch_size"])
        )
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            insample_path,
            x=fit.x,
            y=fit.y,
            residual=fit.y - prediction,
            prediction=prediction,
            series=fit.series,
            origin=fit.origin,
            latency_seconds=np.asarray([seconds]),
        )
    if crossfit_path.exists():
        return
    train_start, train_end = (float(value) for value in config.payload["crossfit"]["train"])
    hold_start, hold_end = (float(value) for value in config.payload["crossfit"]["holdout"])
    train = make_range_windows(data, train_start, train_end, 18, "crossfit_train")
    holdout = make_range_windows(
        data,
        hold_start,
        hold_end,
        int(config.payload["windows"]["crossfit_per_fold"]),
        "crossfit_holdout",
    )
    crossfit_paths: list[Path] = []
    for seed in config.payload["study"]["seeds"]:
        path = ROOT / "artifacts" / "models" / dataset / f"crossfit_seed_{seed}.pt"
        crossfit_paths.append(path)
        if not path.exists():
            train_patchtst(
                train=train,
                validation=None,
                spec=config.payload["patchtst"],
                seed=int(seed),
                output=path,
                max_epochs=int(config.payload["patchtst"]["crossfit_epochs"]),
            )
    prediction, _ = predict_ensemble(
        crossfit_paths, holdout, int(config.payload["patchtst"]["batch_size"])
    )
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        crossfit_path,
        x=holdout.x,
        y=holdout.y,
        residual=holdout.y - prediction,
        prediction=prediction,
        series=holdout.series,
        origin=holdout.origin,
        fold=np.zeros(len(holdout), dtype=np.int8),
    )


def _base_artifact(dataset: str, role: str) -> Path:
    return ROOT / "artifacts" / "base" / role / f"{dataset}.npz"


def build_patchtst_inputs(config: StudyConfig | None = None) -> list[dict[str, object]]:
    config = config or load_config()
    ensure_layout()
    rows: list[dict[str, object]] = []
    for dataset in config.datasets:
        paths = _train_final_models(dataset, config)
        _build_residual_banks(dataset, config, paths)
        data = load_prepared(dataset, config)
        for role in ("calibration_fit", "calibration_threshold", "development"):
            path = _base_artifact(dataset, role)
            if path.exists():
                continue
            batch = make_role_windows(data, role, config)
            members: list[np.ndarray] = []
            total_seconds = 0.0
            for member_path in paths:
                prediction, seconds = predict_ensemble(
                    [member_path], batch, int(config.payload["patchtst"]["batch_size"])
                )
                members.append(prediction)
                total_seconds += seconds
            ensemble = np.mean(np.stack(members), axis=0).astype(np.float32)
            forecasts = {"patchtst": ensemble}
            extras = {
                "patchtst_members": np.stack(members).astype(np.float32),
                "patchtst_latency_seconds": np.asarray([total_seconds], dtype=np.float64),
            }
            save_prediction_artifact(path, batch, forecasts, extras)
        rows.append(
            {
                "dataset": dataset,
                "final_models": len(paths),
                "model_hashes": ";".join(sha256_file(path) for path in paths),
                "crossfit_bank": str(
                    (ROOT / "artifacts" / "banks" / f"{dataset}_crossfit.npz").relative_to(ROOT)
                ),
            }
        )
    frame = pd.DataFrame(rows)
    output = ROOT / "artifacts" / "base" / "patchtst_manifest.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return rows


def add_development_references(config: StudyConfig | None = None) -> None:
    config = config or load_config()
    chronos = None
    for dataset, spec in config.datasets.items():
        path = _base_artifact(dataset, "development")
        batch, forecasts, extras = load_prediction_artifact(path, dataset, "development")
        forecasts.update(
            {
                "last_value": last_value(batch),
                "drift": drift(batch),
                "seasonal_naive": seasonal_naive(batch, int(spec["seasonal_period"])),
            }
        )
        theta, theta_seconds = auto_theta(batch, int(spec["seasonal_period"]))
        forecasts["auto_theta"] = theta
        extras["auto_theta_latency_seconds"] = np.asarray([theta_seconds], dtype=np.float64)
        if chronos is None:
            chronos = load_chronos(str(config.payload["chronos"]["model"]))
        quantiles, mean, chronos_seconds = predict_chronos(
            chronos,
            batch,
            [float(value) for value in config.payload["chronos"]["quantiles"]],
            int(config.payload["chronos"]["batch_size"]),
        )
        forecasts["chronos"] = mean
        extras["chronos_quantiles"] = quantiles
        extras["chronos_latency_seconds"] = np.asarray([chronos_seconds], dtype=np.float64)
        save_prediction_artifact(path, batch, forecasts, extras)
    metadata = {
        "created_at": utc_now(),
        "chronos_model": config.payload["chronos"]["model"],
        "quantiles": config.payload["chronos"]["quantiles"],
    }
    atomic_json(ROOT / "artifacts" / "base" / "reference_manifest.json", metadata)


def build_confirmation_inputs(config: StudyConfig | None = None) -> None:
    config = config or load_config()
    if not (ROOT / "CONFIRMATION_LOCK.json").exists():
        raise RuntimeError("Confirmation is sealed until CONFIRMATION_LOCK.json exists")
    chronos = None
    for dataset, spec in config.datasets.items():
        output = _base_artifact(dataset, "confirmation")
        if output.exists():
            continue
        data = load_prepared(dataset, config)
        batch = make_role_windows(data, "confirmation", config)
        paths = _model_paths(dataset, config)
        patchtst, patch_seconds = predict_ensemble(
            paths, batch, int(config.payload["patchtst"]["batch_size"])
        )
        forecasts = {
            "patchtst": patchtst,
            "last_value": last_value(batch),
            "drift": drift(batch),
            "seasonal_naive": seasonal_naive(batch, int(spec["seasonal_period"])),
        }
        theta, theta_seconds = auto_theta(batch, int(spec["seasonal_period"]))
        forecasts["auto_theta"] = theta
        if chronos is None:
            chronos = load_chronos(str(config.payload["chronos"]["model"]))
        quantiles, mean, chronos_seconds = predict_chronos(
            chronos,
            batch,
            [float(value) for value in config.payload["chronos"]["quantiles"]],
            int(config.payload["chronos"]["batch_size"]),
        )
        forecasts["chronos"] = mean
        extras = {
            "patchtst_latency_seconds": np.asarray([patch_seconds]),
            "auto_theta_latency_seconds": np.asarray([theta_seconds]),
            "chronos_latency_seconds": np.asarray([chronos_seconds]),
            "chronos_quantiles": quantiles,
        }
        save_prediction_artifact(output, batch, forecasts, extras)


def base_manifest() -> dict[str, object]:
    paths = sorted((ROOT / "artifacts" / "models").rglob("*.pt"))
    return {
        "created_at": utc_now(),
        "checkpoints": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
    }
