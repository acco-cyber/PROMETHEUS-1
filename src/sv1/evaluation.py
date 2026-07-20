from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .data.dataset import WindowBatch

EPS = 1e-8


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    lower: float
    upper: float
    probability_positive: float
    distribution: np.ndarray


def mase_denominator(context: np.ndarray, period: int) -> np.ndarray:
    context = np.asarray(context, dtype=np.float64)
    lag = period if 1 <= period < context.shape[1] else 1
    denominator = np.mean(np.abs(context[:, lag:] - context[:, :-lag]), axis=1)
    return np.maximum(denominator, EPS)


def rmsse_denominator(context: np.ndarray, period: int) -> np.ndarray:
    context = np.asarray(context, dtype=np.float64)
    lag = period if 1 <= period < context.shape[1] else 1
    denominator = np.mean(np.square(context[:, lag:] - context[:, :-lag]), axis=1)
    return np.maximum(denominator, EPS)


def window_metrics(
    batch: WindowBatch,
    forecasts: dict[str, np.ndarray],
    period: int,
    domain: str,
    role: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    mase_scale = mase_denominator(batch.x, period)
    rmsse_scale = rmsse_denominator(batch.x, period)
    raw_y = batch.y_raw
    for method, prediction_z in forecasts.items():
        prediction_z = np.asarray(prediction_z, dtype=np.float64)
        if prediction_z.shape != batch.y.shape:
            raise ValueError(f"{method} has shape {prediction_z.shape}, expected {batch.y.shape}")
        target_z = np.asarray(batch.y, dtype=np.float64)
        error_z = target_z - prediction_z
        prediction_raw = prediction_z * batch.scale[:, None] + batch.mean[:, None]
        absolute_raw = np.abs(np.asarray(raw_y, dtype=np.float64) - prediction_raw)
        frame = pd.DataFrame(
            {
                "dataset": batch.dataset,
                "domain": domain,
                "role": role,
                "method": method,
                "series": batch.series,
                "origin": batch.origin,
                "mse": np.mean(np.square(error_z), axis=1),
                "mae": np.mean(np.abs(error_z), axis=1),
                "rmse": np.sqrt(np.mean(np.square(error_z), axis=1)),
                "mase": np.mean(np.abs(error_z), axis=1) / mase_scale,
                "rmsse": np.sqrt(np.mean(np.square(error_z), axis=1) / rmsse_scale),
                "smape": np.mean(
                    2.0
                    * absolute_raw
                    / (np.abs(np.asarray(raw_y, dtype=np.float64)) + np.abs(prediction_raw) + EPS),
                    axis=1,
                ),
                "wape_numerator": np.sum(absolute_raw, axis=1),
                "wape_denominator": np.sum(np.abs(raw_y), axis=1) + EPS,
            }
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def aggregate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    by = ["role", "method", "dataset", "domain"]
    result = frame.groupby(by, as_index=False).agg(
        windows=("mse", "size"),
        series=("series", "nunique"),
        mse=("mse", "mean"),
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        mase=("mase", "mean"),
        rmsse=("rmsse", "mean"),
        smape=("smape", "mean"),
        wape_numerator=("wape_numerator", "sum"),
        wape_denominator=("wape_denominator", "sum"),
    )
    result["wape"] = result["wape_numerator"] / result["wape_denominator"]
    baseline = result[result["method"] == "patchtst"][["dataset", "mse"]].rename(
        columns={"mse": "patchtst_mse"}
    )
    result = result.merge(baseline, on="dataset", how="left")
    result["relative_mse_reduction"] = 1.0 - result["mse"] / result["patchtst_mse"]
    return result


def macro_summary(aggregate: pd.DataFrame) -> pd.DataFrame:
    numeric = ["mse", "mae", "rmse", "mase", "rmsse", "smape", "wape", "relative_mse_reduction"]
    return aggregate.groupby(["role", "method"], as_index=False)[numeric].mean()


def _paired_effect(sample: pd.DataFrame, method: str, baseline: str) -> float:
    pivot = sample.pivot_table(
        index=["dataset", "series", "origin"], columns="method", values="mse", aggfunc="mean"
    ).dropna(subset=[method, baseline])
    return float(1.0 - pivot[method].sum() / max(pivot[baseline].sum(), EPS))


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    method: str,
    baseline: str = "patchtst",
    replicates: int = 4000,
    seed: int = 8191,
    block_length: int = 2,
    metric: str = "mse",
) -> BootstrapResult:
    if metric not in frame.columns:
        raise KeyError(f"Metric {metric!r} is absent")
    paired = (
        frame[frame["method"].isin([method, baseline])]
        .pivot_table(
            index=["dataset", "series", "origin"], columns="method", values=metric, aggfunc="mean"
        )
        .dropna(subset=[method, baseline])
        .reset_index()
    )
    datasets = tuple(sorted(paired["dataset"].unique()))
    matrices: list[tuple[np.ndarray, np.ndarray]] = []
    for dataset in datasets:
        data = paired[paired["dataset"] == dataset]
        method_rows = []
        baseline_rows = []
        for _, series_data in data.groupby("series", sort=True):
            ordered = series_data.sort_values("origin")
            method_rows.append(ordered[method].to_numpy(dtype=np.float64))
            baseline_rows.append(ordered[baseline].to_numpy(dtype=np.float64))
        lengths = {len(row) for row in method_rows + baseline_rows}
        if len(lengths) != 1:
            raise RuntimeError(f"{dataset} has unequal paired origin counts within series")
        method_matrix = np.stack(method_rows)
        baseline_matrix = np.stack(baseline_rows)
        matrices.append((method_matrix, baseline_matrix))
    generator = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        effects = []
        for dataset_index in generator.integers(0, len(datasets), size=len(datasets)):
            method_matrix, baseline_matrix = matrices[int(dataset_index)]
            series_count, origin_count = method_matrix.shape
            sampled_series = generator.integers(0, series_count, size=series_count)
            block_count = math.ceil(origin_count / block_length)
            starts = generator.integers(0, origin_count, size=(series_count, block_count))
            offsets = np.arange(block_length)[None, None, :]
            positions = ((starts[:, :, None] + offsets) % origin_count).reshape(series_count, -1)[
                :, :origin_count
            ]
            rows = sampled_series[:, None]
            numerator = float(method_matrix[rows, positions].sum())
            denominator = float(baseline_matrix[rows, positions].sum())
            effects.append(1.0 - numerator / max(denominator, EPS))
        draws[replicate] = float(np.mean(effects))
    dataset_effects = []
    for method_matrix, baseline_matrix in matrices:
        dataset_effects.append(
            1.0 - float(method_matrix.sum()) / max(float(baseline_matrix.sum()), EPS)
        )
    estimate = float(np.mean(dataset_effects))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return BootstrapResult(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        probability_positive=float(np.mean(draws > 0.0)),
        distribution=draws,
    )


def paired_dataset_tests(
    frame: pd.DataFrame, methods: Iterable[str], baseline: str = "patchtst"
) -> pd.DataFrame:
    dataset_mean = frame.groupby(["dataset", "method"], as_index=False)["mse"].mean()
    pivot = dataset_mean.pivot(index="dataset", columns="method", values="mse")
    rows: list[dict[str, float | str | int]] = []
    for method in methods:
        if method == baseline or method not in pivot:
            continue
        available = pivot[[method, baseline]].dropna()
        delta = available[baseline] - available[method]
        try:
            statistic, p_value = stats.wilcoxon(delta, alternative="two-sided")
        except ValueError:
            statistic, p_value = 0.0, 1.0
        relative = 1.0 - available[method] / available[baseline]
        rows.append(
            {
                "method": method,
                "datasets": len(available),
                "wilcoxon_statistic": float(statistic),
                "p_value": float(p_value),
                "wins": int((relative > 0.001).sum()),
                "ties": int((relative.abs() <= 0.001).sum()),
                "losses": int((relative < -0.001).sum()),
                "median_relative_mse_reduction": float(relative.median()),
            }
        )
    result = pd.DataFrame(rows)
    if len(result):
        order = np.argsort(result["p_value"].to_numpy())
        adjusted = np.empty(len(result), dtype=float)
        running = 0.0
        for rank, index in enumerate(order):
            value = min(1.0, float(result.iloc[index]["p_value"]) * (len(result) - rank))
            running = max(running, value)
            adjusted[index] = running
        result["holm_p_value"] = adjusted
    return result


def horizon_metrics(
    batch: WindowBatch,
    forecasts: dict[str, np.ndarray],
    domain: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for method, forecast in forecasts.items():
        squared = np.square(
            np.asarray(batch.y, dtype=np.float64) - np.asarray(forecast, dtype=np.float64)
        )
        for step in range(batch.y.shape[1]):
            rows.append(
                {
                    "dataset": batch.dataset,
                    "domain": domain,
                    "method": method,
                    "horizon_step": step + 1,
                    "normalized_horizon": (step + 1) / batch.y.shape[1],
                    "mse": float(squared[:, step].mean()),
                }
            )
    return pd.DataFrame(rows)


def probabilistic_metrics(
    batch: WindowBatch,
    method: str,
    quantile_forecasts: np.ndarray,
    quantile_levels: list[float],
) -> pd.DataFrame:
    target = np.asarray(batch.y, dtype=np.float64)
    quantile_forecasts = np.asarray(quantile_forecasts, dtype=np.float64)
    losses = []
    for index, level in enumerate(quantile_levels):
        residual = target - quantile_forecasts[:, index]
        losses.append(np.maximum(level * residual, (level - 1.0) * residual))
    pinball = np.stack(losses, axis=1)
    lower = quantile_forecasts[:, 0]
    upper = quantile_forecasts[:, -1]
    return pd.DataFrame(
        {
            "dataset": batch.dataset,
            "method": method,
            "series": batch.series,
            "origin": batch.origin,
            "weighted_quantile_loss": 2.0 * pinball.mean(axis=(1, 2)),
            "coverage_80": np.mean((target >= lower) & (target <= upper), axis=1),
            "normalized_width_80": np.mean(upper - lower, axis=1),
        }
    )
