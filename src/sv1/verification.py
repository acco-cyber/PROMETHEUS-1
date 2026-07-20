from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd
import torch
from PIL import Image

from .common import ROOT, atomic_json, sha256_file, utc_now
from .config import StudyConfig, load_config
from .integrity import verify_confirmation_lock


def _check(condition: bool, name: str, detail: str) -> dict[str, Any]:
    return {"check": name, "passed": bool(condition), "detail": detail}


def _environment() -> None:
    payload = {
        "created_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    atomic_json(ROOT / "environment" / "system.json", payload)
    result = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "pip"), "freeze", "--exclude-editable"],
        check=True,
        capture_output=True,
        text=True,
    )
    (ROOT / "environment" / "requirements-lock.txt").write_text(result.stdout)


def _artifact_manifest() -> list[dict[str, Any]]:
    included_roots = (
        "configs",
        "environment",
        "figures",
        "manuscript",
        "notebooks",
        "protocols",
        "reports",
        "results",
        "src",
        "tests",
    )
    paths: list[Path] = []
    for name in included_roots:
        if (ROOT / name).exists():
            paths.extend(path for path in (ROOT / name).rglob("*") if path.is_file())
    paths.extend(
        path
        for path in ROOT.iterdir()
        if path.is_file() and path.name not in {"ARTIFACT_MANIFEST.json"}
    )
    return [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(set(paths))
    ]


def verify_release(config: StudyConfig | None = None) -> dict[str, Any]:
    config = config or load_config()
    _environment()
    lock = verify_confirmation_lock(config)
    checks: list[dict[str, Any]] = []
    checks.append(_check(bool(lock["lock_hash"]), "confirmation lock", "Protected hashes match."))
    decision = json.loads((ROOT / "CONFIRMATION_DECISION.json").read_text())
    complete = json.loads((ROOT / "CONFIRMATION_COMPLETE.json").read_text())
    checks.append(
        _check(
            decision["decision_hash"] == complete["decision_hash"],
            "confirmation lineage",
            "Decision hash matches completion marker.",
        )
    )
    pngs = sorted((ROOT / "figures" / "png").glob("fig_*.png"))
    pdfs = sorted((ROOT / "figures" / "pdf").glob("fig_*.pdf"))
    checks.append(_check(len(pngs) == 30, "PNG figure count", f"Observed {len(pngs)}; expected 30."))
    checks.append(_check(len(pdfs) == 30, "PDF figure count", f"Observed {len(pdfs)}; expected 30."))
    dimensions_ok = True
    for path in pngs:
        with Image.open(path) as image:
            dimensions_ok &= image.width >= 1500 and image.height >= 700
    checks.append(_check(dimensions_ok, "figure resolution", "All PNGs exceed 1500×700 pixels."))
    plan = pd.read_csv(ROOT / "protocols" / "FIGURE_PLAN.csv")
    observed_slugs = [path.stem[7:] for path in pngs]
    checks.append(
        _check(
            observed_slugs == plan["slug"].tolist(),
            "figure registration",
            "Generated order and slugs match the frozen plan.",
        )
    )
    finite = True
    for path in (ROOT / "results" / "tables").glob("*.csv"):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            finite = False
            continue
        numeric = frame.select_dtypes(include=[np.number]).to_numpy()
        if numeric.size and not np.isfinite(numeric).all():
            finite = False
    checks.append(_check(finite, "finite scalar tables", "All numeric CSV cells are finite."))
    notebook_path = ROOT / "notebooks" / "Sv1_final_evidence.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    executed = all(cell.execution_count is not None for cell in code_cells)
    checks.append(_check(executed, "executed notebook", f"{len(code_cells)} code cells have outputs."))
    paper = (ROOT / "manuscript" / "paper.md").read_text()
    words = len(re.findall(r"\b\w+\b", paper))
    checks.append(
        _check(words >= 5500, "manuscript depth", f"Canonical manuscript has {words:,} words.")
    )
    unresolved = bool(re.search(r"\{\{[A-Z_]+\}\}", paper))
    checks.append(_check(not unresolved, "manuscript placeholders", "No result placeholders remain."))
    build = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), "-m", "build"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    checks.append(
        _check(build.returncode == 0, "package build", build.stdout[-500:] or build.stderr[-500:])
    )
    manifest = _artifact_manifest()
    atomic_json(ROOT / "ARTIFACT_MANIFEST.json", {"created_at": utc_now(), "files": manifest})
    passed = all(row["passed"] for row in checks)
    report = {
        "created_at": utc_now(),
        "protocol_id": config.protocol_id,
        "passed": passed,
        "checks": checks,
        "artifact_count": len(manifest),
    }
    atomic_json(ROOT / "reports" / "VERIFICATION_REPORT.json", report)
    lines = ["# Verification report", "", f"Overall: **{'PASS' if passed else 'FAIL'}**", ""]
    lines.extend(
        f"- {'PASS' if row['passed'] else 'FAIL'} — {row['check']}: {row['detail']}" for row in checks
    )
    (ROOT / "reports" / "VERIFICATION_REPORT.md").write_text("\n".join(lines) + "\n")
    if not passed:
        failed = [row["check"] for row in checks if not row["passed"]]
        raise RuntimeError("Release verification failed: " + ", ".join(failed))
    return report
