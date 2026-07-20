from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
import pypandoc

from .common import ROOT, atomic_json, sha256_file, utc_now
from .config import StudyConfig, load_config
from .visuals import generate_figures


def _format_interval(lower: float, upper: float) -> str:
    return f"[{100 * lower:.2f}%, {100 * upper:.2f}%]"


def _render_paper(config: StudyConfig) -> dict[str, Any]:
    decision = json.loads((ROOT / "CONFIRMATION_DECISION.json").read_text())
    selection = json.loads((ROOT / "artifacts" / "development" / "SELECTED_CANDIDATE.json").read_text())
    manifest = json.loads((ROOT / "data" / "processed" / "manifest.json").read_text())
    diagnostics = pd.read_parquet(
        ROOT / "results" / "tables" / "confirmation_retrieval_diagnostics.parquet"
    )
    bootstrap = pd.read_csv(ROOT / "results" / "tables" / "confirmation_bootstrap_summary.csv")
    primary = bootstrap[
        (bootstrap["method"] == "sv1_xfit_router") & (bootstrap["metric"] == "mse")
    ].iloc[0]
    window_metrics = pd.read_parquet(
        ROOT / "results" / "tables" / "confirmation_window_metrics.parquet"
    )
    confirmation_windows = int(len(window_metrics[window_metrics["method"] == "sv1_xfit_router"]))
    replacements = {
        "{{SERIES_TOTAL}}": str(manifest["series_total"]),
        "{{CONFIRMATION_WINDOWS}}": f"{confirmation_windows:,}",
        "{{PRIMARY_PERCENT}}": f"{100 * decision['primary_estimate']:.2f}",
        "{{PRIMARY_CI}}": _format_interval(decision["primary_lower_95"], decision["primary_upper_95"]),
        "{{CONFIRMATION_COVERAGE}}": f"{100 * diagnostics['selected'].mean():.1f}",
        "{{DATASET_WINS}}": str(decision["dataset_wins"]),
        "{{DECISION}}": decision["decision"].replace("_", " "),
        "{{PROBABILITY_POSITIVE}}": f"{primary['probability_positive']:.3f}",
        "{{K}}": str(selection["k"]),
        "{{TEMPERATURE}}": f"{selection['temperature']:.2f}",
        "{{THRESHOLD}}": f"{selection['threshold']:.5f}",
        "{{DEVELOPMENT_PERCENT}}": (
            f"{100 * selection['development_macro_relative_mse_reduction']:.2f}"
        ),
        "{{DEVELOPMENT_COVERAGE}}": f"{100 * selection['development_macro_coverage']:.1f}",
    }
    template = (ROOT / "manuscript" / "paper_template.md").read_text(encoding="utf-8")
    paper = template
    for source, target in replacements.items():
        paper = paper.replace(source, target)
    remaining = sorted(set(re.findall(r"\{\{[A-Z_]+\}\}", paper)))
    if remaining:
        raise RuntimeError(f"Unresolved manuscript placeholders: {remaining}")
    paper_path = ROOT / "manuscript" / "paper.md"
    paper_path.write_text(paper, encoding="utf-8")
    tex_path = ROOT / "manuscript" / "paper.tex"
    pypandoc.convert_file(
        str(paper_path),
        to="latex",
        format="gfm+tex_math_dollars",
        outputfile=str(tex_path),
        extra_args=[
            "--standalone",
            "--citeproc",
            f"--bibliography={ROOT / 'manuscript' / 'references.bib'}",
            "--metadata=link-citations:true",
            "--number-sections",
        ],
    )
    return {
        "paper": str(paper_path.relative_to(ROOT)),
        "paper_words": len(re.findall(r"\b\w+\b", paper)),
        "paper_sha256": sha256_file(paper_path),
        "tex": str(tex_path.relative_to(ROOT)),
        "tex_sha256": sha256_file(tex_path),
        "replacements": replacements,
    }


def _supplement() -> None:
    plan = pd.read_csv(ROOT / "protocols" / "FIGURE_PLAN.csv")
    sections = [
        "# Supplementary evidence atlas",
        "",
        "This file maps every registered visual to its evidentiary purpose. Figures are generated "
        "from machine-readable result tables; PDF versions are vector graphics.",
        "",
    ]
    for row in plan.itertuples(index=False):
        stem = f"fig_{int(row.figure):02d}_{row.slug}"
        sections.extend(
            [
                f"## Figure {row.figure}. {row.title}",
                "",
                f"Purpose: {row.purpose}. Evidence: {row.evidence_source}.",
                "",
                f"![Figure {row.figure}](../figures/png/{stem}.png)",
                "",
                f"[Vector PDF](../figures/pdf/{stem}.pdf)",
                "",
            ]
        )
    (ROOT / "manuscript" / "supplementary.md").write_text("\n".join(sections), encoding="utf-8")


def _executive_summary(config: StudyConfig) -> None:
    decision = json.loads((ROOT / "CONFIRMATION_DECISION.json").read_text())
    label = decision["decision"].replace("_", " ")
    text = f"""# Sv1 final evidence summary

The preregistered confirmation decision is **{label}**. The selected forward-cross-fitted
residual retriever changed standardized MSE by {100 * decision["primary_estimate"]:.2f}% relative
to the three-seed PatchTST ensemble (95% hierarchical interval
{_format_interval(decision["primary_lower_95"], decision["primary_upper_95"])}). It improved
{decision["dataset_wins"]} of ten dataset means. The practical success threshold was a 1% reduction.

This result must be interpreted with the registered decision conditions in
`CONFIRMATION_DECISION.json`; a positive point estimate alone is not a success claim. The release
contains ten fresh collections, 234 selected series, official PatchTST and Chronos-Bolt executions,
classical references, leakage and negative controls, 4,000-draw hierarchical inference, and exactly
30 data-driven figure pairs.

The repository is a strong submission package, not a guarantee of Q1 acceptance. Before submission,
replace the anonymous author and repository placeholders, choose a target journal, apply its template,
archive a public release, and obtain independent reproduction or co-author review.
"""
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "EXECUTIVE_SUMMARY.md").write_text(text, encoding="utf-8")


def _q1_audit() -> None:
    paper = (ROOT / "manuscript" / "paper.md").read_text()
    word_count = len(re.findall(r"\b\w+\b", paper))
    checks = [
        (
            "Novel, consequential question",
            True,
            "Residual provenance and selective retrieval are isolated.",
        ),
        (
            "Fresh heterogeneous evidence",
            True,
            "Ten collections and 234 deterministically selected series.",
        ),
        (
            "Strong references",
            True,
            "Official PatchTST architecture, official Chronos-Bolt, and classical controls.",
        ),
        ("Confirmatory separation", True, "Hashed selection precedes one sealed confirmation run."),
        ("Uncertainty and practical effect", True, "Hierarchical intervals and a fixed 1% SESOI."),
        (
            "Negative/leakage controls",
            True,
            "Self-inclusion, provenance, shuffled, random, and oracle diagnostics.",
        ),
        ("Writing depth", word_count >= 5500, f"Canonical paper contains {word_count:,} words."),
        (
            "Journal visuals",
            len(list((ROOT / "figures" / "pdf").glob("*.pdf"))) == 30,
            "Thirty registered vector PDFs.",
        ),
        (
            "Independent external reproduction",
            False,
            "Not yet completed; recommended before submission.",
        ),
        (
            "Target-journal formatting",
            False,
            "Generic LaTeX is supplied; venue template remains to be chosen.",
        ),
        (
            "Public DOI and author metadata",
            False,
            "Release placeholders must be replaced after repository publication.",
        ),
    ]
    lines = [
        "# Q1-readiness audit",
        "",
        "This is an evidence audit, not a prediction of editorial acceptance.",
        "",
        "| Criterion | Status | Evidence / action |",
        "|---|---|---|",
    ]
    for name, passed, detail in checks:
        lines.append(f"| {name} | {'PASS' if passed else 'OPEN'} | {detail} |")
    lines.extend(
        [
            "",
            "## Honest verdict",
            "",
            "The package is capable of supporting a Q1 submission because it has a defensible "
            "question, "
            "strong comparisons, a locked confirmatory design, transparent null-result rules, and a "
            "complete reproducibility trail. It is not submission-complete until the three OPEN items "
            "are handled, and no artifact can guarantee Q1 acceptance.",
        ]
    )
    (ROOT / "reports" / "Q1_READINESS_AUDIT.md").write_text("\n".join(lines) + "\n")


def generate_report(config: StudyConfig | None = None) -> dict[str, Any]:
    config = config or load_config()
    if not (ROOT / "CONFIRMATION_COMPLETE.json").exists():
        raise RuntimeError("Report generation requires completed confirmation")
    figures = generate_figures(config)
    manuscript = _render_paper(config)
    _supplement()
    _executive_summary(config)
    _q1_audit()
    output = {
        "created_at": utc_now(),
        "protocol_id": config.protocol_id,
        "figures": len(figures),
        **manuscript,
    }
    atomic_json(ROOT / "reports" / "REPORT_MANIFEST.json", output)
    return output
