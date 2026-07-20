from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

from .common import ROOT


def build_executed_notebook() -> Path:
    if not (ROOT / "CONFIRMATION_COMPLETE.json").exists():
        raise RuntimeError("Notebook generation requires completed confirmation")
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# Sv1: locked evidence walkthrough\n\n"
            "This notebook reads final artifacts; it does not train, select, or rescore confirmation."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json, pandas as pd\n"
            "ROOT = Path.cwd()\n"
            "decision = json.loads((ROOT/'CONFIRMATION_DECISION.json').read_text())\n"
            "decision"
        ),
        nbformat.v4.new_markdown_cell("## Dataset-level confirmatory effects"),
        nbformat.v4.new_code_cell(
            "metrics = pd.read_csv(ROOT/'results/tables/confirmation_dataset_metrics.csv')\n"
            "metrics[metrics.method.isin(['sv1_xfit_router','patchtst'])]"
            ".pivot(index='dataset', columns='method', values='mse')"
            ".assign(relative_reduction=lambda x: 1-x.sv1_xfit_router/x.patchtst)"
        ),
        nbformat.v4.new_markdown_cell("## Bootstrap decision and metric robustness"),
        nbformat.v4.new_code_cell(
            "bootstrap = pd.read_csv(ROOT/'results/tables/confirmation_bootstrap_summary.csv')\n"
            "bootstrap.query(\"method == 'sv1_xfit_router'\")"
        ),
        nbformat.v4.new_markdown_cell("## Leakage audit"),
        nbformat.v4.new_code_cell(
            "leakage = pd.read_csv(ROOT/'results/tables/leakage_audit.csv')\n"
            "leakage.groupby(['bank','mode']).mse.median().unstack()"
        ),
        nbformat.v4.new_markdown_cell("## Registered figure inventory"),
        nbformat.v4.new_code_cell(
            "figures = pd.read_csv(ROOT/'results/tables/figure_manifest.csv')\n"
            "assert len(figures) == 30\nfigures[['figure','slug','png_sha256','pdf_sha256']]"
        ),
        nbformat.v4.new_markdown_cell(
            "## Interpretation boundary\n\n"
            "The machine-generated decision applies the preregistered null, 1% SESOI, dataset-win, "
            "robustness, and integrity conditions. Secondary slices do not alter that decision."
        ),
    ]
    output = ROOT / "notebooks" / "Sv1_final_evidence.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(
        notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    )
    executed = client.execute()
    nbformat.write(executed, output)
    return output
