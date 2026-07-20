from __future__ import annotations

import pandas as pd

from sv1.common import ROOT
from sv1.config import load_config


def test_exactly_thirty_registered_figures() -> None:
    config = load_config()
    plan = pd.read_csv(ROOT / "protocols" / "FIGURE_PLAN.csv")
    assert config.payload["evaluation"]["figure_count"] == 30
    assert len(plan) == 30
    assert plan["figure"].tolist() == list(range(1, 31))
    assert plan["slug"].is_unique


def test_temporal_roles_are_ordered_and_disjoint() -> None:
    roles = load_config().payload["roles"]
    previous = 0.0
    for start, end in roles.values():
        assert start >= previous
        assert start < end
        previous = end
    assert roles["confirmation"][0] == 0.86
