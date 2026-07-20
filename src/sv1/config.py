from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .common import ROOT, canonical_hash


@dataclass(frozen=True)
class StudyConfig:
    payload: dict[str, Any]
    path: Path
    hash: str

    @property
    def protocol_id(self) -> str:
        return str(self.payload["study"]["protocol_id"])

    @property
    def datasets(self) -> dict[str, dict[str, Any]]:
        return self.payload["datasets"]


def load_config(path: Path | None = None) -> StudyConfig:
    resolved = path or ROOT / "configs" / "study.yaml"
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    return StudyConfig(payload=payload, path=resolved, hash=canonical_hash(payload))
