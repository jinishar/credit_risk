"""Small shared helpers used across the ML, rules and talk-to-data layers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def risk_band(probability: float, low_threshold: float, high_threshold: float) -> str:
    """Map a predicted default probability to a business-readable risk band."""
    if probability < low_threshold:
        return "Low"
    if probability < high_threshold:
        return "Medium"
    return "High"
