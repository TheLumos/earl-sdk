"""Parse Lumos generic verifier catalog responses from ``GET /api/v1/additional-verifiers``.

Case-specific *case verifiers* live on each clinical case; this module handles only
platform-wide hard gates + scoring dimensions returned by the verifiers API.
"""

from __future__ import annotations

from typing import Any


def parse_verifiers_list_payload(data: Any) -> tuple[list[str], list[str]]:
    """
    Normalize a verifiers list JSON payload into sorted path lists.

    Accepts several shapes used by the Earl verifiers API and common variants.

    Returns:
        (hard_gate_paths, scoring_dimension_paths) using paths like
        ``hard-gates/...`` and ``scoring-dimensions/...``.
    """
    gates: set[str] = set()
    scoring: set[str] = set()

    if data is None:
        return [], []

    if isinstance(data, list):
        _consume_item_list(data, gates, scoring)
        return _sorted_unique(gates), _sorted_unique(scoring)

    if not isinstance(data, dict):
        return [], []

    # Some APIs wrap under "data"
    if "data" in data:
        inner = data["data"]
    else:
        inner = data
    if inner is None:
        return [], []
    if isinstance(inner, list):
        _consume_item_list(inner, gates, scoring)
        return _sorted_unique(gates), _sorted_unique(scoring)
    if isinstance(inner, dict) and inner is not data:
        # Single nested object — recurse once
        return parse_verifiers_list_payload(inner)

    _ingest_dict_arrays(data, gates, scoring)

    # Top-level combined lists (objects with id + type, or plain paths)
    for key in (
        "verifiers",
        "items",
        "results",
        "catalog",
        "dimensions",
        "all",
    ):
        arr = data.get(key)
        if isinstance(arr, list):
            _consume_item_list(arr, gates, scoring)

    return _sorted_unique(gates), _sorted_unique(scoring)


def _sorted_unique(paths: set[str]) -> list[str]:
    return sorted(paths, key=lambda x: x.lower())


def _ingest_dict_arrays(data: dict[str, Any], gates: set[str], scoring: set[str]) -> None:
    for key in ("hard_gates", "hardGates", "gates", "hardgates"):
        arr = data.get(key)
        if isinstance(arr, list):
            for x in arr:
                vid = _extract_id(x)
                if vid:
                    gates.add(_ensure_prefix(vid, "hard-gates"))

    for key in (
        "scoring_dimensions",
        "scoringDimensions",
        "scoring_dims",
        "scoringDimensionsList",
    ):
        arr = data.get(key)
        if isinstance(arr, list):
            for x in arr:
                vid = _extract_id(x)
                if vid:
                    scoring.add(_ensure_prefix(vid, "scoring-dimensions"))


def _consume_item_list(items: list[Any], gates: set[str], scoring: set[str]) -> None:
    for item in items:
        if isinstance(item, str):
            _classify_path_string(item, gates, scoring)
        elif isinstance(item, dict):
            vid = _extract_id(item)
            if not vid:
                continue
            kind = str(item.get("type") or item.get("kind") or item.get("category") or "").lower()
            if kind in ("hard_gate", "hard-gate", "hardgate", "gate", "boolean", "hard_gates"):
                gates.add(_ensure_prefix(vid, "hard-gates"))
            elif kind in (
                "scoring",
                "scoring_dimension",
                "scoring-dimension",
                "dimension",
                "score",
            ):
                scoring.add(_ensure_prefix(vid, "scoring-dimensions"))
            else:
                _classify_path_string(vid, gates, scoring)


def _extract_id(obj: Any) -> str:
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, dict):
        for k in ("id", "verifier_id", "path", "slug", "name", "key"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def _ensure_prefix(verifier_id: str, prefix: str) -> str:
    if "/" in verifier_id:
        return verifier_id
    return f"{prefix}/{verifier_id}"


def _classify_path_string(vid: str, gates: set[str], scoring: set[str]) -> None:
    if not vid:
        return
    if vid.startswith("hard-gates/"):
        gates.add(vid)
    elif vid.startswith("scoring-dimensions/"):
        scoring.add(vid)
    elif "/" in vid:
        # Unknown namespace — treat as scoring (Lumos-style paths are explicit)
        scoring.add(vid)
    else:
        scoring.add(f"scoring-dimensions/{vid}")
