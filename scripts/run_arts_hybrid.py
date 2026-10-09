#!/usr/bin/env python3
"""Generate one ARTS-Hybrid rubric from an anonymous JSON configuration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.rubrics import ARTSConfig, generate_arts_tree_with_diagnostics


def resolve_key(entry: dict[str, Any]) -> dict[str, Any]:
    result = dict(entry)
    env_name = str(result.pop("api_key_env", "")).strip()
    if env_name:
        result["api_key"] = os.getenv(env_name, "")
    return result


def load_config(path: Path) -> ARTSConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a JSON object")
    generator_key_env = str(raw.pop("generator_api_key_env", "")).strip()
    if generator_key_env:
        raw["generator_api_key"] = os.getenv(generator_key_env, "")
    for key in ("calibration_response_generators", "calibration_judges"):
        if key in raw:
            raw[key] = [resolve_key(item) for item in raw[key]]
    allowed = {field.name for field in fields(ARTSConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown ARTSConfig fields: {', '.join(unknown)}")
    return ARTSConfig(**raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query")
    query_group.add_argument("--query-file", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-logs", action="store_true")
    args = parser.parse_args()

    query = args.query or args.query_file.read_text(encoding="utf-8").strip()
    outcome = generate_arts_tree_with_diagnostics(query, load_config(args.config))
    if outcome.result is None:
        raise RuntimeError(f"{outcome.failure_stage}: {outcome.failure_reason}")
    result = outcome.result
    payload: dict[str, Any] = {
        "query": query,
        "rubric_tree": result.tree.to_json(),
        "leaf_scoring_points": result.leaf_scoring_points,
        "budget": result.budget,
        "max_depth": result.max_depth,
        "leaf_weighting": result.leaf_weighting,
    }
    if args.include_logs:
        payload["logs"] = result.logs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
