#!/usr/bin/env python3
"""Validate the compact Gold Tree and counterfactual release."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def leaf_ids(tree: dict[str, Any]) -> Iterable[str]:
    def walk(node: dict[str, Any]) -> Iterable[str]:
        for leaf in node.get("leaves", []):
            yield str(leaf["leaf_id"])
        for child in node.get("children", []):
            yield from walk(child)

    for dimension in tree.get("dimensions", []):
        yield from walk(dimension)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-tree", type=Path, default=ROOT / "data/gold_tree.jsonl")
    parser.add_argument(
        "--counterfactuals",
        type=Path,
        default=ROOT / "data/counterfactual_responses.jsonl",
    )
    parser.add_argument("--expected-samples", type=int, default=1500)
    args = parser.parse_args()

    trees = {str(row["sample_id"]): row for row in read_jsonl(args.gold_tree)}
    counterfactuals = {
        str(row["sample_id"]): row for row in read_jsonl(args.counterfactuals)
    }
    if len(trees) != args.expected_samples or len(counterfactuals) != args.expected_samples:
        raise ValueError(
            f"Expected {args.expected_samples} samples; found "
            f"{len(trees)} trees and {len(counterfactuals)} counterfactual rows"
        )
    if set(trees) != set(counterfactuals):
        raise ValueError("Gold Tree and counterfactual sample IDs differ")

    counts: Counter[str] = Counter()
    for sample_id, tree_row in trees.items():
        ids = list(leaf_ids(tree_row["gold_tree"]))
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{sample_id}: missing or duplicated Gold leaf IDs")
        valid_ids = set(ids)
        cf_row = counterfactuals[sample_id]
        if not str(cf_row.get("canonical_correct_response", "")).strip():
            raise ValueError(f"{sample_id}: missing canonical correct response")
        cases = cf_row.get("counterfactuals", {})
        expected_types = {
            "single_leaf",
            "same_dimension_multi_leaf",
            "cross_dimension_multi_leaf",
        }
        if set(cases) != expected_types:
            raise ValueError(f"{sample_id}: unexpected counterfactual types")
        for case_type, case in cases.items():
            eligibility = str(case.get("eligibility", ""))
            counts[f"{case_type}:{eligibility}"] += 1
            targets = {str(value) for value in case.get("target_leaf_ids", [])}
            if not targets.issubset(valid_ids):
                raise ValueError(f"{sample_id}/{case_type}: unknown target leaf ID")
            if eligibility == "eligible" and not str(case.get("response", "")).strip():
                raise ValueError(f"{sample_id}/{case_type}: eligible case has no response")

    report = {
        "samples": len(trees),
        "counterfactuals": dict(sorted(counts.items())),
        "status": "ok",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
