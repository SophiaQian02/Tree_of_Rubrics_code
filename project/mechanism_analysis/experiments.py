from __future__ import annotations

import json
import hashlib
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Callable

from .core import JsonModel, Sample, binary_metrics, normalize_ratios, preference_outcome, require_exact_ids
from .prompts import (
    COVERAGE_PROMPT,
    FLAT_GENERATION_PROMPT,
    FLAT_LOCALIZATION_PROMPT,
    FLAT_SINGLE_LEAF_PROMPT,
    FLAT_SINGLE_LEAF_WEIGHT_PROMPT,
    FLAT_WEIGHT_PROMPT,
    FULL_TREE_PATH_LEAF_WEIGHT_PROMPT,
    FULL_TREE_TOP_DOWN_WEIGHT_PROMPT,
    JSON_SYSTEM,
    LOCAL_SPLIT_CHECK_PROMPT,
    LOCAL_SPLIT_REPAIR_PROMPT,
    PATH_LEAF_WEIGHT_PROMPT,
    RESPONSE_CONDITIONED_FLAT_WEIGHT_PROMPT,
    RESPONSE_CONDITIONED_TREE_WEIGHT_PROMPT,
    TREE_ROOT_PROMPT,
    TREE_NODE_GATE_PROMPT,
    TREE_PATH_LEAF_PROMPT,
    TREE_ROUTE_PROMPT,
    TREE_SINGLE_LEAF_PROMPT,
    TREE_SOFT_ROUTER_PROMPT,
    TREE_SPLIT_PROMPT,
    WEIGHT_PROMPT,
)


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _validated_call(
    model: JsonModel,
    prompt: str,
    validator: Callable[[dict[str, Any]], Any],
    label: str,
    repair: Callable[[dict[str, Any], Exception], Any] | None = None,
) -> Any:
    """Retry invalid structured outputs, then optionally repair the last JSON.

    The repair hook is intentionally opt-in.  It is useful for schemas where a
    deterministic, auditable repair can preserve semantics more faithfully
    than dropping an otherwise complete experiment row.
    """
    retries = max(0, int(os.getenv("MECHANISM_SCHEMA_RETRIES", "2")))
    current_prompt = prompt
    last_error: Exception | None = None
    last_data: dict[str, Any] | None = None
    for attempt in range(retries + 1):
        try:
            last_data = model.call(JSON_SYSTEM, current_prompt)
            return validator(last_data)
        except (ValueError, TypeError, KeyError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            print(
                f"[MECHANISM SCHEMA RETRY] label={label} attempt={attempt + 1} "
                f"error={type(exc).__name__}: {str(exc)[:300]}",
                file=sys.stderr,
                flush=True,
            )
            current_prompt = (
                prompt
                + "\n\nYOUR PREVIOUS JSON FAILED SCHEMA VALIDATION.\n"
                + f"Validation error: {type(exc).__name__}: {str(exc)[:500]}\n"
                + "Regenerate the entire JSON object. Return every required field and ID exactly once, "
                + "use only enumerated values from the schema, and do not add prose."
            )
    assert last_error is not None
    if repair is not None and last_data is not None:
        try:
            repaired = repair(last_data, last_error)
            print(
                f"[MECHANISM SCHEMA REPAIR] label={label} "
                f"after={retries + 1} attempts error={type(last_error).__name__}: {str(last_error)[:300]}",
                file=sys.stderr,
                flush=True,
            )
            return repaired
        except (ValueError, TypeError, KeyError) as repair_error:
            raise ValueError(
                f"{label} repair failed after {retries + 1} attempts: {repair_error}"
            ) from repair_error
    raise ValueError(f"{label} remained invalid after {retries + 1} attempts: {last_error}") from last_error


def _validated_children(data: dict[str, Any], minimum: int, maximum: int, label: str) -> list[dict[str, Any]]:
    children = data.get("children")
    if not isinstance(children, list) or not minimum <= len(children) <= maximum:
        raise ValueError(f"{label} must return {minimum}-{maximum} children")
    for index, child in enumerate(children, 1):
        if not isinstance(child, dict):
            raise ValueError(f"{label} child {index} must be an object")
        if not str(child.get("name", "")).strip() or not str(child.get("criterion", "")).strip():
            raise ValueError(f"{label} child {index} must contain non-empty name and criterion")
    return children


def gold_leaf_payload(sample: Sample) -> list[dict[str, Any]]:
    return [
        {
            "leaf_id": leaf["leaf_id"],
            "path": [item["name"] for item in leaf.get("dimension_path", [])],
            "criterion": leaf["criterion"],
            "desired_state": leaf["desired_state"],
        }
        for leaf in sample.gold_flat
    ]


def generate_flat(sample: Sample, model: JsonModel, max_leaves: int) -> dict[str, Any]:
    prompt = FLAT_GENERATION_PROMPT.format(query=sample.query, max_leaves=max_leaves)

    def validate(data: dict[str, Any]) -> list[dict[str, Any]]:
        rubrics = data.get("rubrics")
        if not isinstance(rubrics, list) or not 1 <= len(rubrics) <= max_leaves:
            raise ValueError("Flat generator returned an invalid rubric count")
        if any(not isinstance(item, dict) or not str(item.get("criterion", "")).strip() for item in rubrics):
            raise ValueError("Flat generator returned an empty or invalid rubric")
        return rubrics

    rubrics = _validated_call(model, prompt, validate, "flat_generation")
    output = []
    for index, item in enumerate(rubrics, 1):
        criterion = str(item.get("criterion", "")).strip()
        if not criterion:
            raise ValueError("Flat generator returned an empty criterion")
        output.append({"leaf_id": f"F{index:02d}", "criterion": criterion, "path": [str(item.get("category", "Other"))]})
    return {"representation": "flat", "leaves": output, "calls": 1}


def _local_overlap_findings(
    sample: Sample,
    model: JsonModel,
    *,
    path: list[str],
    criterion: str,
    children: list[dict[str, Any]],
) -> list[dict[str, str]]:
    payload = [
        {"child_id": f"C{index}", "name": str(item.get("name", "")), "criterion": str(item.get("criterion", ""))}
        for index, item in enumerate(children, 1)
    ]
    prompt = LOCAL_SPLIT_CHECK_PROMPT.format(
        query=sample.query,
        path=" > ".join(path),
        criterion=criterion,
        children=compact(payload),
    )

    def validate(data: dict[str, Any]) -> list[dict[str, str]]:
        overlaps = data.get("overlaps", [])
        if not isinstance(overlaps, list) or any(not isinstance(item, dict) for item in overlaps):
            raise ValueError("Local split checker returned invalid overlaps")
        allowed_ids = {item["child_id"] for item in payload}
        allowed_relations = {"duplicate", "subsumes", "partially_overlapping"}
        normalized: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in overlaps:
            left = str(item.get("left_id", ""))
            right = str(item.get("right_id", ""))
            relation = str(item.get("relation", ""))
            if left not in allowed_ids or right not in allowed_ids or left == right:
                raise ValueError("Local split checker returned an unknown or self-paired child ID")
            if relation not in allowed_relations:
                raise ValueError(f"Local split checker returned unknown relation: {relation!r}")
            pair = tuple(sorted((left, right)))
            if pair in seen:
                continue
            seen.add(pair)
            normalized.append({"left_id": left, "right_id": right, "relation": relation, "reason": str(item.get("reason", ""))})
        declared_valid = data.get("valid")
        if not isinstance(declared_valid, bool):
            raise ValueError("Local split checker valid flag must be boolean")
        if declared_valid == bool(normalized):
            raise ValueError("Local split checker valid flag contradicts overlap findings")
        return normalized

    return _validated_call(model, prompt, validate, "local_split_check")


def generate_tree(
    sample: Sample,
    model: JsonModel,
    max_depth: int,
    max_leaves: int,
    *,
    local_check: bool = False,
    local_check_max_repairs: int = 2,
) -> dict[str, Any]:
    root_prompt = TREE_ROOT_PROMPT.format(query=sample.query)
    children = _validated_call(
        model,
        root_prompt,
        lambda data: _validated_children(data, 1, 8, "Tree root"),
        "tree_root",
    )
    if len(children) > max_leaves:
        raise ValueError("Tree root returned more branches than --max-leaves")
    calls = 1
    leaf_count = 0
    check_stats: dict[str, Any] = {
        "enabled": local_check,
        "max_repairs": local_check_max_repairs,
        "checked_splits": 0,
        "flagged_splits": 0,
        "overlap_pairs": 0,
        "repair_requests": 0,
        "repaired_splits": 0,
        "unresolved_splits_collapsed": 0,
        "relation_counts": {"duplicate": 0, "subsumes": 0, "partially_overlapping": 0},
        "events": [],
    }

    def build(item: dict[str, Any], path: list[str], depth: int) -> dict[str, Any]:
        nonlocal calls, leaf_count
        name = str(item.get("name", "")).strip()
        criterion = str(item.get("criterion", "")).strip()
        if not name or not criterion:
            raise ValueError("Tree generator returned an empty node")
        node = {"name": name, "criterion": criterion, "children": []}
        if depth >= max_depth or leaf_count >= max_leaves - 1:
            leaf_count += 1
            return node
        split_prompt = TREE_SPLIT_PROMPT.format(
            query=sample.query,
            path=" > ".join(path + [name]),
            criterion=criterion,
            depth=depth,
            max_depth=max_depth,
        )

        def validate_split(data: dict[str, Any]) -> dict[str, Any]:
            decision_name = data.get("decision")
            if decision_name not in {"atomic", "split"}:
                raise ValueError("Tree split decision must be atomic or split")
            if decision_name == "split":
                _validated_children(data, 2, 6, "Tree split")
            return data

        decision = _validated_call(model, split_prompt, validate_split, "tree_split")
        calls += 1
        proposed = decision.get("children", [])
        should_split = decision.get("decision") == "split" and isinstance(proposed, list) and 2 <= len(proposed) <= 6
        if not should_split or leaf_count + len(proposed) > max_leaves:
            leaf_count += 1
            return node
        if local_check:
            split_path = path + [name]
            check_stats["checked_splits"] += 1
            was_flagged = False
            repaired = False
            event = {"path": split_path, "attempts": []}
            for repair_index in range(local_check_max_repairs + 1):
                findings = _local_overlap_findings(
                    sample,
                    model,
                    path=split_path,
                    criterion=criterion,
                    children=proposed,
                )
                calls += 1
                event["attempts"].append({"repair_index": repair_index, "findings": findings})
                if not findings:
                    if was_flagged:
                        repaired = True
                        check_stats["repaired_splits"] += 1
                    break
                if not was_flagged:
                    check_stats["flagged_splits"] += 1
                    was_flagged = True
                check_stats["overlap_pairs"] += len(findings)
                for finding in findings:
                    check_stats["relation_counts"][finding["relation"]] += 1
                if repair_index >= local_check_max_repairs:
                    should_split = False
                    check_stats["unresolved_splits_collapsed"] += 1
                    break
                repair_prompt = LOCAL_SPLIT_REPAIR_PROMPT.format(
                    query=sample.query,
                    path=" > ".join(split_path),
                    criterion=criterion,
                    children=compact(proposed),
                    overlaps=compact(findings),
                )
                replacement = _validated_call(
                    model,
                    repair_prompt,
                    lambda data: _validated_children(data, 2, 6, "Local split repair"),
                    "local_split_repair",
                )
                calls += 1
                check_stats["repair_requests"] += 1
                proposed = replacement
            event["repaired"] = repaired
            event["accepted_split"] = should_split
            check_stats["events"].append(event)
            if not should_split:
                leaf_count += 1
                return node
        node["children"] = [build(child, path + [name], depth + 1) for child in proposed]
        return node

    tree = [build(item, [], 1) for item in children]
    leaves: list[dict[str, Any]] = []

    def number(node: dict[str, Any], path: list[str]) -> None:
        if node["children"]:
            for child in node["children"]:
                number(child, path + [node["name"]])
        else:
            node["leaf_id"] = f"T{len(leaves) + 1:02d}"
            leaves.append({"leaf_id": node["leaf_id"], "criterion": node["criterion"], "path": path + [node["name"]]})

    for root in tree:
        number(root, [])
    return {
        "representation": "tree",
        "generation_variant": "tree_local_check" if local_check else "tree_raw",
        "tree": tree,
        "leaves": leaves,
        "calls": calls,
        "max_depth": max_depth,
        "local_check": check_stats,
    }


def evaluate_coverage(sample: Sample, generated: dict[str, Any], judge: JsonModel) -> dict[str, Any]:
    gold = gold_leaf_payload(sample)
    leaves = generated["leaves"]
    gold_ids = {item["leaf_id"] for item in gold}
    generated_ids = {item["leaf_id"] for item in leaves}
    allowed = {"matched", "novel_valid", "redundant", "off_task"}
    allowed_categories = {
        "safety_risk", "reasoning_numerical", "context_personalization", "instruction_compliance",
        "factual_technical", "evidence_attribution", "completeness_relevance", "communication_style", "other_valid",
    }
    repair_state: dict[str, Any] = {"applied": False}
    prompt = COVERAGE_PROMPT.format(query=sample.query, gold=compact(gold), generated=compact(leaves))

    def validate(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
        gold_rows = data.get("gold_coverage", [])
        generated_rows = data.get("generated_classification", [])
        require_exact_ids(gold_rows, "gold_id", gold_ids, "gold_coverage")
        require_exact_ids(generated_rows, "generated_id", generated_ids, "generated_classification")
        classes = Counter(item.get("class") for item in generated_rows)
        if set(classes) - allowed:
            raise ValueError(f"Unknown generated rubric classes: {set(classes) - allowed}")
        for item in generated_rows:
            category = item.get("novel_category")
            if item.get("class") == "novel_valid" and category not in allowed_categories:
                raise ValueError(
                    f"novel_valid item has unknown novel_category: {category!r}; "
                    f"allowed={sorted(allowed_categories)}"
                )
            if item.get("class") != "novel_valid" and category is not None:
                raise ValueError("Only novel_valid items may have novel_category; all other items must use null")
        return gold_rows, generated_rows, classes

    def repair(data: dict[str, Any], error: Exception) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
        """Conservatively complete a nearly valid coverage response.

        The model has already exhausted ordinary schema retries. Missing Gold
        IDs are counted as uncovered and missing generated IDs as off-task, so
        repair can never improve the measured result. Unknown valid novelty
        labels are mapped to the schema's catch-all ``other_valid`` category.
        """
        raw_gold = data.get("gold_coverage")
        raw_generated = data.get("generated_classification")
        # If the judge omitted an entire list after exhausting retries, treat
        # it as empty and fill every expected ID with the conservative default
        # below. This can only lower coverage/novelty, never inflate it.
        if not isinstance(raw_gold, list):
            raw_gold = []
        if not isinstance(raw_generated, list):
            raw_generated = []

        gold_by_id: dict[str, dict[str, Any]] = {}
        for item in raw_gold:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("gold_id", ""))
            if item_id in gold_ids and item_id not in gold_by_id:
                gold_by_id[item_id] = item
        generated_by_id: dict[str, dict[str, Any]] = {}
        for item in raw_generated:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("generated_id", ""))
            if item_id in generated_ids and item_id not in generated_by_id:
                generated_by_id[item_id] = item

        repaired_gold: list[dict[str, Any]] = []
        missing_gold: list[str] = []
        for item in gold:
            item_id = str(item["leaf_id"])
            source = gold_by_id.get(item_id)
            if source is None:
                missing_gold.append(item_id)
                repaired_gold.append(
                    {
                        "gold_id": item_id,
                        "covered": False,
                        "generated_ids": [],
                        "reason": "Conservative schema repair: omitted by judge; counted as uncovered.",
                    }
                )
                continue
            repaired_gold.append(
                {
                    **source,
                    "gold_id": item_id,
                    "covered": source.get("covered") is True,
                    "generated_ids": [
                        str(value) for value in source.get("generated_ids", []) if str(value) in generated_ids
                    ],
                }
            )

        repaired_generated: list[dict[str, Any]] = []
        missing_generated: list[str] = []
        category_repairs = 0
        for leaf in leaves:
            item_id = str(leaf["leaf_id"])
            source = generated_by_id.get(item_id)
            if source is None:
                missing_generated.append(item_id)
                repaired_generated.append(
                    {
                        "generated_id": item_id,
                        "class": "off_task",
                        "gold_ids": [],
                        "novel_category": None,
                        "reason": "Conservative schema repair: omitted by judge; counted as off-task.",
                    }
                )
                continue
            class_name = str(source.get("class", ""))
            if class_name not in allowed:
                class_name = "off_task"
            category = source.get("novel_category")
            if class_name == "novel_valid":
                if category not in allowed_categories:
                    category = "other_valid"
                    category_repairs += 1
            else:
                if category is not None:
                    category_repairs += 1
                category = None
            repaired_generated.append(
                {
                    **source,
                    "generated_id": item_id,
                    "class": class_name,
                    "gold_ids": [str(value) for value in source.get("gold_ids", []) if str(value) in gold_ids],
                    "novel_category": category,
                }
            )

        classes = Counter(item["class"] for item in repaired_generated)
        repair_state.update(
            {
                "applied": True,
                "trigger": f"{type(error).__name__}: {error}",
                "missing_gold_ids_counted_uncovered": missing_gold,
                "missing_generated_ids_counted_off_task": missing_generated,
                "category_repairs": category_repairs,
            }
        )
        return repaired_gold, repaired_generated, classes

    gold_rows, generated_rows, classes = _validated_call(
        judge,
        prompt,
        validate,
        "coverage_judgment",
        repair=repair,
    )
    covered = sum(bool(item.get("covered")) for item in gold_rows)
    total_generated = len(generated_rows)
    return {
        "gold_coverage": gold_rows,
        "generated_classification": generated_rows,
        "schema_repair": repair_state if repair_state["applied"] else None,
        "metrics": {
            "gold_covered": covered,
            "gold_total": len(gold_rows),
            "coverage": covered / len(gold_rows) if gold_rows else 0.0,
            "generated_total": total_generated,
            "novel_valid_count": classes["novel_valid"],
            "uniqueness_rate": classes["novel_valid"] / total_generated if total_generated else 0.0,
            "redundancy_rate": classes["redundant"] / total_generated if total_generated else 0.0,
            "off_task_rate": classes["off_task"] / total_generated if total_generated else 0.0,
            "novel_category_counts": dict(Counter(item["novel_category"] for item in generated_rows if item.get("class") == "novel_valid")),
        },
    }


def _flat_leaf_judgment(sample: Sample, response: str, judge: JsonModel) -> tuple[dict[str, bool], dict[str, Any]]:
    leaves = gold_leaf_payload(sample)
    expected = {leaf["leaf_id"] for leaf in leaves}
    try:
        data = judge.call(
            JSON_SYSTEM,
            FLAT_LOCALIZATION_PROMPT.format(query=sample.query, response=response, leaves=compact(leaves)),
        )
        rows = require_exact_ids(data.get("leaf_results", []), "leaf_id", expected, "leaf_results")
        for row in rows:
            if not isinstance(row.get("passes"), bool):
                raise ValueError(f"leaf_results returned non-boolean passes for {row.get('leaf_id')}")
        return {str(row["leaf_id"]): row["passes"] for row in rows}, data
    except (ValueError, TypeError, KeyError) as batch_error:
        # Recovery for truncated or incomplete batched JSON. Judge exactly the
        # same leaves one at a time so a single malformed item cannot discard
        # the whole preference sample.
        recovered: list[dict[str, Any]] = []

        def validate_single(data: dict[str, Any], leaf_id: str) -> dict[str, Any]:
            if str(data.get("leaf_id", leaf_id)) != leaf_id:
                raise ValueError(f"single-leaf recovery must return leaf_id={leaf_id!r}")
            if not isinstance(data.get("passes"), bool):
                raise ValueError(f"single-leaf recovery returned non-boolean passes for {leaf_id}")
            return {
                "leaf_id": leaf_id,
                "passes": data["passes"],
                "evidence": str(data.get("evidence", "")),
            }

        for leaf in leaves:
            leaf_id = str(leaf["leaf_id"])
            result = _validated_call(
                judge,
                FLAT_SINGLE_LEAF_PROMPT.format(
                    query=sample.query,
                    response=response,
                    leaf_id=leaf_id,
                    criterion=leaf["criterion"],
                    desired_state=leaf["desired_state"],
                ),
                lambda data, item=leaf_id: validate_single(data, item),
                f"preference_single_leaf_recovery:{leaf_id}",
            )
            recovered.append(result)
        return (
            {row["leaf_id"]: row["passes"] for row in recovered},
            {
                "leaf_results": recovered,
                "recovery_mode": "single_leaf_after_batch_failure",
                "batch_error": f"{type(batch_error).__name__}: {batch_error}",
            },
        )


def evaluate_counterfactual_flat(sample: Sample, kind: str, judge: JsonModel) -> dict[str, Any]:
    cf = sample.counterfactuals[kind]
    if cf.get("eligibility") != "eligible":
        return {"eligible": False, "reason": cf.get("ineligibility_reason")}
    passes, raw = _flat_leaf_judgment(sample, cf["response"], judge)
    predicted = {leaf_id for leaf_id, value in passes.items() if not value}
    gold = set(cf["target_leaf_ids"])
    metrics = binary_metrics(predicted, gold, set(passes))
    return {"eligible": True, "predicted_failed_leaf_ids": sorted(predicted), "gold_failed_leaf_ids": sorted(gold), "metrics": metrics, "calls": 1, "raw": raw}


def evaluate_counterfactual_flat_single_leaf(sample: Sample, kind: str, judge: JsonModel) -> dict[str, Any]:
    """Judge every flat Gold leaf independently, matching exhaustive Tree call granularity."""
    cf = sample.counterfactuals[kind]
    if cf.get("eligibility") != "eligible":
        return {"eligible": False, "reason": cf.get("ineligibility_reason"), "flat_leaf_mode": "single"}
    response = cf["response"]
    predicted: set[str] = set()
    visited_leaves: set[str] = set()
    trace: list[dict[str, Any]] = []
    calls = 0
    for leaf in gold_leaf_payload(sample):
        leaf_id = str(leaf["leaf_id"])
        result = judge.call(
            JSON_SYSTEM,
            FLAT_SINGLE_LEAF_PROMPT.format(
                query=sample.query,
                response=response,
                leaf_id=leaf_id,
                criterion=leaf["criterion"],
                desired_state=leaf["desired_state"],
            ),
        )
        calls += 1
        if not isinstance(result.get("passes"), bool):
            raise ValueError(f"Flat single-leaf judge returned non-boolean passes for {leaf_id}")
        passes = result["passes"]
        visited_leaves.add(leaf_id)
        if not passes:
            predicted.add(leaf_id)
        trace.append({
            "step": "leaf_judge",
            "leaf_id": leaf_id,
            "passes": passes,
            "evidence": str(result.get("evidence", "")),
        })
    gold = set(cf["target_leaf_ids"])
    universe = {leaf["leaf_id"] for leaf in sample.gold_flat}
    if visited_leaves != universe:
        raise ValueError(
            f"Flat single-leaf evaluation did not visit every Gold leaf: missing={sorted(universe - visited_leaves)} "
            f"extra={sorted(visited_leaves - universe)}"
        )
    metrics = binary_metrics(predicted, gold, universe)
    metrics["visited_leaf_rate"] = 1.0 if universe else 0.0
    return {
        "eligible": True,
        "predicted_failed_leaf_ids": sorted(predicted),
        "gold_failed_leaf_ids": sorted(gold),
        "visited_leaf_ids": sorted(visited_leaves),
        "metrics": metrics,
        "calls": calls,
        "trace": trace,
        "flat_leaf_mode": "single",
    }


def _node_id(node: dict[str, Any]) -> str:
    return str(node.get("dimension_id") or node.get("node_id"))


def _branch_payload(node: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for leaf in node.get("leaves", []):
        rows.append({"branch_id": leaf["leaf_id"], "kind": "leaf", "criterion": leaf["criterion"], "desired_state": leaf["desired_state"]})
    for child in node.get("children", []):
        rows.append({"branch_id": _node_id(child), "kind": "internal", "name": child.get("name"), "description": child.get("description", "")})
    return rows


def evaluate_counterfactual_tree_batch(sample: Sample, kind: str, judge: JsonModel) -> dict[str, Any]:
    cf = sample.counterfactuals[kind]
    if cf.get("eligibility") != "eligible":
        return {"eligible": False, "reason": cf.get("ineligibility_reason")}
    response = cf["response"]
    predicted: set[str] = set()
    visited_leaves: set[str] = set()
    visited_nodes: list[str] = []
    trace: list[dict[str, Any]] = []
    calls = 0

    def route(node: dict[str, Any], path: list[str]) -> None:
        nonlocal calls
        branches = _branch_payload(node)
        if not branches:
            return
        data = judge.call(
            JSON_SYSTEM,
            TREE_ROUTE_PROMPT.format(query=sample.query, response=response, path=" > ".join(path), branches=compact(branches)),
        )
        calls += 1
        rows = data.get("branch_results", [])
        expected = {item["branch_id"] for item in branches}
        require_exact_ids(rows, "branch_id", expected, f"tree branches at {path}")
        trace.append({"path": path, "branch_results": rows})
        row_map = {row["branch_id"]: row for row in rows}
        child_map = {_node_id(child): child for child in node.get("children", [])}
        for branch in branches:
            branch_id = branch["branch_id"]
            row = row_map[branch_id]
            if branch["kind"] == "leaf":
                visited_leaves.add(branch_id)
                if not bool(row.get("leaf_passes")):
                    predicted.add(branch_id)
            elif bool(row.get("possible_failure")):
                visited_nodes.append(branch_id)
                route(child_map[branch_id], path + [str(child_map[branch_id].get("name", branch_id))])

    root = {"node_id": sample.gold_tree.get("root_id", "root"), "leaves": [], "children": sample.gold_tree.get("dimensions", [])}
    route(root, [sample.gold_tree.get("name", "Overall response quality")])
    gold = set(cf["target_leaf_ids"])
    universe = {leaf["leaf_id"] for leaf in sample.gold_flat}
    metrics = binary_metrics(predicted, gold, universe)
    routed_out_targets = gold - visited_leaves
    metrics["routing_false_negative_count"] = len(routed_out_targets)
    metrics["visited_leaf_rate"] = len(visited_leaves) / len(universe) if universe else 0.0
    return {
        "eligible": True,
        "predicted_failed_leaf_ids": sorted(predicted),
        "gold_failed_leaf_ids": sorted(gold),
        "visited_leaf_ids": sorted(visited_leaves),
        "routed_out_target_leaf_ids": sorted(routed_out_targets),
        "visited_internal_node_ids": visited_nodes,
        "metrics": metrics,
        "calls": calls,
        "trace": trace,
        "route_mode": "batch",
    }


def evaluate_counterfactual_tree_sequential(sample: Sample, kind: str, judge: JsonModel) -> dict[str, Any]:
    cf = sample.counterfactuals[kind]
    if cf.get("eligibility") != "eligible":
        return {"eligible": False, "reason": cf.get("ineligibility_reason"), "route_mode": "sequential"}
    response = cf["response"]
    predicted: set[str] = set()
    visited_leaves: set[str] = set()
    visited_nodes: list[str] = []
    trace: list[dict[str, Any]] = []
    calls = 0

    def route(node: dict[str, Any], path: list[str]) -> None:
        nonlocal calls
        node_id = _node_id(node)
        name = str(node.get("name") or node_id)
        gate = judge.call(
            JSON_SYSTEM,
            TREE_NODE_GATE_PROMPT.format(
                query=sample.query,
                response=response,
                path=" > ".join(path),
                name=name,
                description=str(node.get("description", "")),
                assignment_summary=str(node.get("assignment_summary", "")),
            ),
        )
        calls += 1
        decision = str(gate.get("decision", "")).strip().lower()
        if decision not in {"pass", "fail", "uncertain"}:
            raise ValueError(f"Sequential Tree gate returned invalid decision at {path}: {decision!r}")
        visited_nodes.append(node_id)
        trace.append({"step": "node_gate", "node_id": node_id, "path": path, "decision": decision, "evidence": str(gate.get("evidence", ""))})
        if decision == "pass":
            return
        for leaf in node.get("leaves", []):
            leaf_id = str(leaf["leaf_id"])
            result = judge.call(
                JSON_SYSTEM,
                TREE_SINGLE_LEAF_PROMPT.format(
                    query=sample.query,
                    response=response,
                    leaf_id=leaf_id,
                    criterion=leaf["criterion"],
                    desired_state=leaf["desired_state"],
                ),
            )
            calls += 1
            if not isinstance(result.get("passes"), bool):
                raise ValueError(f"Sequential Tree leaf returned non-boolean passes for {leaf_id}")
            visited_leaves.add(leaf_id)
            passes = result["passes"]
            if not passes:
                predicted.add(leaf_id)
            trace.append({"step": "leaf_judge", "leaf_id": leaf_id, "path": path, "passes": passes, "evidence": str(result.get("evidence", ""))})
        for child in node.get("children", []):
            child_name = str(child.get("name") or _node_id(child))
            route(child, path + [child_name])

    root = {
        "node_id": sample.gold_tree.get("root_id", "root"),
        "name": sample.gold_tree.get("name", "Overall response quality"),
        "description": "Overall response quality across the Gold rubric tree.",
        "assignment_summary": "Route only; use uncertain rather than pass when this broad scope cannot be safely verified.",
        "leaves": [],
        "children": sample.gold_tree.get("dimensions", []),
    }
    route(root, [str(root["name"])])
    gold = set(cf["target_leaf_ids"])
    universe = {leaf["leaf_id"] for leaf in sample.gold_flat}
    metrics = binary_metrics(predicted, gold, universe)
    routed_out_targets = gold - visited_leaves
    metrics["routing_false_negative_count"] = len(routed_out_targets)
    metrics["visited_leaf_rate"] = len(visited_leaves) / len(universe) if universe else 0.0
    return {
        "eligible": True,
        "predicted_failed_leaf_ids": sorted(predicted),
        "gold_failed_leaf_ids": sorted(gold),
        "visited_leaf_ids": sorted(visited_leaves),
        "routed_out_target_leaf_ids": sorted(routed_out_targets),
        "visited_internal_node_ids": visited_nodes,
        "metrics": metrics,
        "calls": calls,
        "trace": trace,
        "route_mode": "sequential",
    }


def evaluate_counterfactual_tree_soft(
    sample: Sample,
    kind: str,
    judge: JsonModel,
    *,
    threshold: float = 0.30,
    top_k: int = 2,
    force_safety: bool = True,
) -> dict[str, Any]:
    cf = sample.counterfactuals[kind]
    if cf.get("eligibility") != "eligible":
        return {"eligible": False, "reason": cf.get("ineligibility_reason"), "route_mode": "soft"}
    response = cf["response"]
    predicted: set[str] = set()
    visited_leaves: set[str] = set()
    visited_nodes: list[str] = []
    trace: list[dict[str, Any]] = []
    calls = 0

    def route(node: dict[str, Any], path: list[str], in_safety_subtree: bool = False) -> None:
        nonlocal calls
        branches = _branch_payload(node)
        if not branches:
            return
        node_id = _node_id(node)
        name = str(node.get("name") or node_id)
        data = judge.call(
            JSON_SYSTEM,
            TREE_SOFT_ROUTER_PROMPT.format(
                query=sample.query,
                response=response,
                path=" > ".join(path),
                name=name,
                description=str(node.get("description", "")),
                assignment_summary=str(node.get("assignment_summary", "")),
                branches=compact(branches),
                node_id=node_id,
            ),
        )
        calls += 1
        try:
            node_probability = float(data.get("descendant_violation_probability"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Soft Tree router returned invalid node probability at {path}") from exc
        if not 0.0 <= node_probability <= 1.0:
            raise ValueError(f"Soft Tree router probability is outside [0,1] at {path}")
        uncertainty = str(data.get("uncertainty", "")).strip().lower()
        router_schema_warnings: list[str] = []
        if uncertainty not in {"low", "medium", "high"}:
            router_schema_warnings.append(f"invalid uncertainty {uncertainty!r}; treated as high")
            uncertainty = "high"
        candidates = data.get("candidate_children", [])
        if not isinstance(candidates, list) or any(not isinstance(item, dict) for item in candidates):
            raise ValueError(f"Soft Tree router returned invalid candidate_children at {path}")
        branch_ids = [item["branch_id"] for item in branches]
        branch_id_set = set(branch_ids)
        probabilities = {branch_id: 0.0 for branch_id in branch_ids}
        duplicate_ids: set[str] = set()
        returned_ids: set[str] = set()
        for item in candidates:
            child_id = str(item.get("child_id", ""))
            if child_id not in branch_id_set:
                # Never guess a Gold ID. Conservatively expand every valid
                # sibling so malformed routing cannot introduce a false
                # negative, and retain the warning in the trace.
                router_schema_warnings.append(f"ignored unknown child ID {child_id!r}; treated as high uncertainty")
                uncertainty = "high"
                continue
            if child_id in returned_ids:
                duplicate_ids.add(child_id)
            returned_ids.add(child_id)
            try:
                probability = float(item.get("probability"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Soft Tree router returned invalid child probability for {child_id}") from exc
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"Soft Tree child probability is outside [0,1] for {child_id}")
            probabilities[child_id] = max(probabilities[child_id], probability)
        ranked = sorted(branch_ids, key=lambda branch_id: (-probabilities[branch_id], branch_ids.index(branch_id)))
        selected: set[str] = {branch_id for branch_id, probability in probabilities.items() if probability >= threshold}
        selected.update(ranked[: min(top_k, len(ranked))])
        safety_ids = {
            branch_id
            for branch_id in branch_ids
            if branch_id == "safety_risk" or "safety" in branch_id.lower()
        }
        if force_safety:
            if in_safety_subtree or node_id == "safety_risk":
                selected.update(branch_ids)
            else:
                selected.update(safety_ids)
        # Follow the soft-routing OR rule literally: medium/high uncertainty
        # is sufficient to retain every direct branch for further checking.
        if uncertainty in {"medium", "high"}:
            selected.update(branch_ids)
        visited_nodes.append(node_id)
        trace.append(
            {
                "step": "soft_route",
                "node_id": node_id,
                "path": path,
                "descendant_violation_probability": node_probability,
                "routing_decision": str(data.get("routing_decision", "")),
                "uncertainty": uncertainty,
                "probabilities": probabilities,
                "returned_candidate_ids": sorted(returned_ids),
                "missing_candidate_ids": sorted(branch_id_set - returned_ids),
                "duplicate_candidate_ids": sorted(duplicate_ids),
                "router_schema_warnings": router_schema_warnings,
                "selected_branch_ids": sorted(selected),
                "evidence": data.get("evidence", []),
            }
        )
        child_map = {_node_id(child): child for child in node.get("children", [])}
        leaf_map = {str(leaf["leaf_id"]): leaf for leaf in node.get("leaves", [])}
        for branch_id in branch_ids:
            if branch_id not in selected:
                continue
            if branch_id in leaf_map:
                leaf = leaf_map[branch_id]
                result = judge.call(
                    JSON_SYSTEM,
                    TREE_SINGLE_LEAF_PROMPT.format(
                        query=sample.query,
                        response=response,
                        leaf_id=branch_id,
                        criterion=leaf["criterion"],
                        desired_state=leaf["desired_state"],
                    ),
                )
                calls += 1
                if not isinstance(result.get("passes"), bool):
                    raise ValueError(f"Soft Tree leaf returned non-boolean passes for {branch_id}")
                passes = result["passes"]
                visited_leaves.add(branch_id)
                if not passes:
                    predicted.add(branch_id)
                trace.append({"step": "leaf_judge", "leaf_id": branch_id, "path": path, "passes": passes, "evidence": str(result.get("evidence", ""))})
            else:
                child = child_map[branch_id]
                child_safety = in_safety_subtree or node_id == "safety_risk" or branch_id == "safety_risk"
                route(child, path + [str(child.get("name") or branch_id)], child_safety)

    root = {
        "node_id": sample.gold_tree.get("root_id", "root"),
        "name": sample.gold_tree.get("name", "Overall response quality"),
        "description": "Overall routing root; do not make a final correctness judgment.",
        "assignment_summary": "Prioritize any subtree that may contain a local Gold-rubric violation.",
        "leaves": [],
        "children": sample.gold_tree.get("dimensions", []),
    }
    route(root, [str(root["name"])])
    gold = set(cf["target_leaf_ids"])
    universe = {leaf["leaf_id"] for leaf in sample.gold_flat}
    metrics = binary_metrics(predicted, gold, universe)
    routed_out_targets = gold - visited_leaves
    metrics["routing_false_negative_count"] = len(routed_out_targets)
    metrics["visited_leaf_rate"] = len(visited_leaves) / len(universe) if universe else 0.0
    return {
        "eligible": True,
        "predicted_failed_leaf_ids": sorted(predicted),
        "gold_failed_leaf_ids": sorted(gold),
        "visited_leaf_ids": sorted(visited_leaves),
        "routed_out_target_leaf_ids": sorted(routed_out_targets),
        "visited_internal_node_ids": visited_nodes,
        "metrics": metrics,
        "calls": calls,
        "trace": trace,
        "route_mode": "soft",
        "routing_config": {"threshold": threshold, "top_k": top_k, "force_safety": force_safety},
    }


def evaluate_counterfactual_tree_exhaustive_path(sample: Sample, kind: str, judge: JsonModel) -> dict[str, Any]:
    """Judge every leaf independently with its full semantic path and no parent routing."""
    cf = sample.counterfactuals[kind]
    if cf.get("eligibility") != "eligible":
        return {"eligible": False, "reason": cf.get("ineligibility_reason"), "route_mode": "exhaustive_path"}
    response = cf["response"]
    predicted: set[str] = set()
    visited_leaves: set[str] = set()
    trace: list[dict[str, Any]] = []
    calls = 0

    root_context = {
        "node_id": str(sample.gold_tree.get("root_id", "root")),
        "name": str(sample.gold_tree.get("name", "Overall response quality")),
        "description": str(sample.gold_tree.get("description", "Overall response quality across the Gold rubric tree.")),
        "assignment_summary": str(sample.gold_tree.get("assignment_summary", "")),
    }

    def node_context(node: dict[str, Any]) -> dict[str, str]:
        return {
            "node_id": _node_id(node),
            "name": str(node.get("name", "")),
            "description": str(node.get("description", "")),
            "assignment_summary": str(node.get("assignment_summary", "")),
        }

    def visit(node: dict[str, Any], semantic_path: list[dict[str, str]]) -> None:
        nonlocal calls
        for leaf in node.get("leaves", []):
            leaf_id = str(leaf["leaf_id"])
            full_path = semantic_path + [{
                "leaf_id": leaf_id,
                "criterion": str(leaf["criterion"]),
                "desired_state": str(leaf["desired_state"]),
            }]
            result = judge.call(
                JSON_SYSTEM,
                TREE_PATH_LEAF_PROMPT.format(
                    query=sample.query,
                    response=response,
                    semantic_path=compact(full_path),
                    leaf_id=leaf_id,
                    criterion=leaf["criterion"],
                    desired_state=leaf["desired_state"],
                ),
            )
            calls += 1
            if not isinstance(result.get("passes"), bool):
                raise ValueError(f"Exhaustive-path Tree leaf returned non-boolean passes for {leaf_id}")
            passes = result["passes"]
            visited_leaves.add(leaf_id)
            if not passes:
                predicted.add(leaf_id)
            trace.append({
                "step": "leaf_judge",
                "leaf_id": leaf_id,
                "semantic_path": full_path,
                "passes": passes,
                "evidence": str(result.get("evidence", "")),
            })
        for child in node.get("children", []):
            visit(child, semantic_path + [node_context(child)])

    synthetic_root = {"leaves": [], "children": sample.gold_tree.get("dimensions", [])}
    visit(synthetic_root, [root_context])
    gold = set(cf["target_leaf_ids"])
    universe = {leaf["leaf_id"] for leaf in sample.gold_flat}
    if visited_leaves != universe:
        raise ValueError(
            f"Exhaustive-path Tree did not visit every Gold leaf: missing={sorted(universe - visited_leaves)} "
            f"extra={sorted(visited_leaves - universe)}"
        )
    metrics = binary_metrics(predicted, gold, universe)
    metrics["routing_false_negative_count"] = 0
    metrics["visited_leaf_rate"] = 1.0 if universe else 0.0
    return {
        "eligible": True,
        "predicted_failed_leaf_ids": sorted(predicted),
        "gold_failed_leaf_ids": sorted(gold),
        "visited_leaf_ids": sorted(visited_leaves),
        "routed_out_target_leaf_ids": [],
        "visited_internal_node_ids": [],
        "metrics": metrics,
        "calls": calls,
        "trace": trace,
        "route_mode": "exhaustive_path",
    }


def extract_semantic_leaf_paths(sample: Sample) -> dict[str, list[dict[str, str]]]:
    """Return the canonical full semantic path for every Gold leaf."""
    root_context = {
        "node_id": str(sample.gold_tree.get("root_id", "root")),
        "name": str(sample.gold_tree.get("name", "Overall response quality")),
        "description": str(sample.gold_tree.get("description", "Overall response quality across the Gold rubric tree.")),
        "assignment_summary": str(sample.gold_tree.get("assignment_summary", "")),
    }

    def node_context(node: dict[str, Any]) -> dict[str, str]:
        return {
            "node_id": _node_id(node), "name": str(node.get("name", "")),
            "description": str(node.get("description", "")),
            "assignment_summary": str(node.get("assignment_summary", "")),
        }

    original: dict[str, list[dict[str, str]]] = {}

    def visit(node: dict[str, Any], path: list[dict[str, str]]) -> None:
        for leaf in node.get("leaves", []):
            leaf_id = str(leaf["leaf_id"])
            original[leaf_id] = path + [{
                "leaf_id": leaf_id, "criterion": str(leaf["criterion"]),
                "desired_state": str(leaf["desired_state"]),
            }]
        for child in node.get("children", []):
            visit(child, path + [node_context(child)])

    visit({"children": sample.gold_tree.get("dimensions", []), "leaves": []}, [root_context])
    return original


HARD_DIMENSION_SHUFFLE_MAP = {
    "Factual and technical correctness": "Communication and style",
    "Reasoning and numerical correctness": "Communication and style",
    "Communication and style": "Reasoning and numerical correctness",
    "Safety and risk handling": "Communication and style",
    "Context and personalization": "Reasoning and numerical correctness",
    "Evidence and attribution": "Context and personalization",
    "Completeness and relevance": "Communication and style",
    "Task and instruction compliance": "Reasoning and numerical correctness",
    "Other valid criterion": "Safety and risk handling",
}


def _top_dimension_name(path: list[dict[str, str]]) -> str:
    if len(path) < 3:
        raise ValueError("Root-to-Leaf path has no top-level Dimension")
    return str(path[1].get("name", "")).strip()


def build_cross_sample_path_catalog(samples: list[Sample]) -> dict[int, list[dict[str, Any]]]:
    """Index donor internal paths by exact path length for cross-tree fallback."""
    catalog: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        for leaf_id, path in extract_semantic_leaf_paths(sample).items():
            catalog[len(path)].append({
                "sample_id": sample.sample_id,
                "leaf_id": leaf_id,
                "path": path,
                "internal_node_ids": tuple(str(item.get("node_id", "")) for item in path[1:-1]),
                "top_dimension_name": _top_dimension_name(path),
            })
    return dict(catalog)


def build_shuffled_leaf_paths(
    sample: Sample,
    seed: int,
    fraction: float = 0.5,
    cross_sample_catalog: dict[int, list[dict[str, Any]]] | None = None,
    strategy: str = "random_same_depth",
) -> dict[str, Any]:
    """Deterministically corrupt paths, preferring in-tree equal-depth donors.

    ``hard_dimension`` uses a preregistered mapping between the nine canonical
    Gold Dimensions. It never asks a Judge model to choose its own corruption.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("shuffle fraction must be in [0,1]")
    if strategy not in {"random_same_depth", "hard_dimension"}:
        raise ValueError(f"unknown shuffle strategy: {strategy}")
    original = extract_semantic_leaf_paths(sample)
    leaf_ids = sorted(original)
    seed_material = f"{seed}|{sample.sample_id}" if strategy == "random_same_depth" else f"{seed}|{strategy}|{sample.sample_id}"
    stable_seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
    rng = random.Random(stable_seed)

    def signature(path: list[dict[str, str]]) -> tuple[str, ...]:
        return tuple(str(item.get("node_id", "")) for item in path[1:-1])

    original_dimensions = {leaf_id: _top_dimension_name(original[leaf_id]) for leaf_id in leaf_ids}
    required_dimensions = {}
    if strategy == "hard_dimension":
        unknown = sorted({name for name in original_dimensions.values() if name not in HARD_DIMENSION_SHUFFLE_MAP})
        if unknown:
            raise ValueError(f"hard_dimension has no preregistered mismatch for Dimensions: {unknown}")
        required_dimensions = {
            leaf_id: HARD_DIMENSION_SHUFFLE_MAP[original_dimensions[leaf_id]]
            for leaf_id in leaf_ids
        }

    def dimension_matches(target_leaf_id: str, candidate_path: list[dict[str, str]]) -> bool:
        return (
            strategy != "hard_dimension"
            or _top_dimension_name(candidate_path) == required_dimensions[target_leaf_id]
        )

    local_donors = {
        leaf_id: [
            candidate for candidate in leaf_ids
            if len(original[candidate]) == len(original[leaf_id])
            and signature(original[candidate]) != signature(original[leaf_id])
            and dimension_matches(leaf_id, original[candidate])
        ]
        for leaf_id in leaf_ids
    }
    eligible = [
        leaf_id for leaf_id in leaf_ids
        if local_donors[leaf_id] or (
            cross_sample_catalog is not None
            and any(
                row["sample_id"] != sample.sample_id
                and row["internal_node_ids"] != signature(original[leaf_id])
                and dimension_matches(leaf_id, row["path"])
                for row in cross_sample_catalog.get(len(original[leaf_id]), [])
            )
        )
    ]
    rng.shuffle(eligible)
    target_count = int(len(leaf_ids) * fraction + 0.5)
    selected = set(eligible[: min(target_count, len(eligible))])
    shuffled, mapping = {}, []
    for leaf_id in leaf_ids:
        target = original[leaf_id]
        donor_id = None
        donor_sample_id = None
        donor_scope = None
        donor_path = None
        if leaf_id in selected and local_donors[leaf_id]:
            donor_id = rng.choice(local_donors[leaf_id])
            donor_sample_id = sample.sample_id
            donor_scope = "within_tree"
            donor_path = original[donor_id]
        elif leaf_id in selected and cross_sample_catalog is not None:
            candidates = [
                row for row in cross_sample_catalog.get(len(target), [])
                if row["sample_id"] != sample.sample_id
                and row["internal_node_ids"] != signature(target)
                and dimension_matches(leaf_id, row["path"])
            ]
            if candidates:
                donor = rng.choice(candidates)
                donor_id = str(donor["leaf_id"])
                donor_sample_id = str(donor["sample_id"])
                donor_scope = "cross_tree_same_depth"
                donor_path = donor["path"]
        new_path = target if donor_path is None else [target[0]] + donor_path[1:-1] + [target[-1]]
        if len(new_path) != len(target):
            raise ValueError("shuffle changed path depth")
        donor_dimension = _top_dimension_name(donor_path) if donor_path is not None else None
        if strategy == "hard_dimension" and donor_path is not None:
            expected = required_dimensions[leaf_id]
            if donor_dimension != expected or donor_dimension == original_dimensions[leaf_id]:
                raise ValueError(
                    f"hard_dimension donor mismatch for {leaf_id}: original={original_dimensions[leaf_id]!r}, "
                    f"expected={expected!r}, donor={donor_dimension!r}"
                )
        shuffled[leaf_id] = new_path
        mapping.append({
            "leaf_id": leaf_id, "selected_for_shuffle": leaf_id in selected,
            "actually_shuffled": donor_path is not None, "donor_leaf_id": donor_id,
            "donor_sample_id": donor_sample_id, "donor_scope": donor_scope,
            "path_depth": len(target) - 1,
            "original_internal_node_ids": list(signature(target)),
            "shuffled_internal_node_ids": list(signature(new_path)),
            "original_top_dimension_name": original_dimensions[leaf_id],
            "required_donor_dimension_name": required_dimensions.get(leaf_id),
            "donor_top_dimension_name": donor_dimension,
            "hard_dimension_mismatch": bool(
                donor_path is not None
                and strategy == "hard_dimension"
                and donor_dimension == required_dimensions[leaf_id]
            ),
        })
    return {
        "paths": shuffled, "mapping": mapping, "seed": seed, "strategy": strategy,
        "stable_sample_seed": stable_seed, "requested_fraction": fraction,
        "target_count": target_count, "eligible_count": len(eligible),
        "actual_count": len(selected),
        "actual_fraction": len(selected) / len(leaf_ids) if leaf_ids else 0.0,
    }


def evaluate_counterfactual_tree_shuffled_path(
    sample: Sample, kind: str, judge: JsonModel, *, shuffle_seed: int, shuffle_fraction: float = 0.5,
    cross_sample_catalog: dict[int, list[dict[str, Any]]] | None = None,
    shuffle_strategy: str = "random_same_depth",
) -> dict[str, Any]:
    """Judge all leaves with a deterministic equal-depth path corruption."""
    cf = sample.counterfactuals[kind]
    if cf.get("eligibility") != "eligible":
        return {"eligible": False, "reason": cf.get("ineligibility_reason"), "route_mode": "shuffled_path"}
    plan = build_shuffled_leaf_paths(
        sample, shuffle_seed, shuffle_fraction, cross_sample_catalog, strategy=shuffle_strategy,
    )
    predicted, trace = set(), []
    mapping_by_id = {item["leaf_id"]: item for item in plan["mapping"]}
    for leaf_id in sorted(plan["paths"]):
        full_path = plan["paths"][leaf_id]
        leaf = full_path[-1]
        result = judge.call(
            JSON_SYSTEM,
            TREE_PATH_LEAF_PROMPT.format(
                query=sample.query, response=cf["response"], semantic_path=compact(full_path),
                leaf_id=leaf_id, criterion=leaf["criterion"], desired_state=leaf["desired_state"],
            ),
        )
        if not isinstance(result.get("passes"), bool):
            raise ValueError(f"Shuffled-path Tree leaf returned non-boolean passes for {leaf_id}")
        if not result["passes"]:
            predicted.add(leaf_id)
        trace.append({
            "step": "leaf_judge", "leaf_id": leaf_id, "semantic_path": full_path,
            "passes": result["passes"], "evidence": str(result.get("evidence", "")),
            **mapping_by_id[leaf_id],
        })
    gold = set(cf["target_leaf_ids"])
    universe = set(plan["paths"])
    metrics = binary_metrics(predicted, gold, universe)
    shuffled_ids = {item["leaf_id"] for item in plan["mapping"] if item["actually_shuffled"]}
    target_shuffled = gold & shuffled_ids
    metrics.update({
        "routing_false_negative_count": 0, "visited_leaf_rate": 1.0,
        "target_leaf_count": len(gold), "shuffled_target_count": len(target_shuffled),
        "shuffled_target_fraction": len(target_shuffled) / len(gold) if gold else 0.0,
    })
    return {
        "eligible": True, "predicted_failed_leaf_ids": sorted(predicted),
        "gold_failed_leaf_ids": sorted(gold), "visited_leaf_ids": sorted(universe),
        "routed_out_target_leaf_ids": [], "visited_internal_node_ids": [],
        "metrics": metrics, "calls": len(universe), "trace": trace,
        "route_mode": "shuffled_path",
        "shuffle": {key: value for key, value in plan.items() if key != "paths"},
    }


def evaluate_counterfactual_tree(
    sample: Sample,
    kind: str,
    judge: JsonModel,
    route_mode: str = "batch",
    *,
    routing_threshold: float = 0.30,
    routing_top_k: int = 2,
    force_safety: bool = True,
    shuffle_seed: int = 20260813,
    shuffle_fraction: float = 0.5,
    shuffle_cross_sample_catalog: dict[int, list[dict[str, Any]]] | None = None,
    shuffle_strategy: str = "random_same_depth",
) -> dict[str, Any]:
    if route_mode == "batch":
        return evaluate_counterfactual_tree_batch(sample, kind, judge)
    if route_mode == "sequential":
        return evaluate_counterfactual_tree_sequential(sample, kind, judge)
    if route_mode == "soft":
        return evaluate_counterfactual_tree_soft(
            sample,
            kind,
            judge,
            threshold=routing_threshold,
            top_k=routing_top_k,
            force_safety=force_safety,
        )
    if route_mode == "exhaustive_path":
        return evaluate_counterfactual_tree_exhaustive_path(sample, kind, judge)
    if route_mode == "shuffled_path":
        return evaluate_counterfactual_tree_shuffled_path(
            sample, kind, judge, shuffle_seed=shuffle_seed, shuffle_fraction=shuffle_fraction,
            cross_sample_catalog=shuffle_cross_sample_catalog, shuffle_strategy=shuffle_strategy,
        )
    raise ValueError(f"Unknown Tree route mode: {route_mode}")


def _children_for_weight(node: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for leaf in node.get("leaves", []):
        rows.append({"id": leaf["leaf_id"], "kind": "leaf", "criterion": leaf["criterion"]})
    for child in node.get("children", []):
        rows.append({"id": _node_id(child), "kind": "internal", "name": child.get("name"), "description": child.get("description", "")})
    return rows


def _validate_importance_scores(data: dict[str, Any], expected_ids: set[str], label: str) -> list[dict[str, Any]]:
    returned = require_exact_ids(data.get("weights", []), "id", expected_ids, label)
    for row in returned:
        value = row.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label}: importance value must be an integer from 1 to 5")
        numeric = float(value)
        if not numeric.is_integer() or not 1 <= numeric <= 5:
            raise ValueError(f"{label}: importance value must be an integer from 1 to 5, got {value!r}")
        row["value"] = int(numeric)
    return returned


def _candidate_pair(sample: Sample) -> tuple[str, str]:
    candidates = sample.seed.get("candidate_responses", {})
    response_a = str(candidates.get("A", "")).strip()
    response_b = str(candidates.get("B", "")).strip()
    if not response_a or not response_b:
        raise ValueError("response-conditioned weighting requires complete Candidate A/B responses")
    return response_a, response_b


def assign_tree_weights(
    sample: Sample,
    model: JsonModel,
    response_conditioned: bool = False,
    full_tree_context: bool = False,
) -> dict[str, Any]:
    leaf_weights: dict[str, float] = {}
    groups: list[dict[str, Any]] = []
    calls = 0
    response_a, response_b = _candidate_pair(sample) if response_conditioned else ("", "")

    def visit(node: dict[str, Any], path: list[str], parent_mass: float) -> None:
        nonlocal calls
        children = _children_for_weight(node)
        if not children:
            return
        if response_conditioned:
            prompt = RESPONSE_CONDITIONED_TREE_WEIGHT_PROMPT.format(
                query=sample.query,
                response_a=response_a,
                response_b=response_b,
                path=" > ".join(path),
                children=compact(children),
            )
        elif full_tree_context:
            prompt = FULL_TREE_TOP_DOWN_WEIGHT_PROMPT.format(
                query=sample.query,
                full_tree=compact(sample.gold_tree),
                path=" > ".join(path),
                children=compact(children),
            )
        else:
            prompt = WEIGHT_PROMPT.format(query=sample.query, path=" > ".join(path), children=compact(children))
        expected_ids = {child["id"] for child in children}
        returned = _validated_call(
            model,
            prompt,
            lambda data: _validate_importance_scores(data, expected_ids, f"weights at {path}"),
            "tree_weight_assignment",
        )
        calls += 1
        values = {str(row.get("id")): row.get("value", 0.0) for row in returned}
        ids = [child["id"] for child in children]
        if set(values) != set(ids):
            raise ValueError(f"Weight model at {path} did not return every child")
        ratios = normalize_ratios(ids, values)
        groups.append({"path": path, "parent_mass": parent_mass, "ratios": ratios, "raw": returned})
        child_nodes = {_node_id(child): child for child in node.get("children", [])}
        leaf_ids = {leaf["leaf_id"] for leaf in node.get("leaves", [])}
        for child_id in ids:
            mass = parent_mass * ratios[child_id]
            if child_id in leaf_ids:
                leaf_weights[child_id] = mass
            else:
                child = child_nodes[child_id]
                visit(child, path + [str(child.get("name", child_id))], mass)

    root = {"node_id": sample.gold_tree.get("root_id", "root"), "leaves": [], "children": sample.gold_tree.get("dimensions", [])}
    visit(root, [sample.gold_tree.get("name", "Overall response quality")], 1.0)
    if set(leaf_weights) != {leaf["leaf_id"] for leaf in sample.gold_flat}:
        raise ValueError("Tree weighting did not reach every Gold leaf")
    total = sum(leaf_weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Tree leaf weights sum to {total}, expected 1")
    return {
        "representation": "tree_full_context" if full_tree_context else "tree",
        "weighting_variant": (
            "response_conditioned_positive_integer_1to5" if response_conditioned
            else "full_tree_context_top_down_positive_integer_1to5" if full_tree_context
            else "positive_integer_1to5"
        ),
        "leaf_weights": leaf_weights,
        "groups": groups,
        "calls": calls,
    }


def assign_flat_weights(sample: Sample, model: JsonModel, response_conditioned: bool = False) -> dict[str, Any]:
    leaves = gold_leaf_payload(sample)
    if response_conditioned:
        response_a, response_b = _candidate_pair(sample)
        prompt = RESPONSE_CONDITIONED_FLAT_WEIGHT_PROMPT.format(
            query=sample.query,
            response_a=response_a,
            response_b=response_b,
            leaves=compact(leaves),
        )
    else:
        prompt = FLAT_WEIGHT_PROMPT.format(query=sample.query, leaves=compact(leaves))
    expected_ids = {leaf["leaf_id"] for leaf in leaves}
    returned = _validated_call(
        model,
        prompt,
        lambda data: _validate_importance_scores(data, expected_ids, "flat weights"),
        "flat_weight_assignment",
    )
    values = {str(row.get("id")): row.get("value", 0.0) for row in returned}
    ids = [leaf["leaf_id"] for leaf in leaves]
    if set(values) != set(ids):
        raise ValueError("Flat weight model did not return every Gold leaf")
    weights = normalize_ratios(ids, values)
    return {
        "representation": "flat",
        "weighting_variant": "response_conditioned_positive_integer_1to5" if response_conditioned else "positive_integer_1to5",
        "leaf_weights": weights,
        "raw": returned,
        "calls": 1,
    }


def assign_flat_single_leaf_weights(sample: Sample, model: JsonModel) -> dict[str, Any]:
    """Independently score each flat leaf without Tree or sibling context."""
    raw: list[dict[str, Any]] = []
    leaves = gold_leaf_payload(sample)

    def validate(data: dict[str, Any], leaf_id: str) -> dict[str, Any]:
        if str(data.get("id", "")) != leaf_id:
            raise ValueError(f"flat single-leaf weight must return id={leaf_id!r}")
        value = data.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("flat single-leaf importance must be an integer from 1 to 5")
        numeric = float(value)
        if not numeric.is_integer() or not 1 <= numeric <= 5:
            raise ValueError(f"flat single-leaf importance must be an integer from 1 to 5, got {value!r}")
        return {"id": leaf_id, "value": int(numeric), "reason": str(data.get("reason", ""))}

    for leaf in leaves:
        leaf_id = str(leaf["leaf_id"])
        prompt = FLAT_SINGLE_LEAF_WEIGHT_PROMPT.format(
            query=sample.query,
            leaf_id=leaf_id,
            criterion=leaf["criterion"],
            desired_state=leaf["desired_state"],
        )
        returned = _validated_call(
            model,
            prompt,
            lambda data, item=leaf_id: validate(data, item),
            f"flat_single_leaf_weight_assignment:{leaf_id}",
        )
        raw.append(returned)
    ids = [str(row["id"]) for row in raw]
    expected = {str(leaf["leaf_id"]) for leaf in leaves}
    if len(ids) != len(expected) or len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Flat single-leaf weighting did not reach every Gold leaf exactly once")
    weights = normalize_ratios(ids, {str(row["id"]): int(row["value"]) for row in raw})
    return {
        "representation": "flat_single",
        "weighting_variant": "flat_single_leaf_independent_positive_integer_1to5",
        "leaf_weights": weights,
        "raw": raw,
        "calls": len(raw),
    }


def assign_path_leaf_weights(sample: Sample, model: JsonModel, full_tree_context: bool = False) -> dict[str, Any]:
    """Independently score each leaf with its full root-to-leaf semantic path."""
    raw: list[dict[str, Any]] = []
    expected = {str(leaf["leaf_id"]) for leaf in sample.gold_flat}

    root_context = {
        "node_id": str(sample.gold_tree.get("root_id", "root")),
        "name": str(sample.gold_tree.get("name", "Overall response quality")),
        "description": str(sample.gold_tree.get("description", "")),
        "assignment_summary": str(sample.gold_tree.get("assignment_summary", "")),
    }

    def node_context(node: dict[str, Any]) -> dict[str, str]:
        return {
            "node_id": _node_id(node),
            "name": str(node.get("name", "")),
            "description": str(node.get("description", "")),
            "assignment_summary": str(node.get("assignment_summary", "")),
        }

    def validate(data: dict[str, Any], leaf_id: str) -> dict[str, Any]:
        if str(data.get("id", "")) != leaf_id:
            raise ValueError(f"path leaf weight must return id={leaf_id!r}")
        value = data.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("path leaf importance must be an integer from 1 to 5")
        numeric = float(value)
        if not numeric.is_integer() or not 1 <= numeric <= 5:
            raise ValueError(f"path leaf importance must be an integer from 1 to 5, got {value!r}")
        return {"id": leaf_id, "value": int(numeric), "reason": str(data.get("reason", ""))}

    def visit(node: dict[str, Any], semantic_path: list[dict[str, str]]) -> None:
        for leaf in node.get("leaves", []):
            leaf_id = str(leaf["leaf_id"])
            full_path = semantic_path + [{
                "leaf_id": leaf_id,
                "criterion": str(leaf["criterion"]),
                "desired_state": str(leaf["desired_state"]),
            }]
            prompt_template = FULL_TREE_PATH_LEAF_WEIGHT_PROMPT if full_tree_context else PATH_LEAF_WEIGHT_PROMPT
            prompt = prompt_template.format(
                query=sample.query,
                full_tree=compact(sample.gold_tree),
                semantic_path=compact(full_path),
                leaf_id=leaf_id,
                criterion=leaf["criterion"],
                desired_state=leaf["desired_state"],
            )
            returned = _validated_call(
                model,
                prompt,
                lambda data, item=leaf_id: validate(data, item),
                f"path_leaf_weight_assignment:{leaf_id}",
            )
            raw.append({**returned, "semantic_path": full_path})
        for child in node.get("children", []):
            visit(child, semantic_path + [node_context(child)])

    synthetic_root = {"leaves": [], "children": sample.gold_tree.get("dimensions", [])}
    visit(synthetic_root, [root_context])
    ids = [str(row["id"]) for row in raw]
    if len(ids) != len(expected) or len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Path weighting did not reach every Gold leaf exactly once")
    values = {str(row["id"]): int(row["value"]) for row in raw}
    weights = normalize_ratios(ids, values)
    return {
        "representation": "path_full_context" if full_tree_context else "path",
        "weighting_variant": (
            "full_tree_context_path_leaf_independent_positive_integer_1to5"
            if full_tree_context else "path_leaf_independent_positive_integer_1to5"
        ),
        "leaf_weights": weights,
        "raw": raw,
        "calls": len(raw),
    }


def score_preference(sample: Sample, judge: JsonModel, weights_by_method: dict[str, dict[str, float]]) -> dict[str, Any]:
    candidates = sample.seed.get("candidate_responses", {})
    gold = sample.seed.get("gold_preference")
    if gold not in {"A", "B"} or not candidates.get("A") or not candidates.get("B"):
        return {"eligible": False, "reason": "sample has no complete labeled A/B preference pair"}
    pass_a, raw_a = _flat_leaf_judgment(sample, candidates["A"], judge)
    pass_b, raw_b = _flat_leaf_judgment(sample, candidates["B"], judge)
    outcomes = {}
    for method, weights in weights_by_method.items():
        if set(weights) != set(pass_a):
            raise ValueError(f"{method}: weight/leaf ID mismatch")
        score_a = sum(weights[leaf_id] * float(pass_a[leaf_id]) for leaf_id in weights)
        score_b = sum(weights[leaf_id] * float(pass_b[leaf_id]) for leaf_id in weights)
        outcomes[method] = preference_outcome(score_a, score_b, gold)
    return {"eligible": True, "gold_preference": gold, "outcomes": outcomes, "leaf_passes_a": pass_a, "leaf_passes_b": pass_b, "calls": 2, "raw_a": raw_a, "raw_b": raw_b}


def summarize_rows(rows: list[dict[str, Any]], metric_path: tuple[str, ...]) -> dict[str, Any]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = tuple(str(row.get(field, "")) for field in ("experiment", "representation", "judge"))
        grouped[key].append(row)
    output = {}
    for key, items in grouped.items():
        values = []
        for item in items:
            value: Any = item
            for part in metric_path:
                value = value.get(part, {}) if isinstance(value, dict) else None
            if isinstance(value, (int, float)):
                values.append(float(value))
        output["|".join(key)] = {"n": len(values), "mean": sum(values) / len(values) if values else None}
    return output
