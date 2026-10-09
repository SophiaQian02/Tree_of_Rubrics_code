#!/usr/bin/env python3
"""Run the three pre-registered Gold-Rubric mechanism experiments."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import hashlib
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.mechanism_analysis.core import GoldDataset, JsonModel, JsonlSink, ModelSpec, json_digest, read_jsonl, safe_mean
from project.mechanism_analysis.experiments import (
    assign_flat_weights,
    assign_flat_single_leaf_weights,
    assign_path_leaf_weights,
    assign_tree_weights,
    build_shuffled_leaf_paths,
    build_cross_sample_path_catalog,
    evaluate_counterfactual_flat,
    evaluate_counterfactual_flat_single_leaf,
    evaluate_counterfactual_tree,
    evaluate_coverage,
    generate_flat,
    generate_tree,
    score_preference,
)
from project.mechanism_analysis.prompts import TREE_PATH_LEAF_PROMPT

DATA = REPO_ROOT / "data/gold_rubric_1500"
DEFAULT_OUTPUT = REPO_ROOT / "project/outputs/gold_rubric_mechanisms"
TREE_PATH_LEAF_PROMPT_SHA256 = hashlib.sha256(TREE_PATH_LEAF_PROMPT.encode("utf-8")).hexdigest()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def latest_rows(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        if key not in latest or str(row.get("finished_at", "")) >= str(latest[key].get("finished_at", "")):
            latest[key] = row
    return list(latest.values())


def dataset(args: argparse.Namespace) -> GoldDataset:
    return GoldDataset(args.seed, args.gold_tree, args.gold_flat, args.counterfactuals)


def run_parallel(
    jobs: list[tuple[dict[str, Any], Callable[[], dict[str, Any]]]],
    sink: JsonlSink,
    concurrency: int,
) -> Counter:
    counts: Counter = Counter()

    def execute(meta: dict[str, Any], fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        started = now()
        try:
            payload = fn()
            return {**meta, "status": "ok", "started_at": started, "finished_at": now(), **payload}
        except Exception as exc:
            return {
                **meta,
                "status": "error",
                "started_at": started,
                "finished_at": now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }

    pending = [(meta, fn) for meta, fn in jobs if sink.key(meta) not in sink.done]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(execute, meta, fn) for meta, fn in pending]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            sink.append(row)
            counts[row["status"]] += 1
            if index % 10 == 0 or index == len(futures):
                print(f"[{index}/{len(futures)}] ok={counts['ok']} error={counts['error']}", flush=True)
    counts["skipped_existing"] = len(jobs) - len(pending)
    return counts


def cmd_validate(args: argparse.Namespace) -> int:
    data = dataset(args)
    cf = Counter()
    preference = 0
    depths: list[int] = []

    def depth(node: dict[str, Any]) -> int:
        return 1 + max((depth(child) for child in node.get("children", [])), default=0)

    for sample in data.samples:
        preference += int(sample.seed.get("gold_preference") in {"A", "B"})
        depths.append(max((depth(dim) for dim in sample.gold_tree.get("dimensions", [])), default=0))
        for kind, item in sample.counterfactuals.items():
            cf[f"{kind}:{item.get('eligibility')}"] += 1
    report = {
        "joined_samples": len(data.samples),
        "preference_pairs": preference,
        "counterfactuals": dict(sorted(cf.items())),
        "tree_depth_min": min(depths),
        "tree_depth_max": max(depths),
        "tree_flat_leaf_identity": True,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    if args.max_depth < 1 or args.max_leaves < 2:
        raise SystemExit("--max-depth must be >=1 and --max-leaves must be >=2")
    if args.local_check_max_repairs < 0:
        raise SystemExit("--local-check-max-repairs must be >=0")
    data = dataset(args)
    spec = ModelSpec.parse(args.generator)
    sink = JsonlSink(args.output, ("sample_id", "representation", "generator", "input_sha256"))
    jobs = []
    for sample in data.subset(args.limit):
        base = {
            "experiment": "coverage_generation",
            "sample_id": sample.sample_id,
            "generator": spec.name,
            "generator_model": spec.model,
            "generator_base_urls": list(spec.base_urls),
        }
        if args.representation in {"tree", "both"}:
            tree_config = {
                "variant": "tree_local_check" if args.tree_local_check else "tree_raw",
                "max_depth": args.max_depth,
                "max_leaves": args.max_leaves,
                "local_check_max_repairs": args.local_check_max_repairs,
            }
            meta = {
                **base,
                "representation": "tree",
                "generation_variant": tree_config["variant"],
                "input_sha256": json_digest({"query": sample.query, "config": tree_config}),
            }
            jobs.append(
                (
                    meta,
                    lambda s=sample: {
                        "generated": generate_tree(
                            s,
                            JsonModel(spec, args.temperature),
                            args.max_depth,
                            args.max_leaves,
                            local_check=args.tree_local_check,
                            local_check_max_repairs=args.local_check_max_repairs,
                        )
                    },
                )
            )
        if args.representation in {"flat", "both"}:
            meta = {
                **base,
                "representation": "flat",
                "generation_variant": "flat_raw",
                "input_sha256": json_digest({"query": sample.query, "config": {"variant": "flat_raw", "max_leaves": args.max_leaves}}),
            }
            jobs.append((meta, lambda s=sample: {"generated": generate_flat(s, JsonModel(spec, args.temperature), args.max_leaves)}))
    print(dict(run_parallel(jobs, sink, args.concurrency)))
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    data = dataset(args)
    by_id = {sample.sample_id: sample for sample in data.subset(args.limit)}
    generated_rows = latest_rows(
        [row for row in read_jsonl(args.generated) if row.get("status") == "ok" and row.get("sample_id") in by_id],
        ("sample_id", "representation", "generator"),
    )
    judges = [ModelSpec.parse(value) for value in args.judge]
    sink = JsonlSink(args.output, ("sample_id", "representation", "generator", "judge", "input_sha256"))
    jobs = []
    for source in generated_rows:
        sample = by_id[source["sample_id"]]
        for judge_spec in judges:
            input_hash = json_digest({"query": sample.query, "gold": sample.gold_flat, "generated": source["generated"]})
            generated = source["generated"]
            local_check = generated.get("local_check", {})
            meta = {
                "experiment": "coverage_uniqueness",
                "sample_id": sample.sample_id,
                "representation": source["representation"],
                "generation_variant": source.get("generation_variant") or generated.get("generation_variant") or f"{source['representation']}_raw",
                "generator": source["generator"],
                "judge": judge_spec.name,
                "judge_model": judge_spec.model,
                "judge_base_urls": list(judge_spec.base_urls),
                "generated_sha256": json_digest(source["generated"]),
                "generation_calls": generated.get("calls"),
                "generated_leaf_count": len(generated.get("leaves", [])),
                "local_check_summary": {key: value for key, value in local_check.items() if key != "events"},
                "input_sha256": input_hash,
            }
            jobs.append((meta, lambda s=sample, g=source["generated"], j=judge_spec: evaluate_coverage(s, g, JsonModel(j))))
    print(dict(run_parallel(jobs, sink, args.concurrency)))
    return 0


def cmd_counterfactual(args: argparse.Namespace) -> int:
    if not 0.0 <= args.routing_threshold <= 1.0:
        raise SystemExit("--routing-threshold must be in [0,1]")
    if args.routing_top_k < 0:
        raise SystemExit("--routing-top-k must be >=0")
    if not 0.0 <= args.shuffle_fraction <= 1.0:
        raise SystemExit("--shuffle-fraction must be in [0,1]")
    if not 0.0 < args.shuffle_case_fraction <= 1.0:
        raise SystemExit("--shuffle-case-fraction must be in (0,1]")
    data = dataset(args)
    judges = [ModelSpec.parse(value) for value in args.judge]
    sink = JsonlSink(args.output, ("sample_id", "counterfactual_class", "representation", "judge", "input_sha256"))
    jobs = []
    selected_shuffle_cases: set[tuple[str, str]] | None = None
    shuffle_cross_sample_catalog = None
    shuffle_catalog_sha256 = None
    shuffle_selection_report: dict[str, Any] = {}
    if args.tree_route_mode == "shuffled_path" and (
        args.shuffle_require_all_leaves or args.shuffle_strategy == "hard_dimension"
    ):
        shuffle_cross_sample_catalog = build_cross_sample_path_catalog(data.subset(args.limit))
        shuffle_catalog_sha256 = json_digest({
            str(depth): [
                (row["sample_id"], row["leaf_id"], row["internal_node_ids"], row["top_dimension_name"])
                for row in rows
            ]
            for depth, rows in sorted(shuffle_cross_sample_catalog.items())
        })
    if args.tree_route_mode == "shuffled_path" and args.shuffle_case_fraction < 1.0:
        selected_shuffle_cases = set()
        for kind in sorted({name for sample in data.subset(args.limit) for name in sample.counterfactuals}):
            all_cases = [
                sample for sample in data.subset(args.limit)
                if sample.counterfactuals.get(kind, {}).get("eligibility") == "eligible"
            ]
            target = int(len(all_cases) * args.shuffle_case_fraction + 0.5)
            selection_pool = all_cases
            excluded_unshuffleable = []
            if args.shuffle_require_all_leaves:
                selection_pool = []
                for sample in all_cases:
                    plan = build_shuffled_leaf_paths(
                        sample, args.shuffle_seed, args.shuffle_fraction, shuffle_cross_sample_catalog,
                        strategy=args.shuffle_strategy,
                    )
                    if plan["actual_count"] == len(sample.gold_flat):
                        selection_pool.append(sample)
                    else:
                        excluded_unshuffleable.append(sample.sample_id)
                if len(selection_pool) < target:
                    raise ValueError(
                        f"{kind}: only {len(selection_pool)} fully shuffleable cases are available, "
                        f"fewer than requested target={target}"
                    )
            ranked = sorted(
                selection_pool,
                key=lambda sample: hashlib.sha256(
                    f"{args.shuffle_seed}|{kind}|{sample.sample_id}".encode("utf-8")
                ).digest(),
            )
            chosen = ranked[:target]
            selected_shuffle_cases.update((sample.sample_id, kind) for sample in chosen)
            shuffle_selection_report[kind] = {
                "all_eligible_cases": len(all_cases), "selection_pool": len(selection_pool),
                "excluded_unshuffleable_cases": len(excluded_unshuffleable),
                "selected_cases": len(chosen), "case_fraction": len(chosen) / len(all_cases) if all_cases else 0.0,
            }
        print(json.dumps({"shuffle_case_selection": shuffle_selection_report}, ensure_ascii=False, indent=2), flush=True)
    for sample in data.subset(args.limit):
        for kind in sorted(sample.counterfactuals):
            if sample.counterfactuals[kind].get("eligibility") != "eligible":
                continue
            if selected_shuffle_cases is not None and (sample.sample_id, kind) not in selected_shuffle_cases:
                continue
            for judge_spec in judges:
                base = {
                    "experiment": "counterfactual_localization",
                    "sample_id": sample.sample_id,
                    "counterfactual_class": kind,
                    "judge": judge_spec.name,
                    "judge_model": judge_spec.model,
                    "judge_base_urls": list(judge_spec.base_urls),
                }
                tree_meta = {
                    **base,
                    "representation": "tree",
                    "tree_route_mode": args.tree_route_mode,
                    "routing_threshold": args.routing_threshold,
                    "routing_top_k": args.routing_top_k,
                    "force_safety": args.force_safety,
                    "shuffle_case_fraction": args.shuffle_case_fraction,
                    "shuffle_require_all_leaves": args.shuffle_require_all_leaves,
                    "shuffle_strategy": args.shuffle_strategy,
                    "shuffle_donor_policy": (
                        "fixed_mismatched_dimension_within_tree_then_cross_tree_same_depth"
                        if args.shuffle_strategy == "hard_dimension"
                        else "within_tree_then_cross_tree_same_depth"
                        if shuffle_cross_sample_catalog
                        else "within_tree_only"
                    ),
                    "shuffle_catalog_sha256": shuffle_catalog_sha256,
                    "tree_path_leaf_prompt_sha256": (
                        TREE_PATH_LEAF_PROMPT_SHA256
                        if args.tree_route_mode in {"exhaustive_path", "shuffled_path"}
                        else None
                    ),
                    "input_sha256": json_digest({
                        "query": sample.query,
                        "gold_tree": sample.gold_tree,
                        "counterfactual": sample.counterfactuals[kind],
                        "tree_route_mode": args.tree_route_mode,
                        "routing_threshold": args.routing_threshold,
                        "routing_top_k": args.routing_top_k,
                        "force_safety": args.force_safety,
                        "shuffle_seed": args.shuffle_seed,
                        "shuffle_fraction": args.shuffle_fraction,
                        "shuffle_case_fraction": args.shuffle_case_fraction,
                        "shuffle_require_all_leaves": args.shuffle_require_all_leaves,
                        "shuffle_strategy": args.shuffle_strategy,
                        "shuffle_donor_policy": (
                            "fixed_mismatched_dimension_within_tree_then_cross_tree_same_depth"
                            if args.shuffle_strategy == "hard_dimension"
                            else "within_tree_then_cross_tree_same_depth"
                            if shuffle_cross_sample_catalog
                            else "within_tree_only"
                        ),
                        "shuffle_catalog_sha256": shuffle_catalog_sha256,
                        "tree_path_leaf_prompt_sha256": (
                            TREE_PATH_LEAF_PROMPT_SHA256
                            if args.tree_route_mode in {"exhaustive_path", "shuffled_path"}
                            else None
                        ),
                    }),
                }
                if args.representation in {"tree", "both"}:
                    jobs.append(
                        (
                            tree_meta,
                            lambda s=sample, k=kind, j=judge_spec: evaluate_counterfactual_tree(
                                s,
                                k,
                                JsonModel(j),
                                route_mode=args.tree_route_mode,
                                routing_threshold=args.routing_threshold,
                                routing_top_k=args.routing_top_k,
                                force_safety=args.force_safety,
                                shuffle_seed=args.shuffle_seed,
                                shuffle_fraction=args.shuffle_fraction,
                                shuffle_cross_sample_catalog=shuffle_cross_sample_catalog,
                                shuffle_strategy=args.shuffle_strategy,
                            ),
                        )
                    )
                flat_meta = {
                    **base,
                    "representation": "flat",
                    "flat_leaf_mode": args.flat_leaf_mode,
                    "input_sha256": json_digest({
                        "query": sample.query,
                        "gold_flat": sample.gold_flat,
                        "counterfactual": sample.counterfactuals[kind],
                        "flat_leaf_mode": args.flat_leaf_mode,
                    }),
                }
                if args.representation in {"flat", "both"}:
                    if args.flat_leaf_mode == "single":
                        jobs.append((flat_meta, lambda s=sample, k=kind, j=judge_spec: evaluate_counterfactual_flat_single_leaf(s, k, JsonModel(j))))
                    else:
                        jobs.append((flat_meta, lambda s=sample, k=kind, j=judge_spec: evaluate_counterfactual_flat(s, k, JsonModel(j))))
    print(dict(run_parallel(jobs, sink, args.concurrency)))
    return 0


def cmd_weights(args: argparse.Namespace) -> int:
    data = dataset(args)
    spec = ModelSpec.parse(args.generator)
    weighting_variant = "response_conditioned_positive_integer_1to5" if args.response_conditioned else "positive_integer_1to5"
    sink = JsonlSink(args.output, ("sample_id", "representation", "weight_model", "input_sha256"))
    jobs = []
    for sample in data.subset(args.limit):
        if sample.seed.get("gold_preference") not in {"A", "B"}:
            continue
        base = {
            "experiment": "gold_weight_assignment",
            "sample_id": sample.sample_id,
            "weight_model": spec.name,
            "weight_model_id": spec.model,
            "weight_model_base_urls": list(spec.base_urls),
            "weighting_variant": weighting_variant,
            "input_sha256": json_digest({
                "query": sample.query,
                "gold_tree": sample.gold_tree,
                "gold_flat": sample.gold_flat,
                "candidate_responses": sample.seed.get("candidate_responses") if args.response_conditioned else None,
                "weighting_variant": weighting_variant,
            }),
        }
        jobs.append(({**base, "representation": "tree"}, lambda s=sample: {"weighting": assign_tree_weights(s, JsonModel(spec, args.temperature), response_conditioned=args.response_conditioned)}))
        jobs.append(({**base, "representation": "flat"}, lambda s=sample: {"weighting": assign_flat_weights(s, JsonModel(spec, args.temperature), response_conditioned=args.response_conditioned)}))
    print(dict(run_parallel(jobs, sink, args.concurrency)))
    return 0


def cmd_path_weights(args: argparse.Namespace) -> int:
    data = dataset(args)
    spec = ModelSpec.parse(args.generator)
    weighting_variant = "path_leaf_independent_positive_integer_1to5"
    sink = JsonlSink(args.output, ("sample_id", "representation", "weight_model", "input_sha256"))
    jobs = []
    for sample in data.subset(args.limit):
        if sample.seed.get("gold_preference") not in {"A", "B"}:
            continue
        meta = {
            "experiment": "gold_weight_assignment",
            "sample_id": sample.sample_id,
            "representation": "path",
            "weight_model": spec.name,
            "weight_model_id": spec.model,
            "weight_model_base_urls": list(spec.base_urls),
            "weighting_variant": weighting_variant,
            "input_sha256": json_digest({
                "query": sample.query,
                "gold_tree": sample.gold_tree,
                "weighting_variant": weighting_variant,
            }),
        }
        jobs.append((meta, lambda s=sample: {"weighting": assign_path_leaf_weights(s, JsonModel(spec, args.temperature))}))
    print(dict(run_parallel(jobs, sink, args.concurrency)))
    return 0


def cmd_tree_context_weight_ablation(args: argparse.Namespace) -> int:
    """Run the 2x2 Tree weighting context ablation on preference-eligible samples."""
    data = dataset(args)
    spec = ModelSpec.parse(args.generator)
    sink = JsonlSink(args.output, ("sample_id", "representation", "weight_model", "input_sha256"))
    jobs = []
    variants = (
        ("tree", "top_down_local", lambda sample, model: assign_tree_weights(sample, model)),
        ("tree_full_context", "top_down_full_tree", lambda sample, model: assign_tree_weights(sample, model, full_tree_context=True)),
        ("path", "path_leaf_local", lambda sample, model: assign_path_leaf_weights(sample, model)),
        ("path_full_context", "path_leaf_full_tree", lambda sample, model: assign_path_leaf_weights(sample, model, full_tree_context=True)),
    )
    for sample in data.subset(args.limit):
        if sample.seed.get("gold_preference") not in {"A", "B"}:
            continue
        for representation, variant, assigner in variants:
            meta = {
                "experiment": "gold_weight_tree_context_ablation",
                "sample_id": sample.sample_id,
                "representation": representation,
                "weight_model": spec.name,
                "weight_model_id": spec.model,
                "weight_model_base_urls": list(spec.base_urls),
                "weighting_variant": variant,
                "input_sha256": json_digest({
                    "query": sample.query,
                    "gold_tree": sample.gold_tree,
                    "variant": variant,
                }),
            }
            jobs.append((meta, lambda s=sample, fn=assigner: {"weighting": fn(s, JsonModel(spec, args.temperature))}))
    print(dict(run_parallel(jobs, sink, args.concurrency)))
    return 0


def cmd_flat_context_weight_ablation(args: argparse.Namespace) -> int:
    """Compare one-at-a-time Flat weighting with all-leaves-at-once Flat weighting."""
    data = dataset(args)
    spec = ModelSpec.parse(args.generator)
    sink = JsonlSink(args.output, ("sample_id", "representation", "weight_model", "input_sha256"))
    jobs = []
    variants = (
        ("flat_single", "flat_single_leaf", lambda sample, model: assign_flat_single_leaf_weights(sample, model)),
        ("flat_all", "flat_all_leaves", lambda sample, model: assign_flat_weights(sample, model)),
    )
    for sample in data.subset(args.limit):
        if sample.seed.get("gold_preference") not in {"A", "B"}:
            continue
        for representation, variant, assigner in variants:
            meta = {
                "experiment": "gold_weight_flat_context_ablation",
                "sample_id": sample.sample_id,
                "representation": representation,
                "weight_model": spec.name,
                "weight_model_id": spec.model,
                "weight_model_base_urls": list(spec.base_urls),
                "weighting_variant": variant,
                "input_sha256": json_digest({
                    "query": sample.query,
                    "gold_flat": sample.gold_flat,
                    "variant": variant,
                }),
            }
            jobs.append((meta, lambda s=sample, fn=assigner: {"weighting": fn(s, JsonModel(spec, args.temperature))}))
    print(dict(run_parallel(jobs, sink, args.concurrency)))
    return 0


def cmd_preference(args: argparse.Namespace) -> int:
    data = dataset(args)
    by_id = {sample.sample_id: sample for sample in data.subset(args.limit)}
    weight_rows = latest_rows(
        [row for row in read_jsonl(args.weights) if row.get("status") == "ok" and row.get("sample_id") in by_id],
        ("sample_id", "representation", "weight_model"),
    )
    grouped: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    grouped_variants: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in weight_rows:
        group_key = (row["sample_id"], row["weight_model"])
        grouped[group_key][row["representation"]] = row["weighting"]["leaf_weights"]
        grouped_variants[group_key].add(str(row.get("weighting_variant") or row["weighting"].get("weighting_variant", "legacy_raw")))
    judges = [ModelSpec.parse(value) for value in args.judge]
    sink = JsonlSink(args.output, ("sample_id", "weight_model", "judge", "input_sha256"))
    jobs = []
    for (sample_id, weight_model), methods in grouped.items():
        if set(methods) != {"tree", "flat"}:
            continue
        sample = by_id[sample_id]
        weighting_variant = "+".join(sorted(grouped_variants[(sample_id, weight_model)]))
        for judge_spec in judges:
            input_hash = json_digest({
                "query": sample.query,
                "gold": sample.gold_flat,
                "candidates": sample.seed.get("candidate_responses"),
                "gold_preference": sample.seed.get("gold_preference"),
                "weights": methods,
            })
            meta = {
                "experiment": "weighted_preference",
                "sample_id": sample_id,
                "representation": "shared_leaf_judgment",
                "weight_model": weight_model,
                "weighting_variant": weighting_variant,
                "judge": judge_spec.name,
                "judge_model": judge_spec.model,
                "judge_base_urls": list(judge_spec.base_urls),
                "input_sha256": input_hash,
            }
            jobs.append((meta, lambda s=sample, j=judge_spec, w=methods: score_preference(s, JsonModel(j), w)))
    print(dict(run_parallel(jobs, sink, args.concurrency)))
    return 0


def _mean_metric(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    values = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)):
            values.append(float(value))
    return safe_mean(values)


def cmd_summarize(args: argparse.Namespace) -> int:
    raw_rows = [row for path in args.input for row in read_jsonl(path) if row.get("status") == "ok"]
    # Older coverage outputs predate generation_variant.  Infer it before
    # deduplication so a failed local-check result cannot silently fall back to
    # (and become mixed with) an older raw Tree result for the same sample.
    for row in raw_rows:
        if row.get("experiment") == "coverage_uniqueness" and not row.get("generation_variant"):
            row["generation_variant"] = f"{row.get('representation', 'unknown')}_raw"
        if row.get("experiment") == "counterfactual_localization" and not row.get("tree_route_mode"):
            if row.get("representation") == "flat":
                row["tree_route_mode"] = "flat"
            else:
                row["tree_route_mode"] = str(row.get("route_mode") or "legacy_batch")
    identity_fields = {
        "coverage_uniqueness": (
            "experiment", "sample_id", "representation", "generation_variant", "generator", "judge"
        ),
        "counterfactual_localization": (
            "experiment", "sample_id", "counterfactual_class", "representation", "tree_route_mode", "judge"
        ),
        "weighted_preference": ("experiment", "sample_id", "weight_model", "judge"),
    }
    rows = []
    for experiment, fields in identity_fields.items():
        rows.extend(latest_rows([row for row in raw_rows if row.get("experiment") == experiment], fields))
    report: dict[str, Any] = {"created_at": now(), "files": [str(path) for path in args.input]}
    coverage = [row for row in rows if row.get("experiment") == "coverage_uniqueness"]
    report["coverage_uniqueness"] = {}
    for key in sorted({(row["generator"], row["representation"], row["generation_variant"], row["judge"]) for row in coverage}):
        items = [
            row for row in coverage
            if (row["generator"], row["representation"], row["generation_variant"], row["judge"]) == key
        ]
        report["coverage_uniqueness"]["|".join(key)] = {
            "n": len(items),
            "generation_variant": key[2],
            "mean_generation_calls": safe_mean(float(row["generation_calls"]) for row in items if isinstance(row.get("generation_calls"), (int, float))),
            "mean_generated_leaves": safe_mean(float(row["generated_leaf_count"]) for row in items if isinstance(row.get("generated_leaf_count"), (int, float))),
            "macro_coverage": _mean_metric(items, ("metrics", "coverage")),
            "uniqueness_rate": _mean_metric(items, ("metrics", "uniqueness_rate")),
            "redundancy_rate": _mean_metric(items, ("metrics", "redundancy_rate")),
            "off_task_rate": _mean_metric(items, ("metrics", "off_task_rate")),
            "micro_coverage": sum(row["metrics"]["gold_covered"] for row in items) / max(1, sum(row["metrics"]["gold_total"] for row in items)),
            "novel_category_counts": dict(sum((Counter(row["metrics"].get("novel_category_counts", {})) for row in items), Counter())),
            "local_check_totals": {
                field: sum(int(row.get("local_check_summary", {}).get(field, 0)) for row in items)
                for field in (
                    "checked_splits", "flagged_splits", "overlap_pairs", "repair_requests",
                    "repaired_splits", "unresolved_splits_collapsed",
                )
            },
        }
    counter = [row for row in rows if row.get("experiment") == "counterfactual_localization" and row.get("eligible")]
    report["counterfactual_localization"] = {}
    keys = {
        (row["representation"], row["tree_route_mode"], row["judge"], row["counterfactual_class"])
        for row in counter
    }
    for key in sorted(keys):
        items = [
            row for row in counter
            if (row["representation"], row["tree_route_mode"], row["judge"], row["counterfactual_class"]) == key
        ]
        report["counterfactual_localization"]["|".join(key)] = {
            "n": len(items),
            "tree_route_mode": key[1],
            "exact_set_accuracy": _mean_metric(items, ("metrics", "exact_match")),
            "leaf_precision": _mean_metric(items, ("metrics", "precision")),
            "leaf_recall": _mean_metric(items, ("metrics", "recall")),
            "leaf_f1": _mean_metric(items, ("metrics", "f1")),
            "mean_calls": _mean_metric(items, ("calls",)),
            "visited_leaf_rate": _mean_metric(items, ("metrics", "visited_leaf_rate")),
            "routing_false_negatives": sum(row["metrics"].get("routing_false_negative_count", 0) for row in items),
        }
    preference = [row for row in rows if row.get("experiment") == "weighted_preference" and row.get("eligible")]
    report["weighted_preference"] = {}
    for key in sorted({(row["weight_model"], row["judge"]) for row in preference}):
        items = [row for row in preference if (row["weight_model"], row["judge"]) == key]
        for method in ("tree", "flat"):
            report["weighted_preference"]["|".join((*key, method))] = {
                "n": len(items),
                "preference_accuracy_tie_half": safe_mean(row["outcomes"][method]["correct_tie_half"] for row in items),
                "tie_rate": safe_mean(float(row["outcomes"][method]["predicted"] == "tie") for row in items),
                "mean_score_margin": safe_mean(abs(row["outcomes"][method]["delta"]) for row in items),
            }
    paired: dict[str, Any] = {"tree_minus_flat": {}, "judge_agreement": {}}
    coverage_pairs: dict[tuple[str, str, str], dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for row in coverage:
        coverage_pairs[(row["sample_id"], row["generator"], row["judge"])][
            (row["representation"], row["generation_variant"])
        ] = row
    for generator, judge in sorted({(key[1], key[2]) for key in coverage_pairs}):
        cases = [value for key, value in coverage_pairs.items() if key[1:] == (generator, judge)]
        tree_variants = sorted({variant for value in cases for representation, variant in value if representation == "tree"})
        flat_variants = sorted({variant for value in cases for representation, variant in value if representation == "flat"})
        for tree_variant in tree_variants:
            for flat_variant in flat_variants:
                pairs = [
                    value for value in cases
                    if ("tree", tree_variant) in value and ("flat", flat_variant) in value
                ]
                tree_key = ("tree", tree_variant)
                flat_key = ("flat", flat_variant)
                paired["tree_minus_flat"][
                    f"coverage|{generator}|{judge}|{tree_variant}_vs_{flat_variant}"
                ] = {
                    "n": len(pairs),
                    "coverage_delta": safe_mean(
                        value[tree_key]["metrics"]["coverage"] - value[flat_key]["metrics"]["coverage"]
                        for value in pairs
                    ),
                    "uniqueness_delta": safe_mean(
                        value[tree_key]["metrics"]["uniqueness_rate"] - value[flat_key]["metrics"]["uniqueness_rate"]
                        for value in pairs
                    ),
                }
    counter_pairs: dict[tuple[str, str, str], dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for row in counter:
        counter_pairs[(row["sample_id"], row["counterfactual_class"], row["judge"])][
            (row["representation"], row["tree_route_mode"])
        ] = row
    for kind, judge in sorted({(key[1], key[2]) for key in counter_pairs}):
        cases = [value for key, value in counter_pairs.items() if key[1:] == (kind, judge)]
        tree_modes = sorted({mode for value in cases for representation, mode in value if representation == "tree"})
        for tree_mode in tree_modes:
            tree_key = ("tree", tree_mode)
            flat_key = ("flat", "flat")
            pairs = [value for value in cases if tree_key in value and flat_key in value]
            paired["tree_minus_flat"][f"localization|{kind}|{judge}|{tree_mode}_vs_flat"] = {
                "n": len(pairs),
                "exact_accuracy_delta": safe_mean(
                    value[tree_key]["metrics"]["exact_match"] - value[flat_key]["metrics"]["exact_match"]
                    for value in pairs
                ),
                "f1_delta": safe_mean(
                    value[tree_key]["metrics"]["f1"] - value[flat_key]["metrics"]["f1"]
                    for value in pairs
                ),
                "recall_delta": safe_mean(
                    value[tree_key]["metrics"]["recall"] - value[flat_key]["metrics"]["recall"]
                    for value in pairs
                ),
            }
    for key in sorted({(row["weight_model"], row["judge"]) for row in preference}):
        items = [row for row in preference if (row["weight_model"], row["judge"]) == key]
        paired["tree_minus_flat"][f"preference|{'|'.join(key)}"] = {
            "n": len(items),
            "accuracy_delta": safe_mean(row["outcomes"]["tree"]["correct_tie_half"] - row["outcomes"]["flat"]["correct_tie_half"] for row in items),
        }
    coverage_by_case: dict[tuple[str, str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in coverage:
        coverage_by_case[
            (
                row["sample_id"], row["generator"], row["representation"],
                row["generation_variant"], str(row.get("generated_sha256", "")),
            )
        ][row["judge"]] = row
    agreement_values = []
    class_agreement_values = []
    agreement_id_mismatch_cases = 0
    for by_judge in coverage_by_case.values():
        if len(by_judge) != 2:
            continue
        first, second = [by_judge[name] for name in sorted(by_judge)]
        first_gold = {item["gold_id"]: bool(item["covered"]) for item in first["gold_coverage"]}
        second_gold = {item["gold_id"]: bool(item["covered"]) for item in second["gold_coverage"]}
        first_gen = {item["generated_id"]: item["class"] for item in first["generated_classification"]}
        second_gen = {item["generated_id"]: item["class"] for item in second["generated_classification"]}
        if set(first_gold) != set(second_gold) or set(first_gen) != set(second_gen):
            agreement_id_mismatch_cases += 1
        agreement_values.extend(float(first_gold[key] == second_gold[key]) for key in sorted(set(first_gold) & set(second_gold)))
        class_agreement_values.extend(float(first_gen[key] == second_gen[key]) for key in sorted(set(first_gen) & set(second_gen)))
    paired["judge_agreement"]["coverage_leaf_decisions"] = {
        "n": len(agreement_values), "agreement": safe_mean(agreement_values),
        "id_mismatch_cases": agreement_id_mismatch_cases,
    }
    paired["judge_agreement"]["generated_rubric_classes"] = {
        "n": len(class_agreement_values), "agreement": safe_mean(class_agreement_values),
        "id_mismatch_cases": agreement_id_mismatch_cases,
    }
    report["paired_comparisons"] = paired
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=Path, default=DATA / "processed/gold_rubric_seed_1500.jsonl")
    parser.add_argument("--gold-tree", type=Path, default=DATA / "annotation/exports/gold_tree.jsonl")
    parser.add_argument("--gold-flat", type=Path, default=DATA / "annotation/exports/gold_flat.jsonl")
    parser.add_argument("--counterfactuals", type=Path, default=DATA / "annotation/exports/counterfactual_responses.jsonl")
    parser.add_argument("--limit", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-data")
    add_data_args(validate)
    validate.set_defaults(func=cmd_validate)

    generation = sub.add_parser("generate-rubrics")
    add_data_args(generation)
    generation.add_argument("--generator", required=True, help="name|base_url|model[|API_KEY_ENV]")
    generation.add_argument("--representation", choices=("tree", "flat", "both"), default="both")
    generation.add_argument("--max-depth", type=int, default=3)
    generation.add_argument("--max-leaves", type=int, default=16)
    generation.add_argument("--temperature", type=float, default=0.2)
    generation.add_argument("--tree-local-check", action="store_true", help="Check and repair sibling overlap after every proposed Tree split.")
    generation.add_argument("--local-check-max-repairs", type=int, default=2)
    generation.add_argument("--concurrency", type=int, default=8)
    generation.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "generated_rubrics.jsonl")
    generation.set_defaults(func=cmd_generate)

    coverage = sub.add_parser("coverage")
    add_data_args(coverage)
    coverage.add_argument("--generated", type=Path, default=DEFAULT_OUTPUT / "generated_rubrics.jsonl")
    coverage.add_argument("--judge", action="append", required=True, help="name|base_url|model[|API_KEY_ENV]")
    coverage.add_argument("--concurrency", type=int, default=16)
    coverage.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "coverage.jsonl")
    coverage.set_defaults(func=cmd_coverage)

    counter = sub.add_parser("counterfactual")
    add_data_args(counter)
    counter.add_argument("--judge", action="append", required=True)
    counter.add_argument("--representation", choices=("tree", "flat", "both"), default="both")
    counter.add_argument("--tree-route-mode", choices=("batch", "sequential", "soft", "exhaustive_path", "shuffled_path"), default="batch")
    counter.add_argument("--flat-leaf-mode", choices=("batch", "single"), default="batch")
    counter.add_argument("--routing-threshold", type=float, default=0.30)
    counter.add_argument("--routing-top-k", type=int, default=2)
    counter.add_argument("--force-safety", action=argparse.BooleanOptionalAction, default=True)
    counter.add_argument("--shuffle-seed", type=int, default=20260813)
    counter.add_argument("--shuffle-fraction", type=float, default=0.5)
    counter.add_argument("--shuffle-case-fraction", type=float, default=1.0)
    counter.add_argument(
        "--shuffle-strategy", choices=("random_same_depth", "hard_dimension"),
        default="random_same_depth",
    )
    counter.add_argument("--shuffle-require-all-leaves", action=argparse.BooleanOptionalAction, default=False)
    counter.add_argument("--concurrency", type=int, default=16)
    counter.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "counterfactual.jsonl")
    counter.set_defaults(func=cmd_counterfactual)

    weights = sub.add_parser("assign-weights")
    add_data_args(weights)
    weights.add_argument("--generator", required=True)
    weights.add_argument("--temperature", type=float, default=0.0)
    weights.add_argument("--response-conditioned", action="store_true", help="Condition both Tree and Flat weights on Candidate A and Candidate B (never the gold preference label).")
    weights.add_argument("--concurrency", type=int, default=8)
    weights.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "weights.jsonl")
    weights.set_defaults(func=cmd_weights)

    path_weights = sub.add_parser("assign-path-weights")
    add_data_args(path_weights)
    path_weights.add_argument("--generator", required=True)
    path_weights.add_argument("--temperature", type=float, default=0.0)
    path_weights.add_argument("--concurrency", type=int, default=8)
    path_weights.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "path_weights.jsonl")
    path_weights.set_defaults(func=cmd_path_weights)

    ablation = sub.add_parser("tree-context-weight-ablation")
    add_data_args(ablation)
    ablation.add_argument("--generator", required=True)
    ablation.add_argument("--temperature", type=float, default=0.0)
    ablation.add_argument("--concurrency", type=int, default=12)
    ablation.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "tree_context_weights.jsonl")
    ablation.set_defaults(func=cmd_tree_context_weight_ablation)

    flat_ablation = sub.add_parser("flat-context-weight-ablation")
    add_data_args(flat_ablation)
    flat_ablation.add_argument("--generator", required=True)
    flat_ablation.add_argument("--temperature", type=float, default=0.0)
    flat_ablation.add_argument("--concurrency", type=int, default=16)
    flat_ablation.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "flat_context_weights.jsonl")
    flat_ablation.set_defaults(func=cmd_flat_context_weight_ablation)

    pref = sub.add_parser("preference")
    add_data_args(pref)
    pref.add_argument("--weights", type=Path, default=DEFAULT_OUTPUT / "weights.jsonl")
    pref.add_argument("--judge", action="append", required=True)
    pref.add_argument("--concurrency", type=int, default=16)
    pref.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "preference.jsonl")
    pref.set_defaults(func=cmd_preference)

    summary = sub.add_parser("summarize")
    summary.add_argument("--input", type=Path, action="append", required=True)
    summary.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "summary.json")
    summary.set_defaults(func=cmd_summarize)
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    raise SystemExit(parsed_args.func(parsed_args))
