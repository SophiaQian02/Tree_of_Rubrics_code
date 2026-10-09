#!/usr/bin/env python3
"""Experiment 4: semantic and calibration-guided Gold-rubric rewriting."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.data.calibration_responses import cached_responses_for_sample, load_calibration_response_cache
from project.mechanism_analysis.core import (
    GoldDataset,
    JsonModel,
    JsonlSink,
    ModelSpec,
    json_digest,
    read_jsonl,
)
from project.mechanism_analysis.experiment4 import (
    CALIBRATION_REWRITE_PROMPT,
    CALIBRATION_SINGLE_RUBRIC_SCORE_PROMPT,
    FINAL_QUALITY_PROMPT,
    SINGLE_RUBRIC_RESPONSE_SCORE_PROMPT,
    SEMANTIC_REWRITE_PROMPT,
    calibration_diagnostics,
    final_quality_rating,
    gold_rubrics,
    response_panel,
    revise_with_calibration,
    score_calibration_rubric,
    score_response_with_rubric,
    semantic_rewrite,
)

DATA = REPO_ROOT / "data/gold_rubric_1500"
METHODS = ("gold", "semantic", "calibration", "semantic_calibration")
CALIBRATION_METHODS = ("calibration", "semantic_calibration")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=Path, default=DATA / "processed/gold_rubric_seed_1500.jsonl")
    parser.add_argument("--gold-tree", type=Path, default=DATA / "annotation/exports/gold_tree.jsonl")
    parser.add_argument("--gold-flat", type=Path, default=DATA / "annotation/exports/gold_flat.jsonl")
    parser.add_argument("--counterfactuals", type=Path, default=DATA / "annotation/exports/counterfactual_responses.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-shard-index", type=int, default=0)
    parser.add_argument("--sample-shard-count", type=int, default=1)


def dataset(args: argparse.Namespace) -> GoldDataset:
    return GoldDataset(args.seed, args.gold_tree, args.gold_flat, args.counterfactuals)


def preference_samples(args: argparse.Namespace):
    samples = [item for item in dataset(args).samples if item.seed.get("gold_preference") in {"A", "B"}]
    if args.limit > 0:
        samples = samples[: args.limit]
    shard_count = int(getattr(args, "sample_shard_count", 1))
    shard_index = int(getattr(args, "sample_shard_index", 0))
    if shard_count < 1:
        raise ValueError("--sample-shard-count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--sample-shard-index must be in [0, sample-shard-count)")
    if shard_count > 1:
        samples = [
            sample for global_index, sample in enumerate(samples)
            if global_index % shard_count == shard_index
        ]
    return samples


def latest_ok(path: Path, fields: tuple[str, ...]) -> dict[tuple[Any, ...], dict[str, Any]]:
    output: dict[tuple[Any, ...], dict[str, Any]] = {}
    if not path.exists():
        return output
    for row in read_jsonl(path):
        if row.get("status") != "ok":
            continue
        key = tuple(row.get(field) for field in fields)
        if key not in output or str(row.get("finished_at", "")) >= str(output[key].get("finished_at", "")):
            output[key] = row
    return output


def run_jobs(
    jobs: list[tuple[dict[str, Any], Callable[[], dict[str, Any]]]],
    sink: JsonlSink,
    concurrency: int,
) -> Counter:
    pending = [(meta, fn) for meta, fn in jobs if sink.key(meta) not in sink.done]
    counts: Counter = Counter()

    def execute(meta: dict[str, Any], fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            return {**meta, "status": "ok", "finished_at": now(), **fn()}
        except Exception as exc:
            return {
                **meta,
                "status": "error",
                "finished_at": now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=10),
            }

    if not pending:
        counts["skipped_existing"] = len(jobs)
        return counts
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(execute, meta, fn) for meta, fn in pending]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            sink.append(row)
            counts[row["status"]] += 1
            if index <= 10 or index % 100 == 0 or index == len(futures):
                print(
                    f"[{index}/{len(futures)}] ok={counts['ok']} error={counts['error']}",
                    flush=True,
                )
    counts["skipped_existing"] = len(jobs) - len(pending)
    return counts


def run_jobs_with_round_retries(
    jobs: list[tuple[dict[str, Any], Callable[[], dict[str, Any]]]],
    sink: JsonlSink,
    concurrency: int,
    retry_rounds: int,
    label: str,
) -> Counter:
    """Retry only jobs that still lack a successful sink row."""
    aggregate_counts: Counter = Counter()
    for retry_round in range(max(0, retry_rounds) + 1):
        counts = run_jobs(jobs, sink, concurrency)
        aggregate_counts.update({key: value for key, value in counts.items() if key != "skipped_existing"})
        remaining = sum(sink.key(meta) not in sink.done for meta, _fn in jobs)
        if remaining == 0:
            break
        if retry_round < retry_rounds:
            print(
                f"[EXPERIMENT4 JOB RETRY] label={label} "
                f"round={retry_round + 1}/{retry_rounds} remaining={remaining}",
                flush=True,
            )
    aggregate_counts["remaining_without_success"] = sum(
        sink.key(meta) not in sink.done for meta, _fn in jobs
    )
    return aggregate_counts


def append_success(sink: JsonlSink, row: dict[str, Any]) -> None:
    if sink.key(row) not in sink.done:
        sink.append({**row, "status": "ok", "finished_at": now()})


def cache_panels(samples, cache_path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    cache = load_calibration_response_cache(cache_path)
    panels: dict[str, list[dict[str, Any]]] = {}
    rows: dict[str, dict[str, Any]] = {}
    for sample in samples:
        responses, row = cached_responses_for_sample(
            cache,
            sample_id=sample.sample_id,
            question=sample.query,
            required_count=6,
        )
        row = {**row, "responses": responses}
        panels[sample.sample_id] = response_panel(row, 6)
        rows[sample.sample_id] = row
    return panels, rows


def model_payload(spec: ModelSpec) -> dict[str, Any]:
    return {"name": spec.name, "model": spec.model, "base_urls": list(spec.base_urls)}


def cmd_validate(args: argparse.Namespace) -> int:
    samples = preference_samples(args)
    report: dict[str, Any] = {
        "preference_samples": len(samples),
        "sample_shard_index": int(getattr(args, "sample_shard_index", 0)),
        "sample_shard_count": int(getattr(args, "sample_shard_count", 1)),
        "gold_leaf_count": {
            "min": min(len(item.gold_flat) for item in samples),
            "max": max(len(item.gold_flat) for item in samples),
            "mean": statistics.fmean(len(item.gold_flat) for item in samples),
        },
    }
    if args.calibration_responses:
        panels, _rows = cache_panels(samples, args.calibration_responses)
        model_counts: Counter = Counter()
        quality_counts: Counter = Counter()
        for panel in panels.values():
            model_counts.update(item["model"] for item in panel)
            quality_counts.update(item["quality"] for item in panel)
        report["calibration_cache"] = {
            "path": str(args.calibration_responses),
            "samples": len(panels),
            "responses": sum(len(item) for item in panels.values()),
            "model_counts": dict(model_counts),
            "quality_counts": dict(quality_counts),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    samples = preference_samples(args)
    generator_spec = ModelSpec.parse(args.generator)
    sink = JsonlSink(args.output, ("sample_id", "method", "input_sha256"))
    jobs: list[tuple[dict[str, Any], Callable[[], dict[str, Any]]]] = []
    semantic_prompt_sha = json_digest(SEMANTIC_REWRITE_PROMPT)
    for sample in samples:
        gold = gold_rubrics(sample)
        gold_hash = json_digest({"query": sample.query, "rubrics": gold})
        append_success(
            sink,
            {
                "stage": "rubric_construction",
                "sample_id": sample.sample_id,
                "method": "gold",
                "generator": "immutable_gold",
                "generator_model": None,
                "input_sha256": gold_hash,
                "rubric_sha256": json_digest(gold),
                "rubrics": gold,
                "rubric_count": len(gold),
            },
        )
        input_hash = json_digest(
            {
                "query": sample.query,
                "gold": gold,
                "generator": model_payload(generator_spec),
                "max_items": args.max_items,
                "prompt": semantic_prompt_sha,
            }
        )
        meta = {
            "stage": "rubric_construction",
            "sample_id": sample.sample_id,
            "method": "semantic",
            "generator": generator_spec.name,
            "generator_model": generator_spec.model,
            "generator_base_urls": list(generator_spec.base_urls),
            "input_sha256": input_hash,
            "prompt_sha256": semantic_prompt_sha,
        }

        def build(s=sample):
            result = semantic_rewrite(s, JsonModel(generator_spec, args.temperature, args.max_tokens), args.max_items)
            rubrics = result["rubrics"]
            return {
                **result,
                "rubric_sha256": json_digest(rubrics),
                "rubric_count": len(rubrics),
            }

        jobs.append((meta, build))
    counts = run_jobs(jobs, sink, args.concurrency)
    rows = latest_ok(args.output, ("sample_id", "method"))
    expected = len(samples) * 2
    print(json.dumps({"run": dict(counts), "complete": len(rows), "expected": expected}, ensure_ascii=False))
    return 0 if len(rows) >= expected else 1


def _base_rows(path: Path, sample_ids: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = latest_ok(path, ("sample_id", "method"))
    return {
        key: row
        for key, row in rows.items()
        if key[0] in sample_ids and key[1] in {"gold", "semantic"}
    }


def _calibration_config(args: argparse.Namespace, generator: ModelSpec, judges: list[ModelSpec]) -> tuple[dict[str, Any], str]:
    payload = {
        "version": 2,
        "calibration_scoring_mode": "per_rubric_v1",
        "generator": model_payload(generator),
        "judges": [model_payload(item) for item in judges],
        "max_rewrites": args.max_rewrites,
        "max_items": args.max_items,
        "min_discrimination_variance": args.min_discrimination_variance,
        "max_judge_instability_variance": args.max_judge_instability_variance,
        "redundancy_correlation": args.redundancy_correlation,
        "judge_max_tokens": args.judge_max_tokens,
        "score_prompt_sha256": json_digest(CALIBRATION_SINGLE_RUBRIC_SCORE_PROMPT),
        "rewrite_prompt_sha256": json_digest(CALIBRATION_REWRITE_PROMPT),
    }
    return payload, json_digest(payload)


def cmd_calibrate(args: argparse.Namespace) -> int:
    samples = preference_samples(args)
    by_id = {sample.sample_id: sample for sample in samples}
    sample_ids = set(by_id)
    bases = _base_rows(args.base_rubrics, sample_ids)
    if len(bases) != len(samples) * 2:
        raise SystemExit(
            f"Base rubric file is incomplete: found {len(bases)}, expected {len(samples) * 2}. Rerun prepare."
        )
    panels, cache_rows = cache_panels(samples, args.calibration_responses)
    generator_spec = ModelSpec.parse(args.generator)
    judge_specs = [ModelSpec.parse(value) for value in args.judge]
    if len(judge_specs) != 2:
        raise SystemExit("Experiment 4 calibration requires exactly two independent --judge models")
    config, config_sha = _calibration_config(args, generator_spec, judge_specs)

    score_sink = JsonlSink(
        args.scores_output,
        ("sample_id", "method", "iteration", "judge", "rubric_id", "input_sha256"),
    )
    revision_sink = JsonlSink(
        args.revisions_output,
        ("sample_id", "method", "iteration", "input_sha256"),
    )
    final_sink = JsonlSink(args.final_output, ("sample_id", "method", "configuration_sha256"))

    for sample in samples:
        for method in ("gold", "semantic"):
            row = bases[(sample.sample_id, method)]
            append_success(
                final_sink,
                {
                    "stage": "final_rubric",
                    "sample_id": sample.sample_id,
                    "method": method,
                    "configuration_sha256": "baseline_v1",
                    "source_rubric_sha256": row["rubric_sha256"],
                    "rubric_sha256": row["rubric_sha256"],
                    "rubrics": row["rubrics"],
                    "rubric_count": len(row["rubrics"]),
                    "final_reason": "immutable_baseline" if method == "gold" else "semantic_only_baseline",
                },
            )

    finalized = {
        (row.get("sample_id"), row.get("method"))
        for row in read_jsonl(args.final_output)
        if row.get("status") == "ok"
        and row.get("configuration_sha256") == config_sha
        and row.get("method") in CALIBRATION_METHODS
    }
    current: dict[tuple[str, str], dict[str, Any]] = {}
    for sample in samples:
        if (sample.sample_id, "calibration") not in finalized:
            current[(sample.sample_id, "calibration")] = bases[(sample.sample_id, "gold")]
        if (sample.sample_id, "semantic_calibration") not in finalized:
            current[(sample.sample_id, "semantic_calibration")] = bases[(sample.sample_id, "semantic")]

    for iteration in range(args.max_rewrites + 1):
        if not current:
            break
        print(f"[CALIBRATION ROUND {iteration}] active={len(current)}", flush=True)
        score_jobs: list[tuple[dict[str, Any], Callable[[], dict[str, Any]]]] = []
        score_hashes: dict[tuple[str, str, str, str], str] = {}
        for (sample_id, method), rubric_row in current.items():
            sample = by_id[sample_id]
            rubrics = rubric_row["rubrics"]
            rubric_sha = str(rubric_row.get("rubric_sha256") or json_digest(rubrics))
            panel = panels[sample_id]
            panel_sha = json_digest(panel)
            for judge_spec in judge_specs:
                for rubric in rubrics:
                    rubric_id = str(rubric["rubric_id"])
                    input_hash = json_digest(
                        {
                            "query": sample.query,
                            "rubric": rubric,
                            "panel_sha256": panel_sha,
                            "judge": model_payload(judge_spec),
                            "prompt": config["score_prompt_sha256"],
                            "scoring_mode": config["calibration_scoring_mode"],
                        }
                    )
                    score_hashes[(sample_id, method, judge_spec.name, rubric_id)] = input_hash
                    meta = {
                        "stage": "calibration_single_rubric_scoring",
                        "scoring_mode": "per_rubric_v1",
                        "sample_id": sample_id,
                        "method": method,
                        "iteration": iteration,
                        "judge": judge_spec.name,
                        "judge_model": judge_spec.model,
                        "judge_base_urls": list(judge_spec.base_urls),
                        "rubric_id": rubric_id,
                        "rubric_sha256": rubric_sha,
                        "panel_sha256": panel_sha,
                        "input_sha256": input_hash,
                        "configuration_sha256": config_sha,
                    }
                    score_jobs.append(
                        (
                            meta,
                            lambda s=sample, r=rubric, p=panel, j=judge_spec: score_calibration_rubric(
                                s, r, p, JsonModel(j, 0.0, args.judge_max_tokens)
                            ),
                        )
                    )
        print(
            dict(
                run_jobs_with_round_retries(
                    score_jobs,
                    score_sink,
                    args.concurrency,
                    args.job_retry_rounds,
                    f"calibration_score_round_{iteration}",
                )
            ),
            flush=True,
        )
        score_rows = latest_ok(
            args.scores_output,
            ("sample_id", "method", "iteration", "judge", "rubric_id", "input_sha256"),
        )

        revision_jobs: list[tuple[dict[str, Any], Callable[[], dict[str, Any]]]] = []
        next_meta_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for (sample_id, method), rubric_row in current.items():
            judge_scores: dict[str, dict[str, Any]] = {}
            for judge_spec in judge_specs:
                response_ids = sorted(str(item["response_id"]) for item in panels[sample_id])
                matrix: dict[str, dict[str, int]] = {response_id: {} for response_id in response_ids}
                complete_judge = True
                for rubric in rubric_row["rubrics"]:
                    rubric_id = str(rubric["rubric_id"])
                    input_hash = score_hashes[(sample_id, method, judge_spec.name, rubric_id)]
                    row = score_rows.get(
                        (sample_id, method, iteration, judge_spec.name, rubric_id, input_hash)
                    )
                    if row is None:
                        complete_judge = False
                        break
                    response_scores = row.get("response_scores")
                    if not isinstance(response_scores, dict) or set(response_scores) != set(response_ids):
                        complete_judge = False
                        break
                    for response_id in response_ids:
                        matrix[response_id][rubric_id] = int(response_scores[response_id])
                if complete_judge:
                    judge_scores[judge_spec.name] = {
                        "matrix": matrix,
                        "response_ids": response_ids,
                        "rubric_ids": sorted(str(item["rubric_id"]) for item in rubric_row["rubrics"]),
                        "calls": len(rubric_row["rubrics"]),
                        "scoring_mode": "per_rubric_v1",
                    }
            if len(judge_scores) != len(judge_specs):
                continue
            rubrics = rubric_row["rubrics"]
            matrices = {name: value["matrix"] for name, value in judge_scores.items()}
            diagnostics = calibration_diagnostics(
                rubrics,
                matrices,
                min_discrimination_variance=args.min_discrimination_variance,
                max_judge_instability_variance=args.max_judge_instability_variance,
                redundancy_correlation=args.redundancy_correlation,
            )
            should_stop = diagnostics["issue_count"] == 0 or iteration >= args.max_rewrites
            if should_stop:
                append_success(
                    final_sink,
                    {
                        "stage": "final_rubric",
                        "sample_id": sample_id,
                        "method": method,
                        "configuration_sha256": config_sha,
                        "configuration": config,
                        "source_rubric_sha256": rubric_row.get("rubric_sha256"),
                        "rubric_sha256": json_digest(rubrics),
                        "rubrics": rubrics,
                        "rubric_count": len(rubrics),
                        "final_iteration": iteration,
                        "final_reason": "thresholds_satisfied" if diagnostics["issue_count"] == 0 else "max_rewrites_reached",
                        "calibration_diagnostics": diagnostics,
                        "calibration_response_cache": str(args.calibration_responses),
                    },
                )
                continue

            next_iteration = iteration + 1
            sample = by_id[sample_id]
            input_hash = json_digest(
                {
                    "query": sample.query,
                    "current_rubrics": rubrics,
                    "panel": panels[sample_id],
                    "diagnostics": diagnostics,
                    "generator": model_payload(generator_spec),
                    "max_items": args.max_items,
                    "prompt": config["rewrite_prompt_sha256"],
                }
            )
            meta = {
                "stage": "calibration_revision",
                "sample_id": sample_id,
                "method": method,
                "iteration": next_iteration,
                "generator": generator_spec.name,
                "generator_model": generator_spec.model,
                "generator_base_urls": list(generator_spec.base_urls),
                "source_rubric_sha256": json_digest(rubrics),
                "input_sha256": input_hash,
                "configuration_sha256": config_sha,
                "calibration_diagnostics": diagnostics,
            }
            prefix = f"C{next_iteration}_" if method == "calibration" else f"SC{next_iteration}_"
            next_meta_by_key[(sample_id, method)] = meta

            def revise(s=sample, r=rubrics, p=panels[sample_id], js=judge_scores, d=diagnostics, pref=prefix):
                result = revise_with_calibration(
                    s,
                    r,
                    p,
                    js,
                    d,
                    JsonModel(generator_spec, args.temperature, args.generator_max_tokens),
                    id_prefix=pref,
                    max_items=args.max_items,
                )
                revised = result["rubrics"]
                return {
                    **result,
                    "rubric_sha256": json_digest(revised),
                    "rubric_count": len(revised),
                }

            revision_jobs.append((meta, revise))
        print(
            dict(
                run_jobs_with_round_retries(
                    revision_jobs,
                    revision_sink,
                    args.generator_concurrency,
                    args.job_retry_rounds,
                    f"calibration_revision_round_{iteration + 1}",
                )
            ),
            flush=True,
        )
        revision_rows = latest_ok(
            args.revisions_output,
            ("sample_id", "method", "iteration", "input_sha256"),
        )
        next_current: dict[tuple[str, str], dict[str, Any]] = {}
        for key, meta in next_meta_by_key.items():
            row = revision_rows.get((key[0], key[1], iteration + 1, meta["input_sha256"]))
            if row is not None:
                next_current[key] = row
        current = next_current

    final_rows = latest_ok(args.final_output, ("sample_id", "method"))
    complete = sum((sample.sample_id, method) in final_rows for sample in samples for method in METHODS)
    expected = len(samples) * len(METHODS)
    print(json.dumps({"configuration_sha256": config_sha, "complete": complete, "expected": expected}))
    return 0 if complete >= expected else 1


def final_rubric_rows(
    path: Path,
    sample_ids: set[str],
    methods: tuple[str, ...] = METHODS,
) -> dict[tuple[str, str], dict[str, Any]]:
    rows = latest_ok(path, ("sample_id", "method"))
    selected = set(methods)
    return {key: row for key, row in rows.items() if key[0] in sample_ids and key[1] in selected}


def cmd_quality(args: argparse.Namespace) -> int:
    samples = preference_samples(args)
    by_id = {sample.sample_id: sample for sample in samples}
    methods = tuple(args.method or METHODS)
    rubrics = final_rubric_rows(args.rubrics, set(by_id), methods)
    expected_rubrics = len(samples) * len(methods)
    if len(rubrics) != expected_rubrics:
        raise SystemExit(f"Final rubrics incomplete: found {len(rubrics)}, expected {expected_rubrics}")
    judges = [ModelSpec.parse(value) for value in args.judge]
    if not 1 <= len(judges) <= 2:
        raise SystemExit("Final quality evaluation requires one or two --judge models")
    sink = JsonlSink(args.output, ("sample_id", "method", "judge", "input_sha256"))
    jobs = []
    prompt_sha = json_digest(FINAL_QUALITY_PROMPT)
    for (sample_id, method), rubric_row in rubrics.items():
        sample = by_id[sample_id]
        for judge_spec in judges:
            input_hash = json_digest(
                {
                    "query": sample.query,
                    "gold": sample.gold_flat,
                    "rubric_sha256": rubric_row["rubric_sha256"],
                    "judge": model_payload(judge_spec),
                    "prompt": prompt_sha,
                }
            )
            meta = {
                "stage": "final_quality_evaluation",
                "sample_id": sample_id,
                "method": method,
                "judge": judge_spec.name,
                "judge_model": judge_spec.model,
                "judge_base_urls": list(judge_spec.base_urls),
                "rubric_sha256": rubric_row["rubric_sha256"],
                "input_sha256": input_hash,
                "prompt_sha256": prompt_sha,
            }
            jobs.append(
                (
                    meta,
                    lambda s=sample, r=rubric_row["rubrics"], j=judge_spec: final_quality_rating(
                        s, r, JsonModel(j, 0.0, args.max_tokens)
                    ),
                )
            )
    counts = run_jobs(jobs, sink, args.concurrency)
    selected_judges = {judge.name for judge in judges}
    selected_methods = set(methods)
    complete = sum(
        key[1] in selected_methods and key[2] in selected_judges
        for key in latest_ok(args.output, ("sample_id", "method", "judge"))
    )
    expected = len(samples) * len(methods) * len(judges)
    print(json.dumps({"run": dict(counts), "complete": complete, "expected": expected}))
    return 0 if complete >= expected else 1


def cmd_preference(args: argparse.Namespace) -> int:
    samples = preference_samples(args)
    by_id = {sample.sample_id: sample for sample in samples}
    methods = tuple(args.method or METHODS)
    rubrics = final_rubric_rows(args.rubrics, set(by_id), methods)
    expected_rubrics = len(samples) * len(methods)
    if len(rubrics) != expected_rubrics:
        raise SystemExit(f"Final rubrics incomplete: found {len(rubrics)}, expected {expected_rubrics}")
    judges = [ModelSpec.parse(value) for value in args.judge]
    if not 1 <= len(judges) <= 2:
        raise SystemExit("Preference evaluation requires one or two --judge models")
    atomic_sink = JsonlSink(
        args.output,
        ("sample_id", "method", "judge", "candidate", "rubric_id", "input_sha256"),
    )
    aggregate_sink = JsonlSink(
        args.output,
        ("sample_id", "method", "judge", "candidate", "input_sha256"),
    )
    jobs: list[tuple[dict[str, Any], Callable[[], dict[str, Any]]]] = []
    atomic_hashes: dict[tuple[str, str, str, str, str], str] = {}
    aggregate_hashes: dict[tuple[str, str, str, str], str] = {}
    prompt_sha = json_digest(SINGLE_RUBRIC_RESPONSE_SCORE_PROMPT)
    for (sample_id, method), rubric_row in rubrics.items():
        sample = by_id[sample_id]
        candidates = sample.seed.get("candidate_responses") or {}
        for judge_spec in judges:
            for candidate in ("A", "B"):
                response = str(candidates.get(candidate) or "")
                if not response:
                    raise ValueError(f"{sample_id} is missing candidate {candidate}")
                aggregate_input_hash = json_digest(
                    {
                        "query": sample.query,
                        "response": response,
                        "rubric_sha256": rubric_row["rubric_sha256"],
                        "judge": model_payload(judge_spec),
                        "prompt": prompt_sha,
                        "scoring_mode": "per_rubric_v1",
                    }
                )
                aggregate_hashes[(sample_id, method, judge_spec.name, candidate)] = aggregate_input_hash
                for rubric in rubric_row["rubrics"]:
                    rubric_id = str(rubric["rubric_id"])
                    input_hash = json_digest(
                        {
                            "query": sample.query,
                            "response": response,
                            "rubric": rubric,
                            "judge": model_payload(judge_spec),
                            "prompt": prompt_sha,
                            "scoring_mode": "per_rubric_v1",
                        }
                    )
                    atomic_hashes[(sample_id, method, judge_spec.name, candidate, rubric_id)] = input_hash
                    meta = {
                        "stage": "preference_single_rubric_scoring",
                        "scoring_mode": "per_rubric_v1",
                        "sample_id": sample_id,
                        "method": method,
                        "judge": judge_spec.name,
                        "judge_model": judge_spec.model,
                        "judge_base_urls": list(judge_spec.base_urls),
                        "candidate": candidate,
                        "gold_preference": sample.seed["gold_preference"],
                        "rubric_id": rubric_id,
                        "rubric_sha256": rubric_row["rubric_sha256"],
                        "input_sha256": input_hash,
                        "prompt_sha256": prompt_sha,
                    }
                    jobs.append(
                        (
                            meta,
                            lambda s=sample, text=response, r=rubric, j=judge_spec: score_response_with_rubric(
                                s, text, r, JsonModel(j, 0.0, args.max_tokens)
                            ),
                        )
                    )
    counts = run_jobs_with_round_retries(
        jobs,
        atomic_sink,
        args.concurrency,
        args.job_retry_rounds,
        "preference_single_rubric_scoring",
    )
    atomic_rows = latest_ok(
        args.output,
        ("sample_id", "method", "judge", "candidate", "rubric_id", "input_sha256"),
    )
    for (sample_id, method), rubric_row in rubrics.items():
        sample = by_id[sample_id]
        for judge_spec in judges:
            for candidate in ("A", "B"):
                scores: dict[str, int] = {}
                for rubric in rubric_row["rubrics"]:
                    rubric_id = str(rubric["rubric_id"])
                    input_hash = atomic_hashes[(sample_id, method, judge_spec.name, candidate, rubric_id)]
                    row = atomic_rows.get(
                        (sample_id, method, judge_spec.name, candidate, rubric_id, input_hash)
                    )
                    if row is None:
                        break
                    scores[rubric_id] = int(row["score"])
                if len(scores) != len(rubric_row["rubrics"]):
                    continue
                aggregate_input_hash = aggregate_hashes[(sample_id, method, judge_spec.name, candidate)]
                append_success(
                    aggregate_sink,
                    {
                        "stage": "preference_response_scoring",
                        "scoring_mode": "per_rubric_v1",
                        "sample_id": sample_id,
                        "method": method,
                        "judge": judge_spec.name,
                        "judge_model": judge_spec.model,
                        "judge_base_urls": list(judge_spec.base_urls),
                        "candidate": candidate,
                        "gold_preference": sample.seed["gold_preference"],
                        "rubric_sha256": rubric_row["rubric_sha256"],
                        "input_sha256": aggregate_input_hash,
                        "prompt_sha256": prompt_sha,
                        "rubric_scores": scores,
                        "uniform_mean_score": statistics.fmean(scores.values()),
                        "calls": len(scores),
                    },
                )
    aggregate_rows = latest_ok(
        args.output,
        ("sample_id", "method", "judge", "candidate", "input_sha256"),
    )
    complete = sum(
        (sample_id, method, judge, candidate, input_hash) in aggregate_rows
        for (sample_id, method, judge, candidate), input_hash in aggregate_hashes.items()
    )
    expected = len(samples) * len(methods) * len(judges) * 2
    print(
        json.dumps(
            {
                "scoring_mode": "per_rubric_v1",
                "atomic_run": dict(counts),
                "atomic_expected": len(jobs),
                "aggregate_complete": complete,
                "aggregate_expected": expected,
            }
        )
    )
    return 0 if complete >= expected else 1


def preference_outcome(score_a: float, score_b: float, gold: str) -> dict[str, Any]:
    delta = score_a - score_b
    predicted = "tie" if abs(delta) <= 1e-12 else ("A" if delta > 0 else "B")
    return {
        "score_a": score_a,
        "score_b": score_b,
        "delta": delta,
        "predicted": predicted,
        "correct_tie_half": 0.5 if predicted == "tie" else float(predicted == gold),
        "strict_correct": float(predicted == gold),
    }


def summarize_preference(path: Path) -> dict[str, Any]:
    rows = latest_ok(path, ("sample_id", "method", "judge", "candidate"))
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for (sample_id, method, judge, candidate), row in rows.items():
        grouped[(sample_id, method, judge)][candidate] = row
    outcomes: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, pair in grouped.items():
        if set(pair) != {"A", "B"}:
            continue
        outcomes[key] = preference_outcome(
            float(pair["A"]["uniform_mean_score"]),
            float(pair["B"]["uniform_mean_score"]),
            str(pair["A"]["gold_preference"]),
        )
    report: dict[str, Any] = {}
    for method, judge in sorted({(key[1], key[2]) for key in outcomes}):
        items = [value for key, value in outcomes.items() if key[1:] == (method, judge)]
        decisive = [item for item in items if item["predicted"] != "tie"]
        report[f"{method}|{judge}"] = {
            "n": len(items),
            "accuracy_tie_half": statistics.fmean(item["correct_tie_half"] for item in items) if items else None,
            "strict_accuracy": statistics.fmean(item["strict_correct"] for item in items) if items else None,
            "decisive_accuracy": statistics.fmean(item["strict_correct"] for item in decisive) if decisive else None,
            "tie_rate": 1.0 - len(decisive) / len(items) if items else None,
        }
    by_sample_method: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for (sample_id, method, judge), pair in grouped.items():
        if set(pair) == {"A", "B"}:
            by_sample_method[(sample_id, method)].append((judge, pair))
    for method in METHODS:
        items = []
        for (sample_id, item_method), judge_pairs in by_sample_method.items():
            if item_method != method or len(judge_pairs) < 2:
                continue
            score_a = statistics.fmean(float(pair["A"]["uniform_mean_score"]) for _judge, pair in judge_pairs)
            score_b = statistics.fmean(float(pair["B"]["uniform_mean_score"]) for _judge, pair in judge_pairs)
            gold = str(judge_pairs[0][1]["A"]["gold_preference"])
            items.append(preference_outcome(score_a, score_b, gold))
        decisive = [item for item in items if item["predicted"] != "tie"]
        report[f"{method}|two_judge_ensemble"] = {
            "n": len(items),
            "accuracy_tie_half": statistics.fmean(item["correct_tie_half"] for item in items) if items else None,
            "strict_accuracy": statistics.fmean(item["strict_correct"] for item in items) if items else None,
            "decisive_accuracy": statistics.fmean(item["strict_correct"] for item in decisive) if decisive else None,
            "tie_rate": 1.0 - len(decisive) / len(items) if items else None,
        }
    return report


def summarize_quality(path: Path) -> dict[str, Any]:
    rows = latest_ok(path, ("sample_id", "method", "judge"))
    report: dict[str, Any] = {}
    dimensions = ("coverage", "discriminability", "redundancy", "novelty")
    for method, judge in sorted({(key[1], key[2]) for key in rows}):
        items = [row for key, row in rows.items() if key[1:] == (method, judge)]
        report[f"{method}|{judge}"] = {
            "n": len(items),
            **{
                name: statistics.fmean(float(item["ratings"][name]) for item in items) if items else None
                for name in dimensions
            },
        }
    for method in METHODS:
        by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for (sample_id, item_method, _judge), row in rows.items():
            if item_method == method:
                by_sample[sample_id].append(row)
        complete = [items for items in by_sample.values() if len(items) >= 2]
        report[f"{method}|two_judge_mean"] = {
            "n": len(complete),
            **{
                name: statistics.fmean(
                    statistics.fmean(float(item["ratings"][name]) for item in items)
                    for items in complete
                ) if complete else None
                for name in dimensions
            },
        }
    return report


def error_summary(paths: list[Path]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for path in paths:
        if not path.exists():
            output[str(path)] = {"missing": True}
            continue
        rows = read_jsonl(path)
        errors = [row for row in rows if row.get("status") == "error"]
        output[str(path)] = {
            "rows": len(rows),
            "error_rows": len(errors),
            "error_types": dict(Counter(str(row.get("error_type") or "unknown") for row in errors)),
        }
    return output


def cmd_summarize(args: argparse.Namespace) -> int:
    finals = latest_ok(args.rubrics, ("sample_id", "method"))
    rubric_counts: dict[str, list[int]] = defaultdict(list)
    final_iterations: dict[str, Counter] = defaultdict(Counter)
    for (_sample_id, method), row in finals.items():
        rubric_counts[method].append(int(row.get("rubric_count") or len(row.get("rubrics", []))))
        if method in CALIBRATION_METHODS:
            final_iterations[method][str(row.get("final_iteration"))] += 1
    report = {
        "created_at": now(),
        "methods": list(METHODS),
        "rubrics": {
            method: {
                "n": len(values),
                "mean_count": statistics.fmean(values) if values else None,
                "min_count": min(values) if values else None,
                "max_count": max(values) if values else None,
                "final_iteration_counts": dict(final_iterations.get(method, {})),
            }
            for method, values in rubric_counts.items()
        },
        "quality_1to5": summarize_quality(args.quality),
        "preference_accuracy": summarize_preference(args.preference),
        "files": {
            "rubrics": str(args.rubrics),
            "quality": str(args.quality),
            "preference": str(args.preference),
        },
        "raw_error_audit": error_summary(args.audit_file),
        "score_directions": {
            "coverage": "higher_is_better",
            "discriminability": "higher_is_better",
            "redundancy": "higher_is_better; this is redundancy-control/non-redundancy",
            "novelty": "higher_is_better",
            "preference_tie": "reported both as strict=0 and tie_half=0.5",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    add_data_args(validate)
    validate.add_argument("--calibration-responses", type=Path)
    validate.set_defaults(func=cmd_validate)

    prepare = sub.add_parser("prepare")
    add_data_args(prepare)
    prepare.add_argument("--generator", required=True)
    prepare.add_argument("--temperature", type=float, default=0.1)
    prepare.add_argument("--max-items", type=int, default=24)
    prepare.add_argument("--max-tokens", type=int, default=8192)
    prepare.add_argument("--concurrency", type=int, default=8)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(func=cmd_prepare)

    calibrate = sub.add_parser("calibrate")
    add_data_args(calibrate)
    calibrate.add_argument("--base-rubrics", type=Path, required=True)
    calibrate.add_argument("--calibration-responses", type=Path, required=True)
    calibrate.add_argument("--generator", required=True)
    calibrate.add_argument("--judge", action="append", required=True)
    calibrate.add_argument("--temperature", type=float, default=0.1)
    calibrate.add_argument("--max-rewrites", type=int, default=3)
    calibrate.add_argument("--max-items", type=int, default=24)
    calibrate.add_argument("--min-discrimination-variance", type=float, default=0.50)
    calibrate.add_argument("--max-judge-instability-variance", type=float, default=0.50)
    calibrate.add_argument("--redundancy-correlation", type=float, default=0.90)
    calibrate.add_argument("--concurrency", type=int, default=24)
    calibrate.add_argument("--generator-concurrency", type=int, default=8)
    calibrate.add_argument("--judge-max-tokens", type=int, default=8192)
    calibrate.add_argument("--generator-max-tokens", type=int, default=8192)
    calibrate.add_argument("--job-retry-rounds", type=int, default=2)
    calibrate.add_argument("--scores-output", type=Path, required=True)
    calibrate.add_argument("--revisions-output", type=Path, required=True)
    calibrate.add_argument("--final-output", type=Path, required=True)
    calibrate.set_defaults(func=cmd_calibrate)

    quality = sub.add_parser("quality")
    add_data_args(quality)
    quality.add_argument("--rubrics", type=Path, required=True)
    quality.add_argument("--judge", action="append", required=True)
    quality.add_argument("--method", action="append", choices=METHODS)
    quality.add_argument("--concurrency", type=int, default=32)
    quality.add_argument("--max-tokens", type=int, default=4096)
    quality.add_argument("--output", type=Path, required=True)
    quality.set_defaults(func=cmd_quality)

    preference = sub.add_parser("preference")
    add_data_args(preference)
    preference.add_argument("--rubrics", type=Path, required=True)
    preference.add_argument("--judge", action="append", required=True)
    preference.add_argument("--method", action="append", choices=METHODS)
    preference.add_argument("--concurrency", type=int, default=32)
    preference.add_argument("--max-tokens", type=int, default=4096)
    preference.add_argument("--job-retry-rounds", type=int, default=2)
    preference.add_argument("--output", type=Path, required=True)
    preference.set_defaults(func=cmd_preference)

    summarize = sub.add_parser("summarize")
    summarize.add_argument("--rubrics", type=Path, required=True)
    summarize.add_argument("--quality", type=Path, required=True)
    summarize.add_argument("--preference", type=Path, required=True)
    summarize.add_argument("--audit-file", type=Path, action="append", default=[])
    summarize.add_argument("--output", type=Path, required=True)
    summarize.set_defaults(func=cmd_summarize)
    return root


def main() -> None:
    args = parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
