#!/usr/bin/env python3
"""RQ2 mechanism study: rubric discriminability and cross-judge stability.

The experiment deliberately changes only the rubric-generation instruction.
It never exposes candidate responses or preference labels to the generator.
All response scoring uses uniform aggregation so this study does not confound
RQ2 with the weighting mechanism studied in RQ3.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import html
import json
import math
import os
import random
import statistics
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.mechanism_analysis.core import (  # noqa: E402
    JsonModel,
    JsonlSink,
    ModelSpec,
    json_digest,
    prompt_text,
    read_jsonl,
)


DATASET_DEFAULT = REPO_ROOT / "data/gold_rubric_1500/processed/gold_rubric_seed_1500.jsonl"
METHODS = ("direct", "discriminability", "stability")
JUDGE_METHODS = METHODS
QUALITY_DIMENSIONS = (
    "task_relevance",
    "coverage",
    "atomicity",
    "clarity",
    "observability",
    "executability",
    "nonredundancy",
)

BASE_GENERATION_PROMPT = """You construct a complete flat rubric for evaluating responses to a user query.
Use ONLY the query. You cannot see candidate responses, preference labels, Gold rubrics, or calibration data.

Generate exactly {rubric_count} criteria. The cardinality is fixed across experimental conditions.
Each criterion must assess one task-grounded aspect and must be usable independently on unseen responses.
Do not assign weights: every criterion will receive equal weight in this experiment.
Do not reveal or solve the task itself.

Return only this JSON schema:
{{"rubrics":[{{"criterion":"criterion text","desired_state":"criterion_met|criterion_not_met","category":"short category","rationale":"brief query-grounded reason"}}]}}
"""

COUNT_PLANNING_PROMPT = """You estimate the natural number of independently judgeable rubric criteria needed for a user query.
Use ONLY the query. You cannot see candidate responses, preference labels, Gold rubrics, calibration data, or downstream method names.

Choose the SMALLEST count that can cover all explicit requirements and the task-grounded quality dimensions needed to distinguish weak, partial, and strong responses. Do not pad the plan with generic style criteria. Keep each planned aspect atomic and distinct.
The count must be between {minimum} and {maximum}, inclusive, and must equal the number of returned aspects.

Return only JSON:
{{"target_rubric_count":4,"aspects":["short atomic aspect"],"reason":"brief query-only complexity rationale"}}
"""

DIRECT_CONDITION = """CONDITION: ordinary direct generation.
Write a reasonable, balanced set of evaluation criteria for the query. Cover the important requirements and use clear language."""

DISCRIMINABILITY_CONDITION = """CONDITION: deliberately maximize discriminability.
The rubric must create SHARP SCORE SEPARATION between plausibly stronger and weaker responses to this query.
- Replace broad criteria such as "is good/correct/helpful/clear" with concrete requirements whose satisfaction can vary across responses.
- Prioritize independently fail-able task requirements, omissions, factual or logical defects, constraint violations, and meaningful quality thresholds.
- Avoid criteria that nearly every plausible response passes, generic style preferences, and duplicate signals.
- Each criterion must define what evidence separates weak, partial, and strong satisfaction, so a judge can use the full 1--5 scale.
This condition is intentionally strong: discriminative decision boundaries take priority over elegant wording."""

STABILITY_CONDITION = """CONDITION: deliberately maximize cross-judge stability.
The same response should receive nearly the same score from independently trained judge-model families.
- Use self-contained, literal, operational criteria grounded in observable response evidence.
- State inclusion/exclusion boundaries and avoid vague adjectives, latent intent, taste, or unspecified external assumptions.
- Keep exactly one independently judgeable claim per criterion; identify the concrete evidence a judge should inspect.
- Phrase criteria so that two judges can apply the same 1--5 threshold without inferring hidden standards.
This condition is intentionally strong: reproducible interpretation across judges takes priority over stylistic elegance."""

SCORE_SYSTEM_PROMPT = """You are a strict response evaluator. Score one response against every supplied rubric item independently.
For each item, return an integer from 1 to 5 indicating how well the response satisfies the item's DESIRED state:
1 = clearly fails, 2 = mostly fails, 3 = partially satisfies or ambiguous, 4 = mostly satisfies, 5 = clearly satisfies.
Use only evidence in the response. Do not compare against another candidate and do not infer a preference label.
Return every rubric_id exactly once and no extra IDs.
Return only JSON: {"scores":[{"rubric_id":"R01","score":1,"evidence":"short observable evidence"}]}"""

SCORE_USER_PROMPT = """QUERY:
{query}

RESPONSE:
{response}

RUBRICS:
{rubrics}
"""

QUALITY_SYSTEM_PROMPT = """You are an independent expert auditing a complete LLM evaluation rubric.
You cannot see responses, method names, prompt conditions, preference labels, or downstream results.
Evaluate only whether the rubric is a high-quality measurement instrument for the supplied query.

Rate each dimension from 1 (poor) to 5 (excellent): task_relevance, coverage, atomicity, clarity,
observability, executability, and nonredundancy. Then give an overall_quality integer from 0 to 100.
The overall score is a holistic audit, not an arithmetic reward for having more items.
Return only JSON:
{"ratings":{"task_relevance":1,"coverage":1,"atomicity":1,"clarity":1,"observability":1,"executability":1,"nonredundancy":1},"overall_quality":0,"diagnosis":"brief reason"}"""

QUALITY_USER_PROMPT = """QUERY:
{query}

RUBRIC SET (uniform weights; order randomized for blinding):
{rubrics}
"""


@dataclass(frozen=True)
class PreferenceSample:
    index: int
    sample_id: str
    query: str
    response_a: str
    response_b: str
    gold: str


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def load_samples(path: Path, limit: int = 0) -> list[PreferenceSample]:
    rows = read_jsonl(path)
    samples: list[PreferenceSample] = []
    for row in rows:
        gold = str(row.get("gold_preference", "")).upper()
        responses = row.get("candidate_responses")
        if gold not in {"A", "B"} or not isinstance(responses, dict):
            continue
        a, b = str(responses.get("A", "")), str(responses.get("B", ""))
        if not a.strip() or not b.strip():
            continue
        samples.append(
            PreferenceSample(
                index=len(samples),
                sample_id=str(row["sample_id"]),
                query=prompt_text(row["prompt"]),
                response_a=a,
                response_b=b,
                gold=gold,
            )
        )
        if limit > 0 and len(samples) >= limit:
            break
    return samples


def latest_ok(path: Path, fields: tuple[str, ...]) -> dict[tuple[Any, ...], dict[str, Any]]:
    output: dict[tuple[Any, ...], dict[str, Any]] = {}
    if not path.exists():
        return output
    for row in read_jsonl(path):
        if row.get("status") != "ok":
            continue
        key = tuple(row.get(field) for field in fields)
        previous = output.get(key)
        if previous is None or str(row.get("finished_at", "")) >= str(previous.get("finished_at", "")):
            output[key] = row
    return output


def _run_jobs(
    jobs: list[tuple[dict[str, Any], Callable[[], dict[str, Any]]]],
    sink: JsonlSink,
    concurrency: int,
) -> Counter:
    pending = [(meta, fn) for meta, fn in jobs if sink.key(meta) not in sink.done]
    counts: Counter = Counter()

    def execute(meta: dict[str, Any], fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            return {**meta, "status": "ok", "finished_at": now(), **fn()}
        except Exception as exc:  # preserve a resumable error audit
            return {
                **meta,
                "status": "error",
                "finished_at": now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(execute, meta, fn) for meta, fn in pending]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            sink.append(row)
            counts[row["status"]] += 1
            if index % 50 == 0 or index == len(futures):
                print(json.dumps({"finished": index, "pending": len(futures), "status": dict(counts)}), flush=True)
    counts["already_ok"] = len(jobs) - len(pending)
    return counts


def _run_with_round_retries(
    jobs: list[tuple[dict[str, Any], Callable[[], dict[str, Any]]]],
    sink: JsonlSink,
    concurrency: int,
    rounds: int,
    label: str,
) -> Counter:
    total: Counter = Counter()
    for round_index in range(max(1, rounds)):
        counts = _run_jobs(jobs, sink, concurrency)
        total.update(counts)
        remaining = sum(sink.key(meta) not in sink.done for meta, _ in jobs)
        print(json.dumps({"stage": label, "round": round_index + 1, "remaining": remaining}), flush=True)
        if remaining == 0:
            break
    return total


def _condition_text(method: str) -> str:
    return {
        "direct": DIRECT_CONDITION,
        "discriminability": DISCRIMINABILITY_CONDITION,
        "stability": STABILITY_CONDITION,
    }[method]


def _normalize_rubrics(data: dict[str, Any], rubric_count: int) -> list[dict[str, Any]]:
    rows = data.get("rubrics")
    if not isinstance(rows, list) or len(rows) != rubric_count:
        raise ValueError(f"rubrics must contain exactly {rubric_count} objects")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"rubric {index} is not an object")
        criterion = " ".join(str(row.get("criterion", "")).split())
        if len(criterion) < 12:
            raise ValueError(f"rubric {index} criterion is too short")
        signature = criterion.casefold()
        if signature in seen:
            raise ValueError(f"rubric {index} exactly duplicates another criterion")
        seen.add(signature)
        desired = str(row.get("desired_state", "criterion_met"))
        if desired not in {"criterion_met", "criterion_not_met"}:
            raise ValueError(f"rubric {index} has invalid desired_state")
        output.append(
            {
                "rubric_id": f"R{index:02d}",
                "criterion": criterion,
                "desired_state": desired,
                "category": " ".join(str(row.get("category", "task_quality")).split())[:120],
                "rationale": " ".join(str(row.get("rationale", "")).split())[:500],
            }
        )
    return output


def _generate_rubrics_with_schema_retries(
    model: JsonModel,
    system: str,
    user: str,
    rubric_count: int,
    label: str,
) -> list[dict[str, Any]]:
    """Retry parseable responses that violate this experiment's schema."""
    retries = max(0, int(os.getenv("MECHANISM_SCHEMA_RETRIES", "4")))
    current_user = user
    last_error: ValueError | None = None
    for attempt in range(retries + 1):
        data = model.call(system, current_user)
        try:
            return _normalize_rubrics(data, rubric_count)
        except ValueError as exc:
            last_error = exc
            if attempt >= retries:
                break
            print(
                f"[RQ2 SCHEMA RETRY] label={label} attempt={attempt + 1} error={exc}",
                file=sys.stderr,
                flush=True,
            )
            current_user = (
                user
                + "\n\nYOUR PREVIOUS JSON WAS PARSEABLE BUT FAILED THE RUBRIC SCHEMA.\n"
                + f"Validation error: {exc}\n"
                + f"Regenerate the entire panel with exactly {rubric_count} rubric objects. "
                + "Count the objects before returning. Every criterion text must be non-empty "
                + "and distinct; do not repeat or lightly rephrase a criterion merely to reach "
                + "the required count. Keep desired_state restricted to criterion_met or "
                + "criterion_not_met. Return only the corrected JSON object."
            )
    assert last_error is not None
    raise last_error


def _normalize_count_plan(data: dict[str, Any], minimum: int, maximum: int) -> dict[str, Any]:
    value = data.get("target_rubric_count")
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"target_rubric_count must be an integer in [{minimum},{maximum}]")
    raw_aspects = data.get("aspects")
    if not isinstance(raw_aspects, list) or len(raw_aspects) != value:
        raise ValueError("aspects must contain exactly target_rubric_count strings")
    aspects: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_aspects, 1):
        aspect = " ".join(str(raw).split())
        if len(aspect) < 4:
            raise ValueError(f"aspect {index} is too short")
        signature = aspect.casefold()
        if signature in seen:
            raise ValueError(f"aspect {index} exactly duplicates another aspect")
        seen.add(signature)
        aspects.append(aspect[:500])
    return {
        "target_rubric_count": value,
        "aspects": aspects,
        "reason": " ".join(str(data.get("reason", "")).split())[:1000],
    }


def _plan_count_with_schema_retries(
    model: JsonModel,
    system: str,
    user: str,
    minimum: int,
    maximum: int,
    label: str,
) -> dict[str, Any]:
    retries = max(0, int(os.getenv("MECHANISM_SCHEMA_RETRIES", "4")))
    current_user = user
    last_error: ValueError | None = None
    for attempt in range(retries + 1):
        data = model.call(system, current_user)
        try:
            return _normalize_count_plan(data, minimum, maximum)
        except ValueError as exc:
            last_error = exc
            if attempt >= retries:
                break
            print(
                f"[RQ2 COUNT SCHEMA RETRY] label={label} attempt={attempt + 1} error={exc}",
                file=sys.stderr,
                flush=True,
            )
            current_user = (
                user
                + "\n\nYOUR PREVIOUS JSON FAILED CARDINALITY-PLAN VALIDATION.\n"
                + f"Validation error: {exc}\n"
                + f"Return an integer count in [{minimum},{maximum}] and exactly that many "
                + "distinct, atomic aspect strings. Return only the corrected JSON object."
            )
    assert last_error is not None
    raise last_error


def cmd_plan(args: argparse.Namespace) -> int:
    if args.min_count > args.max_count:
        raise ValueError("--min-count cannot exceed --max-count")
    samples = load_samples(args.dataset, args.limit)
    if args.require_samples and len(samples) != args.require_samples:
        raise RuntimeError(f"Expected {args.require_samples} preference samples, found {len(samples)}")
    spec = ModelSpec.parse(args.generator)
    system = COUNT_PLANNING_PROMPT.format(minimum=args.min_count, maximum=args.max_count)
    sink = JsonlSink(args.output, ("sample_id", "input_sha256"))
    jobs: list[tuple[dict[str, Any], Callable[[], dict[str, Any]]]] = []
    for sample in samples:
        input_hash = json_digest(
            {
                "sample_id": sample.sample_id,
                "query": sample.query,
                "system": system,
                "generator": args.generator,
                "temperature": args.temperature,
                "min_count": args.min_count,
                "max_count": args.max_count,
            }
        )
        meta = {
            "stage": "query_only_cardinality_plan",
            "sample_index": sample.index,
            "sample_id": sample.sample_id,
            "generator": spec.name,
            "generator_model": spec.model,
            "generator_base_urls": list(spec.base_urls),
            "input_sha256": input_hash,
            "prompt_sha256": json_digest(system),
            "query_only": True,
            "candidate_responses_visible": False,
            "gold_preference_visible": False,
            "count_policy": "adaptive_query_only_matched_v1",
            "min_count": args.min_count,
            "max_count": args.max_count,
        }

        def build(s=sample):
            model = JsonModel(spec, args.temperature, args.max_tokens)
            plan = _plan_count_with_schema_retries(
                model, system, f"QUERY:\n{s.query}", args.min_count, args.max_count, s.sample_id
            )
            return {"query": s.query, **plan, "plan_sha256": json_digest(plan)}

        jobs.append((meta, build))
    counts = _run_with_round_retries(jobs, sink, args.concurrency, args.retry_rounds, "plan")
    complete = sum(sink.key(meta) in sink.done for meta, _ in jobs)
    print(json.dumps({"counts": dict(counts), "complete": complete, "expected": len(jobs)}), flush=True)
    return 0 if complete == len(jobs) else 1


def cmd_generate(args: argparse.Namespace) -> int:
    samples = load_samples(args.dataset, args.limit)
    if args.require_samples and len(samples) != args.require_samples:
        raise RuntimeError(f"Expected {args.require_samples} preference samples, found {len(samples)}")
    spec = ModelSpec.parse(args.generator)
    if args.rubric_count:
        if not 3 <= args.rubric_count <= 32:
            raise ValueError("--rubric-count must be in [3,32]")
        count_plans: dict[tuple[Any, ...], dict[str, Any]] = {}
        count_policy = "fixed_global_count_v1"
    else:
        if args.count_plan is None:
            raise ValueError("adaptive generation requires --count-plan (or set --rubric-count for a fixed-count run)")
        count_plans = latest_ok(args.count_plan, ("sample_id",))
        missing_plans = [sample.sample_id for sample in samples if (sample.sample_id,) not in count_plans]
        if missing_plans:
            raise RuntimeError(f"Missing {len(missing_plans)} count plans; first={missing_plans[0]}")
        count_policy = "adaptive_query_only_matched_v1"
    sink = JsonlSink(args.output, ("sample_id", "method", "input_sha256"))
    jobs: list[tuple[dict[str, Any], Callable[[], dict[str, Any]]]] = []
    for sample in samples:
        plan = count_plans.get((sample.sample_id,), {})
        target_count = int(args.rubric_count or plan["target_rubric_count"])
        plan_hash = str(plan.get("plan_sha256") or "fixed")
        for method in METHODS:
            system = BASE_GENERATION_PROMPT.format(rubric_count=target_count) + "\n\n" + _condition_text(method)
            input_hash = json_digest(
                {
                    "sample_id": sample.sample_id,
                    "query": sample.query,
                    "method": method,
                    "system": system,
                    "generator": args.generator,
                    "temperature": args.temperature,
                    "target_rubric_count": target_count,
                    "count_policy": count_policy,
                    "count_plan_sha256": plan_hash,
                }
            )
            meta = {
                "stage": "rubric_generation",
                "sample_index": sample.index,
                "sample_id": sample.sample_id,
                "method": method,
                "generator": spec.name,
                "generator_model": spec.model,
                "generator_base_urls": list(spec.base_urls),
                "input_sha256": input_hash,
                "prompt_sha256": json_digest(system),
                "query_only": True,
                "candidate_responses_visible": False,
                "gold_preference_visible": False,
                "aggregation": "uniform_mean",
                "count_policy": count_policy,
                "target_rubric_count": target_count,
                "count_plan_sha256": plan_hash,
            }

            def build(s=sample, sys_prompt=system, condition=method, expected_count=target_count):
                model = JsonModel(spec, args.temperature, args.max_tokens)
                rubrics = _generate_rubrics_with_schema_retries(
                    model,
                    sys_prompt,
                    f"QUERY:\n{s.query}",
                    expected_count,
                    f"{s.sample_id}/{condition}",
                )
                return {"query": s.query, "rubrics": rubrics, "rubric_count": len(rubrics), "rubric_sha256": json_digest(rubrics)}

            jobs.append((meta, build))
    counts = _run_with_round_retries(jobs, sink, args.concurrency, args.retry_rounds, "generate")
    complete = sum(sink.key(meta) in sink.done for meta, _ in jobs)
    print(json.dumps({"counts": dict(counts), "complete": complete, "expected": len(jobs)}), flush=True)
    return 0 if complete == len(jobs) else 1


def _normalize_scores(data: dict[str, Any], rubric_ids: list[str]) -> tuple[dict[str, int], dict[str, str]]:
    rows = data.get("scores")
    if not isinstance(rows, list) or len(rows) != len(rubric_ids):
        raise ValueError("scores must return every rubric exactly once")
    scores: dict[str, int] = {}
    evidence: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("score row is not an object")
        # Whitespace around an otherwise exact ID is a serialization defect,
        # not a different scoring decision.
        rubric_id = str(row.get("rubric_id", "")).strip()
        value = row.get("score")
        if rubric_id in scores or rubric_id not in rubric_ids:
            raise ValueError(f"invalid or duplicate rubric_id={rubric_id!r}")
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"invalid score for {rubric_id}: {value!r}")
        scores[rubric_id] = value
        evidence[rubric_id] = " ".join(str(row.get("evidence", "")).split())[:500]
    if set(scores) != set(rubric_ids):
        raise ValueError("score response omitted rubric IDs")
    return scores, evidence


def _score_with_schema_retries(
    model: JsonModel,
    user: str,
    rubric_ids: list[str],
    label: str,
) -> tuple[dict[str, int], dict[str, str]]:
    retries = max(0, int(os.getenv("MECHANISM_SCHEMA_RETRIES", "4")))
    current_user = user
    last_error: ValueError | None = None
    for attempt in range(retries + 1):
        data = model.call(SCORE_SYSTEM_PROMPT, current_user)
        try:
            return _normalize_scores(data, rubric_ids)
        except ValueError as exc:
            last_error = exc
            if attempt >= retries:
                break
            print(
                f"[RQ2 SCORE SCHEMA RETRY] label={label} attempt={attempt + 1} error={exc}",
                file=sys.stderr,
                flush=True,
            )
            current_user = (
                user
                + "\n\nYOUR PREVIOUS JSON WAS PARSEABLE BUT FAILED SCORE VALIDATION.\n"
                + f"Validation error: {exc}\n"
                + "Return every supplied rubric_id exactly once, with no leading/trailing "
                + "spaces and an integer score from 1 to 5. Keep evidence brief and avoid "
                + "verbatim quotations that can break JSON escaping. Return only JSON."
            )
    assert last_error is not None
    raise last_error


def cmd_judge(args: argparse.Namespace) -> int:
    samples = load_samples(args.dataset, args.limit)
    if args.require_samples and len(samples) != args.require_samples:
        raise RuntimeError(f"Expected {args.require_samples} preference samples, found {len(samples)}")
    by_id = {s.sample_id: s for s in samples}
    rubric_rows = latest_ok(args.rubrics, ("sample_id", "method"))
    methods = tuple(args.method or JUDGE_METHODS)
    missing = [(s.sample_id, m) for s in samples for m in methods if (s.sample_id, m) not in rubric_rows]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} generated rubric sets; first={missing[0]}")
    judges = [ModelSpec.parse(value) for value in args.judge]
    if len({j.name for j in judges}) != len(judges):
        raise ValueError("judge names must be unique")
    sink = JsonlSink(args.output, ("sample_id", "method", "judge", "candidate", "input_sha256"))
    jobs: list[tuple[dict[str, Any], Callable[[], dict[str, Any]]]] = []
    for (sample_id, method), rubric_row in rubric_rows.items():
        if sample_id not in by_id or method not in methods:
            continue
        sample = by_id[sample_id]
        rubrics = rubric_row["rubrics"]
        public = [
            {"rubric_id": r["rubric_id"], "criterion": r["criterion"], "desired_state": r["desired_state"]}
            for r in rubrics
        ]
        rubric_ids = [r["rubric_id"] for r in rubrics]
        for judge in judges:
            for candidate, response in (("A", sample.response_a), ("B", sample.response_b)):
                user = SCORE_USER_PROMPT.format(
                    query=sample.query,
                    response=response,
                    rubrics=json.dumps(public, ensure_ascii=False, separators=(",", ":")),
                )
                input_hash = json_digest(
                    {
                        "sample_id": sample_id,
                        "method": method,
                        "judge": {"name": judge.name, "model": judge.model, "urls": judge.base_urls},
                        "candidate": candidate,
                        "rubric_sha256": rubric_row["rubric_sha256"],
                        "prompt": SCORE_SYSTEM_PROMPT,
                        "response": response,
                    }
                )
                meta = {
                    "stage": "preference_response_scoring",
                    "sample_index": sample.index,
                    "sample_id": sample_id,
                    "method": method,
                    "judge": judge.name,
                    "judge_model": judge.model,
                    "judge_base_urls": list(judge.base_urls),
                    "candidate": candidate,
                    "gold_preference": sample.gold,
                    "rubric_sha256": rubric_row["rubric_sha256"],
                    "rubric_count": len(rubrics),
                    "count_policy": rubric_row.get("count_policy", "unknown"),
                    "target_rubric_count": rubric_row.get("target_rubric_count", len(rubrics)),
                    "input_sha256": input_hash,
                    "prompt_sha256": json_digest(SCORE_SYSTEM_PROMPT),
                    "scoring_mode": "independent_response_full_panel_1to5_v1",
                    "aggregation": "uniform_mean",
                }

                def score(
                    model_spec=judge,
                    prompt=user,
                    ids=rubric_ids,
                    score_label=f"{sample_id}/{method}/{judge.name}/{candidate}",
                ):
                    model = JsonModel(model_spec, 0.0, args.max_tokens)
                    scores, evidence = _score_with_schema_retries(model, prompt, ids, score_label)
                    return {
                        "rubric_scores": scores,
                        "evidence": evidence,
                        "uniform_mean_score": statistics.fmean(scores.values()),
                        "calls": 1,
                    }

                jobs.append((meta, score))
    counts = _run_with_round_retries(jobs, sink, args.concurrency, args.retry_rounds, "judge")
    complete = sum(sink.key(meta) in sink.done for meta, _ in jobs)
    print(json.dumps({"counts": dict(counts), "complete": complete, "expected": len(jobs)}), flush=True)
    return 0 if complete == len(jobs) else 1


def _normalize_quality(data: dict[str, Any]) -> dict[str, Any]:
    ratings = data.get("ratings")
    if not isinstance(ratings, dict) or set(ratings) != set(QUALITY_DIMENSIONS):
        raise ValueError(f"ratings must contain exactly {QUALITY_DIMENSIONS}")
    normalized: dict[str, int] = {}
    for name in QUALITY_DIMENSIONS:
        value = ratings[name]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"rating {name} must be an integer in [1,5]")
        normalized[name] = value
    overall = data.get("overall_quality")
    if isinstance(overall, bool) or not isinstance(overall, int) or not 0 <= overall <= 100:
        raise ValueError("overall_quality must be an integer in [0,100]")
    return {"ratings": normalized, "overall_quality": overall, "diagnosis": " ".join(str(data.get("diagnosis", "")).split())[:1000]}


def cmd_quality(args: argparse.Namespace) -> int:
    samples = load_samples(args.dataset, args.limit)
    if args.require_samples and len(samples) != args.require_samples:
        raise RuntimeError(f"Expected {args.require_samples} preference samples, found {len(samples)}")
    by_id = {s.sample_id: s for s in samples}
    rubric_rows = latest_ok(args.rubrics, ("sample_id", "method"))
    methods = tuple(args.method or METHODS)
    judge = ModelSpec.parse(args.quality_judge)
    sink = JsonlSink(args.output, ("sample_id", "method", "judge", "input_sha256"))
    jobs: list[tuple[dict[str, Any], Callable[[], dict[str, Any]]]] = []
    for (sample_id, method), row in rubric_rows.items():
        if sample_id not in by_id or method not in methods:
            continue
        sample = by_id[sample_id]
        # Deterministic order blinding prevents method-specific positional artifacts.
        blinded = list(row["rubrics"])
        random.Random(int(json_digest([sample_id, method])[:16], 16)).shuffle(blinded)
        public = [{"criterion": r["criterion"], "desired_state": r["desired_state"]} for r in blinded]
        user = QUALITY_USER_PROMPT.format(
            query=sample.query,
            rubrics=json.dumps(public, ensure_ascii=False, separators=(",", ":")),
        )
        input_hash = json_digest(
            {
                "sample_id": sample_id,
                "method": method,
                "rubric_sha256": row["rubric_sha256"],
                "judge": {"name": judge.name, "model": judge.model, "urls": judge.base_urls},
                "prompt": QUALITY_SYSTEM_PROMPT,
                "blinded_order": public,
            }
        )
        meta = {
            "stage": "blinded_rubric_quality",
            "sample_index": sample.index,
            "sample_id": sample_id,
            "method": method,
            "judge": judge.name,
            "judge_model": judge.model,
            "judge_base_urls": list(judge.base_urls),
            "rubric_sha256": row["rubric_sha256"],
            "input_sha256": input_hash,
            "prompt_sha256": json_digest(QUALITY_SYSTEM_PROMPT),
            "responses_visible": False,
            "method_visible": False,
        }

        def evaluate(prompt=user):
            return _normalize_quality(JsonModel(judge, 0.0, args.max_tokens).call(QUALITY_SYSTEM_PROMPT, prompt))

        jobs.append((meta, evaluate))
    expected = len(samples) * len(methods)
    if len(jobs) != expected:
        raise RuntimeError(f"Expected {expected} quality jobs, built {len(jobs)}; generation rows are incomplete")
    counts = _run_with_round_retries(jobs, sink, args.concurrency, args.retry_rounds, "quality")
    complete = sum(sink.key(meta) in sink.done for meta, _ in jobs)
    print(json.dumps({"counts": dict(counts), "complete": complete, "expected": len(jobs)}), flush=True)
    return 0 if complete == len(jobs) else 1


def _mean(values: Iterable[float]) -> float | None:
    xs = list(values)
    return statistics.fmean(xs) if xs else None


def _median(values: Iterable[float]) -> float | None:
    xs = list(values)
    return statistics.median(xs) if xs else None


def _bootstrap_ci(values: list[float], seed: int = 13, draws: int = 2000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(draws))
    return [means[int(0.025 * (draws - 1))], means[int(0.975 * (draws - 1))]]


def _paired_sign_permutation_p(differences: list[float], seed: int = 17, draws: int = 20000) -> float | None:
    nonzero = [x for x in differences if abs(x) > 1e-12]
    if not nonzero:
        return 1.0 if differences else None
    observed = abs(statistics.fmean(nonzero))
    rng = random.Random(seed)
    exceed = 0
    for _ in range(draws):
        value = abs(statistics.fmean(x if rng.random() < 0.5 else -x for x in nonzero))
        exceed += value >= observed
    return (exceed + 1) / (draws + 1)


def _paired_summary(base: list[float], treatment: list[float]) -> dict[str, Any]:
    if len(base) != len(treatment):
        raise ValueError("paired vectors differ in length")
    delta = [b - a for a, b in zip(base, treatment)]
    sd = statistics.stdev(delta) if len(delta) > 1 else 0.0
    return {
        "n": len(delta),
        "baseline_mean": _mean(base),
        "treatment_mean": _mean(treatment),
        "baseline_median": _median(base),
        "treatment_median": _median(treatment),
        "paired_delta_mean": _mean(delta),
        "paired_delta_median": _median(delta),
        "paired_delta_bootstrap_95ci": _bootstrap_ci(delta),
        "paired_sign_permutation_p": _paired_sign_permutation_p(delta),
        "paired_effect_dz": (statistics.fmean(delta) / sd if sd > 0 else None),
    }


def _score_records(
    samples: list[PreferenceSample], rows: dict[tuple[Any, ...], dict[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    judges = sorted({str(key[2]) for key in rows})
    for sample in samples:
        for method in JUDGE_METHODS:
            for judge in judges:
                a = rows.get((sample.sample_id, method, judge, "A"))
                b = rows.get((sample.sample_id, method, judge, "B"))
                if not a or not b:
                    continue
                score_a = float(a["uniform_mean_score"])
                score_b = float(b["uniform_mean_score"])
                margin = score_a - score_b
                aligned = margin if sample.gold == "A" else -margin
                output.append(
                    {
                        "sample_index": sample.index,
                        "sample_id": sample.sample_id,
                        "method": method,
                        "judge": judge,
                        "gold": sample.gold,
                        "score_a": score_a,
                        "score_b": score_b,
                        "margin_a_minus_b": margin,
                        "absolute_gap": abs(margin),
                        "gold_aligned_margin": aligned,
                        "correct": 0.5 if abs(margin) <= 1e-12 else float(aligned > 0),
                    }
                )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _histogram(values: list[float], bins: int, low: float, high: float) -> list[int]:
    counts = [0] * bins
    span = high - low or 1.0
    for value in values:
        index = int((value - low) / span * bins)
        counts[max(0, min(bins - 1, index))] += 1
    return counts


def _svg_histogram(
    path: Path,
    title: str,
    xlabel: str,
    series: list[tuple[str, list[float], str]],
    bins: int = 24,
    domain: tuple[float, float] | None = None,
) -> None:
    all_values = [x for _, values, _ in series for x in values]
    if not all_values:
        return
    low, high = domain or (min(all_values), max(all_values))
    if math.isclose(low, high):
        high = low + 1.0
    width, height = 920, 560
    left, right, top, bottom = 82, 30, 60, 78
    plot_w, plot_h = width - left - right, height - top - bottom
    histograms = [_histogram(values, bins, low, high) for _, values, _ in series]
    densities = [[count / max(1, len(values)) for count in counts] for counts, (_, values, _) in zip(histograms, series)]
    ymax = max(max(row) for row in densities) * 1.12 or 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="20">{html.escape(title)}</text>',
    ]
    for tick in range(6):
        y = top + plot_h * tick / 5
        value = ymax * (1 - tick / 5)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.3f}</text>')
    bar_w = plot_w / bins
    for series_index, ((label, values, color), density) in enumerate(zip(series, densities)):
        for index, value in enumerate(density):
            x = left + index * bar_w
            h = value / ymax * plot_h
            lines.append(
                f'<rect x="{x:.2f}" y="{top+plot_h-h:.2f}" width="{bar_w:.2f}" height="{h:.2f}" '
                f'fill="{color}" fill-opacity="0.34" stroke="{color}" stroke-opacity="0.70"/>'
            )
        lx = left + 18 + series_index * 220
        lines.append(f'<rect x="{lx}" y="{top+8}" width="18" height="12" fill="{color}" fill-opacity="0.45"/>')
        lines.append(f'<text x="{lx+25}" y="{top+19}" font-family="Arial" font-size="13">{html.escape(label)} (n={len(values)})</text>')
    lines.extend(
        [
            f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111827"/>',
            f'<text x="{width/2}" y="{height-24}" text-anchor="middle" font-family="Arial" font-size="15">{html.escape(xlabel)}</text>',
            f'<text x="20" y="{top+plot_h/2}" transform="rotate(-90 20 {top+plot_h/2})" text-anchor="middle" font-family="Arial" font-size="15">Proportion per bin</text>',
        ]
    )
    for tick in range(6):
        x = left + plot_w * tick / 5
        value = low + (high - low) * tick / 5
        lines.append(f'<text x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="Arial" font-size="12">{value:.2f}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _subset_summary(
    samples: list[PreferenceSample],
    score_records: list[dict[str, Any]],
    quality_rows: dict[tuple[Any, ...], dict[str, Any]],
    size: int,
    output_dir: Path,
) -> dict[str, Any]:
    chosen = samples[:size]
    ids = {s.sample_id for s in chosen}
    records = [r for r in score_records if r["sample_id"] in ids]
    judges = sorted({r["judge"] for r in records})
    by_key = {(r["sample_id"], r["method"], r["judge"]): r for r in records}

    direct_abs, disc_abs = [], []
    direct_aligned, disc_aligned = [], []
    per_judge_discrimination: dict[str, Any] = {}
    for judge in judges:
        base, treatment, ba, ta = [], [], [], []
        for sample in chosen:
            d = by_key.get((sample.sample_id, "direct", judge))
            x = by_key.get((sample.sample_id, "discriminability", judge))
            if d and x:
                base.append(d["absolute_gap"]); treatment.append(x["absolute_gap"])
                ba.append(d["gold_aligned_margin"]); ta.append(x["gold_aligned_margin"])
        direct_abs.extend(base); disc_abs.extend(treatment)
        direct_aligned.extend(ba); disc_aligned.extend(ta)
        per_judge_discrimination[judge] = {
            "absolute_gap": _paired_summary(base, treatment),
            "gold_aligned_margin_diagnostic": _paired_summary(ba, ta),
            "direct_accuracy_tie_half": _mean(float(by_key[(s.sample_id, "direct", judge)]["correct"]) for s in chosen if (s.sample_id, "direct", judge) in by_key),
            "discriminability_accuracy_tie_half": _mean(float(by_key[(s.sample_id, "discriminability", judge)]["correct"]) for s in chosen if (s.sample_id, "discriminability", judge) in by_key),
        }

    if len(judges) != 2:
        raise RuntimeError(f"Stability analysis requires exactly two judges, found {judges}")
    j1, j2 = judges
    stability_rows: list[dict[str, Any]] = []
    per_method_stability: dict[str, Any] = {}
    for method in ("direct", "stability"):
        raw_margin_gap, response_gap, direction_agree, reversals = [], [], [], []
        for sample in chosen:
            a = by_key.get((sample.sample_id, method, j1))
            b = by_key.get((sample.sample_id, method, j2))
            if not a or not b:
                continue
            margin_gap = abs(a["margin_a_minus_b"] - b["margin_a_minus_b"])
            score_gap = 0.5 * (abs(a["score_a"] - b["score_a"]) + abs(a["score_b"] - b["score_b"]))
            sign_a = 0 if abs(a["margin_a_minus_b"]) <= 1e-12 else (1 if a["margin_a_minus_b"] > 0 else -1)
            sign_b = 0 if abs(b["margin_a_minus_b"]) <= 1e-12 else (1 if b["margin_a_minus_b"] > 0 else -1)
            agree = float(sign_a == sign_b)
            reversal = float(sign_a * sign_b == -1)
            raw_margin_gap.append(margin_gap); response_gap.append(score_gap); direction_agree.append(agree); reversals.append(reversal)
            stability_rows.append(
                {
                    "sample_index": sample.index,
                    "sample_id": sample.sample_id,
                    "method": method,
                    "judge_1": j1,
                    "judge_2": j2,
                    "cross_judge_abs_margin_gap": margin_gap,
                    "cross_judge_mean_response_score_gap": score_gap,
                    "preference_direction_agreement": agree,
                    "strict_rank_reversal": reversal,
                }
            )
        per_method_stability[method] = {
            "n": len(raw_margin_gap),
            "cross_judge_abs_margin_gap_mean": _mean(raw_margin_gap),
            "cross_judge_abs_margin_gap_median": _median(raw_margin_gap),
            "cross_judge_mean_response_score_gap": _mean(response_gap),
            "preference_direction_agreement": _mean(direction_agree),
            "strict_rank_reversal_rate": _mean(reversals),
        }
    direct_stability = [r["cross_judge_abs_margin_gap"] for r in stability_rows if r["method"] == "direct"]
    stable_stability = [r["cross_judge_abs_margin_gap"] for r in stability_rows if r["method"] == "stability"]

    quality: dict[str, Any] = {}
    quality_plot: list[tuple[str, list[float], str]] = []
    colors = {"direct": "#64748b", "discriminability": "#dc2626", "stability": "#2563eb"}
    for method in METHODS:
        rows = [row for (sid, m, _judge), row in quality_rows.items() if sid in ids and m == method]
        overall = [float(row["overall_quality"]) for row in rows]
        quality[method] = {
            "n": len(rows),
            "overall_quality_mean": _mean(overall),
            "overall_quality_median": _median(overall),
            "dimension_means": {name: _mean(float(row["ratings"][name]) for row in rows) for name in QUALITY_DIMENSIONS},
        }
        if overall:
            quality_plot.append((method, overall, colors[method]))

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "score_records.csv", [r for r in records])
    _write_csv(output_dir / "stability_records.csv", stability_rows)
    _svg_histogram(
        output_dir / "exp1_absolute_score_gap_histogram.svg",
        f"RQ2 Discriminability Manipulation (N={size})",
        "|uniform score(A) - uniform score(B)|",
        [("Direct", direct_abs, colors["direct"]), ("Discriminability prompt", disc_abs, colors["discriminability"])],
        domain=(0.0, 4.0),
    )
    _svg_histogram(
        output_dir / "exp1_paired_delta_histogram.svg",
        f"Paired Change in Absolute Gap (N={size}; sample × judge)",
        "discriminability absolute gap - direct absolute gap",
        [("Paired delta", [b - a for a, b in zip(direct_abs, disc_abs)], colors["discriminability"])],
        domain=(-4.0, 4.0),
    )
    _svg_histogram(
        output_dir / "exp2_cross_judge_margin_gap_histogram.svg",
        f"RQ2 Cross-Judge Stability Manipulation (N={size})",
        "|margin(Qwen) - margin(Llama)| (lower is more stable)",
        [("Direct", direct_stability, colors["direct"]), ("Stability prompt", stable_stability, colors["stability"])],
        domain=(0.0, 8.0),
    )
    if quality_plot:
        _svg_histogram(
            output_dir / "quality_overall_histogram.svg",
            f"Blinded Text-Only Rubric Quality (N={size})",
            "DeepSeek overall quality (0-100)",
            quality_plot,
            domain=(0.0, 100.0),
        )

    result = {
        "sample_size": len(chosen),
        "sample_selection": "deterministic prefix of the fixed 750-sample RubricBench preference order",
        "experiment_1_discriminability": {
            "primary_metric": "absolute uniform-mean A/B score gap",
            "pooled_sample_judge_pairs": _paired_summary(direct_abs, disc_abs),
            "per_judge": per_judge_discrimination,
            "warning": "Absolute gap measures separation strength, not correctness; gold-aligned margin and accuracy are diagnostic safeguards.",
        },
        "experiment_2_stability": {
            "primary_metric": "absolute difference between the two judges' A-minus-B margins",
            "paired_direct_vs_stability": _paired_summary(direct_stability, stable_stability),
            "per_method": per_method_stability,
        },
        "blinded_text_only_quality": quality,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def cmd_summarize(args: argparse.Namespace) -> int:
    samples = load_samples(args.dataset, args.limit)
    if args.require_samples and len(samples) != args.require_samples:
        raise RuntimeError(f"Expected {args.require_samples} preference samples, found {len(samples)}")
    score_rows = latest_ok(args.scores, ("sample_id", "method", "judge", "candidate"))
    records = _score_records(samples, score_rows)
    expected = len(samples) * len(JUDGE_METHODS) * 2
    if len(records) != expected:
        raise RuntimeError(f"Need {expected} paired method/judge records, found {len(records)}")
    quality_rows = latest_ok(args.quality, ("sample_id", "method", "judge")) if args.quality.exists() else {}
    output: dict[str, Any] = {
        "dataset": str(args.dataset),
        "available_samples": len(samples),
        "judge_names": sorted({r["judge"] for r in records}),
        "aggregation": "uniform mean; no learned or assigned weights",
        "subsets": {},
    }
    if args.rubrics is not None:
        rubric_rows = latest_ok(args.rubrics, ("sample_id", "method"))
        missing = [
            (sample.sample_id, method)
            for sample in samples
            for method in METHODS
            if (sample.sample_id, method) not in rubric_rows
        ]
        if missing:
            raise RuntimeError(f"Cardinality audit is missing {len(missing)} rubric sets; first={missing[0]}")
        distribution: Counter = Counter()
        mismatches: list[dict[str, Any]] = []
        policies: Counter = Counter()
        for sample in samples:
            method_counts: dict[str, int] = {}
            targets: dict[str, int] = {}
            for method in METHODS:
                row = rubric_rows[(sample.sample_id, method)]
                count = len(row.get("rubrics") or [])
                target = int(row.get("target_rubric_count", count))
                method_counts[method] = count
                targets[method] = target
                policies[str(row.get("count_policy", "unknown"))] += 1
            if len(set(method_counts.values())) != 1 or len(set(targets.values())) != 1 or method_counts != targets:
                mismatches.append(
                    {"sample_id": sample.sample_id, "method_counts": method_counts, "targets": targets}
                )
            else:
                distribution[next(iter(method_counts.values()))] += 1
        if mismatches:
            raise RuntimeError(
                f"Cardinality matching failed for {len(mismatches)} samples; first={mismatches[0]}"
            )
        expanded = [count for count, frequency in distribution.items() for _ in range(frequency)]
        output["cardinality_control"] = {
            "policy": "within-sample matched; across-sample adaptive",
            "method_panel": list(METHODS),
            "matched_samples": len(samples),
            "mismatched_samples": 0,
            "minimum": min(expanded),
            "maximum": max(expanded),
            "mean": statistics.fmean(expanded),
            "median": statistics.median(expanded),
            "distribution": {str(key): distribution[key] for key in sorted(distribution)},
            "recorded_policies": dict(policies),
        }
    for size in args.summary_sizes:
        actual = min(size, len(samples))
        output["subsets"][str(actual)] = _subset_summary(
            samples, records, quality_rows, actual, args.output_dir / f"n{actual}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary_all.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--require-samples", type=int, default=750)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan")
    add_data_args(plan)
    plan.add_argument("--generator", required=True)
    plan.add_argument("--min-count", type=int, default=4, choices=range(3, 33))
    plan.add_argument("--max-count", type=int, default=12, choices=range(3, 33))
    plan.add_argument("--temperature", type=float, default=0.0)
    plan.add_argument("--max-tokens", type=int, default=2048)
    plan.add_argument("--concurrency", type=int, default=8)
    plan.add_argument("--retry-rounds", type=int, default=4)
    plan.add_argument("--output", type=Path, required=True)
    plan.set_defaults(func=cmd_plan)

    generate = sub.add_parser("generate")
    add_data_args(generate)
    generate.add_argument("--generator", required=True)
    generate.add_argument("--count-plan", type=Path)
    generate.add_argument("--rubric-count", type=int, default=0, help="fixed-count robustness mode; omit for adaptive planning")
    generate.add_argument("--temperature", type=float, default=0.1)
    generate.add_argument("--max-tokens", type=int, default=8192)
    generate.add_argument("--concurrency", type=int, default=8)
    generate.add_argument("--retry-rounds", type=int, default=4)
    generate.add_argument("--output", type=Path, required=True)
    generate.set_defaults(func=cmd_generate)

    judge = sub.add_parser("judge")
    add_data_args(judge)
    judge.add_argument("--rubrics", type=Path, required=True)
    judge.add_argument("--method", action="append", help="Rubric method to score; repeat for multiple methods")
    judge.add_argument("--judge", action="append", required=True)
    judge.add_argument("--max-tokens", type=int, default=8192)
    judge.add_argument("--concurrency", type=int, default=32)
    judge.add_argument("--retry-rounds", type=int, default=4)
    judge.add_argument("--output", type=Path, required=True)
    judge.set_defaults(func=cmd_judge)

    quality = sub.add_parser("quality")
    add_data_args(quality)
    quality.add_argument("--rubrics", type=Path, required=True)
    quality.add_argument("--method", action="append", help="Rubric method to audit; repeat for multiple methods")
    quality.add_argument("--quality-judge", required=True)
    quality.add_argument("--max-tokens", type=int, default=2048)
    quality.add_argument("--concurrency", type=int, default=12)
    quality.add_argument("--retry-rounds", type=int, default=4)
    quality.add_argument("--output", type=Path, required=True)
    quality.set_defaults(func=cmd_quality)

    summarize = sub.add_parser("summarize")
    add_data_args(summarize)
    summarize.add_argument("--scores", type=Path, required=True)
    summarize.add_argument("--quality", type=Path, required=True)
    summarize.add_argument("--rubrics", type=Path)
    summarize.add_argument("--summary-sizes", type=int, nargs="+", default=[500, 750])
    summarize.add_argument("--output-dir", type=Path, required=True)
    summarize.set_defaults(func=cmd_summarize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
