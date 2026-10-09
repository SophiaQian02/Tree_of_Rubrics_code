from __future__ import annotations

import json
import math
import os
import re
import statistics
from itertools import combinations
from typing import Any

from project.mechanism_analysis.core import JsonModel, Sample, require_exact_ids
from project.mechanism_analysis.experiments import _validated_call
from project.mechanism_analysis.prompts import JSON_SYSTEM
from project.utils.openai_client import clip_text_middle


SEMANTIC_REWRITE_PROMPT = """Rewrite the immutable Gold rubric into a high-quality atomic rubric set using semantic analysis only.

QUERY:
{query}

IMMUTABLE GOLD RUBRICS:
{gold}

Goals:
1. Preserve the complete meaning and coverage of every Gold rubric.
2. Make every criterion operational and consistently scoreable from observable response content.
3. Improve discrimination between strong, medium, and weak plausible responses.
4. Remove semantic duplication. You may merge overlapping Gold items or split a compound Gold item.
5. You may add a small number of genuinely useful, query-grounded novel criteria, but never add generic style filler.

You do not have calibration responses, candidate A/B responses, or a preference label. Do not guess them.
Every Gold ID must occur in source_gold_ids of at least one returned item. Multiple output items may cite the same Gold ID after a useful split; one item may cite multiple Gold IDs after a justified merge. A novel item uses an empty source_gold_ids list.

Return 1-{max_items} rubrics:
{{"rubrics":[{{"criterion":"observable scoring criterion","desired_state":"criterion_met|criterion_not_met","category":"short semantic category","source_gold_ids":["L01"],"operation":"preserve|rewrite|split|merge|novel","rationale":"brief reason"}}]}}
"""


CALIBRATION_SINGLE_RUBRIC_SCORE_PROMPT = """Score every calibration response against the ONE rubric below independently.

QUERY:
{query}

RUBRIC:
{rubric}

CALIBRATION RESPONSES:
{responses}

For desired_state=criterion_met, a high score means the response satisfies the criterion.
For desired_state=criterion_not_met, a high score means the undesirable behavior is absent.
Use this fixed scale:
1 = clearly violates the desired state;
2 = mostly violates it;
3 = mixed, partial, or genuinely borderline;
4 = mostly satisfies it with a minor defect;
5 = clearly and fully satisfies it.

Judge only observable response content. Do not use the response's declared quality target as a substitute for inspection.
Return every response ID exactly once, with no missing, duplicate, or additional IDs. Return integers only:
{{"response_scores":[{{"response_id":"C01","score":1}}]}}
"""

# Backward-compatible import name.  Experiment 4 v2 uses one request per
# rubric; this is no longer a whole-matrix prompt.
CALIBRATION_SCORE_PROMPT = CALIBRATION_SINGLE_RUBRIC_SCORE_PROMPT


CALIBRATION_REWRITE_PROMPT = """Revise a rubric set using calibration-response evidence. This is calibration, not preference-label fitting.

QUERY:
{query}

IMMUTABLE GOLD RUBRICS THAT MUST REMAIN COVERED:
{gold}

CURRENT RUBRICS:
{rubrics}

CALIBRATION RESPONSES (fixed across methods and rounds):
{responses}

TWO-JUDGE SCORE MATRICES AND DERIVED DIAGNOSTICS:
{diagnostics}

Revision rules:
- High judge-instability means the criterion is not operational enough: rewrite its decision boundary using observable evidence.
- Low response-discrimination means the criterion does not separate plausible response strengths: split it into finer independently scoreable requirements when a meaningful split exists.
- A highly correlated pair is only a merge candidate. Merge it only when its semantic content or required evidence also overlaps; do not merge distinct requirements merely because six responses happened to receive similar scores.
- Preserve every Gold requirement. Every Gold ID must remain represented in source_gold_ids.
- Do not inspect candidate A/B responses and do not infer a preference label.
- Do not add generic style criteria. Novel items must be useful and grounded in this exact query.

Return the complete replacement rubric set, not a patch. Return 1-{max_items} rubrics:
{{"rubrics":[{{"criterion":"observable scoring criterion","desired_state":"criterion_met|criterion_not_met","category":"short semantic category","source_gold_ids":["L01"],"operation":"preserve|rewrite|split|merge|novel","rationale":"brief calibration-based reason"}}]}}
"""


FINAL_QUALITY_PROMPT = """Evaluate one candidate rubric set for the query. The method name is hidden. Compare semantic coverage against the immutable Gold reference, but assess the candidate on its own wording.

QUERY:
{query}

IMMUTABLE GOLD REFERENCE:
{gold}

CANDIDATE RUBRIC SET:
{candidate}

Rate each dimension with an INTEGER from 1 to 5:
- coverage: 1 misses most Gold requirements; 5 fully preserves all Gold requirements.
- discriminability: 1 is vague and unlikely to separate plausible response quality; 5 has concrete atomic boundaries that distinguish strong, medium, and weak responses.
- redundancy: this is a REDUNDANCY-CONTROL score where higher is better. 1 has severe duplication/subsumption; 5 has minimal semantic or evidence overlap.
- novelty: 1 adds no useful distinction beyond the Gold reference; 5 adds multiple valid, useful, query-grounded distinctions absent from Gold. Do not reward paraphrases or generic style filler.

Do not reward a rubric merely for having more items. Return:
{{"ratings":{{"coverage":1,"discriminability":1,"redundancy":1,"novelty":1}},"reasons":{{"coverage":"...","discriminability":"...","redundancy":"...","novelty":"..."}}}}
"""


SINGLE_RUBRIC_RESPONSE_SCORE_PROMPT = """Score one response against the ONE rubric below. You are not selecting between A and B, and no preference label is available.

QUERY:
{query}

RESPONSE:
{response}

RUBRIC:
{rubric}

For desired_state=criterion_met, a high score means the response satisfies the criterion.
For desired_state=criterion_not_met, a high score means the undesirable behavior is absent.
Use integers only:
1 = clearly violates the desired state;
2 = mostly violates it;
3 = mixed, partial, or borderline;
4 = mostly satisfies it;
5 = clearly and fully satisfies it.

Judge observable content only. Return exactly one integer score and no prose:
{{"score":1}}
"""

# Backward-compatible import name.  Preference scoring now persists one
# request per rubric and assembles the response-level score vector in code.
RESPONSE_RUBRIC_SCORE_PROMPT = SINGLE_RUBRIC_RESPONSE_SCORE_PROMPT


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def gold_rubrics(sample: Sample) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for leaf in sample.gold_flat:
        path = [str(item.get("name") or item.get("node_id") or "") for item in leaf.get("dimension_path", [])]
        output.append(
            {
                "rubric_id": str(leaf["leaf_id"]),
                "criterion": str(leaf["criterion"]).strip(),
                "desired_state": str(leaf.get("desired_state") or "criterion_met"),
                "category": " > ".join(item for item in path if item) or str(leaf.get("dimension_id") or "Other"),
                "source_gold_ids": [str(leaf["leaf_id"])],
                "operation": "gold",
            }
        )
    return output


def public_rubrics(rubrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hide construction lineage from judges to avoid giving methods credit by metadata."""
    return [
        {
            "rubric_id": str(item["rubric_id"]),
            "criterion": str(item["criterion"]),
            "desired_state": str(item.get("desired_state") or "criterion_met"),
            "category": str(item.get("category") or "Other"),
        }
        for item in rubrics
    ]


def _normalize_generated_rubrics(
    data: dict[str, Any],
    *,
    gold_ids: set[str],
    id_prefix: str,
    max_items: int,
) -> list[dict[str, Any]]:
    rows = data.get("rubrics")
    if not isinstance(rows, list) or not 1 <= len(rows) <= max_items:
        raise ValueError(f"rubrics must contain 1-{max_items} objects")
    desired_values = {"criterion_met", "criterion_not_met"}
    operations = {"preserve", "rewrite", "split", "merge", "novel"}
    output: list[dict[str, Any]] = []
    covered: set[str] = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"rubric {index} is not an object")
        criterion = str(row.get("criterion") or "").strip()
        if not criterion:
            raise ValueError(f"rubric {index} has an empty criterion")
        desired_state = str(row.get("desired_state") or "")
        if desired_state not in desired_values:
            raise ValueError(f"rubric {index} has invalid desired_state={desired_state!r}")
        source_ids = row.get("source_gold_ids")
        if not isinstance(source_ids, list) or any(str(item) not in gold_ids for item in source_ids):
            raise ValueError(f"rubric {index} has invalid source_gold_ids")
        source_ids = list(dict.fromkeys(str(item) for item in source_ids))
        operation = str(row.get("operation") or "")
        if operation not in operations:
            raise ValueError(f"rubric {index} has invalid operation={operation!r}")
        if not source_ids and operation != "novel":
            raise ValueError(f"rubric {index} without Gold lineage must use operation=novel")
        covered.update(source_ids)
        output.append(
            {
                "rubric_id": f"{id_prefix}{index:02d}",
                "criterion": criterion,
                "desired_state": desired_state,
                "category": str(row.get("category") or "Other").strip() or "Other",
                "source_gold_ids": source_ids,
                "operation": operation,
                "rationale": str(row.get("rationale") or "").strip(),
            }
        )
    missing = gold_ids - covered
    if missing:
        raise ValueError(f"generated rubric lost Gold lineage IDs: {sorted(missing)}")
    return output


def _repair_generated_rubrics(
    data: dict[str, Any],
    *,
    gold: list[dict[str, Any]],
    fallback_rubrics: list[dict[str, Any]],
    id_prefix: str,
    max_items: int,
    validation_error: Exception,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically repair a final invalid GLM rubric replacement.

    Valid generated criteria are retained. Unknown lineage IDs are removed,
    and any Gold IDs still missing are restored from the previous valid rubric
    set (or immutable Gold). Optional split/novel rows are only retained when
    they fit after reserving space for complete Gold coverage.
    """
    gold_ids_ordered = [str(item["rubric_id"]) for item in gold]
    gold_ids = set(gold_ids_ordered)
    if len(gold_ids) > max_items:
        raise ValueError(
            f"cannot preserve {len(gold_ids)} Gold requirements with max_items={max_items}"
        )

    raw_rows = data.get("rubrics")
    rows = raw_rows if isinstance(raw_rows, list) else []
    desired_values = {"criterion_met", "criterion_not_met"}
    operations = {"preserve", "rewrite", "split", "merge", "novel"}
    invalid_source_ids: set[str] = set()
    structurally_invalid_rows = 0
    normalized: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            structurally_invalid_rows += 1
            continue
        criterion = str(row.get("criterion") or "").strip()
        if not criterion:
            structurally_invalid_rows += 1
            continue

        raw_sources = row.get("source_gold_ids")
        if isinstance(raw_sources, list):
            candidates = [str(item) for item in raw_sources]
        elif isinstance(raw_sources, str):
            candidates = re.findall(r"L\d+", raw_sources)
        else:
            candidates = []
        invalid_source_ids.update(item for item in candidates if item not in gold_ids)
        source_ids = list(dict.fromkeys(item for item in candidates if item in gold_ids))

        desired_state = str(row.get("desired_state") or "")
        if desired_state not in desired_values:
            desired_state = "criterion_met"
        operation = str(row.get("operation") or "")
        if operation not in operations:
            operation = "rewrite" if source_ids else "novel"
        if not source_ids:
            operation = "novel"

        normalized.append(
            {
                "criterion": criterion,
                "desired_state": desired_state,
                "category": str(row.get("category") or "Other").strip() or "Other",
                "source_gold_ids": source_ids,
                "operation": operation,
                "rationale": str(row.get("rationale") or "").strip(),
            }
        )

    # Keep the first generated row that introduces each Gold ID as mandatory;
    # duplicate split rows and novel rows are optional extras.
    covered: set[str] = set()
    mandatory: list[dict[str, Any]] = []
    extras: list[dict[str, Any]] = []
    for row in normalized:
        introduced = set(row["source_gold_ids"]) - covered
        if introduced:
            mandatory.append(row)
            covered.update(row["source_gold_ids"])
        else:
            extras.append(row)

    missing = set(gold_ids) - covered
    fallback_pool = list(fallback_rubrics) + list(gold)
    appended: list[dict[str, Any]] = []
    appended_ids: set[str] = set()
    for item in fallback_pool:
        raw_sources = item.get("source_gold_ids")
        if isinstance(raw_sources, list):
            item_sources = [str(value) for value in raw_sources if str(value) in gold_ids]
        else:
            rubric_id = str(item.get("rubric_id") or "")
            item_sources = [rubric_id] if rubric_id in gold_ids else []
        useful = set(item_sources) & missing
        if not useful:
            continue
        appended.append(
            {
                "criterion": str(item.get("criterion") or "").strip(),
                "desired_state": (
                    str(item.get("desired_state"))
                    if str(item.get("desired_state")) in desired_values
                    else "criterion_met"
                ),
                "category": str(item.get("category") or "Other").strip() or "Other",
                "source_gold_ids": list(dict.fromkeys(item_sources)),
                "operation": "preserve",
                "rationale": "Deterministic schema repair: retained from the previous valid rubric set.",
            }
        )
        covered.update(item_sources)
        appended_ids.update(item_sources)
        missing = set(gold_ids) - covered
        if not missing:
            break
    if missing:
        raise ValueError(f"repair could not restore Gold lineage IDs: {sorted(missing)}")

    extra_capacity = max(0, max_items - len(mandatory) - len(appended))
    kept_extras = extras[:extra_capacity]
    repaired_input = mandatory + kept_extras + appended
    repaired = _normalize_generated_rubrics(
        {"rubrics": repaired_input},
        gold_ids=gold_ids,
        id_prefix=id_prefix,
        max_items=max_items,
    )
    audit = {
        "applied": True,
        "mode": "partial_generated_repair" if mandatory else "previous_valid_rubric_fallback",
        "primary_validation_error": f"{type(validation_error).__name__}: {validation_error}",
        "input_rubric_count": len(rows),
        "kept_generated_count": len(mandatory) + len(kept_extras),
        "appended_fallback_count": len(appended),
        "appended_fallback_gold_ids": sorted(appended_ids),
        "dropped_generated_count": max(0, len(rows) - len(mandatory) - len(kept_extras)),
        "structurally_invalid_rows": structurally_invalid_rows,
        "invalid_source_gold_ids_removed": sorted(invalid_source_ids),
    }
    return repaired, audit


def semantic_rewrite(sample: Sample, generator: JsonModel, max_items: int) -> dict[str, Any]:
    gold = gold_rubrics(sample)
    gold_ids = {item["rubric_id"] for item in gold}
    prompt = SEMANTIC_REWRITE_PROMPT.format(
        query=sample.query,
        gold=compact(public_rubrics(gold)),
        max_items=max_items,
    )
    repair_audit: dict[str, Any] = {}

    def repair(data: dict[str, Any], error: Exception) -> list[dict[str, Any]]:
        repaired, audit = _repair_generated_rubrics(
            data,
            gold=gold,
            fallback_rubrics=gold,
            id_prefix="S",
            max_items=max_items,
            validation_error=error,
        )
        repair_audit.update(audit)
        return repaired

    rubrics = _validated_call(
        generator,
        prompt,
        lambda data: _normalize_generated_rubrics(
            data, gold_ids=gold_ids, id_prefix="S", max_items=max_items
        ),
        "experiment4_semantic_rewrite",
        repair=repair,
    )
    result: dict[str, Any] = {"rubrics": rubrics, "generator_raw_calls": 1}
    if repair_audit:
        result["schema_repair"] = repair_audit
    return result


def response_panel(cache_row: dict[str, Any], required_count: int = 6) -> list[dict[str, Any]]:
    responses = [str(item).strip() for item in cache_row.get("responses", []) if str(item).strip()]
    specs = cache_row.get("response_specs")
    if len(responses) < required_count:
        raise ValueError(f"calibration cache contains only {len(responses)} responses; need {required_count}")
    if not isinstance(specs, list) or len(specs) < required_count:
        raise ValueError("calibration cache must include response_specs for every response")
    output = []
    for index, (response, spec) in enumerate(zip(responses[:required_count], specs[:required_count]), 1):
        if not isinstance(spec, dict):
            raise ValueError("response_specs entries must be objects")
        output.append(
            {
                "response_id": f"C{index:02d}",
                "model": str(spec.get("model") or "unknown"),
                "quality": str(spec.get("quality") or "unknown"),
                "prompt_modifier": spec.get("prompt_modifier"),
                "response": response,
            }
        )
    return output


def _score_matrix(data: dict[str, Any], response_ids: set[str], rubric_ids: set[str]) -> dict[str, dict[str, int]]:
    response_rows = require_exact_ids(data.get("response_scores", []), "response_id", response_ids, "response_scores")
    matrix: dict[str, dict[str, int]] = {}
    for response_row in response_rows:
        response_id = str(response_row["response_id"])
        rubric_rows = require_exact_ids(
            response_row.get("rubric_scores", []), "rubric_id", rubric_ids, f"rubric_scores:{response_id}"
        )
        values: dict[str, int] = {}
        for row in rubric_rows:
            value = row.get("score")
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                raise ValueError(f"{response_id}/{row.get('rubric_id')} score must be an integer in [1,5]")
            values[str(row["rubric_id"])] = value
        matrix[response_id] = values
    return matrix


def score_calibration_rubric(
    sample: Sample,
    rubric: dict[str, Any],
    panel: list[dict[str, Any]],
    judge: JsonModel,
) -> dict[str, Any]:
    """Score all fixed calibration responses for exactly one rubric.

    Keeping the model-facing unit to one rubric avoids asking a model to copy
    a large, repetitive rubric-ID matrix.  The caller persists each unit and
    assembles the full matrix deterministically.
    """
    rubric_id = str(rubric["rubric_id"])
    response_ids = {str(item["response_id"]) for item in panel}
    prompt_responses = [
        {
            "response_id": item["response_id"],
            "model": item["model"],
            "quality_target": item["quality"],
            "response": item["response"],
        }
        for item in panel
    ]
    prompt = CALIBRATION_SINGLE_RUBRIC_SCORE_PROMPT.format(
        query=sample.query,
        rubric=compact(public_rubrics([rubric])[0]),
        responses=compact(prompt_responses),
    )

    def validate(data: dict[str, Any]) -> dict[str, int]:
        rows = require_exact_ids(data.get("response_scores", []), "response_id", response_ids, "response_scores")
        scores: dict[str, int] = {}
        for row in rows:
            value = row.get("score")
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                raise ValueError(f"{row.get('response_id')}/{rubric_id} score must be an integer in [1,5]")
            scores[str(row["response_id"])] = value
        return scores

    scores = _validated_call(judge, prompt, validate, "experiment4_calibration_single_rubric_score")
    return {"rubric_id": rubric_id, "response_scores": scores, "calls": 1}


def score_calibration_panel(
    sample: Sample,
    rubrics: list[dict[str, Any]],
    panel: list[dict[str, Any]],
    judge: JsonModel,
) -> dict[str, Any]:
    response_ids = sorted(str(item["response_id"]) for item in panel)
    rubric_ids = [str(item["rubric_id"]) for item in rubrics]
    matrix: dict[str, dict[str, int]] = {response_id: {} for response_id in response_ids}
    for rubric in rubrics:
        result = score_calibration_rubric(sample, rubric, panel, judge)
        rubric_id = result["rubric_id"]
        for response_id, score in result["response_scores"].items():
            matrix[response_id][rubric_id] = score
    return {
        "matrix": matrix,
        "response_ids": response_ids,
        "rubric_ids": sorted(rubric_ids),
        "calls": len(rubrics),
        "scoring_mode": "per_rubric_v1",
    }


def _pvariance(values: list[float]) -> float:
    return statistics.pvariance(values) if len(values) >= 2 else 0.0


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_scale = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_scale = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if left_scale <= 0 or right_scale <= 0:
        return None
    return numerator / (left_scale * right_scale)


def calibration_diagnostics(
    rubrics: list[dict[str, Any]],
    judge_matrices: dict[str, dict[str, dict[str, int]]],
    *,
    min_discrimination_variance: float,
    max_judge_instability_variance: float,
    redundancy_correlation: float,
) -> dict[str, Any]:
    if len(judge_matrices) < 2:
        raise ValueError("calibration diagnostics require at least two independent judges")
    judges = sorted(judge_matrices)
    response_ids = sorted(next(iter(judge_matrices.values())))
    rubric_ids = [str(item["rubric_id"]) for item in rubrics]
    for judge, matrix in judge_matrices.items():
        if set(matrix) != set(response_ids):
            raise ValueError(f"judge {judge} returned a different response set")
        if any(set(row) != set(rubric_ids) for row in matrix.values()):
            raise ValueError(f"judge {judge} returned a different rubric set")

    item_metrics: list[dict[str, Any]] = []
    mean_vectors: dict[str, list[float]] = {}
    low_discrimination_ids: list[str] = []
    unstable_ids: list[str] = []
    for rubric_id in rubric_ids:
        by_judge = {
            judge: [float(judge_matrices[judge][response_id][rubric_id]) for response_id in response_ids]
            for judge in judges
        }
        discrimination_by_judge = {judge: _pvariance(values) for judge, values in by_judge.items()}
        discrimination = statistics.fmean(discrimination_by_judge.values())
        instability_by_response = {
            response_id: _pvariance(
                [float(judge_matrices[judge][response_id][rubric_id]) for judge in judges]
            )
            for response_id in response_ids
        }
        instability = statistics.fmean(instability_by_response.values())
        vector = [
            statistics.fmean(float(judge_matrices[judge][response_id][rubric_id]) for judge in judges)
            for response_id in response_ids
        ]
        mean_vectors[rubric_id] = vector
        if discrimination < min_discrimination_variance:
            low_discrimination_ids.append(rubric_id)
        if instability > max_judge_instability_variance:
            unstable_ids.append(rubric_id)
        item_metrics.append(
            {
                "rubric_id": rubric_id,
                "discrimination_variance": discrimination,
                "discrimination_variance_by_judge": discrimination_by_judge,
                "judge_instability_variance": instability,
                "judge_instability_variance_by_response": instability_by_response,
                "mean_score": statistics.fmean(vector),
                "score_range": max(vector) - min(vector),
            }
        )

    correlations: list[dict[str, Any]] = []
    redundant_pairs: list[dict[str, Any]] = []
    for left, right in combinations(rubric_ids, 2):
        correlation = _pearson(mean_vectors[left], mean_vectors[right])
        row = {"left_id": left, "right_id": right, "pearson_correlation": correlation}
        correlations.append(row)
        if correlation is not None and correlation >= redundancy_correlation:
            redundant_pairs.append(row)

    return {
        "judge_names": judges,
        "response_ids": response_ids,
        "thresholds": {
            "min_discrimination_variance": min_discrimination_variance,
            "max_judge_instability_variance": max_judge_instability_variance,
            "redundancy_correlation": redundancy_correlation,
        },
        "item_metrics": item_metrics,
        "pair_correlations": correlations,
        "issues": {
            "low_discrimination_ids": low_discrimination_ids,
            "unstable_ids": unstable_ids,
            "redundant_pairs": redundant_pairs,
        },
        "issue_count": len(low_discrimination_ids) + len(unstable_ids) + len(redundant_pairs),
        "mean_discrimination_variance": statistics.fmean(
            item["discrimination_variance"] for item in item_metrics
        ) if item_metrics else None,
        "mean_judge_instability_variance": statistics.fmean(
            item["judge_instability_variance"] for item in item_metrics
        ) if item_metrics else None,
    }


def _revision_diagnostics_payload(
    judge_scores: dict[str, dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Keep all actionable calibration evidence without quadratic prompt bulk.

    The full audit file still stores every per-response variance and every
    pairwise correlation.  GLM revision only needs the score matrices,
    aggregate item metrics, issue lists, and correlations above the registered
    redundancy threshold.
    """
    compact_items = [
        {
            key: item.get(key)
            for key in (
                "rubric_id",
                "discrimination_variance",
                "judge_instability_variance",
                "mean_score",
                "score_range",
            )
        }
        for item in diagnostics.get("item_metrics", [])
        if isinstance(item, dict)
    ]
    return {
        "judge_score_matrices": {name: value["matrix"] for name, value in judge_scores.items()},
        "metrics": {
            "thresholds": diagnostics.get("thresholds", {}),
            "item_metrics": compact_items,
            "issues": diagnostics.get("issues", {}),
            "issue_count": diagnostics.get("issue_count", 0),
            "mean_discrimination_variance": diagnostics.get("mean_discrimination_variance"),
            "mean_judge_instability_variance": diagnostics.get("mean_judge_instability_variance"),
        },
    }


def revise_with_calibration(
    sample: Sample,
    rubrics: list[dict[str, Any]],
    panel: list[dict[str, Any]],
    judge_scores: dict[str, dict[str, Any]],
    diagnostics: dict[str, Any],
    generator: JsonModel,
    *,
    id_prefix: str,
    max_items: int,
) -> dict[str, Any]:
    gold = gold_rubrics(sample)
    gold_ids = {item["rubric_id"] for item in gold}
    response_char_limit = max(500, int(os.getenv("EXPERIMENT4_REVISION_RESPONSE_CHARS", "3000")))
    prompt_panel = [
        {
            "response_id": item["response_id"],
            "model": item["model"],
            "quality_target": item["quality"],
            "response": clip_text_middle(item["response"], response_char_limit),
        }
        for item in panel
    ]
    diagnostic_payload = _revision_diagnostics_payload(judge_scores, diagnostics)
    prompt = CALIBRATION_REWRITE_PROMPT.format(
        query=sample.query,
        gold=compact(public_rubrics(gold)),
        rubrics=compact(rubrics),
        responses=compact(prompt_panel),
        diagnostics=compact(diagnostic_payload),
        max_items=max_items,
    )
    repair_audit: dict[str, Any] = {}

    def repair(data: dict[str, Any], error: Exception) -> list[dict[str, Any]]:
        repaired, audit = _repair_generated_rubrics(
            data,
            gold=gold,
            fallback_rubrics=rubrics,
            id_prefix=id_prefix,
            max_items=max_items,
            validation_error=error,
        )
        repair_audit.update(audit)
        return repaired

    revised = _validated_call(
        generator,
        prompt,
        lambda data: _normalize_generated_rubrics(
            data, gold_ids=gold_ids, id_prefix=id_prefix, max_items=max_items
        ),
        "experiment4_calibration_rewrite",
        repair=repair,
    )
    result: dict[str, Any] = {
        "rubrics": revised,
        "generator_raw_calls": 1,
        "rewrite_prompt_chars": len(prompt),
        "revision_response_char_limit": response_char_limit,
        "revision_diagnostics_payload_version": 2,
    }
    if repair_audit:
        result["schema_repair"] = repair_audit
    return result


def final_quality_rating(
    sample: Sample,
    rubrics: list[dict[str, Any]],
    judge: JsonModel,
) -> dict[str, Any]:
    prompt = FINAL_QUALITY_PROMPT.format(
        query=sample.query,
        gold=compact(public_rubrics(gold_rubrics(sample))),
        candidate=compact(public_rubrics(rubrics)),
    )

    def validate(data: dict[str, Any]) -> dict[str, Any]:
        ratings = data.get("ratings")
        if not isinstance(ratings, dict):
            raise ValueError("ratings must be an object")
        expected = {"coverage", "discriminability", "redundancy", "novelty"}
        if set(ratings) != expected:
            raise ValueError(f"ratings must contain exactly {sorted(expected)}")
        normalized: dict[str, int] = {}
        for name in expected:
            value = ratings[name]
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                raise ValueError(f"rating {name} must be an integer in [1,5]")
            normalized[name] = value
        reasons = data.get("reasons")
        if not isinstance(reasons, dict):
            reasons = {}
        return {"ratings": normalized, "reasons": {name: str(reasons.get(name, "")) for name in expected}}

    result = _validated_call(judge, prompt, validate, "experiment4_final_quality")
    result["calls"] = 1
    return result


def score_response_with_rubric(
    sample: Sample,
    response: str,
    rubric: dict[str, Any],
    judge: JsonModel,
) -> dict[str, Any]:
    rubric_id = str(rubric["rubric_id"])
    prompt = SINGLE_RUBRIC_RESPONSE_SCORE_PROMPT.format(
        query=sample.query,
        response=response,
        rubric=compact(public_rubrics([rubric])[0]),
    )

    def validate(data: dict[str, Any]) -> int:
        value = data.get("score")
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"score for {rubric_id} must be an integer in [1,5]")
        return value

    score = _validated_call(judge, prompt, validate, "experiment4_preference_single_rubric_score")
    return {"rubric_id": rubric_id, "score": score, "calls": 1}


def score_response_with_rubrics(
    sample: Sample,
    response: str,
    rubrics: list[dict[str, Any]],
    judge: JsonModel,
) -> dict[str, Any]:
    scores = {
        result["rubric_id"]: result["score"]
        for result in (score_response_with_rubric(sample, response, rubric, judge) for rubric in rubrics)
    }
    return {
        "rubric_scores": scores,
        "uniform_mean_score": statistics.fmean(scores.values()),
        "calls": len(rubrics),
        "scoring_mode": "per_rubric_v1",
    }
