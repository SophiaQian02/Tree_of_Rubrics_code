#!/usr/bin/env python3
"""Semantic-feasibility-first calibration selection on a frozen 750-query pool."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import project.scripts.run_gold_rubric_experiment4 as shared
from project.mechanism_analysis.core import JsonModel, JsonlSink, ModelSpec, json_digest, read_jsonl
from project.mechanism_analysis.experiment4 import calibration_diagnostics, score_calibration_rubric
from project.scripts.run_rq2_discriminability_stability import (
    _bootstrap_ci,
    _paired_sign_permutation_p,
    _run_with_round_retries,
    latest_ok,
    load_samples,
)


POLICY_METHODS = ("semantic_only", "calibration_only", "semantic_first_calibration")
METHODS = ("direct",) + POLICY_METHODS
COLORS = {
    "direct": "#111827",
    "semantic_only": "#64748b",
    "calibration_only": "#dc2626",
    "semantic_first_calibration": "#2563eb",
}
LABELS = {
    "direct": "Direct",
    "semantic_only": "Semantic-only",
    "calibration_only": "Calibration-only",
    "semantic_first_calibration": "Semantic-first + Calibration",
}


def _base_rows(path: Path, sample_ids: set[str]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = latest_ok(path, ("sample_id", "method"))
    direct = {sid: row for (sid, method), row in rows.items() if sid in sample_ids and method == "direct"}
    semantic = {sid: row for (sid, method), row in rows.items() if sid in sample_ids and method == "semantic"}
    if len(direct) != len(sample_ids) or len(semantic) != len(sample_ids):
        raise RuntimeError(
            f"Frozen base file incomplete: direct={len(direct)}, semantic={len(semantic)}, expected={len(sample_ids)}"
        )
    return direct, semantic


def _hard_semantic_valid(candidate: dict[str, Any]) -> bool:
    acceptance = candidate.get("acceptance") or {}
    checks = acceptance.get("checks") or {}
    return bool(checks.get("no_dimension_worse") and checks.get("no_candidate_regressions"))


def _strict_semantic_improvement(candidate: dict[str, Any]) -> bool:
    return bool((candidate.get("acceptance") or {}).get("pass"))


def _direct_local_rows(direct: list[dict[str, Any]], node: dict[str, Any]) -> list[dict[str, Any]]:
    targets = {str(value) for value in node.get("target_direct_ids") or []}
    if not targets:
        return []
    return [
        dict(row)
        for row in direct
        if {str(value) for value in row.get("source_direct_ids") or []} & targets
    ]


def _candidate_entries(direct: list[dict[str, Any]], node: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = _direct_local_rows(direct, node)
    output = [
        {
            "candidate_index": 0,
            "local_rubrics": baseline,
            "local_rubric_sha256": json_digest(baseline),
            "acceptance": None,
            "hard_semantic_valid": True,
            "strict_semantic_improvement": False,
        }
    ]
    for candidate in node.get("candidates") or []:
        local = list(candidate.get("local_rubrics") or [])
        output.append(
            {
                **candidate,
                "local_rubrics": local,
                "local_rubric_sha256": str(candidate.get("local_rubric_sha256") or json_digest(local)),
                "hard_semantic_valid": _hard_semantic_valid(candidate),
                "strict_semantic_improvement": _strict_semantic_improvement(candidate),
            }
        )
    return output


def _old_score_indices(path: Path) -> tuple[dict[tuple[Any, ...], dict[str, Any]], dict[tuple[Any, ...], dict[str, Any]]]:
    local: dict[tuple[Any, ...], dict[str, Any]] = {}
    direct: dict[tuple[Any, ...], dict[str, Any]] = {}
    if not path.exists():
        return local, direct
    for row in read_jsonl(path):
        if row.get("status") != "ok" or not isinstance(row.get("response_scores"), dict):
            continue
        model = str(row.get("judge_model") or "")
        if row.get("stage") == "local_ambiguity_calibration_scoring":
            key = (
                str(row.get("sample_id")),
                str(row.get("local_node_id")),
                int(row.get("selection_candidate_index") or -1),
                model,
                str(row.get("rubric_id")),
                str(row.get("rubric_sha256")),
            )
            local[key] = row
        elif row.get("stage") == "calibration_single_rubric_scoring" and int(row.get("iteration") or 0) == 0:
            key = (str(row.get("sample_id")), model, str(row.get("rubric_id")))
            direct.setdefault(key, row)
    return local, direct


def cmd_score_calibration(args: argparse.Namespace) -> int:
    samples = shared.preference_samples(args)
    if args.require_samples and len(samples) != args.require_samples:
        raise RuntimeError(f"Expected {args.require_samples} samples, found {len(samples)}")
    sample_ids = {sample.sample_id for sample in samples}
    direct_rows, semantic_rows = _base_rows(args.base_rubrics, sample_ids)
    panels, _ = shared.cache_panels(samples, args.calibration_responses)
    judges = [ModelSpec.parse(value) for value in args.judge]
    if len(judges) != 2:
        raise ValueError("Exactly two calibration judges are required")
    old_local, old_direct = _old_score_indices(args.reuse_scores)
    sink = JsonlSink(
        args.output,
        ("sample_id", "local_node_id", "candidate_index", "judge", "rubric_id", "input_sha256"),
    )
    jobs = []
    reused = 0
    for sample in samples:
        direct = direct_rows[sample.sample_id]["rubrics"]
        panel = panels[sample.sample_id]
        panel_sha = json_digest(panel)
        for node in semantic_rows[sample.sample_id].get("semantic_local_nodes") or []:
            node_id = str(node["local_node_id"])
            for candidate in _candidate_entries(direct, node):
                candidate_index = int(candidate["candidate_index"])
                candidate_sha = str(candidate["local_rubric_sha256"])
                for judge in judges:
                    for rubric in candidate["local_rubrics"]:
                        rubric_id = str(rubric["rubric_id"])
                        input_hash = json_digest(
                            {
                                "sample_id": sample.sample_id,
                                "query": sample.query,
                                "local_node_id": node_id,
                                "candidate_index": candidate_index,
                                "rubric": rubric,
                                "panel_sha256": panel_sha,
                                "judge_model": judge.model,
                                "prompt": "CALIBRATION_SINGLE_RUBRIC_SCORE_PROMPT",
                            }
                        )
                        meta = {
                            "stage": "frozen_pool_calibration_scoring",
                            "sample_id": sample.sample_id,
                            "local_node_id": node_id,
                            "candidate_index": candidate_index,
                            "candidate_hard_semantic_valid": bool(candidate["hard_semantic_valid"]),
                            "candidate_strict_semantic_improvement": bool(candidate["strict_semantic_improvement"]),
                            "local_rubric_sha256": candidate_sha,
                            "judge": judge.name,
                            "judge_model": judge.model,
                            "judge_base_urls": list(judge.base_urls),
                            "rubric_id": rubric_id,
                            "input_sha256": input_hash,
                            "panel_sha256": panel_sha,
                        }
                        if sink.key(meta) in sink.done:
                            continue
                        source = None
                        if candidate_index == 0:
                            source = old_direct.get((sample.sample_id, judge.model, rubric_id))
                        else:
                            source = old_local.get(
                                (sample.sample_id, node_id, candidate_index, judge.model, rubric_id, candidate_sha)
                            )
                        if source is not None:
                            shared.append_success(
                                sink,
                                {
                                    **meta,
                                    "response_scores": source["response_scores"],
                                    "calls": source.get("calls", 1),
                                    "reused_from": str(args.reuse_scores),
                                    "reused_from_input_sha256": source.get("input_sha256"),
                                },
                            )
                            reused += 1
                            continue

                        def score(s=sample, r=rubric, p=panel, j=judge):
                            return score_calibration_rubric(s, r, p, JsonModel(j, 0.0, args.max_tokens))

                        jobs.append((meta, score))
    print(json.dumps({"reused": reused, "new_jobs": len(jobs)}), flush=True)
    counts = _run_with_round_retries(jobs, sink, args.concurrency, args.retry_rounds, "frozen_pool_calibration")
    remaining = sum(sink.key(meta) not in sink.done for meta, _ in jobs)
    print(json.dumps({"counts": dict(counts), "remaining": remaining}), flush=True)
    return 0 if remaining == 0 else 1


def _candidate_metrics(
    candidate: dict[str, Any],
    sample_id: str,
    node_id: str,
    judges: list[str],
    scores: dict[tuple[Any, ...], dict[str, Any]],
    *,
    redundancy_weight: float,
    size_weight: float,
) -> dict[str, Any]:
    rubrics = candidate["local_rubrics"]
    if not rubrics:
        return {
            "leaf_count": 0,
            "mean_discrimination_variance": 0.0,
            "mean_judge_instability_variance": 0.0,
            "empirical_redundancy_rate": 0.0,
            "noise_adjusted_discrimination": 0.0,
            "utility": 0.0,
        }
    response_ids: set[str] | None = None
    matrices: dict[str, dict[str, dict[str, int]]] = {}
    candidate_index = int(candidate["candidate_index"])
    for judge in judges:
        matrix: dict[str, dict[str, int]] = {}
        for rubric in rubrics:
            row = scores.get((sample_id, node_id, candidate_index, judge, str(rubric["rubric_id"])))
            if row is None:
                raise RuntimeError(
                    f"Missing calibration score: {sample_id}/{node_id}/C{candidate_index}/{judge}/{rubric['rubric_id']}"
                )
            values = {str(key): int(value) for key, value in row["response_scores"].items()}
            if response_ids is None:
                response_ids = set(values)
            if set(values) != response_ids:
                raise RuntimeError("Calibration response IDs differ across score rows")
            for response_id, value in values.items():
                matrix.setdefault(response_id, {})[str(rubric["rubric_id"])] = value
        matrices[judge] = matrix
    diagnostics = calibration_diagnostics(
        rubrics,
        matrices,
        min_discrimination_variance=0.35,
        max_judge_instability_variance=0.35,
        redundancy_correlation=0.90,
    )
    leaf_count = len(rubrics)
    pair_count = leaf_count * (leaf_count - 1) // 2
    issues = diagnostics["issues"]
    redundancy_rate = len(issues["redundant_pairs"]) / max(1, pair_count)
    discrimination = float(diagnostics["mean_discrimination_variance"] or 0.0)
    instability = float(diagnostics["mean_judge_instability_variance"] or 0.0)
    adjusted = discrimination - instability
    utility = adjusted - redundancy_weight * redundancy_rate - size_weight * max(0, leaf_count - 1)
    return {
        "leaf_count": leaf_count,
        "mean_discrimination_variance": discrimination,
        "mean_judge_instability_variance": instability,
        "empirical_redundancy_rate": redundancy_rate,
        "noise_adjusted_discrimination": adjusted,
        "utility": utility,
    }


def _compose(
    direct: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    choices: dict[str, int],
    *,
    max_items: int,
    prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    current = [dict(row) for row in direct]
    applied: dict[str, int] = {}
    rejected: dict[str, str] = {}
    for node in nodes:
        node_id = str(node["local_node_id"])
        choice = int(choices.get(node_id, 0))
        if choice == 0:
            continue
        candidate = next((x for x in node.get("candidates") or [] if int(x["candidate_index"]) == choice), None)
        if candidate is None:
            rejected[node_id] = "candidate_missing"
            continue
        patch = [dict(row) for row in candidate.get("local_rubrics") or []]
        targets = {str(value) for value in node.get("target_direct_ids") or []}
        proposed: list[dict[str, Any]] = []
        insertion: int | None = None
        for row in current:
            lineage = {str(value) for value in row.get("source_direct_ids") or []}
            if targets and lineage & targets:
                insertion = len(proposed) if insertion is None else insertion
                continue
            proposed.append(row)
        if insertion is None:
            proposed.extend(patch)
        else:
            proposed[insertion:insertion] = patch
        if len(proposed) > max_items:
            rejected[node_id] = "max_items_exceeded"
            continue
        direct_ids = {str(row["rubric_id"]) for row in direct}
        covered = {str(value) for row in proposed for value in row.get("source_direct_ids") or []}
        if direct_ids - covered:
            rejected[node_id] = "direct_lineage_lost"
            continue
        current = proposed
        applied[node_id] = choice
    output = []
    for index, row in enumerate(current, 1):
        normalized = dict(row)
        normalized["rubric_id"] = f"{prefix}{index:02d}"
        output.append(normalized)
    return output, {"applied": applied, "rejected": rejected}


def _best_by_utility(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        candidates,
        key=lambda item: (
            float(item["metrics"]["utility"]),
            -float(item["metrics"]["mean_judge_instability_variance"]),
            float(item["metrics"]["mean_discrimination_variance"]),
            -int(item["candidate_index"]),
        ),
    )


def cmd_select(args: argparse.Namespace) -> int:
    samples = shared.preference_samples(args)
    if args.require_samples and len(samples) != args.require_samples:
        raise RuntimeError(f"Expected {args.require_samples} samples, found {len(samples)}")
    sample_ids = {sample.sample_id for sample in samples}
    direct_rows, semantic_rows = _base_rows(args.base_rubrics, sample_ids)
    score_rows = latest_ok(
        args.scores,
        ("sample_id", "local_node_id", "candidate_index", "judge", "rubric_id"),
    )
    judges = sorted({str(key[3]) for key in score_rows})
    if len(judges) != 2:
        raise RuntimeError(f"Expected two judges, found {judges}")
    sink = JsonlSink(args.output, ("sample_id", "method", "selection_config_sha256"))
    config = {
        "version": "semantic_hard_feasible_then_calibration_v1",
        "utility_margin": args.utility_margin,
        "redundancy_weight": args.redundancy_weight,
        "size_weight": args.size_weight,
        "max_items": args.max_items,
        "candidate_pool": str(args.base_rubrics),
        "calibration_scores": str(args.scores),
    }
    config_sha = json_digest(config)
    for sample in samples:
        sid = sample.sample_id
        direct = direct_rows[sid]["rubrics"]
        semantic = semantic_rows[sid]
        nodes = list(semantic.get("semantic_local_nodes") or [])
        policy_choices: dict[str, dict[str, int]] = {method: {} for method in POLICY_METHODS}
        node_audits: list[dict[str, Any]] = []
        for node in nodes:
            node_id = str(node["local_node_id"])
            candidates = _candidate_entries(direct, node)
            scored = []
            for candidate in candidates:
                metrics = _candidate_metrics(
                    candidate,
                    sid,
                    node_id,
                    judges,
                    score_rows,
                    redundancy_weight=args.redundancy_weight,
                    size_weight=args.size_weight,
                )
                scored.append({**candidate, "metrics": metrics})
            by_index = {int(item["candidate_index"]): item for item in scored}
            baseline = by_index[0]

            semantic_choice = node.get("semantic_selected_candidate_index")
            semantic_choice = int(semantic_choice) if semantic_choice is not None else 0
            policy_choices["semantic_only"][node_id] = semantic_choice

            calibration_top = _best_by_utility(scored)
            calibration_choice = (
                int(calibration_top["candidate_index"])
                if float(calibration_top["metrics"]["utility"]) >= float(baseline["metrics"]["utility"]) + args.utility_margin
                else 0
            )
            policy_choices["calibration_only"][node_id] = calibration_choice

            strict = [item for item in scored if item["strict_semantic_improvement"]]
            hard = [item for item in scored if item["hard_semantic_valid"] and int(item["candidate_index"]) != 0]
            if len(strict) == 1:
                sf_choice = int(strict[0]["candidate_index"])
                resolution = "unique_strict_semantic_winner"
            else:
                default_index = semantic_choice if strict else 0
                default = by_index[default_index]
                pool = strict if len(strict) > 1 else hard
                top = _best_by_utility(pool) if pool else default
                if pool and float(top["metrics"]["utility"]) >= float(default["metrics"]["utility"]) + args.utility_margin:
                    sf_choice = int(top["candidate_index"])
                    resolution = "calibration_resolved_semantic_ambiguity"
                else:
                    sf_choice = default_index
                    resolution = "calibration_indistinguishable_kept_semantic_default"
            policy_choices["semantic_first_calibration"][node_id] = sf_choice

            node_audits.append(
                {
                    "local_node_id": node_id,
                    "semantic_decision_old": node.get("semantic_decision"),
                    "hard_candidate_count": len(hard),
                    "strict_candidate_count": len(strict),
                    "semantic_only_choice": semantic_choice,
                    "calibration_only_choice": calibration_choice,
                    "semantic_first_calibration_choice": sf_choice,
                    "semantic_first_resolution": resolution,
                    "calibration_only_selected_semantically_invalid": bool(
                        calibration_choice != 0 and not by_index[calibration_choice]["hard_semantic_valid"]
                    ),
                    "candidates": [
                        {
                            "candidate_index": int(item["candidate_index"]),
                            "hard_semantic_valid": bool(item["hard_semantic_valid"]),
                            "strict_semantic_improvement": bool(item["strict_semantic_improvement"]),
                            "metrics": item["metrics"],
                        }
                        for item in scored
                    ],
                }
            )

        for method in METHODS:
            if method == "direct":
                rubrics = direct
                composition = {"applied": {}, "rejected": {}}
                local_choices = {str(node["local_node_id"]): 0 for node in nodes}
            elif method == "semantic_only":
                rubrics = semantic["rubrics"]
                composition = semantic.get("semantic_composition") or {}
                local_choices = policy_choices[method]
            else:
                rubrics, composition = _compose(
                    direct,
                    nodes,
                    policy_choices[method],
                    max_items=args.max_items,
                    prefix=("CO_" if method == "calibration_only" else "SFC_"),
                )
                local_choices = policy_choices[method]
            selected_invalid = sum(
                audit["calibration_only_selected_semantically_invalid"]
                for audit in node_audits
            ) if method == "calibration_only" else 0
            shared.append_success(
                sink,
                {
                    "stage": "frozen_pool_policy_selection",
                    "sample_id": sid,
                    "method": method,
                    "selection_config_sha256": config_sha,
                    "selection_config": config,
                    "source_direct_rubric_sha256": direct_rows[sid]["rubric_sha256"],
                    "source_semantic_rubric_sha256": semantic["rubric_sha256"],
                    "rubrics": rubrics,
                    "rubric_count": len(rubrics),
                    "rubric_sha256": json_digest(rubrics),
                    "local_choices": local_choices,
                    "composition": composition,
                    "node_audits": node_audits,
                    "selected_semantically_invalid_node_count": selected_invalid,
                    "gold_visible_during_selection": False,
                    "preference_pair_visible_during_selection": False,
                },
            )
    rows = latest_ok(args.output, ("sample_id", "method", "selection_config_sha256"))
    complete = sum((sample.sample_id, method, config_sha) in rows for sample in samples for method in METHODS)
    expected = len(samples) * len(METHODS)
    print(json.dumps({"complete": complete, "expected": expected, "config_sha256": config_sha}))
    return 0 if complete == expected else 1


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _hist(values: list[float], bins: int, low: float, high: float) -> list[float]:
    counts = [0] * bins
    for value in values:
        index = int((value - low) / max(1e-9, high - low) * bins)
        counts[max(0, min(bins - 1, index))] += 1
    return [count / max(1, len(values)) for count in counts]


def _svg_triptych(path: Path, panels: list[dict[str, Any]]) -> None:
    width, height = 1500, 510
    margin, gap, top, bottom = 58, 38, 65, 70
    panel_w = (width - 2 * margin - 2 * gap) / 3
    plot_h = height - top - bottom
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>']
    for p_index, panel in enumerate(panels):
        x0 = margin + p_index * (panel_w + gap)
        low, high = panel["domain"]
        bins = 22
        densities = {method: _hist(panel["values"].get(method, []), bins, low, high) for method in METHODS}
        ymax = max([value for row in densities.values() for value in row] or [1.0]) * 1.12
        lines.append(f'<text x="{x0+panel_w/2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="17">{html.escape(panel["title"])}</text>')
        for tick in range(5):
            y = top + plot_h * tick / 4
            lines.append(f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0+panel_w:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        bar_w = panel_w / bins
        for method in METHODS:
            for index, value in enumerate(densities[method]):
                h = value / max(1e-9, ymax) * plot_h
                x = x0 + index * bar_w
                fill_opacity = "0.04" if method == "direct" else "0.25"
                stroke_opacity = "0.95" if method == "direct" else "0.55"
                lines.append(f'<rect x="{x:.2f}" y="{top+plot_h-h:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{COLORS[method]}" fill-opacity="{fill_opacity}" stroke="{COLORS[method]}" stroke-opacity="{stroke_opacity}"/>')
        lines.append(f'<line x1="{x0:.1f}" y1="{top+plot_h}" x2="{x0+panel_w:.1f}" y2="{top+plot_h}" stroke="#111827"/>')
        for tick in range(5):
            x = x0 + panel_w * tick / 4; value = low + (high-low)*tick/4
            lines.append(f'<text x="{x:.1f}" y="{top+plot_h+20}" text-anchor="middle" font-family="Arial" font-size="11">{value:.2f}</text>')
        lines.append(f'<text x="{x0+panel_w/2:.1f}" y="{height-25}" text-anchor="middle" font-family="Arial" font-size="12">{html.escape(panel["xlabel"])}</text>')
    for index, method in enumerate(METHODS):
        x = width/2 - 330 + index*245
        lines.append(f'<rect x="{x:.1f}" y="43" width="15" height="10" fill="{COLORS[method]}" fill-opacity="0.55"/>')
        lines.append(f'<text x="{x+21:.1f}" y="52" font-family="Arial" font-size="12">{html.escape(LABELS[method])}</text>')
    lines.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text('\n'.join(lines), encoding='utf-8')


def _svg_gate(path: Path, invalid_rates: dict[str, float], resolutions: Counter) -> None:
    width, height = 980, 500
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="490" y="32" text-anchor="middle" font-family="Arial" font-size="20">Why the Semantic Gate Is Necessary (N=750)</text>']
    methods = list(POLICY_METHODS); max_rate = max([invalid_rates.get(m, 0.0) for m in methods] + [0.01])
    for index, method in enumerate(methods):
        x = 90 + index*175; h = invalid_rates.get(method,0.0)/max_rate*260
        lines.append(f'<rect x="{x}" y="{360-h:.1f}" width="90" height="{h:.1f}" fill="{COLORS[method]}"/>')
        lines.append(f'<text x="{x+45}" y="{350-h:.1f}" text-anchor="middle" font-family="Arial" font-size="14">{invalid_rates.get(method,0.0):.1%}</text>')
        lines.append(f'<text x="{x+45}" y="385" text-anchor="middle" font-family="Arial" font-size="11">{html.escape(LABELS[method])}</text>')
    total = sum(resolutions.values()) or 1
    lines.append('<text x="750" y="80" text-anchor="middle" font-family="Arial" font-size="15">Semantic-first node decisions</text>')
    y=115
    for label,count in resolutions.most_common():
        lines.append(f'<text x="590" y="{y}" font-family="Arial" font-size="12">{html.escape(label)}</text>')
        lines.append(f'<rect x="590" y="{y+8}" width="{280*count/total:.1f}" height="15" fill="#2563eb" fill-opacity="0.65"/>')
        lines.append(f'<text x="880" y="{y+20}" text-anchor="end" font-family="Arial" font-size="11">{count} ({count/total:.1%})</text>')
        y += 55
    lines.append('</svg>'); path.parent.mkdir(parents=True, exist_ok=True); path.write_text('\n'.join(lines),encoding='utf-8')


def _paired_summary(base: list[float], treatment: list[float]) -> dict[str, Any]:
    delta = [b-a for a,b in zip(base,treatment)]
    return {"n":len(delta),"baseline_mean":statistics.fmean(base),"treatment_mean":statistics.fmean(treatment),"paired_delta_mean":statistics.fmean(delta),"paired_delta_bootstrap_95ci":_bootstrap_ci(delta),"paired_sign_permutation_p":_paired_sign_permutation_p(delta)}


def cmd_summarize(args: argparse.Namespace) -> int:
    samples = load_samples(args.dataset, args.limit)
    if args.require_samples and len(samples) != args.require_samples:
        raise RuntimeError(f"Expected {args.require_samples}, found {len(samples)}")
    ids = {sample.sample_id for sample in samples}
    score_rows = latest_ok(args.scores, ("sample_id","method","judge","candidate"))
    judges = sorted({str(key[2]) for key in score_rows if key[0] in ids and key[1] in METHODS})
    if len(judges)!=2: raise RuntimeError(f"Need two judges, found {judges}")
    records=[]; values={"discriminability":defaultdict(list),"instability":defaultdict(list),"quality":defaultdict(list)}
    for sample in samples:
        for method in METHODS:
            margins=[]; absolute=[]; rows_by_judge={}
            for judge in judges:
                a=score_rows.get((sample.sample_id,method,judge,"A")); b=score_rows.get((sample.sample_id,method,judge,"B"))
                if not a or not b: raise RuntimeError(f"Missing held-out score {sample.sample_id}/{method}/{judge}")
                rows_by_judge[judge]={"A":a,"B":b}
                margin=float(a["uniform_mean_score"])-float(b["uniform_mean_score"]); margins.append(margin); absolute.append(abs(margin))
            response_variances=[]
            aggregate_response_variances=[]
            for candidate in ("A","B"):
                rubric_id_sets=[set((rows_by_judge[judge][candidate].get("rubric_scores") or {}).keys()) for judge in judges]
                if not rubric_id_sets[0] or any(ids != rubric_id_sets[0] for ids in rubric_id_sets[1:]):
                    raise RuntimeError(
                        f"Cross-judge rubric IDs differ for {sample.sample_id}/{method}/{candidate}"
                    )
                per_rubric_variance=[]
                for rubric_id in sorted(rubric_id_sets[0]):
                    per_rubric_variance.append(
                        statistics.pvariance(
                            float(rows_by_judge[judge][candidate]["rubric_scores"][rubric_id])
                            for judge in judges
                        )
                    )
                response_variances.append(statistics.fmean(per_rubric_variance))
                aggregate_response_variances.append(
                    statistics.pvariance(
                        float(rows_by_judge[judge][candidate]["uniform_mean_score"])
                        for judge in judges
                    )
                )
            d=statistics.fmean(absolute)
            instability=statistics.fmean(response_variances)
            margin_disagreement=abs(margins[0]-margins[1])
            values["discriminability"][method].append(d); values["instability"][method].append(instability)
            records.append({
                "sample_id":sample.sample_id,
                "method":method,
                "mean_absolute_ab_gap":d,
                "mean_same_response_cross_judge_rubric_variance":instability,
                "mean_same_response_cross_judge_aggregate_variance":statistics.fmean(aggregate_response_variances),
                "cross_judge_margin_disagreement_secondary":margin_disagreement,
            })
    quality_rows=latest_ok(args.quality,("sample_id","method","judge")) if args.quality.exists() else {}
    for (sid,method,_judge),row in quality_rows.items():
        if sid in ids and method in METHODS: values["quality"][method].append(float(row["overall_quality"]))
    selected=latest_ok(args.rubrics,("sample_id","method")); invalid_rates={}; resolutions=Counter()
    for method in POLICY_METHODS:
        rows=[row for (sid,m),row in selected.items() if sid in ids and m==method]
        node_count=sum(len(row.get("node_audits") or []) for row in rows)
        invalid=sum(int(row.get("selected_semantically_invalid_node_count") or 0) for row in rows)
        invalid_rates[method]=invalid/max(1,node_count)
        if method=="semantic_first_calibration":
            for row in rows:
                resolutions.update(a.get("semantic_first_resolution") for a in row.get("node_audits") or [])
    summary={
        "sample_size":len(samples),
        "query_split":"none; all 750 queries",
        "response_role_separation":"six calibration responses select candidates; A/B preference responses evaluate only",
        "judges":judges,
        "metric_definitions":{
            "discriminability":"For each judge, abs(score(A)-score(B)); then mean across judges. Higher is better.",
            "instability":"For each held-out response and rubric leaf, population variance of scores across judges; then mean across leaves and A/B. Lower is better.",
            "cross_judge_margin_disagreement":"Retained only as a secondary per-query diagnostic; not used as the stability metric.",
        },
        "metrics":{},
        "semantic_invalid_selection_rate":invalid_rates,
        "semantic_first_resolutions":dict(resolutions),
    }
    for metric in ("discriminability","instability"):
        summary["metrics"][metric]={"direct_vs_semantic_only":_paired_summary(values[metric]["direct"],values[metric]["semantic_only"]),"direct_vs_calibration_only":_paired_summary(values[metric]["direct"],values[metric]["calibration_only"]),"direct_vs_semantic_first_calibration":_paired_summary(values[metric]["direct"],values[metric]["semantic_first_calibration"]),"semantic_only_vs_calibration_only":_paired_summary(values[metric]["semantic_only"],values[metric]["calibration_only"]),"semantic_only_vs_semantic_first_calibration":_paired_summary(values[metric]["semantic_only"],values[metric]["semantic_first_calibration"])}
    if all(values["quality"][m] for m in METHODS):
        summary["metrics"]["quality"]={"means":{m:statistics.fmean(values["quality"][m]) for m in METHODS}}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    _write_csv(args.output_dir/"per_query_metrics.csv",records)
    (args.output_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    panels=[{"title":"(a) Held-out Discriminability ↑","xlabel":"Mean |score(A)-score(B)|","domain":(0.0,4.0),"values":values["discriminability"]},{"title":"(b) Same-Response Cross-Judge Variance ↓","xlabel":"Mean Var_judge(score(response, rubric))","domain":(0.0,4.0),"values":values["instability"]},{"title":"(c) Blinded Overall Quality ↑","xlabel":"DeepSeek overall quality (0-100)","domain":(0.0,100.0),"values":values["quality"]}]
    _svg_triptych(args.output_dir/"semantic_calibration_triptych.svg",panels)
    _svg_gate(args.output_dir/"semantic_gate_diagnostics.svg",invalid_rates,resolutions)
    print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0


def add_data(parser: argparse.ArgumentParser) -> None:
    shared.add_data_args(parser); parser.add_argument("--require-samples",type=int,default=750)


def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    score=sub.add_parser("score-calibration"); add_data(score)
    score.add_argument("--base-rubrics",type=Path,required=True); score.add_argument("--calibration-responses",type=Path,required=True); score.add_argument("--reuse-scores",type=Path,required=True); score.add_argument("--judge",action="append",required=True); score.add_argument("--max-tokens",type=int,default=4096); score.add_argument("--concurrency",type=int,default=32); score.add_argument("--retry-rounds",type=int,default=4); score.add_argument("--output",type=Path,required=True); score.set_defaults(func=cmd_score_calibration)
    select=sub.add_parser("select"); add_data(select)
    select.add_argument("--base-rubrics",type=Path,required=True); select.add_argument("--scores",type=Path,required=True); select.add_argument("--utility-margin",type=float,default=0.05); select.add_argument("--redundancy-weight",type=float,default=0.10); select.add_argument("--size-weight",type=float,default=0.02); select.add_argument("--max-items",type=int,default=24); select.add_argument("--output",type=Path,required=True); select.set_defaults(func=cmd_select)
    summarize=sub.add_parser("summarize"); summarize.add_argument("--dataset",type=Path,default=REPO_ROOT/"data/gold_rubric_1500/processed/gold_rubric_seed_1500.jsonl"); summarize.add_argument("--limit",type=int,default=0); summarize.add_argument("--require-samples",type=int,default=750); summarize.add_argument("--rubrics",type=Path,required=True); summarize.add_argument("--scores",type=Path,required=True); summarize.add_argument("--quality",type=Path,required=True); summarize.add_argument("--output-dir",type=Path,required=True); summarize.set_defaults(func=cmd_summarize)
    return parser


def main() -> int:
    args=build_parser().parse_args(); return int(args.func(args))


if __name__ == "__main__": raise SystemExit(main())
