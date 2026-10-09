from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from project.utils.json_utils import loads_json_with_escape_repair
from project.utils.openai_client import chat_completion_with_recovery, create_openai_client


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def json_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_json(text: str) -> dict[str, Any]:
    value = (text or "").strip()
    if value.startswith("```"):
        value = value[value.find("\n") + 1 :]
        if value.rstrip().endswith("```"):
            value = value.rstrip()[:-3]
    value = value.strip()
    try:
        parsed = loads_json_with_escape_repair(value)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = value.find("{")
    if start < 0:
        raise ValueError(f"Model output contains no JSON object: {value[:300]!r}")
    depth = 0
    in_string = False
    escaped = False
    end = -1
    for index in range(start, len(value)):
        char = value[index]
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index
                break
    if end < start:
        recovered_scores = _recover_score_rows(value[start:])
        if recovered_scores is not None:
            return recovered_scores
        # Some OpenAI-compatible endpoints occasionally stop immediately after
        # the useful JSON content but before the final quote/braces.  Repair
        # only a structurally traceable object; downstream schema validators
        # still decide whether the recovered payload is usable.
        repaired = _close_truncated_json(value[start:])
        if repaired is not None:
            parsed = loads_json_with_escape_repair(repaired)
            if isinstance(parsed, dict):
                return parsed
        raise ValueError(f"Model output contains no complete JSON object: {value[:300]!r}")
    candidate = value[start : end + 1]
    try:
        parsed = loads_json_with_escape_repair(candidate)
    except json.JSONDecodeError:
        recovered = _recover_leaf_passes(candidate)
        if recovered is not None:
            return recovered
        recovered_scores = _recover_score_rows(candidate)
        if recovered_scores is not None:
            return recovered_scores
        raise
    if not isinstance(parsed, dict):
        raise ValueError("Model output JSON is not an object")
    return parsed


def _recover_leaf_passes(value: str) -> dict[str, Any] | None:
    """Recover only an unambiguous leaf boolean when evidence JSON is malformed."""
    matches = re.findall(r'"passes"\s*:\s*(true|false)\b', value, flags=re.IGNORECASE)
    if len(matches) != 1:
        return None
    # Restrict the fallback to the leaf schema. Other structured responses must
    # still pass normal JSON parsing and their full schema validators.
    forbidden = ('"leaf_results"', '"weights"', '"candidate_children"', '"branch_results"')
    if any(token in value for token in forbidden):
        return None
    return {
        "passes": matches[0].lower() == "true",
        "evidence": "[evidence omitted: malformed JSON; unambiguous passes boolean recovered]",
    }


def _recover_score_rows(value: str) -> dict[str, Any] | None:
    """Recover unambiguous rubric IDs/scores when only evidence JSON is broken.

    The experiment consumes the integer scores; evidence is an audit field.
    Recovery remains conservative because the downstream schema validator must
    still observe every expected rubric ID exactly once.
    """
    if not re.search(r'"scores"\s*:', value):
        return None
    matches = re.findall(
        r'"rubric_id"\s*:\s*"([^"\r\n]+)"\s*,\s*"score"\s*:\s*"?([1-5])"?',
        value,
    )
    if not matches:
        return None
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_id, raw_score in matches:
        rubric_id = raw_id.strip()
        if not rubric_id or rubric_id in seen:
            return None
        seen.add(rubric_id)
        rows.append(
            {
                "rubric_id": rubric_id,
                "score": int(raw_score),
                "evidence": "[evidence omitted: malformed JSON; ID and integer score recovered]",
            }
        )
    return {"scores": rows}


def _close_truncated_json(value: str) -> str | None:
    """Conservatively close an otherwise well-formed truncated JSON object."""
    source = value.strip()
    if not source.startswith("{"):
        return None
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in source:
        if escaped:
            escaped = False
            continue
        if in_string and char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            # Raw newlines/control characters require a model retry rather than
            # speculative rewriting.
            if ord(char) < 0x20:
                return None
            continue
        if char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                return None
            expected = "{" if char == "}" else "["
            if stack.pop() != expected:
                return None
    if not stack:
        return None
    repaired = source
    if escaped:
        # A terminal backslash cannot safely identify the intended evidence.
        return None
    if in_string:
        repaired += '"'
    repaired += "".join("}" if opener == "{" else "]" for opener in reversed(stack))
    return repaired


def _completion_text(completion: Any) -> str:
    """Extract content from SDK objects and malformed string/dict gateways."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        choices = completion.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict):
                    return str(message.get("content") or "")
        for key in ("content", "text", "response"):
            if isinstance(completion.get(key), str):
                return str(completion[key])
        raise ValueError("Completion dictionary contains no textual model response")
    choices = getattr(completion, "choices", None)
    if not choices:
        raise ValueError(f"Completion object contains no choices: {type(completion).__name__}")
    message = getattr(choices[0], "message", None)
    return str(getattr(message, "content", "") or "")


def prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n\n".join(
            f"{str(item.get('role', 'user')).upper()}: {item.get('content', '')}"
            for item in prompt
            if isinstance(item, dict)
        )
    return json.dumps(prompt, ensure_ascii=False)


def iter_tree_leaves(tree: dict[str, Any]) -> Iterable[tuple[dict[str, Any], list[dict[str, str]]]]:
    def walk(node: dict[str, Any], path: list[dict[str, str]]) -> Iterable[tuple[dict[str, Any], list[dict[str, str]]]]:
        for leaf in node.get("leaves", []):
            yield leaf, path
        for child in node.get("children", []):
            item = {
                "node_id": child["node_id"],
                "node_type": child.get("node_type", "subdimension"),
                "name": child.get("name", child["node_id"]),
            }
            yield from walk(child, path + [item])

    for dimension in tree.get("dimensions", []):
        root = {
            "node_id": dimension["dimension_id"],
            "node_type": "primary_dimension",
            "name": dimension.get("name", dimension["dimension_id"]),
        }
        yield from walk(dimension, [root])


@dataclass(frozen=True)
class Sample:
    sample_id: str
    query: str
    seed: dict[str, Any]
    gold_tree: dict[str, Any]
    gold_flat: list[dict[str, Any]]
    counterfactuals: dict[str, Any]


class GoldDataset:
    def __init__(self, seed_path: Path, tree_path: Path, flat_path: Path, counterfactual_path: Path):
        seeds = {row["sample_id"]: row for row in read_jsonl(seed_path)}
        trees = {row["sample_id"]: row for row in read_jsonl(tree_path)}
        flats = {row["sample_id"]: row for row in read_jsonl(flat_path)}
        cfs = {row["sample_id"]: row for row in read_jsonl(counterfactual_path)}
        common = set(trees) & set(flats) & set(cfs) & set(seeds)
        if not common:
            raise ValueError("No sample_id is shared by seed, Tree, Flat, and counterfactual exports")
        self.samples: list[Sample] = []
        for sample_id in sorted(common):
            source_hash = json_digest(seeds[sample_id])
            if trees[sample_id].get("source_record_sha256") != source_hash:
                raise ValueError(f"{sample_id}: Tree export is stale relative to the seed row")
            if flats[sample_id].get("source_record_sha256") != source_hash:
                raise ValueError(f"{sample_id}: Flat export is stale relative to the seed row")
            tree = trees[sample_id]["gold_tree"]
            flat = flats[sample_id]["flat_rubric"]
            tree_ids = [leaf["leaf_id"] for leaf, _ in iter_tree_leaves(tree)]
            flat_ids = [leaf["leaf_id"] for leaf in flat]
            if len(tree_ids) != len(set(tree_ids)) or len(flat_ids) != len(set(flat_ids)):
                raise ValueError(f"{sample_id}: duplicate Gold leaf ID")
            if set(tree_ids) != set(flat_ids):
                raise ValueError(f"{sample_id}: Tree/Flat Gold leaves differ")
            seed = seeds[sample_id]
            self.samples.append(
                Sample(
                    sample_id=sample_id,
                    query=prompt_text(seed["prompt"]),
                    seed=seed,
                    gold_tree=tree,
                    gold_flat=flat,
                    counterfactuals=cfs[sample_id]["counterfactuals"],
                )
            )

    def subset(self, limit: int = 0) -> list[Sample]:
        return self.samples[:limit] if limit > 0 else self.samples


@dataclass(frozen=True)
class ModelSpec:
    name: str
    base_urls: tuple[str, ...]
    model: str
    api_key: str

    @property
    def base_url(self) -> str:
        """First endpoint, retained for single-endpoint callers and display."""
        return self.base_urls[0]

    @classmethod
    def parse(cls, value: str) -> "ModelSpec":
        parts = value.split("|")
        if len(parts) not in {3, 4}:
            raise ValueError("Model spec must be name|base_url|model[|API_KEY_ENV]")
        name, base_url_field, model = (part.strip() for part in parts[:3])
        base_urls = tuple(url.strip().rstrip("/") for url in base_url_field.split(",") if url.strip())
        key = os.getenv(parts[3].strip(), "") if len(parts) == 4 else "EMPTY"
        if not name or not base_urls or not model:
            raise ValueError(f"Invalid model spec: {value!r}")
        return cls(name=name, base_urls=base_urls, model=model, api_key=key or "EMPTY")


def no_thinking_body(model: str) -> dict[str, Any]:
    name = model.lower()
    if "glm" in name:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if "qwen" in name:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


_endpoint_pool_lock = threading.Lock()
_endpoint_pool_counters: dict[tuple[str, str, tuple[str, ...]], int] = {}


def _endpoint_order(spec: ModelSpec) -> tuple[str, ...]:
    pool_key = (spec.name, spec.model, spec.base_urls)
    with _endpoint_pool_lock:
        index = _endpoint_pool_counters.get(pool_key, 0)
        _endpoint_pool_counters[pool_key] = index + 1
    start = index % len(spec.base_urls)
    return spec.base_urls[start:] + spec.base_urls[:start]


class JsonModel:
    def __init__(self, spec: ModelSpec, temperature: float = 0.0, max_tokens: int = 8192):
        self.spec = spec
        self.temperature = temperature
        self.max_tokens = max_tokens

    def call(self, system: str, user: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        extra = no_thinking_body(self.spec.model)
        if extra:
            kwargs["extra_body"] = extra
        last_error: Exception | None = None
        format_retries = max(0, int(os.getenv("MECHANISM_JSON_FORMAT_RETRIES", "2")))
        current_user = user
        for format_attempt in range(format_retries + 1):
            endpoints = _endpoint_order(self.spec)
            endpoint_retries = int(os.getenv("MODEL_POOL_ENDPOINT_RETRIES", "2")) if len(endpoints) > 1 else None
            format_error: Exception | None = None
            for base_url in endpoints:
                try:
                    client = create_openai_client(base_url, self.spec.api_key)
                    completion = chat_completion_with_recovery(
                        client,
                        model=self.spec.model,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": current_user}],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        request_retries=endpoint_retries,
                        **kwargs,
                    )
                except Exception as exc:
                    last_error = exc
                    continue
                try:
                    return extract_json(_completion_text(completion))
                except (json.JSONDecodeError, ValueError) as exc:
                    format_error = exc
                    last_error = exc
            if format_error is None or format_attempt >= format_retries:
                break
            print(
                f"[MECHANISM JSON RETRY] attempt={format_attempt + 1} error={type(format_error).__name__}: {str(format_error)[:300]}",
                file=sys.stderr,
                flush=True,
            )
            current_user = (
                user
                + "\n\nYOUR PREVIOUS RESPONSE WAS NOT VALID JSON. "
                + f"Parser error: {type(format_error).__name__}: {str(format_error)[:500]}\n"
                + "Return exactly one complete JSON object matching the requested schema. "
                + "Do not use Markdown fences or explanatory text. Escape backslashes and quotes correctly."
            )
        assert last_error is not None
        raise last_error


class JsonlSink:
    """Append-only, thread-safe result sink with resume keys."""

    def __init__(self, path: Path, key_fields: tuple[str, ...]):
        self.path = path
        self.key_fields = key_fields
        self.lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.done: set[tuple[Any, ...]] = set()
        if path.exists():
            for row in read_jsonl(path):
                if row.get("status") == "ok":
                    self.done.add(tuple(row.get(field) for field in key_fields))

    def key(self, row: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(row.get(field) for field in self.key_fields)

    def append(self, row: dict[str, Any]) -> None:
        with self.lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            if row.get("status") == "ok":
                self.done.add(self.key(row))


def normalize_ratios(ids: list[str], values: dict[str, Any]) -> dict[str, float]:
    raw = {item: max(0.0, float(values.get(item, 0.0))) for item in ids}
    total = sum(raw.values())
    if total <= 0:
        return {item: 1.0 / len(ids) for item in ids}
    return {item: value / total for item, value in raw.items()}


def require_exact_ids(rows: Any, field: str, expected: set[str], label: str) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} is not a list of objects")
    ids = [str(row.get(field, "")) for row in rows]
    if len(ids) != len(expected) or len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError(f"{label} must return every expected ID exactly once")
    return rows


def binary_metrics(predicted: set[str], gold: set[str], universe: set[str]) -> dict[str, float | int]:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    tn = len(universe - predicted - gold)
    precision = tp / (tp + fp) if tp + fp else float(gold == predicted)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "exact_match": float(predicted == gold),
    }


def preference_outcome(score_a: float, score_b: float, gold: str, epsilon: float = 1e-9) -> dict[str, Any]:
    delta = score_a - score_b
    predicted = "tie" if abs(delta) <= epsilon else ("A" if delta > 0 else "B")
    correct = 0.5 if predicted == "tie" else float(predicted == gold)
    return {"score_a": score_a, "score_b": score_b, "delta": delta, "predicted": predicted, "correct_tie_half": correct}


def safe_mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None
