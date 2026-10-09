from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class CalibrationResponseCache(dict[tuple[str, str], dict[str, Any]]):
    """Exact cache keys plus a fallback index for datasets that remap IDs."""

    def __init__(self) -> None:
        super().__init__()
        self.by_question_hash: dict[str, dict[str, Any]] = {}


def question_hash(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def calibration_cache_key(sample_id: str, question: str) -> tuple[str, str]:
    return str(sample_id), question_hash(str(question))


def load_calibration_response_cache(path: str | Path) -> CalibrationResponseCache:
    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Calibration response cache does not exist: {cache_path}")

    entries = CalibrationResponseCache()
    with cache_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {cache_path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {cache_path}:{line_number}")
            sample_id = str(row.get("sample_id", ""))
            stored_hash = str(row.get("question_hash", ""))
            question = str(row.get("question", ""))
            responses = row.get("responses")
            if not sample_id or not question or not isinstance(responses, list):
                raise ValueError(
                    f"Cache row {cache_path}:{line_number} must contain sample_id, question, and a responses list"
                )
            actual_hash = question_hash(question)
            if stored_hash and stored_hash != actual_hash:
                raise ValueError(f"Question hash mismatch in {cache_path}:{line_number}")
            row["question_hash"] = actual_hash
            entries[(sample_id, actual_hash)] = row
            entries.by_question_hash.setdefault(actual_hash, row)
    return entries


def cached_responses_for_sample(
    entries: dict[tuple[str, str], dict[str, Any]],
    *,
    sample_id: str,
    question: str,
    required_count: int,
) -> tuple[list[str], dict[str, Any]]:
    key = calibration_cache_key(sample_id, question)
    row = entries.get(key)
    if row is None and isinstance(entries, CalibrationResponseCache):
        # Cleaned/subsampled datasets may preserve the exact question and
        # responses while assigning a new sample ID. Reuse the already sampled
        # responses by question hash instead of requiring online resampling.
        row = entries.by_question_hash.get(key[1])
    if row is None:
        raise KeyError(
            f"No pre-sampled calibration responses for sample_id={sample_id!r} "
            f"question_hash={key[1]}. Run the pre-sampling command for the same dataset/split/sample count."
        )
    responses = [str(item).strip() for item in row.get("responses", []) if str(item).strip()]
    if len(responses) < required_count:
        raise ValueError(
            f"Pre-sampled entry for sample_id={sample_id!r} has {len(responses)} valid responses; "
            f"ARTS requires {required_count}."
        )
    return responses[:required_count], row


def completed_cache_keys(path: str | Path) -> set[tuple[str, str]]:
    cache_path = Path(path)
    if not cache_path.exists():
        return set()
    return set(load_calibration_response_cache(cache_path))


def append_calibration_cache_row(path: str | Path, row: dict[str, Any]) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
