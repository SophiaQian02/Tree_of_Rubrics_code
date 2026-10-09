from __future__ import annotations

import json
import string
from typing import Any


_VALID_SIMPLE_ESCAPES = {'"', "\\", "/", "b", "f", "n", "r", "t"}
_PLAIN_TEXT_ESCAPES = {"(", ")", "[", "]", "_", "%", "$"}
_HEX_DIGITS = set(string.hexdigits)


def repair_invalid_json_escapes(text: str) -> str:
    """Repair invalid backslash escapes inside JSON strings without changing valid escapes."""
    output: list[str] = []
    in_string = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            in_string = not in_string
            output.append(char)
            index += 1
            continue
        if not in_string or char != "\\":
            output.append(char)
            index += 1
            continue
        if index + 1 >= len(text):
            output.append("\\\\")
            index += 1
            continue
        next_char = text[index + 1]
        if next_char in _VALID_SIMPLE_ESCAPES:
            output.extend((char, next_char))
            index += 2
            continue
        if next_char == "u" and index + 5 < len(text):
            unicode_digits = text[index + 2 : index + 6]
            if len(unicode_digits) == 4 and all(digit in _HEX_DIGITS for digit in unicode_digits):
                output.append(text[index : index + 6])
                index += 6
                continue
        if next_char in _PLAIN_TEXT_ESCAPES:
            output.append(next_char)
        else:
            output.extend(("\\\\", next_char))
        index += 2
    return "".join(output)


def loads_json_with_escape_repair(text: str) -> Any:
    """Parse JSON, retrying once after repairing invalid string escapes."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as original_error:
        repaired = repair_invalid_json_escapes(text)
        if repaired == text:
            raise original_error
        return json.loads(repaired)
