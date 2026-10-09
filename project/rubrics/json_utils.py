"""Compatibility imports for rubric code using the historical module path."""

from project.utils.json_utils import loads_json_with_escape_repair, repair_invalid_json_escapes

__all__ = ["loads_json_with_escape_repair", "repair_invalid_json_escapes"]
