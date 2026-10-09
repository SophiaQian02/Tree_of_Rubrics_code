from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .json_utils import loads_json_with_escape_repair


@dataclass
class RubricNode:
    node_id: str
    text: str
    children: list["RubricNode"] = field(default_factory=list)
    weight: float = 1.0

    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass
class RubricTree:
    root: RubricNode

    def get_leaves(self) -> list[RubricNode]:
        leaves: list[RubricNode] = []

        def walk(node: RubricNode) -> None:
            if node.is_leaf():
                leaves.append(node)
                return
            for child in node.children:
                walk(child)

        walk(self.root)
        return leaves

    def get_depth(self) -> int:
        """Depth excludes the root by convention: root → ... → leaves counts as D levels below root."""

        def depth(node: RubricNode) -> int:
            if node.is_leaf():
                return 0
            return 1 + max(depth(child) for child in node.children)

        return depth(self.root)

    def get_max_breadth(self) -> int:
        breadths: list[int] = []

        def walk(node: RubricNode) -> None:
            if not node.is_leaf():
                breadths.append(len(node.children))
                for child in node.children:
                    walk(child)

        walk(self.root)
        return max(breadths, default=0)

    def to_json(self) -> Dict:
        def encode(node: RubricNode) -> Dict:
            return {
                "node_id": node.node_id,
                "text": node.text,
                "weight": node.weight,
                "children": [encode(child) for child in node.children],
            }

        return {"root": encode(self.root)}

    @staticmethod
    def from_json(data: Dict) -> "RubricTree":
        def decode(node_data: Dict) -> RubricNode:
            return RubricNode(
                node_id=str(node_data.get("node_id", "node")),
                text=str(node_data.get("text", "")),
                weight=float(node_data.get("weight", 1.0)),
                children=[decode(child) for child in node_data.get("children", [])],
            )

        return RubricTree(root=decode(data["root"]))

    def validate_shape(self, target_D: int, target_W: int) -> bool:
        def valid(node: RubricNode, remaining_depth: int) -> bool:
            if remaining_depth == 0:
                return node.is_leaf()
            return len(node.children) == target_W and all(
                valid(child, remaining_depth - 1) for child in node.children
            )

        return valid(self.root, target_D)


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text


def _extract_json_object(text: str) -> Optional[str]:
    text = _strip_json_fences(text)
    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
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
                return text[start : index + 1]
    return None


def _coerce_text_list(items: object) -> Optional[list[str]]:
    if not isinstance(items, list):
        return None
    values: list[str] = []
    for item in items:
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, dict):
            text = item.get("text") or item.get("criterion") or item.get("checkpoint") or item.get("name")
            if isinstance(text, str):
                values.append(text)
            else:
                return None
        else:
            return None
    return values


def _children_from_dimension(dimension: Dict, *keys: str) -> object:
    for key in keys:
        if key in dimension:
            return dimension[key]
    return []


def _decode_level(items: object, remaining_depth: int, W: int, prefix: str) -> Optional[list[RubricNode]]:
    if remaining_depth == 1:
        values = _coerce_text_list(items)
        if values is None or len(values) != W:
            return None
        return [
            RubricNode(node_id=f"{prefix}_leaf_{index}", text=value)
            for index, value in enumerate(values)
        ]

    if not isinstance(items, list) or len(items) != W:
        return None

    nodes: list[RubricNode] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            return None
        child_items = _children_from_dimension(
            item,
            "children",
            "subdimensions",
            "sub_dimensions",
            "subcriteria",
            "checkpoints",
            "criteria",
        )
        children = _decode_level(child_items, remaining_depth - 1, W, f"{prefix}_{index}")
        if children is None:
            return None
        nodes.append(
            RubricNode(
                node_id=f"{prefix}_{index}",
                text=str(item.get("name", f"level_{remaining_depth}_{index}")),
                children=children,
            )
        )
    return nodes


def parse_tree_json(text: str, D: int, W: int) -> Optional[RubricTree]:
    """Parse generator JSON output into a shaped rubric tree."""
    json_text = _extract_json_object(text)
    if json_text is None:
        return None

    try:
        data = loads_json_with_escape_repair(json_text)
    except json.JSONDecodeError:
        return None

    root = RubricNode(node_id="root", text="root")
    dimensions = data.get("dimensions", data.get("criteria", data.get("rubric", [])))
    if D < 1 or W < 1:
        return None

    root.children = _decode_level(dimensions, D, W, "dim")
    return RubricTree(root=root) if root.children is not None else None
