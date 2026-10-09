"""Tree-Regularized Whitened Uniform (TR-WU) pure matrix utilities.

All optimization happens globally over leaves.  The tree contributes only a
covariance prior; it never gates leaves or distributes a parent budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TreeIndex:
    leaf_ids: tuple[str, ...]
    leaf_paths: dict[str, tuple[dict[str, str], ...]]
    descendants: dict[str, tuple[str, ...]]
    root_id: str


def _node_id(node: dict[str, Any]) -> str:
    return str(node.get("node_id") or node.get("dimension_id") or node.get("root_id") or "")


def _node_context(node: dict[str, Any]) -> dict[str, str]:
    return {
        "node_id": _node_id(node),
        "name": str(node.get("name", "")),
        "description": str(node.get("description", "")),
        "assignment_summary": str(node.get("assignment_summary", "")),
    }


def flatten_tree(tree: dict[str, Any], flat_leaves: list[dict[str, Any]] | None = None) -> TreeIndex:
    """Build one stable global leaf order, full paths, and internal descendants."""
    leaf_ids = tuple(str(leaf["leaf_id"]) for leaf in flat_leaves) if flat_leaves is not None else ()
    if len(leaf_ids) != len(set(leaf_ids)):
        raise ValueError("Gold Flat contains duplicate leaf IDs")
    paths: dict[str, tuple[dict[str, str], ...]] = {}
    descendants: dict[str, tuple[str, ...]] = {}
    root_id = str(tree.get("root_id", "root"))
    root = {
        "node_id": root_id,
        "name": str(tree.get("name", "Overall response quality")),
        "description": str(tree.get("description", "")),
        "assignment_summary": str(tree.get("assignment_summary", "")),
        "children": tree.get("dimensions", []),
        "leaves": [],
    }

    discovered: list[str] = []

    def visit(node: dict[str, Any], path: tuple[dict[str, str], ...]) -> list[str]:
        node_id = _node_id(node)
        current = path + (_node_context(node),)
        found: list[str] = []
        for leaf in node.get("leaves", []):
            leaf_id = str(leaf["leaf_id"])
            if leaf_id in paths:
                raise ValueError(f"Tree contains duplicate leaf ID: {leaf_id}")
            leaf_context = {
                "leaf_id": leaf_id,
                "criterion": str(leaf.get("criterion", "")),
                "desired_state": str(leaf.get("desired_state", "criterion_met")),
            }
            paths[leaf_id] = current + (leaf_context,)
            discovered.append(leaf_id)
            found.append(leaf_id)
        for child in node.get("children", []):
            found.extend(visit(child, current))
        descendants[node_id] = tuple(found)
        return found

    visit(root, ())
    if not leaf_ids:
        leaf_ids = tuple(discovered)
    if set(leaf_ids) != set(discovered):
        raise ValueError("Tree and Flat leaf sets differ")
    return TreeIndex(leaf_ids, paths, descendants, root_id)


def build_calibration_matrix(records: list[dict[str, float]], leaf_ids: tuple[str, ...]) -> np.ndarray:
    expected = set(leaf_ids)
    rows = []
    for index, record in enumerate(records):
        if set(record) != expected:
            raise ValueError(f"calibration row {index} leaf IDs differ from Tree index")
        rows.append([float(record[leaf_id]) for leaf_id in leaf_ids])
    matrix = np.asarray(rows, dtype=float)
    if matrix.shape != (len(records), len(leaf_ids)) or matrix.shape[0] < 1:
        raise ValueError("calibration matrix has invalid shape")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("calibration matrix contains NaN or Inf")
    return matrix


def empirical_covariance(scores: np.ndarray) -> np.ndarray:
    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ValueError("scores must have shape [responses, leaves]")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / matrix.shape[0]
    return (covariance + covariance.T) / 2.0


def build_descendant_vectors(index: TreeIndex) -> dict[str, np.ndarray]:
    columns = {leaf_id: position for position, leaf_id in enumerate(index.leaf_ids)}
    output = {}
    for node_id, leaves in index.descendants.items():
        if node_id == index.root_id or len(leaves) < 2:
            continue
        vector = np.zeros(len(index.leaf_ids), dtype=float)
        for leaf_id in leaves:
            vector[columns[leaf_id]] = 1.0
        output[node_id] = vector
    return output


def build_tree_kernel(index: TreeIndex, alpha0: float = 1.0) -> np.ndarray:
    if alpha0 <= 0:
        raise ValueError("alpha0 must be positive")
    kernel = alpha0 * np.eye(len(index.leaf_ids), dtype=float)
    for vector in build_descendant_vectors(index).values():
        kernel += np.outer(vector, vector)
    diagonal = np.diag(kernel)
    correlation = kernel / np.sqrt(np.outer(diagonal, diagonal))
    return (correlation + correlation.T) / 2.0


def build_tree_covariance_prior(
    covariance: np.ndarray,
    tree_correlation: np.ndarray,
    variance_floor: float = 1e-4,
) -> np.ndarray:
    variances = np.maximum(np.diag(covariance), variance_floor)
    scale = np.sqrt(variances)
    prior = tree_correlation * np.outer(scale, scale)
    return (prior + prior.T) / 2.0


def inverse_sqrt_psd(matrix: np.ndarray, eigen_floor: float) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    floored = np.maximum(eigenvalues, eigen_floor)
    inverse_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(floored)) @ eigenvectors.T
    return inverse_sqrt, eigenvalues


def _normalize_positive(raw: np.ndarray) -> np.ndarray:
    positive = np.maximum(np.asarray(raw, dtype=float), 0.0)
    if not np.any(positive):
        positive = np.ones_like(positive)
    return positive / positive.sum()


def rrd_wu_weights(scores: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Numerically identical to RRD_project.rrd.weighting for the same scores."""
    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("scores must have shape [responses, leaves]")
    leaves = matrix.shape[1]
    if leaves == 0:
        return np.empty(0)
    if leaves == 1:
        return np.ones(1)
    covariance = empirical_covariance(matrix) + ridge * np.eye(leaves)
    inverse_sqrt, _ = inverse_sqrt_psd(covariance, ridge)
    return _normalize_positive(inverse_sqrt @ np.ones(leaves))


def hierarchical_local_wu_weights(
    scores: np.ndarray,
    tree: dict[str, Any],
    index: TreeIndex,
    ridge: float = 1e-3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute local WU bottom-up, then expand leaf budgets top-down.

    Every internal node treats its direct internal children and direct leaves as
    sibling branches.  An internal child's calibration vector is its already
    aggregated bottom-up score.  The second pass starts with root budget one
    and multiplies the local sibling weights along each Root-to-Leaf path.
    """
    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ValueError("scores must have shape [responses, leaves]")
    if matrix.shape[1] != len(index.leaf_ids):
        raise ValueError("score columns do not match Tree leaf index")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("scores contain NaN or Inf")
    if ridge <= 0:
        raise ValueError("ridge must be positive")

    columns = {leaf_id: position for position, leaf_id in enumerate(index.leaf_ids)}
    root = {
        "node_id": index.root_id,
        "name": str(tree.get("name", "Overall response quality")),
        "children": tree.get("dimensions", []),
        "leaves": tree.get("leaves", []),
    }
    seen_nodes: set[str] = set()
    seen_leaves: set[str] = set()
    runtime_branches: dict[str, list[tuple[str, str, dict[str, Any] | None]]] = {}
    node_scores: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, Any]] = {}

    def solve(node: dict[str, Any]) -> np.ndarray:
        node_id = _node_id(node)
        if not node_id:
            raise ValueError("every internal Tree node must have a stable ID")
        if node_id in seen_nodes:
            raise ValueError(f"duplicate internal Tree node ID: {node_id}")
        seen_nodes.add(node_id)
        branches: list[tuple[str, str, dict[str, Any] | None]] = []
        vectors: list[np.ndarray] = []

        # Internal children must be solved first because their aggregated
        # calibration scores are the columns used by the current parent.
        for child in node.get("children", []):
            child_id = _node_id(child)
            child_score = solve(child)
            branches.append(("internal", child_id, child))
            vectors.append(child_score)
        for leaf in node.get("leaves", []):
            leaf_id = str(leaf.get("leaf_id", ""))
            if leaf_id not in columns:
                raise ValueError(f"Tree leaf is missing from the global leaf index: {leaf_id}")
            if leaf_id in seen_leaves:
                raise ValueError(f"duplicate Tree leaf ID: {leaf_id}")
            seen_leaves.add(leaf_id)
            branches.append(("leaf", leaf_id, None))
            vectors.append(matrix[:, columns[leaf_id]])
        if not branches:
            raise ValueError(f"internal Tree node has no direct branches: {node_id}")

        child_matrix = np.column_stack(vectors)
        local_weights = rrd_wu_weights(child_matrix, ridge)
        node_score = child_matrix @ local_weights
        runtime_branches[node_id] = branches
        node_scores[node_id] = node_score
        branch_labels = [f"{kind}:{branch_id}" for kind, branch_id, _ in branches]
        diagnostics[node_id] = {
            "node_id": node_id,
            "branch_ids": branch_labels,
            "local_weights": dict(zip(branch_labels, local_weights.tolist())),
            "calibration_child_means": dict(zip(branch_labels, child_matrix.mean(axis=0).tolist())),
            "local_covariance": empirical_covariance(child_matrix).tolist(),
            "calibration_node_scores": node_score.tolist(),
        }
        return node_score

    root_scores = solve(root)
    if seen_leaves != set(index.leaf_ids):
        raise ValueError(
            f"Tree-WU did not reach every leaf exactly once: missing={sorted(set(index.leaf_ids) - seen_leaves)} "
            f"extra={sorted(seen_leaves - set(index.leaf_ids))}"
        )

    final_weights = np.zeros(len(index.leaf_ids), dtype=float)

    def distribute(node: dict[str, Any], budget: float) -> None:
        node_id = _node_id(node)
        branches = runtime_branches[node_id]
        local = diagnostics[node_id]["local_weights"]
        diagnostics[node_id]["global_budget"] = float(budget)
        child_budgets: dict[str, float] = {}
        for kind, branch_id, child in branches:
            label = f"{kind}:{branch_id}"
            child_budget = float(budget * local[label])
            child_budgets[label] = child_budget
            if kind == "leaf":
                final_weights[columns[branch_id]] = child_budget
            else:
                assert child is not None
                distribute(child, child_budget)
        diagnostics[node_id]["child_global_budgets"] = child_budgets

    distribute(root, 1.0)
    if np.any(final_weights < 0) or not np.all(np.isfinite(final_weights)):
        raise ValueError("Tree-WU produced invalid leaf weights")
    if not np.isclose(final_weights.sum(), 1.0, atol=1e-10, rtol=0):
        raise ValueError(f"Tree-WU leaf weights do not sum to one: {final_weights.sum()}")
    # The two-pass expansion and recursive bottom-up root score must be exactly
    # equivalent for all calibration responses.
    expanded_scores = matrix @ final_weights
    if not np.allclose(expanded_scores, root_scores, atol=1e-10, rtol=0):
        raise ValueError("Tree-WU top-down leaf expansion disagrees with bottom-up aggregation")
    return final_weights, {
        "method": "hierarchical_local_wu",
        "passes": ["bottom_up_local_wu", "top_down_budget_expansion"],
        "ridge": ridge,
        "root_id": index.root_id,
        "root_calibration_scores": root_scores.tolist(),
        "final_leaf_weights": dict(zip(index.leaf_ids, final_weights.tolist())),
        "nodes": diagnostics,
    }


def tree_regularized_wu_weights(
    scores: np.ndarray,
    index: TreeIndex,
    tree_strength: float = 0.5,
    ridge: float = 1e-3,
    variance_floor: float = 1e-4,
    alpha0: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not 0.0 <= tree_strength <= 1.0:
        raise ValueError("tree_strength must be in [0, 1]")
    matrix = np.asarray(scores, dtype=float)
    if matrix.shape[1] != len(index.leaf_ids):
        raise ValueError("score columns do not match Tree leaf index")
    if len(index.leaf_ids) == 1:
        return np.ones(1), {"fallback": "single_leaf"}
    empirical = empirical_covariance(matrix)
    correlation = build_tree_kernel(index, alpha0)
    prior = build_tree_covariance_prior(empirical, correlation, variance_floor)
    mixed = (1.0 - tree_strength) * empirical + tree_strength * prior
    regularized = (mixed + mixed.T) / 2.0 + ridge * np.eye(len(index.leaf_ids))
    inverse_sqrt, eigenvalues = inverse_sqrt_psd(regularized, ridge)
    raw = inverse_sqrt @ np.ones(len(index.leaf_ids))
    weights = _normalize_positive(raw)
    diagnostics = {
        "empirical_covariance": empirical.tolist(),
        "tree_correlation": correlation.tolist(),
        "tree_covariance_prior": prior.tolist(),
        "mixed_covariance": mixed.tolist(),
        "eigenvalues_before_floor": eigenvalues.tolist(),
        "raw_wu_vector": raw.tolist(),
        "negative_component_count": int(np.sum(raw < 0)),
        "tree_strength": tree_strength,
        "ridge": ridge,
        "variance_floor": variance_floor,
        "alpha0": alpha0,
    }
    return weights, diagnostics


def subtree_effective_mass(index: TreeIndex, weights: np.ndarray) -> dict[str, float]:
    columns = {leaf_id: position for position, leaf_id in enumerate(index.leaf_ids)}
    return {
        node_id: float(sum(weights[columns[leaf_id]] for leaf_id in leaves))
        for node_id, leaves in index.descendants.items()
    }
