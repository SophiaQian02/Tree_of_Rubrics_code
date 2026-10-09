from __future__ import annotations

import heapq
import hashlib
import json
import math
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Callable, Optional

from project.rubrics.json_utils import loads_json_with_escape_repair
from project.rubrics.tree import RubricNode, RubricTree


ARTS_SYSTEM_PROMPT = """You output only valid JSON. Do not include markdown fences or prose.
Do not use backslashes unless they are valid JSON escapes. Avoid LaTeX delimiters such as \\( \\) or \\[ \\]. Write formulas as plain text."""

ARTS_LEAF_WEIGHTING = "direct_leaf_global_v1"


CALIBRATION_RESPONSE_PROMPT = """Generate one calibration response to the user query below.

User query:
<query>
{calibration_question}
</query>

Quality target: {quality}

Instruction:
{quality_instruction}

Prompt modifier:
{prompt_modifier}

Do not label the response or explain that it is a calibration response.

Output ONLY JSON:
{{"response": "..."}}
"""


def get_calibration_response_prompt(dataset_prompt: str, prompt_modifier: Optional[str] = None) -> str:
    """Wrap the dataset prompt for ARTS calibration with explicit chain-of-thought style answering."""
    cot_instruction = """

Think step by step about this question, then provide your answer.
End your response with "Final answer: X" where X is the answer letter (for multiple choice) or the answer value.
"""
    base_prompt = dataset_prompt + cot_instruction
    if prompt_modifier:
        return f"[Style: {prompt_modifier}]\n\n{base_prompt}"
    return base_prompt


GENERATOR_PROMPT = """User query:
<query>
{query}
</query>

Parent rubric: "{parent_text}"

Generate {n_candidates} sub-rubrics that together cover as much of the parent rubric as possible for this specific query.

Each sub-rubric must be:
- Concrete and verifiable
- Specific enough that different judges agree
- Independently meaningful
- Specific to the user query above, so the resulting rubric can judge responses to this exact task
- Focus on correctness and reasoning before superficial output format; include a pure formatting criterion only when format is central to task success

The full set of generated sub-rubrics should be coverage-oriented: include all major task-specific aspects needed to judge the parent rubric, avoid leaving obvious gaps, and avoid redundant near-duplicates.

Use AND semantics: the selected sub-rubrics should jointly preserve the parent meaning, and each selected sub-rubric should represent a necessary aspect of the parent.

Do not assign weights here. Final leaf criteria are independently assigned an importance score later and globally normalized across leaves.

Output JSON:
{{
  "sub_rubrics": [
    {{"text": "..."}},
    {{"text": "..."}}
  ]
}}
"""


ROOT_GENERATOR_PROMPT = """User query:
<query>
{query}
</query>

Root rubric: "{parent_text}"

Generate {n_candidates} child rubrics by decomposing the user query itself.

Do not freely invent generic dimensions of "overall quality". Instead, identify what a correct response to this exact query must accomplish, and turn those task requirements into concrete child rubrics.

Each child rubric must be:
- Concrete and verifiable
- Specific enough that different judges agree
- Independently meaningful
- Directly tied to an explicit or strongly implied requirement of the user query
- Focused on correctness, reasoning, domain knowledge, or required task constraints before style or format

The full set should cover the main task-specific requirements needed to judge responses to this query. Avoid generic criteria like "overall clarity" unless the query itself makes clarity a central requirement.

Use AND semantics: the selected child rubrics should jointly define what it means to satisfy the query.

Do not assign weights here. Final leaf criteria are independently assigned an importance score later and globally normalized across leaves.

Output JSON:
{{
  "sub_rubrics": [
    {{"text": "..."}},
    {{"text": "..."}}
  ]
}}
"""


REGENERATE_PROMPT = """Rewrite this rubric criterion to be more concrete and verifiable.

Previous criterion:
<criterion>
{criterion}
</criterion>

Hint:
{hint}

Keep the same intended evaluation requirement, but reduce ambiguity so independent judges are more likely to agree.

Output ONLY JSON:
{{"criterion": "..."}}
"""


IMPORTANCE_PROMPT = """TASK: {query}
CRITERION: {rubric_text}

Score the importance of this criterion on a 1-5 scale.

Judge this criterion directly against the task. Do not allocate a parent budget, do not normalize against siblings, and do not multiply by path or depth weights. For criteria that remain final leaves, global normalization is performed later in code.

KEY PRINCIPLE:
- Substantive criteria (correctness, understanding, reasoning) are inherently more important.
- Surface/format criteria (output format, length compliance, syntactic patterns) should only be rated highly if they are EXPLICITLY critical to the task definition (e.g., a task that strictly requires JSON output).
- For most evaluation tasks, surface criteria should score 1-2 even if mentioned in the prompt.

Examples:
- "Does the response correctly identify the legal doctrine?" -> 5 (essential substance)
- "Does the response provide step-by-step reasoning?" -> 4 (important reasoning)
- "Does the response end with 'Final answer: X'?" -> 2 (minor format)
- "Does the response use exactly 5 paragraphs?" -> 1 (trivial format)

Output a single digit 1-5.
"""


ASPECTS_PROMPT = """TASK: {query}
PARENT RUBRIC: {parent_text}

List every independently fail-able semantic requirement that must be evaluated to
decide whether a response satisfies the parent rubric for this task.

Rules:
- Preserve every explicit constraint in the user query. Never combine two
  requirements when a response can satisfy one while violating the other.
- Each aspect must describe exactly one independently judgeable requirement.
- Never return paraphrases or contextual restatements of an aspect already in
  the list. The same requirement mentioned in the process, hypothesis,
  question, or answer is still one aspect unless it can be violated
  independently in those contexts.
- Aspects should be task-specific and substantive.
- Prefer correctness, reasoning, understanding, and explicit task constraints.
- A format, length, keyword-count, JSON-shape, or style constraint explicitly
  stated in TASK is a first-class requirement and must be preserved as its own
  aspect when it can be violated independently.
- Do not invent response requirements from the evaluator's system instruction
  or from this prompt's JSON output schema. In particular, "output only JSON",
  "no markdown fences", escape rules, and similar evaluator-output directions
  are NOT task aspects unless those words occur inside TASK itself.
- For rewrite, transform, summarize, translate, classify, or edit tasks,
  distinguish the OPERATION and its explicit constraints from the SOURCE
  MATERIAL being operated on. Sentences, topics, facts, examples, names, and
  claims inside source material are input data, not separate evaluation aspects,
  unless the user explicitly requires their meaning or details to be preserved.
- In a destructive transformation such as "turn this text into gibberish",
  do not extract semantic-content aspects from the text that is intentionally
  being destroyed. Extract only the requested transformation properties and
  preservation constraints (for example, gibberish and sentence count).
- Do not over-emphasize surface format that is not explicitly required by TASK.
- Return 1-12 aspects. The number must follow the query rather than an arbitrary
  target size.

Output ONLY JSON:
{{"aspects": ["...", "..."]}}
"""


ASPECT_COVERAGE_PROMPT = """Parent aspect:
{aspect}

Child rubric:
{rubric_text}

Question:
Can this child rubric, by itself, evaluate whether the response satisfies the parent aspect?

Return True only if:
- the child directly evaluates the same semantic requirement as the aspect;
- satisfying the child would provide strong evidence that the aspect is satisfied;
- the child is not merely about formatting, style, or general answer presence unless the aspect is also about formatting.

Return False if:
- the child is only loosely related;
- the child checks format while the aspect is about correctness/reasoning/domain knowledge;
- the child is more general or more superficial than the aspect;
- the child only checks whether an answer exists, not whether it is correct.

Output ONLY JSON:
{{"covered": true}}
"""


GLOBAL_COVERAGE_PROMPT = """Check whether the final leaf rubric set adequately covers the evaluation needs for this task.

User query:
<query>
{query}
</query>

Root rubric:
<root>
{root_text}
</root>

Final leaf rubrics:
<leaves>
{leaves_json}
</leaves>

A covered leaf set must be concrete enough to score responses, not merely restate "overall quality".
A covered leaf set must preserve every explicit query constraint as an
independently judgeable leaf whenever it can be satisfied or violated
independently. A final leaf containing multiple independently fail-able
requirements is an atomicity violation.
A format-only leaf is valid and necessary when it directly checks an explicit
user constraint such as sentence count, keyword count, markdown, or required
JSON shape. Do not require an invented substantive criterion for a query whose
entire task consists of explicit format constraints.
A covered leaf set must not contain true semantic duplicates or subsumption
duplicates. Similar calibration scores are not evidence of semantic redundancy.
A covered leaf set must include substantive task-specific correctness/reasoning/domain criteria when the task has substantive content.
For rewrite or transformation tasks, treat quoted/source text as input data,
not as a list of requirements. Do not demand preservation or factual coverage
of source-text details when the user did not request it; in destructive
transformations such as converting text to gibberish, source semantics may be
intentionally removed.
Do not count format-only leaves such as duplicated-letter, single-string, no-prose, or JSON-shape checks as covering substantive legal/factual/math/reasoning or answer-correctness aspects.
Do not treat this evaluator prompt's own JSON output instructions as user-query
requirements.
If multiple leaves cover only output formatting while substantive requirements are missing, return covered=false and list the missing substantive aspects.

Return covered=true only if the leaves jointly cover the main correctness, reasoning, task-specific constraints, and important output requirements implied by the query, every leaf is atomic, and no pair is semantically redundant.
If covered=false, list the missing aspects that should become additional leaf rubrics.

Output ONLY JSON:
{{"covered": true, "reason": "...", "missing_aspects": [], "atomicity_violations": [{{"leaf_id":"...","reason":"..."}}], "redundant_pairs": [{{"left_id":"...","right_id":"...","reason":"..."}}]}}
"""


MISSING_ASPECT_LEAF_PROMPT = """Generate one concrete leaf rubric for the missing evaluation aspect below.

User query:
<query>
{query}
</query>

Root rubric:
<root>
{root_text}
</root>

Existing final leaf rubrics:
<leaves>
{leaves_json}
</leaves>

Missing aspect:
<missing_aspect>
{missing_aspect}
</missing_aspect>

The new leaf rubric must be:
- Concrete and verifiable
- Specific to this user query
- Non-redundant with the existing leaves
- Directly usable as a YES/NO scoring criterion

Output ONLY JSON:
{{"rubric": "..."}}
"""


FORCED_ATOMIC_SPLIT_PROMPT = """Split one non-atomic rubric into atomic leaf rubrics.

User query:
<query>
{query}
</query>

Rubric that must be split:
<rubric>
{rubric_text}
</rubric>

Why it is non-atomic:
<atomicity_reason>
{atomicity_reason}
</atomicity_reason>

You may return at most {max_children} leaves.

Requirements:
- Each returned leaf must contain exactly ONE independently fail-able requirement.
- The returned leaves must jointly preserve every requirement in the original rubric.
- Do not add requirements that are absent from the user query or original rubric.
- Do not return paraphrase duplicates or one leaf that subsumes another.
- Each leaf must be a concrete, task-specific YES/NO scoring criterion.
- Return at least two leaves. If the rubric truly has only one independently
  fail-able requirement, return an empty list rather than inventing a split.

Output ONLY JSON:
{{"atomic_leaves": [{{"rubric": "...", "weight": 3}}], "reason": "..."}}
"""


ATOMIC_LEAF_FOR_ASPECT_PROMPT = """Write exactly one atomic leaf rubric for one extracted requirement.

User query:
<query>
{query}
</query>

Compound source rubric:
<source_rubric>
{rubric_text}
</source_rubric>

The ONE requirement this leaf must test:
<requirement>
{aspect}
</requirement>

Other independently fail-able requirements are handled by other leaves. Do not
include them in this leaf. Preserve the requirement faithfully, make it
concrete and task-specific, and phrase it as a direct YES/NO scoring criterion.

Output ONLY JSON:
{{"rubric": "...", "weight": 3}}
"""


FORCED_SPLIT_SUFFICIENCY_PROMPT = """Validate whether a proposed atomic split is semantically sufficient.

User query:
<query>
{query}
</query>

Compound source rubric:
<source_rubric>
{rubric_text}
</source_rubric>

Proposed atomic child rubrics:
<children>
{children_json}
</children>

Return sufficient=true only if:
- every explicit independently fail-able requirement in the user instruction is covered;
- every child tests exactly one requirement;
- no two children are true semantic duplicates or subsumption duplicates;
- the children do not invent requirements.

For rewrite/transform tasks, source material is input data rather than a list of
requirements. Do not demand preservation of source facts, names, topics, or
structure unless the user explicitly asks for it. Do not include requirements
from this evaluator's JSON/schema/output instructions. An operational property
such as "complete gibberish" may be one atomic requirement; do not require a
separate leaf for every paraphrase or implied definition of that property.

Output ONLY JSON:
{{"sufficient": true, "reason": "..."}}
"""


FINAL_SEMANTIC_REWRITE_PROMPT = """Repair a final rubric leaf set that failed strict semantic validation.

User query:
<query>
{query}
</query>

Root rubric:
<root>
{root_text}
</root>

Current leaf rubrics:
<leaves>
{leaves_json}
</leaves>

Validation diagnostics:
<diagnostics>
{diagnostics_json}
</diagnostics>

Return a COMPLETE replacement leaf set, with at most {budget} leaves.

Requirements:
- Preserve every explicit and substantive requirement needed to evaluate the query.
- Every leaf must contain exactly ONE independently fail-able requirement.
- Split compound criteria; remove true duplicates and subsumption duplicates.
- Do not invent criteria unrelated to the query.
- Each leaf must be concrete, task-specific, and directly judgeable as YES/NO.
- source_leaf_ids must contain the IDs of current leaves from which the repaired
  leaf was derived. It may contain multiple IDs for a true merge.

Output ONLY JSON:
{{"rubrics": [{{"rubric": "...", "source_leaf_ids": ["..."], "weight": 3}}], "reason": "..."}}
"""


REPAIR_PARENT_PROMPT = """Choose where a new missing-aspect rubric should be inserted in the existing rubric tree.

User query:
<query>
{query}
</query>

Missing aspect:
<missing_aspect>
{missing_aspect}
</missing_aspect>

New leaf rubric:
<new_leaf>
{new_leaf}
</new_leaf>

Eligible parent nodes:
<parents>
{parents_json}
</parents>

Choose the most specific eligible parent whose children should include this new leaf.
Choose the root only when the missing aspect is broad or cross-cutting.

Output ONLY JSON:
{{"parent_id": "...", "reason": "..."}}
"""


LOCAL_COVERAGE_REPAIR_PROMPT = """Generate one additional sub-rubric to cover a missing parent aspect.

User query:
<query>
{query}
</query>

Parent rubric:
<parent>
{parent_text}
</parent>

Missing parent aspect:
<missing_aspect>
{missing_aspect}
</missing_aspect>

Existing candidate sub-rubrics:
<candidates>
{candidates_json}
</candidates>

The new sub-rubric must be:
- Concrete and verifiable
- Specific to the user query and parent rubric
- Non-redundant with existing candidates
- Directly usable as a YES/NO scoring criterion

Output ONLY JSON:
{{"rubric": "..."}}
"""


SEMANTIC_RUBRIC_QUALITY_PROMPT = """Evaluate whether a rubric criterion is suitable for this exact task.

User query:
<query>
{query}
</query>

Criterion:
<criterion>
{criterion}
</criterion>

Rate each field on a 0.0-1.0 scale:
- concrete: the criterion is specific, observable, and not vague.
- verifiable: an evaluator can answer YES/NO from the response.
- unambiguous: independent judges are likely to interpret it the same way.
- task_specific: it is tied to the query, not generic response quality.
- distinctive: it can help distinguish correct, partially correct, and incorrect responses.
- leaf_quality: it is suitable as a final YES/NO scoring point if not decomposed.
- atomic: it contains exactly one independently fail-able requirement. Return
  atomic=false whenever a response could satisfy one clause but violate another.
- independent_requirement_count: number of independently fail-able requirements
  contained in the criterion.
- substantive: it evaluates task substance such as correctness, reasoning, domain knowledge, factual/legal/mathematical requirements, or answer choice correctness.
- format_only: it evaluates only output shape, length, duplicated letters, JSON/prose restrictions, or other surface formatting.
- explicit_query_requirement: the criterion directly checks a requirement
  explicitly stated by the user query. This may be true for a format-only
  criterion, such as an exact sentence count, keyword count, markdown rule, or
  required JSON shape. Do not mark evaluator/system output instructions as user
  requirements unless they literally occur in the user query.

The pass field means semantic admissibility as a tree node: concrete, verifiable,
unambiguous, task-specific, and not merely superficial format unless format is
central to the task. A semantically valid but non-atomic criterion may pass only
as an internal node; it must never become a final leaf.
For multiple-choice or answer-format tasks, format criteria may pass, but they should be marked format_only=true and substantive=false unless they also verify the correctness of the chosen answer.

IMPORTANT CONSISTENCY RULES:
- The numeric values in the JSON schema below are placeholders, not defaults.
  Evaluate the criterion and return actual scores.
- pass=true requires concrete, verifiable, unambiguous, and task_specific to
  each be at least 0.6. A central explicit format constraint must still be
  concrete, verifiable, unambiguous, and task-specific.
- An atomic explicit_query_requirement is a legitimate final leaf even when it
  is format_only=true and substantive=false. In that case leaf_quality must
  reflect whether it is a clear YES/NO check, not whether it tests domain facts.
- Never return pass=true with all-zero scores.
- The numeric scores, booleans, pass decision, and written reason must agree.

Output ONLY JSON:
{{
  "concrete": 0.0,
  "verifiable": 0.0,
  "unambiguous": 0.0,
  "task_specific": 0.0,
  "distinctive": 0.0,
  "leaf_quality": 0.0,
  "atomic": true,
  "independent_requirement_count": 1,
  "atomicity_reason": "...",
  "substantive": true,
  "format_only": false,
  "explicit_query_requirement": true,
  "pass": true,
  "reason": "..."
}}
"""


SEMANTIC_NONREDUNDANCY_PROMPT = """Decide whether two child rubrics are semantically redundant for this parent rubric.

User query:
<query>
{query}
</query>

Parent rubric:
<parent>
{parent_text}
</parent>

Rubric A:
<a>
{left}
</a>

Rubric B:
<b>
{right}
</b>

Return redundant=true if they mainly test the same requirement or one mostly subsumes the other.
Return redundant=false if they test complementary requirements.

Output ONLY JSON:
{{"redundant": false, "reason": "..."}}
"""


SEMANTIC_SELECT_CHILDREN_PROMPT = """Select child rubrics for an adaptive rubric tree.

User query:
<query>
{query}
</query>

Parent rubric:
<parent>
{parent_text}
</parent>

Parent aspects that selected children should cover:
<aspects>
{aspects_json}
</aspects>

Candidate child rubrics:
<candidates>
{candidates_json}
</candidates>

Coverage matrix:
<coverage>
{coverage_json}
</coverage>

Hybrid score redundancy diagnostics, if available:
<score_redundancy>
{score_redundancy_json}
</score_redundancy>

Construct one or more semantically feasible child sets that jointly cover the
parent aspects. Semantic validity is the only consideration in this stage.

Rules:
- Prefer substantive correctness, reasoning, domain knowledge, factual/legal/mathematical requirements, and answer-correctness criteria.
- Preserve every explicit user requirement, including independently fail-able
  format, length, keyword-count, markdown, and JSON-shape constraints.
- Every selected child must add independent semantic coverage.
- Do not select redundant children: if two children mainly test the same requirement, select the clearer and more task-specific one.
- Do not select generic "clear explanation" criteria when a more specific task criterion is available.
- If the task has substantive requirements, selected children must include
  substantive children that cover them. A task containing only explicit format
  constraints does not need an invented substantive child.
- Multiple format-only children are allowed when each represents a different
  explicit user requirement. At most one non-explicit format/style child may be
  selected.
- Format-only children such as duplicated letters, single string, no prose, or JSON shape do not cover substantive aspects.
- If a core aspect is missing from all candidates, list it in missing_aspects.
- Ignore calibration scores completely in this stage. Score correlation is not
  semantic redundancy. Two requirements that can be violated independently must
  remain separate even if their calibration scores are perfectly correlated.
- Return 2-3 distinct feasible sets whenever multiple wording choices or
  decomposition structures satisfy all semantic constraints. Return only one
  set only when no genuine semantic alternative exists. Assign semantic_score
  and confidence using semantic evidence only.

Output ONLY JSON:
{{
  "selected_ids": ["..."],
  "feasible_sets": [
    {{"selected_ids": ["..."], "semantic_score": 0.0, "confidence": 0.0, "reason": "..."}}
  ],
  "missing_aspects": ["..."],
  "score_used": false,
  "score_reason": "Calibration is forbidden during semantic feasibility.",
  "rejected": [
    {{"id": "...", "reason": "redundant / too generic / format-only / weak coverage"}}
  ],
  "reason": "..."
}}
"""


SEMANTIC_IMPROVE_SELECTED_PROMPT = """Improve a selected child-rubric set.

User query:
<query>
{query}
</query>

Parent rubric:
<parent>
{parent_text}
</parent>

Parent aspects:
<aspects>
{aspects_json}
</aspects>

Currently selected children:
<selected>
{selected_json}
</selected>

Remaining candidate children:
<remaining>
{remaining_json}
</remaining>

Coverage matrix:
<coverage>
{coverage_json}
</coverage>

Pairwise redundancy analysis among selected children:
<redundancy>
{redundancy_json}
</redundancy>

Hybrid score redundancy diagnostics, if available:
<score_redundancy>
{score_redundancy_json}
</score_redundancy>

Choose exactly one action:
- prune: remove a selected child that is redundant, too generic, nonessential, or duplicate format-only.
- replace: replace a weak/redundant selected child with a better remaining candidate.
- add: add a remaining candidate only if it covers a missing or weakly covered substantive aspect and is not redundant.
- keep: only if coverage is sufficient, no redundancy exists, every selected child has independent semantic value, and format-only is not overrepresented.

Priorities:
1. Remove redundancy.
2. Preserve or improve substantive coverage.
3. Prefer clearer, more specific, more task-grounded criteria.
4. Keep the set compact.

Calibration scores are unavailable and forbidden in this stage. Never prune or
merge independently fail-able requirements because their observed scores are
correlated.

Output ONLY JSON:
{{
  "redundancy_found": false,
  "redundant_pairs": [
    {{"left_id": "...", "right_id": "...", "reason": "..."}}
  ],
  "coverage_sufficient": true,
  "action": "keep",
  "remove_ids": [],
  "add_ids": [],
  "score_used": false,
  "score_reason": "Calibration is forbidden during semantic feasibility.",
  "reason": "..."
}}
"""


SEMANTIC_DECOMPOSE_PROMPT = """Decide whether selected child rubrics should replace the parent rubric in an adaptive rubric tree.

User query:
<query>
{query}
</query>

Parent rubric:
<parent>
{parent_text}
</parent>

Parent aspects:
<aspects>
{aspects_json}
</aspects>

Selected child rubrics:
<children>
{children_json}
</children>

Choose decompose=true only if the children are collectively clearer, more complete, less ambiguous, and more useful for scoring than keeping the parent as a single YES/NO leaf.
Choose decompose=false if the children are redundant, miss important substantive aspects, overfit superficial details, or make scoring less coherent.

Important:
- If the parent contains multiple independently fail-able requirements, it is
  not a valid leaf and decompose must be true whenever the selected children
  preserve coverage without semantic redundancy.
- If the task has substantive requirements, selected children must include
  substantive task criteria that cover them.
- Explicit, independently fail-able format requirements are valid children and
  must not be collapsed merely because they are format-only.
- Do not approve a decomposition dominated by non-explicit superficial style
  children.
- A child that only checks "single string", "duplicated letters", "no prose", or similar output shape does not cover legal/factual/reasoning/answer-correctness aspects.

Output ONLY JSON:
{{"decompose": true, "confidence": 0.0, "reason": "..."}}
"""


DEFAULT_CALIBRATION_RESPONSE_GENERATORS = [
    {"model": "Qwen3-30B-A3B-Instruct-2507", "quality": "strong", "prompt_modifier": None},
    {
        "model": "Qwen3-30B-A3B-Instruct-2507",
        "quality": "strong",
        "prompt_modifier": "be brief, omit examples",
    },
    {"model": "Llama-3.1-70B-Instruct", "quality": "medium", "prompt_modifier": None},
    {
        "model": "Llama-3.1-70B-Instruct",
        "quality": "medium",
        "prompt_modifier": "verbose with elaboration",
    },
    {"model": "Mistral-7B-Instruct-v0.2", "quality": "weak", "prompt_modifier": None},
    {
        "model": "Mistral-7B-Instruct-v0.2",
        "quality": "weak",
        "prompt_modifier": "careless with one minor inaccuracy",
    },
]


DEFAULT_CALIBRATION_JUDGES = [
    {"model": "Qwen2.5-72B-Instruct"},
    {"model": "Llama-3.3-70B-Instruct"},
]


QUALITY_INSTRUCTIONS = {
    "strong": (
        "Write a strong answer that directly satisfies the user query. It should be accurate, complete, "
        "well-structured, and follow all explicit constraints. Do not intentionally include mistakes."
    ),
    "medium": (
        "Write a medium-quality answer. It should mostly address the user query, but may omit some details, "
        "be somewhat shallow, or fail to fully explain one or two secondary points. Do not make it obviously bad."
    ),
    "weak": (
        "Write a weak answer. It should attempt to answer the user query but be incomplete, vague, poorly organized, "
        "or contain one minor inaccuracy. Keep it plausible rather than nonsensical."
    ),
}


@dataclass
class ARTSConfig:
    decision_policy: str = "hybrid"
    leaf_weighting: str = ARTS_LEAF_WEIGHTING
    budget: int = 16
    max_depth: int = 3
    gamma: float = 0.85
    k_calibration: int = 6
    n_candidates_per_node: int = 10
    tau_scor: float = 0.80
    max_regen_attempts: int = 4
    margin: float = 0.00
    greedy_margin: float = 0.0
    low_distinct_threshold: float = 5e-5
    tau_cov: float = 1.0
    lambda_red: float = 0.05
    lambda_size: float = 0.0
    use_distinct: bool = True
    beta: float = 8.0
    min_children: int = 3
    generator_model: str = ""
    generator_base_url: str = ""
    generator_api_key: str = "EMPTY"
    judge_configs: list[dict[str, Any]] = field(default_factory=list)
    calibration_response_generators: list[dict[str, Any]] = field(
        default_factory=lambda: [dict(item) for item in DEFAULT_CALIBRATION_RESPONSE_GENERATORS]
    )
    calibration_judges: list[dict[str, Any]] = field(
        default_factory=lambda: [dict(item) for item in DEFAULT_CALIBRATION_JUDGES]
    )
    generator_temperature: float = 0.2
    max_tokens: int = 2048
    max_retries: int = 3
    root_text: str = "Overall quality of response to the given task"
    coverage_check: bool = True
    max_steps: int = 128
    max_final_repair_attempts: int = 2
    redundancy_threshold: float = 0.8
    calibration_response_concurrency: int = 6
    max_calibration_refill_rounds: int = 6
    calibration_judge_concurrency: int = 12
    candidate_concurrency: int = 2
    # Hybrid is constrained optimization: semantics defines a hard feasible
    # set at every intermediate node; calibration ranks only members of that
    # set and can never rescue a semantically invalid candidate.
    semantic_score_epsilon: float = 0.05
    semantic_confidence_threshold: float = 0.70
    hard_semantic_constraints: bool = True
    # Fail safe for semantic-first hybrid search.  If the strict set-level
    # gate cannot produce a complete decomposition, retain the best currently
    # available individually semantic-valid atomic rubric instead of dropping
    # the entire sample.
    semantic_failure_fallback: bool = True


@dataclass
class ARTSNode:
    id: str
    text: str
    depth: int
    parent_id: Optional[str] = None
    children: list[str] = field(default_factory=list)
    status: str = "frontier"
    importance: float = 1.0
    local_weight: float = 1.0
    scorability: Optional[float] = None
    scores: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ARTSResult:
    tree: RubricTree
    internal_nodes: dict[str, dict[str, Any]]
    leaf_scoring_points: dict[str, dict[str, Any]]
    raw_nodes: dict[str, dict[str, Any]]
    calibration_responses: list[str]
    logs: list[dict[str, Any]]
    budget: int
    max_depth: int
    leaf_weighting: str


@dataclass
class ARTSGenerationOutcome:
    result: Optional[ARTSResult]
    failure_stage: str = ""
    failure_reason: str = ""
    logs: list[dict[str, Any]] = field(default_factory=list)


GateFn = Callable[[ARTSNode, ARTSConfig], bool]


def _no_thinking_extra_body(model: str) -> dict[str, Any]:
    """Force direct output for local reasoning models used by ARTS."""
    normalized = model.strip().lower().replace("/", "-")
    if "qwen3" in normalized or "glm-4.7" in normalized or "glm4.7" in normalized:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


def default_gate(node: ARTSNode, config: ARTSConfig) -> bool:
    if node.depth >= config.max_depth:
        return False
    return True


class _ARTSLogList(list[dict[str, Any]]):
    def __init__(self, callback: Optional[Callable[[dict[str, Any]], None]] = None) -> None:
        super().__init__()
        self.callback = callback

    def append(self, item: dict[str, Any]) -> None:
        super().append(item)
        if self.callback is None:
            return
        try:
            self.callback(item)
        except Exception:
            pass


class ARTSRunner:
    def __init__(
        self,
        question: str,
        config: ARTSConfig,
        calibration_responses: Optional[list[str]] = None,
        gate: GateFn = default_gate,
        log_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        self.question = question
        self.config = config
        self.calibration_responses = calibration_responses or []
        self.gate = gate
        self.log_callback = log_callback
        from project.utils import create_openai_client

        self.client = create_openai_client(
            base_url=config.generator_base_url,
            api_key=config.generator_api_key,
        )
        self.nodes: dict[str, ARTSNode] = {}
        self.root_id = ""
        self.frontier: list[tuple[float, int, str]] = []
        self.leaf_scoring_points: dict[str, ARTSNode] = {}
        self.internal_nodes: dict[str, ARTSNode] = {}
        self.logs: list[dict[str, Any]] = _ARTSLogList(log_callback)
        self.failure_stage = ""
        self.failure_reason = ""
        self._push_order = 0
        self._score_cache: dict[str, float] = {}
        self._client_cache: dict[tuple[str, str], Any] = {}
        self._aspect_coverage_cache: dict[tuple[str, str], bool] = {}
        self._semantic_quality_cache: dict[str, dict[str, Any]] = {}
        self._semantic_redundancy_cache: dict[tuple[str, str, str], bool] = {}
        self._semantic_fallback_candidates: dict[str, list[ARTSNode]] = {}
        self._last_global_coverage_diagnostics: dict[str, Any] = {}
        self._score_cache_lock = threading.Lock()
        self._client_cache_lock = threading.Lock()

    def uses_score_policy(self) -> bool:
        return self.config.decision_policy in {"score", "hybrid"}

    def uses_semantic_policy(self) -> bool:
        return self.config.decision_policy in {"semantic", "hybrid"}

    def uses_hard_semantic_constraints(self) -> bool:
        """Return whether this cell must enforce the shared semantic feasible set.

        The semantic-only ablation removes calibration, not semantic validity.
        It therefore has to use the same atomicity/coverage/non-redundancy
        constraints and best-feasible fallback as the full hybrid method.
        """
        return self.config.hard_semantic_constraints and self.uses_semantic_policy()

    def uses_semantic_aspect_routing(self) -> bool:
        """Whether query aspects may influence local candidate selection.

        A score-only run with coverage disabled is the strict no-semantic
        ablation: candidate retention, set construction, and split decisions
        must come only from calibration-response measurements.
        """
        return self.uses_semantic_policy() or self.config.coverage_check

    def _new_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def _set_failure(self, stage: str, reason: str) -> None:
        self.failure_stage = stage
        self.failure_reason = reason
        self.logs.append({"step": "failure", "stage": stage, "reason": reason})

    def _call_json(self, prompt: str, temperature: Optional[float] = None) -> Optional[dict[str, Any]]:
        return self._call_json_with_endpoint(
            prompt=prompt,
            base_url=self.config.generator_base_url,
            api_key=self.config.generator_api_key,
            model=self.config.generator_model,
            temperature=self.config.generator_temperature if temperature is None else temperature,
        )

    def _client_for_endpoint(self, base_url: str, api_key: str) -> Any:
        key = (base_url, api_key)
        with self._client_cache_lock:
            if key not in self._client_cache:
                from project.utils import create_openai_client

                self._client_cache[key] = create_openai_client(base_url=base_url, api_key=api_key)
            return self._client_cache[key]

    def _reduced_max_tokens_from_error(self, error: Exception, current_max_tokens: int) -> Optional[int]:
        message = str(error)
        if "max_tokens" not in message and "max_completion_tokens" not in message:
            return None
        if "maximum context length" not in message and "context" not in message:
            return None
        match = re.search(r"(\d+)\s*>\s*(\d+)\s*-\s*(\d+)", message)
        if match:
            context_limit = int(match.group(2))
            input_tokens = int(match.group(3))
            available = context_limit - input_tokens
            reduced = max(128, min(current_max_tokens // 2, available - 64))
        else:
            reduced = max(128, current_max_tokens // 2)
        if reduced >= current_max_tokens:
            reduced = max(128, current_max_tokens // 2)
        if reduced < 128 or reduced == current_max_tokens:
            return None
        return reduced

    def _prompt_error_log_fields(self, json_prompt: str) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "prompt_chars": len(json_prompt),
            "prompt_excerpt": json_prompt[:4000],
        }
        if os.getenv("ARTS_LOG_FULL_PROMPT_ON_ERROR", "0").strip().lower() in {"1", "true", "yes", "on"}:
            fields["prompt"] = json_prompt
        return fields

    def _call_json_with_endpoint(
        self,
        prompt: str,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
    ) -> Optional[dict[str, Any]]:
        raw = ""
        client = self._client_for_endpoint(base_url, api_key)
        json_prompt = f"{ARTS_SYSTEM_PROMPT}\n\n{prompt}"
        max_tokens = int(self.config.max_tokens)
        for attempt in range(self.config.max_retries):
            try:
                from project.utils import chat_completion_with_recovery

                completion = chat_completion_with_recovery(
                    client,
                    retry_log=self.logs.append,
                    model=model,
                    messages=[{"role": "user", "content": json_prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body=_no_thinking_extra_body(model),
                )
            except Exception as exc:
                reduced = self._reduced_max_tokens_from_error(exc, max_tokens)
                if reduced is None or reduced == max_tokens:
                    raise
                self.logs.append(
                    {
                        "step": "max_tokens_reduced_retry",
                        "attempt": attempt + 1,
                        "model": model,
                        "old_max_tokens": max_tokens,
                        "new_max_tokens": reduced,
                        "error_excerpt": " ".join(str(exc).split())[:500],
                        **self._prompt_error_log_fields(json_prompt),
                    }
                )
                max_tokens = reduced
                continue
            raw = completion.choices[0].message.content or ""
            parsed = _json_object(raw)
            if parsed is not None:
                return parsed
            self.logs.append(
                {
                    "step": "json_parse_retry",
                    "attempt": attempt + 1,
                    "raw_excerpt": " ".join(raw.split())[:1000],
                }
            )
        return None

    def _call_calibration_response_with_endpoint(
        self,
        prompt: str,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
    ) -> tuple[str, bool, str]:
        raw = ""
        client = self._client_for_endpoint(base_url, api_key)
        json_prompt = f"{ARTS_SYSTEM_PROMPT}\n\n{prompt}"
        max_tokens = int(self.config.max_tokens)
        for attempt in range(self.config.max_retries):
            try:
                from project.utils import chat_completion_with_recovery

                completion = chat_completion_with_recovery(
                    client,
                    retry_log=self.logs.append,
                    model=model,
                    messages=[{"role": "user", "content": json_prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body=_no_thinking_extra_body(model),
                )
            except Exception as exc:
                reduced = self._reduced_max_tokens_from_error(exc, max_tokens)
                if reduced is None or reduced == max_tokens:
                    raise
                self.logs.append(
                    {
                        "step": "max_tokens_reduced_retry",
                        "attempt": attempt + 1,
                        "model": model,
                        "old_max_tokens": max_tokens,
                        "new_max_tokens": reduced,
                        "error_excerpt": " ".join(str(exc).split())[:500],
                        **self._prompt_error_log_fields(json_prompt),
                    }
                )
                max_tokens = reduced
                continue
            raw = completion.choices[0].message.content or ""
            parsed = _json_object(raw)
            if parsed is not None:
                response = str(parsed.get("response", "")).strip()
                if response:
                    return response, True, raw
            fallback = _plain_text_response(raw)
            if fallback:
                self.logs.append(
                    {
                        "step": "calibration_response_text_fallback",
                        "attempt": attempt + 1,
                        "model": model,
                        "raw_excerpt": " ".join(raw.split())[:1000],
                    }
                )
                return fallback, False, raw
            self.logs.append(
                {
                    "step": "calibration_response_parse_retry",
                    "attempt": attempt + 1,
                    "model": model,
                    "raw_excerpt": " ".join(raw.split())[:1000],
                }
            )
        return "", False, raw

    def probe_importance(self, rubric_text: str, fallback: float = 3.0) -> float:
        raw = ""
        prompt = IMPORTANCE_PROMPT.format(query=self.question, rubric_text=rubric_text)
        client = self._client_for_endpoint(self.config.generator_base_url, self.config.generator_api_key)
        from project.utils import chat_completion_with_recovery

        for attempt in range(self.config.max_retries):
            completion = chat_completion_with_recovery(
                client,
                retry_log=self.logs.append,
                model=self.config.generator_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8,
                temperature=0.0,
                extra_body=_no_thinking_extra_body(self.config.generator_model),
            )
            raw = completion.choices[0].message.content or ""
            match = re.search(r"\b([1-5])\b", raw)
            if match:
                score = float(match.group(1))
                self.logs.append(
                    {
                        "step": "importance_probe",
                        "rubric_text": rubric_text,
                        "importance_score": score,
                        "raw_excerpt": " ".join(raw.split())[:300],
                    }
                )
                return score
            self.logs.append(
                {
                    "step": "importance_probe_retry",
                    "attempt": attempt + 1,
                    "rubric_text": rubric_text,
                    "raw_excerpt": " ".join(raw.split())[:300],
                }
            )
        fallback_score = max(1.0, min(5.0, fallback))
        self.logs.append(
            {
                "step": "importance_probe_fallback",
                "rubric_text": rubric_text,
                "importance_score": fallback_score,
                "raw_excerpt": " ".join(raw.split())[:300],
            }
        )
        return fallback_score

    def priority(self, node: ARTSNode) -> float:
        return node.importance * (self.config.gamma ** node.depth)

    def push_frontier(self, node: ARTSNode) -> None:
        self._push_order += 1
        heapq.heappush(self.frontier, (-self.priority(node), self._push_order, node.id))

    def pop_frontier(self) -> Optional[ARTSNode]:
        while self.frontier:
            _, _, node_id = heapq.heappop(self.frontier)
            node = self.nodes.get(node_id)
            if node is not None and node.status == "frontier":
                return node
        return None

    def initialize(self) -> bool:
        if self.uses_score_policy() and not self.calibration_responses:
            self.calibration_responses = self.sample_calibration_responses()
            if not self.calibration_responses:
                self._set_failure("calibration_sampling", "No calibration response generator returned a valid response.")
                return False
        if self.uses_score_policy() and not self.calibration_responses:
            self._set_failure("calibration_sampling", "No calibration responses available.")
            return False

        root = ARTSNode(
            id=self._new_id(),
            text=self.config.root_text,
            depth=0,
            importance=1.0,
            local_weight=1.0,
            status="frontier",
        )
        if self.uses_semantic_policy():
            self.initialize_virtual_root(root)
        else:
            root = self.filter_candidate(root)
        if root is None:
            self._set_failure("root_filter", "Root rubric did not pass ARTS filtering.")
            return False
        self.root_id = root.id
        self.nodes[root.id] = root
        self.push_frontier(root)
        self.logs.append(
            {
                "step": "initialize",
                "root": asdict(root),
                "decision_policy": self.config.decision_policy,
                "calibration_count": len(self.calibration_responses),
                "calibration_response_scope": "query_level_shared_across_all_tree_nodes",
            }
        )
        return True

    def initialize_virtual_root(self, root: ARTSNode) -> None:
        quality = {
            "concrete": 1.0,
            "verifiable": 1.0,
            "unambiguous": 1.0,
            "task_specific": 1.0,
            "distinctive": 1.0,
            "leaf_quality": 0.0,
            "atomic": False,
            "independent_requirement_count": 0,
            "atomicity_reason": "Virtual root is structural and can never be a leaf.",
            "substantive": True,
            "format_only": False,
            "pass": True,
            "reason": "Virtual search root; candidate filtering starts at generated children.",
        }
        root.scorability = 1.0
        root.scores = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        root.metadata["semantic_quality"] = quality
        root.metadata["semantic_distinctive"] = 1.0
        root.metadata["semantic_leaf_quality"] = 0.0
        root.metadata["semantic_atomic"] = False
        root.metadata["independent_requirement_count"] = 0
        root.metadata["semantic_substantive"] = True
        root.metadata["semantic_format_only"] = False
        root.metadata["score_confidence_penalty"] = 1.0
        root.metadata["virtual_root"] = True

    def sample_calibration_responses(self) -> list[str]:
        requested_count = max(0, self.config.k_calibration)
        all_generators = self.config.calibration_response_generators
        generators = all_generators[:requested_count]
        if not generators or requested_count == 0:
            return []

        def sample_one(
            index: int,
            generator: dict[str, Any],
            *,
            refill_source_index: Optional[int] = None,
        ) -> tuple[int, str, dict[str, Any]]:
            model = str(generator.get("model") or self.config.generator_model)
            base_url = str(generator.get("base_url") or self.config.generator_base_url)
            api_key = str(generator.get("api_key") or self.config.generator_api_key)
            quality = str(generator.get("quality", "medium")).strip().lower()
            quality_instruction = str(generator.get("quality_instruction") or QUALITY_INSTRUCTIONS.get(quality, QUALITY_INSTRUCTIONS["medium"]))
            prompt_modifier = generator.get("prompt_modifier")
            prompt_modifier_text = "None" if prompt_modifier in (None, "") else str(prompt_modifier)
            calibration_question = get_calibration_response_prompt(
                self.question,
                None if prompt_modifier in (None, "") else str(prompt_modifier),
            )
            try:
                response, parsed_json, raw_response = self._call_calibration_response_with_endpoint(
                    prompt=CALIBRATION_RESPONSE_PROMPT.format(
                        calibration_question=calibration_question,
                        quality=quality,
                        quality_instruction=quality_instruction,
                        prompt_modifier=prompt_modifier_text,
                    ),
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    temperature=float(generator.get("temperature", self.config.generator_temperature)),
                )
            except Exception as exc:
                return index, "", {
                    "step": "calibration_response_error",
                    "index": index,
                    "model": model,
                    "base_url": base_url,
                    "quality": quality,
                    "prompt_modifier": prompt_modifier,
                    "refill_source_index": refill_source_index,
                    "error_type": type(exc).__name__,
                    "error_excerpt": " ".join(str(exc).split())[:1000],
                }
            return index, response, (
                {
                    "step": "calibration_response",
                    "index": index,
                    "model": model,
                    "quality": quality,
                    "prompt_modifier": prompt_modifier,
                    "base_url": base_url,
                    "refill_source_index": refill_source_index,
                    "ok": bool(response),
                    "parsed_json": parsed_json,
                    "calibration_prompt_excerpt": " ".join(calibration_question.split())[:500],
                    "response_excerpt": response[:300],
                    "raw_excerpt": " ".join(raw_response.split())[:500],
                }
            )

        def sample_many(
            indexed_generators: list[tuple[int, dict[str, Any], Optional[int]]],
        ) -> list[tuple[int, str, dict[str, Any]]]:
            if not indexed_generators:
                return []
            max_workers = min(len(indexed_generators), max(1, self.config.calibration_response_concurrency))
            if max_workers == 1:
                return [
                    sample_one(index, generator, refill_source_index=refill_source_index)
                    for index, generator, refill_source_index in indexed_generators
                ]
            results: list[tuple[int, str, dict[str, Any]]] = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        sample_one,
                        index,
                        generator,
                        refill_source_index=refill_source_index,
                    ): index
                    for index, generator, refill_source_index in indexed_generators
                }
                for future in as_completed(futures):
                    results.append(future.result())
            return results

        indexed_results = sample_many(
            [(index, generator, None) for index, generator in enumerate(generators)]
        )

        successful_primary = [
            (index, generators[index])
            for index, response, _log_entry in sorted(indexed_results, key=lambda item: item[0])
            if response
        ]
        missing_count = requested_count - len(successful_primary)
        if missing_count > 0:
            reserve_generators = [
                (index, generator, None)
                for index, generator in enumerate(all_generators[requested_count:], start=requested_count)
            ]
            reserve_results = sample_many(reserve_generators[:missing_count])
            indexed_results.extend(reserve_results)
            successful_reserve_count = sum(1 for _index, response, _log_entry in reserve_results if response)
            remaining_missing = missing_count - successful_reserve_count
            successful_generators = successful_primary + [
                (index, all_generators[index])
                for index, response, _log_entry in reserve_results
                if response
            ]
            next_index = len(all_generators)
            refill_round = 0
            while (
                remaining_missing > 0
                and successful_generators
                and refill_round < max(0, self.config.max_calibration_refill_rounds)
            ):
                refill_round += 1
                refill_generators: list[tuple[int, dict[str, Any], Optional[int]]] = []
                for refill_offset in range(remaining_missing):
                    source_index, source_generator = successful_generators[
                        (refill_round - 1 + refill_offset) % len(successful_generators)
                    ]
                    refill_generator = dict(source_generator)
                    original_modifier = refill_generator.get("prompt_modifier")
                    distinct_modifier = (
                        "Generate a distinct alternative calibration response "
                        f"(refill round {refill_round}, item {refill_offset + 1})."
                    )
                    refill_generator["prompt_modifier"] = (
                        f"{original_modifier}; {distinct_modifier}" if original_modifier else distinct_modifier
                    )
                    refill_generators.append((next_index, refill_generator, source_index))
                    next_index += 1
                refill_results = sample_many(refill_generators)
                indexed_results.extend(refill_results)
                remaining_missing -= sum(
                    1 for _index, response, _log_entry in refill_results if response
                )

        responses = []
        for _index, response, log_entry in sorted(indexed_results, key=lambda item: item[0]):
            if response:
                responses.append(response)
            self.logs.append(log_entry)
        self.logs.append(
            {
                "step": "calibration_sampling_summary",
                "requested": requested_count,
                "returned": min(len(responses), requested_count),
                "failed": sum(1 for _index, response, _log_entry in indexed_results if not response),
                "refill_attempted": sum(
                    1 for _index, _response, log_entry in indexed_results if log_entry.get("refill_source_index") is not None
                ),
            }
        )
        return responses[:requested_count]

    def llm_score(self, criterion: str, response: str, judge_config: dict[str, Any]) -> float:
        key = _score_cache_key(self.question, response, criterion, judge_config)
        with self._score_cache_lock:
            cached = self._score_cache.get(key)
        if cached is not None:
            return cached

        from project.utils import chat_completion_with_recovery, clip_text_middle, create_openai_client

        base_url = str(judge_config.get("base_url") or self.config.generator_base_url)
        api_key = str(judge_config.get("api_key") or self.config.generator_api_key)
        client = create_openai_client(
            base_url=base_url,
            api_key=api_key,
        )
        question_for_judge = clip_text_middle(
            self.question,
            int(os.getenv("ARTS_JUDGE_MAX_QUESTION_CHARS", "6000")),
        )
        response_for_judge = clip_text_middle(
            response,
            int(os.getenv("ARTS_JUDGE_MAX_RESPONSE_CHARS", "12000")),
        )
        criterion_for_judge = clip_text_middle(
            criterion,
            int(os.getenv("ARTS_JUDGE_MAX_CRITERION_CHARS", "4000")),
        )
        prompt = (
            "You are evaluating whether a response satisfies a specific evaluation criterion.\n\n"
            f"Question: {question_for_judge}\n\n"
            f"Response: {response_for_judge}\n\n"
            f"Criterion: {criterion_for_judge}\n\n"
            'Does the response satisfy this criterion? Answer with ONLY "YES" or "NO" in uppercase, nothing else.'
        )
        try:
            completion = chat_completion_with_recovery(
                client,
                retry_log=self.logs.append,
                model=str(judge_config["model"]),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=int(judge_config.get("max_tokens", 1)),
                temperature=float(judge_config.get("temperature", 0.0)),
                logprobs=True,
                top_logprobs=int(judge_config.get("top_logprobs", 20)),
                extra_body=_no_thinking_extra_body(str(judge_config["model"])),
            )
            result = _yes_no_probability(completion)
        except Exception:
            completion = chat_completion_with_recovery(
                client,
                retry_log=self.logs.append,
                model=str(judge_config["model"]),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max(2, int(judge_config.get("max_tokens", 1))),
                temperature=float(judge_config.get("temperature", 0.0)),
                extra_body=_no_thinking_extra_body(str(judge_config["model"])),
            )
            result = _yes_no_from_text(completion.choices[0].message.content or "")
        with self._score_cache_lock:
            self._score_cache[key] = result
        return result

    def score_matrix(self, criterion: str) -> list[list[float]]:
        judge_configs = self.config.calibration_judges or self.config.judge_configs
        if not judge_configs:
            raise ValueError("ARTSConfig.judge_configs must contain at least one judge configuration.")
        rows = len(self.calibration_responses)
        cols = len(judge_configs)
        scores = [[0.0 for _ in range(cols)] for _ in range(rows)]
        tasks = [
            (row_idx, col_idx, response, judge_config)
            for row_idx, response in enumerate(self.calibration_responses)
            for col_idx, judge_config in enumerate(judge_configs)
        ]
        max_workers = min(len(tasks), max(1, self.config.calibration_judge_concurrency))
        if max_workers == 1:
            for row_idx, col_idx, response, judge_config in tasks:
                scores[row_idx][col_idx] = self.llm_score(criterion, response, judge_config)
            return scores
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.llm_score, criterion, response, judge_config): (row_idx, col_idx)
                for row_idx, col_idx, response, judge_config in tasks
            }
            for future in as_completed(futures):
                row_idx, col_idx = futures[future]
                scores[row_idx][col_idx] = future.result()
        return scores

    def attach_calibration_diagnostics(self, candidate: ARTSNode) -> ARTSNode:
        """Measure one semantically feasible candidate on the shared responses.

        The semantic filter has already admitted the candidate. These scores
        are used to optimize among semantically valid local structures; they
        never make an invalid candidate valid.
        """
        judge_scores = self.score_matrix(candidate.text)
        within_response_vars = [_variance(row) for row in judge_scores]
        judge_response_vars = judge_response_variances(judge_scores)
        candidate.scorability = 1.0 / (1.0 + _mean(within_response_vars))
        candidate.scores = [_mean(row) for row in judge_scores]
        candidate.metadata["judge_scores"] = judge_scores
        candidate.metadata["within_response_vars"] = within_response_vars
        candidate.metadata["judge_response_variances"] = judge_response_vars
        candidate.metadata["distinct"] = compute_distinct_matrix(judge_scores)
        candidate.metadata["score_confidence_penalty"] = 1.0
        candidate.metadata["calibration_role"] = "semantic_feasible_set_optimization"
        self.logs.append(
            {
                "step": "calibration_diagnostics",
                "node_id": candidate.id,
                "text": candidate.text,
                "scorability": candidate.scorability,
                "scores": candidate.scores,
                "distinct": candidate.metadata["distinct"],
                "role": "semantic_feasible_set_optimization",
                "judge_models": [
                    str(judge_config.get("model", ""))
                    for judge_config in (self.config.calibration_judges or self.config.judge_configs)
                ],
                "judge_scores": judge_scores,
                "within_response_vars": within_response_vars,
                "judge_response_variances": judge_response_vars,
            }
        )
        return candidate

    def scorability_filter(self, candidate: ARTSNode) -> Optional[ARTSNode]:
        attempts = 0
        current = candidate
        best = candidate
        best_scorability = -1.0
        while attempts <= self.config.max_regen_attempts:
            judge_scores = self.score_matrix(current.text)
            within_response_vars = [_variance(row) for row in judge_scores]
            judge_response_vars = judge_response_variances(judge_scores)
            current.scorability = 1.0 / (1.0 + _mean(within_response_vars))
            current.scores = [_mean(row) for row in judge_scores]
            current.metadata["judge_scores"] = judge_scores
            current.metadata["within_response_vars"] = within_response_vars
            current.metadata["judge_response_variances"] = judge_response_vars
            current.metadata["distinct"] = compute_distinct_matrix(judge_scores)
            self.logs.append(
                {
                    "step": "scorability",
                    "node_id": current.id,
                    "text": current.text,
                    "attempt": attempts + 1,
                    "scorability": current.scorability,
                    "scores": current.scores,
                    "distinct": current.metadata["distinct"],
                    "judge_models": [
                        str(judge_config.get("model", ""))
                        for judge_config in (self.config.calibration_judges or self.config.judge_configs)
                    ],
                    "judge_scores": judge_scores,
                    "within_response_vars": within_response_vars,
                    "judge_response_variances": judge_response_vars,
                }
            )
            if current.scorability >= self.config.tau_scor:
                current.metadata["score_confidence_penalty"] = 1.0
                current.metadata["low_scorability_retained"] = False
                return current
            if float(current.scorability or 0.0) > best_scorability:
                best = current
                best_scorability = float(current.scorability or 0.0)
            if attempts < self.config.max_regen_attempts:
                regenerated = self.regenerate_node(
                    current,
                    hint=(
                        "Previous version was ambiguous (judges disagreed). "
                        "Rewrite as a more concrete, verifiable criterion."
                    ),
                )
                if regenerated is None:
                    if (
                        self.config.decision_policy == "hybrid"
                        or (
                            self.config.decision_policy == "score"
                            and candidate.parent_id is None
                        )
                    ):
                        break
                    return None
                if self.config.decision_policy == "hybrid":
                    regenerated = self.semantic_filter(regenerated)
                    if regenerated is None:
                        break
                current = regenerated
            attempts += 1
        if self.config.decision_policy == "hybrid" and self.should_retain_low_scorability(best):
            penalty = max(0.25, float(best.scorability or 0.0) / max(self.config.tau_scor, 1e-6))
            best.metadata["score_confidence_penalty"] = penalty
            best.metadata["low_scorability_retained"] = True
            best.metadata["low_scorability_retain_reason"] = "semantic_pass_candidate_retained_after_rewrite"
            self.logs.append(
                {
                    "step": "low_scorability_retained",
                    "node_id": best.id,
                    "text": best.text,
                    "scorability": best.scorability,
                    "tau_scor": self.config.tau_scor,
                    "score_confidence_penalty": penalty,
                    "semantic_quality": best.metadata.get("semantic_quality", {}),
                }
            )
            return best
        # The root is a routing anchor rather than a final scoring leaf.  In
        # the score-only ablation, rejecting it makes the whole sample fail
        # before any calibration-driven child decision can be tested.  Keep
        # the most stable regenerated root after exhausting retries, while
        # continuing to enforce tau_scor strictly for every generated child.
        if (
            self.config.decision_policy == "score"
            and candidate.parent_id is None
            and best_scorability >= 0.0
        ):
            penalty = max(
                0.25,
                float(best.scorability or 0.0) / max(self.config.tau_scor, 1e-6),
            )
            best.metadata["score_confidence_penalty"] = penalty
            best.metadata["low_scorability_retained"] = True
            best.metadata["low_scorability_retain_reason"] = (
                "score_only_root_anchor_after_rewrite_exhausted"
            )
            self.logs.append(
                {
                    "step": "score_only_root_fallback",
                    "node_id": best.id,
                    "text": best.text,
                    "scorability": best.scorability,
                    "tau_scor": self.config.tau_scor,
                    "score_confidence_penalty": penalty,
                    "reason": "root_is_routing_anchor_not_final_leaf",
                }
            )
            return best
        return None

    def should_retain_low_scorability(self, node: ARTSNode) -> bool:
        if self.config.decision_policy != "hybrid":
            return False
        quality = node.metadata.get("semantic_quality")
        if not isinstance(quality, dict):
            return False
        return bool(quality.get("pass"))

    def semantic_quality(self, criterion: str) -> dict[str, Any]:
        cached = self._semantic_quality_cache.get(criterion)
        if cached is not None:
            return cached
        prompt = SEMANTIC_RUBRIC_QUALITY_PROMPT.format(
            query=self.question,
            criterion=criterion,
        )

        required_scores = (
            "concrete",
            "verifiable",
            "unambiguous",
            "task_specific",
            "distinctive",
            "leaf_quality",
        )

        def inconsistent_reason(candidate: Optional[dict[str, Any]]) -> str:
            if not isinstance(candidate, dict):
                return "missing_or_invalid_json"
            parsed: dict[str, float] = {}
            for name in required_scores:
                try:
                    value = float(candidate[name])
                except (KeyError, TypeError, ValueError):
                    return f"missing_or_non_numeric:{name}"
                if not 0.0 <= value <= 1.0:
                    return f"out_of_range:{name}={value}"
                parsed[name] = value
            if candidate.get("pass") is True:
                core = [parsed[name] for name in ("concrete", "verifiable", "unambiguous", "task_specific")]
                if min(core) < 0.6:
                    return "pass_true_but_core_semantic_score_below_0.6"
                if max(parsed.values()) == 0.0:
                    return "pass_true_but_all_scores_zero"
                if (
                    candidate.get("atomic") is True
                    and candidate.get("explicit_query_requirement") is True
                    and parsed["leaf_quality"] < 0.6
                ):
                    return "explicit_atomic_requirement_but_leaf_quality_below_0.6"
                if candidate.get("format_only") is True and "explicit_query_requirement" not in candidate:
                    return "format_only_evaluation_missing_explicit_query_requirement"
            return ""

        data: Optional[dict[str, Any]] = None
        inconsistency = ""
        for attempt in range(1, 4):
            attempt_prompt = prompt
            if attempt > 1:
                attempt_prompt += (
                    "\n\nYour previous evaluation was internally inconsistent: "
                    f"{inconsistency}. Re-evaluate the criterion from scratch. "
                    "Do not copy placeholder zeros. If pass=true, every core "
                    "semantic score must be at least 0.6 and the explanation "
                    "must support those scores. Return only corrected JSON."
                )
            data = self._call_json(attempt_prompt, temperature=0.0)
            inconsistency = inconsistent_reason(data)
            if not inconsistency:
                break
            self.logs.append(
                {
                    "step": "semantic_quality_consistency_retry",
                    "criterion": criterion,
                    "attempt": attempt,
                    "reason": inconsistency,
                    "raw": data,
                }
            )
        if inconsistency and isinstance(data, dict):
            # A self-contradictory semantic evaluation must never admit a
            # candidate into the hard feasible set.  Mark it invalid rather
            # than trusting pass=true or inventing scores in code.
            data = {**data, "pass": False}
            self.logs.append(
                {
                    "step": "semantic_quality_consistency_rejected",
                    "criterion": criterion,
                    "reason": inconsistency,
                }
            )
        if data is None:
            data = {
                "concrete": 0.0,
                "verifiable": 0.0,
                "unambiguous": 0.0,
                "task_specific": 0.0,
                "distinctive": 0.0,
                "leaf_quality": 0.0,
                "atomic": False,
                "independent_requirement_count": 0,
                "atomicity_reason": "Invalid semantic quality JSON.",
                "substantive": False,
                "format_only": False,
                "explicit_query_requirement": False,
                "pass": False,
                "reason": "Invalid semantic quality JSON.",
            }
        def field_score(name: str) -> float:
            try:
                return _clamp01(float(data.get(name, 0.0) or 0.0))
            except (TypeError, ValueError):
                return 0.0
        try:
            independent_requirement_count = max(
                0,
                int(data.get("independent_requirement_count", 1) or 0),
            )
        except (TypeError, ValueError):
            independent_requirement_count = 0
        quality = {
            "concrete": field_score("concrete"),
            "verifiable": field_score("verifiable"),
            "unambiguous": field_score("unambiguous"),
            "task_specific": field_score("task_specific"),
            "distinctive": field_score("distinctive"),
            "leaf_quality": field_score("leaf_quality"),
            "atomic": bool(data.get("atomic") is True),
            "independent_requirement_count": independent_requirement_count,
            "atomicity_reason": str(data.get("atomicity_reason", "")),
            "substantive": bool(data.get("substantive") is True),
            "format_only": bool(data.get("format_only") is True),
            "explicit_query_requirement": bool(data.get("explicit_query_requirement") is True),
            "pass": bool(data.get("pass") is True),
            "reason": str(data.get("reason", "")),
        }
        self._semantic_quality_cache[criterion] = quality
        self.logs.append(
            {
                "step": "semantic_quality",
                "criterion": criterion,
                **quality,
            }
        )
        return quality

    def semantic_filter(self, candidate: ARTSNode) -> Optional[ARTSNode]:
        quality = self.semantic_quality(candidate.text)
        candidate.metadata["semantic_quality"] = quality
        candidate.scorability = _mean(
            [
                float(quality["concrete"]),
                float(quality["verifiable"]),
                float(quality["unambiguous"]),
                float(quality["task_specific"]),
            ]
        )
        candidate.metadata["semantic_distinctive"] = float(quality["distinctive"])
        candidate.metadata["semantic_leaf_quality"] = float(quality["leaf_quality"])
        candidate.metadata["semantic_atomic"] = bool(quality["atomic"])
        candidate.metadata["independent_requirement_count"] = int(quality["independent_requirement_count"])
        candidate.metadata["atomicity_reason"] = str(quality["atomicity_reason"])
        candidate.metadata["semantic_substantive"] = bool(quality["substantive"])
        candidate.metadata["semantic_format_only"] = bool(quality["format_only"])
        candidate.metadata["semantic_explicit_query_requirement"] = bool(
            quality["explicit_query_requirement"]
        )
        candidate.scores = [
            float(quality["concrete"]),
            float(quality["verifiable"]),
            float(quality["unambiguous"]),
            float(quality["task_specific"]),
            float(quality["distinctive"]),
            float(quality["leaf_quality"]),
        ]
        # In hybrid semantic-first search, tau_scor belongs to calibration
        # diagnostics/ranking; it must not erase a semantically valid explicit
        # requirement before calibration is attached. The semantic evaluator's
        # pass contract already enforces a 0.6 floor on all core fields.
        semantic_scorability_floor = 0.6 if self.uses_hard_semantic_constraints() else self.config.tau_scor
        passed = (
            quality["pass"]
            and candidate.scorability >= semantic_scorability_floor
            and float(quality["distinctive"]) >= self.config.low_distinct_threshold
            and (
                not quality["format_only"]
                or quality["substantive"]
                or quality["explicit_query_requirement"]
            )
        )
        self.logs.append(
            {
                "step": "semantic_filter",
                "node_id": candidate.id,
                "text": candidate.text,
                "passed": passed,
                "semantic_scorability": candidate.scorability,
                "semantic_distinctive": quality["distinctive"],
                "semantic_leaf_quality": quality["leaf_quality"],
                "semantic_atomic": quality["atomic"],
                "independent_requirement_count": quality["independent_requirement_count"],
                "atomicity_reason": quality["atomicity_reason"],
                "semantic_substantive": quality["substantive"],
                "semantic_format_only": quality["format_only"],
                "semantic_explicit_query_requirement": quality["explicit_query_requirement"],
                "semantic_scorability_floor": semantic_scorability_floor,
                "reason": quality["reason"],
            }
        )
        return candidate if passed else None

    def filter_candidate(self, candidate: ARTSNode) -> Optional[ARTSNode]:
        current: Optional[ARTSNode] = candidate
        if self.uses_semantic_policy():
            current = self.semantic_filter(candidate)
            if current is None:
                return None
        if self.config.decision_policy == "hybrid":
            # Semantic admissibility is the hard feasible set. Every admitted
            # candidate is then measured by both calibration judges on the same
            # query-level responses for local constrained optimization.
            current = self.attach_calibration_diagnostics(current)
        elif self.uses_score_policy():
            current = self.scorability_filter(current)
        return current

    def regenerate_node(self, node: ARTSNode, hint: str) -> Optional[ARTSNode]:
        data = self._call_json(REGENERATE_PROMPT.format(criterion=node.text, hint=hint))
        text = str(data.get("criterion", "")).strip() if data else ""
        if not text:
            self.logs.append({"step": "regenerate_failed", "node_id": node.id, "reason": "Missing criterion text."})
            return None
        return ARTSNode(
            id=node.id,
            text=text,
            depth=node.depth,
            parent_id=node.parent_id,
            importance=node.importance,
            local_weight=node.local_weight,
            metadata={**node.metadata, "regenerated_from": node.text},
        )

    def generator(self, node: ARTSNode) -> list[ARTSNode]:
        prompt_mode = "root_query" if node.parent_id is None else "parent_rubric"
        prompt_template = ROOT_GENERATOR_PROMPT if node.parent_id is None else GENERATOR_PROMPT
        data = self._call_json(
            prompt_template.format(
                query=self.question,
                parent_text=node.text,
                n_candidates=self.config.n_candidates_per_node,
            )
        )
        if data is None:
            return []
        raw = data.get("sub_rubrics")
        if not isinstance(raw, list):
            self.logs.append({"step": "invalid_generator_output", "node_id": node.id, "reason": "sub_rubrics was not a list."})
            return []

        candidate_inputs = []
        seen = set()
        for item in raw[: self.config.n_candidates_per_node]:
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("rubric") or item.get("criterion") or "").strip()
                generator_weight = _positive_float(item.get("weight"), default=3.0)
            else:
                text = str(item).strip()
                generator_weight = 3.0
            normalized = text.lower()
            if not text or normalized in seen:
                continue
            seen.add(normalized)
            candidate_inputs.append((text, generator_weight, self._new_id()))

        def process_candidate(item: tuple[str, float, str]) -> Optional[ARTSNode]:
            text, generator_weight, candidate_id = item
            importance_score = self.probe_importance(text, fallback=generator_weight)
            candidate = ARTSNode(
                id=candidate_id,
                text=text,
                depth=node.depth + 1,
                parent_id=node.id,
                importance=importance_score,
                local_weight=importance_score,
                metadata={
                    "llm_weight": importance_score,
                    "importance_score": importance_score,
                    "generator_weight": generator_weight,
                    "weight_basis": "direct_independent_importance_1_to_5",
                },
            )
            return self.filter_candidate(candidate)

        filtered = []
        max_workers = min(len(candidate_inputs), max(1, self.config.candidate_concurrency))
        if max_workers == 1:
            filtered = [candidate for candidate in (process_candidate(item) for item in candidate_inputs) if candidate is not None]
        elif candidate_inputs:
            results: list[tuple[int, ARTSNode]] = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_candidate, item): index
                    for index, item in enumerate(candidate_inputs)
                }
                for future in as_completed(futures):
                    candidate = future.result()
                    if candidate is not None:
                        results.append((futures[future], candidate))
            filtered = [candidate for _index, candidate in sorted(results, key=lambda item: item[0])]
        self.logs.append(
            {
                "step": "generator",
                "node_id": node.id,
                "parent_text": node.text,
                "prompt_mode": prompt_mode,
                "raw_candidates": [str(item).strip() for item in raw[: self.config.n_candidates_per_node]],
                "raw_count": len(raw),
                "filtered_count": len(filtered),
                "filtered_candidates": [
                    {
                        "node_id": child.id,
                        "text": child.text,
                        "scorability": child.scorability,
                        "scores": child.scores,
                        "distinct": node_distinct(child),
                        "llm_weight": child.metadata.get("llm_weight", 1.0),
                        "importance_score": child.metadata.get("importance_score"),
                        "generator_weight": child.metadata.get("generator_weight"),
                    }
                    for child in filtered
                ],
            }
        )
        if self.uses_semantic_policy() and filtered:
            self._record_semantic_fallback_candidates(node.id, filtered)
        return filtered

    def _record_semantic_fallback_candidates(self, node_id: str, candidates: list[ARTSNode]) -> None:
        """Retain semantic-admissible candidates for a later fail-safe."""
        cache = getattr(self, "_semantic_fallback_candidates", None)
        if cache is None:
            cache = {}
            self._semantic_fallback_candidates = cache
        existing = {candidate.id: candidate for candidate in cache.get(node_id, [])}
        for candidate in candidates:
            existing[candidate.id] = candidate
        cache[node_id] = list(existing.values())

    def _semantic_fallback_leaf_ok(self, candidate: ARTSNode) -> bool:
        """Check individual semantic feasibility without requiring set coverage."""
        quality = candidate.metadata.get("semantic_quality", {})
        if not isinstance(quality, dict) or quality.get("pass") is not True:
            return False
        if not semantic_atomic(candidate):
            return False
        if int(candidate.metadata.get("independent_requirement_count", 1) or 1) > 1:
            return False
        explicit = semantic_explicit_query_requirement(candidate)
        if not semantic_substantive(candidate) and not explicit:
            return False
        if semantic_format_only(candidate) and not explicit:
            return False
        return True

    def best_semantic_fallback_candidate(self, node: ARTSNode) -> Optional[ARTSNode]:
        """Return the best atomic rubric already admitted by the semantic gate.

        Semantic quality is the primary order.  Calibration is only a
        deterministic tie-break inside that individually feasible set.
        """
        cache = getattr(self, "_semantic_fallback_candidates", {})
        candidates = [
            candidate
            for candidate in cache.get(node.id, [])
            if self._semantic_fallback_leaf_ok(candidate)
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda candidate: (
                semantic_candidate_score(candidate),
                semantic_leaf_quality(candidate),
                hybrid_score_selection_priority(candidate),
                float(candidate.importance),
                candidate.id,
            ),
        )

    def apply_best_semantic_fallback(self, node: ARTSNode, trigger_reason: str) -> Optional[ARTSNode]:
        """Commit one best available atomic child without failing the sample."""
        if not (
            self.config.semantic_failure_fallback
            and self.uses_hard_semantic_constraints()
        ):
            return None
        candidate = self.best_semantic_fallback_candidate(node)
        if candidate is None:
            self.logs.append(
                {
                    "step": "semantic_best_feasible_fallback_unavailable",
                    "node_id": node.id,
                    "reason": trigger_reason,
                }
            )
            return None

        if node.depth < self.config.max_depth:
            node.status = "internal"
            self.internal_nodes[node.id] = node
            self.assign_child_weights(node, [candidate])
            candidate.parent_id = node.id
            candidate.depth = node.depth + 1
            candidate.status = "leaf"
            self.nodes[candidate.id] = candidate
            if candidate.id not in node.children:
                node.children.append(candidate.id)
            self.leaf_scoring_points[candidate.id] = candidate
        else:
            parent = self.nodes.get(node.parent_id or "")
            if parent is None:
                return None
            parent.children = [child_id for child_id in parent.children if child_id != node.id]
            node.status = "removed"
            self.internal_nodes.pop(node.id, None)
            self.leaf_scoring_points.pop(node.id, None)
            self.assign_child_weights(parent, [candidate])
            candidate.parent_id = parent.id
            candidate.depth = node.depth
            candidate.status = "leaf"
            self.nodes[candidate.id] = candidate
            parent.children.append(candidate.id)
            self.leaf_scoring_points[candidate.id] = candidate

        candidate.metadata["semantic_failure_fallback"] = True
        candidate.metadata["semantic_failure_fallback_from"] = node.id
        self.logs.append(
            {
                "step": "semantic_best_feasible_fallback",
                "node_id": node.id,
                "node_text": node.text,
                "fallback_leaf_id": candidate.id,
                "fallback_leaf_text": candidate.text,
                "semantic_score": semantic_candidate_score(candidate),
                "calibration_tiebreak": hybrid_score_selection_priority(candidate),
                "reason": trigger_reason,
            }
        )
        return candidate

    def precompute_aspects(self, node: ARTSNode) -> list[str]:
        data = self._call_json(
            ASPECTS_PROMPT.format(
                query=self.question,
                parent_text=node.text,
            ),
            temperature=0.0,
        )
        raw_aspects = data.get("aspects", []) if data else []
        raw_clean = [str(item).strip() for item in raw_aspects if str(item).strip()] if isinstance(raw_aspects, list) else []
        aspects = []
        removed_aspects: list[dict[str, str]] = []
        seen = set()
        for aspect in raw_clean:
            pollution_reason = self.aspect_pollution_reason(aspect)
            if pollution_reason:
                removed_aspects.append({"aspect": aspect, "reason": pollution_reason})
                continue
            normalized = " ".join(aspect.casefold().split()).rstrip(". ;:")
            if normalized in seen:
                continue
            seen.add(normalized)
            aspects.append(aspect)
        self.logs.append(
            {
                "step": "precompute_aspects",
                "node_id": node.id,
                "node_text": node.text,
                "aspects": aspects,
                "raw_aspects": raw_clean,
                "removed_aspects": removed_aspects,
            }
        )
        return aspects

    def aspect_pollution_reason(self, aspect: str) -> str:
        """Reject evaluator-schema leakage and source-data descriptors.

        The JSON-only system prompt is transport metadata, not a user-task
        rubric.  Local GLM models occasionally copy it into the extracted task
        aspects despite explicit prompt instructions.  Keep such an aspect only
        when the user's own query independently asks for that format.
        """
        normalized = " ".join(str(aspect).casefold().split())
        query = " ".join(self.question.casefold().split())
        source_prefixes = (
            "the input text ",
            "input text ",
            "the source text ",
            "source text ",
            "the source material ",
            "source material ",
            "the provided text ",
            "provided text ",
            "the original text says ",
            "the original text states ",
            "the original text mentions ",
            "the original text is about ",
        )
        if normalized.startswith(source_prefixes):
            return "source_material_descriptor_not_requirement"

        evaluator_rules = (
            (("valid json", "output only json", "json object"), ("json",)),
            (("markdown fence", "markdown fences"), ("markdown", "fence")),
            (("backslash", "json escape"), ("backslash", "escape")),
            (("latex delimiter", "latex delimiters"), ("latex",)),
            (("plain text formula", "formulas as plain text"), ("formula", "plain text")),
        )
        for aspect_markers, query_markers in evaluator_rules:
            if not any(marker in normalized for marker in aspect_markers):
                continue
            if any(marker in query for marker in query_markers):
                return ""
            return "evaluator_json_transport_instruction"
        return ""

    def candidate_covers_aspect(self, candidate: ARTSNode, aspect: str) -> bool:
        key = (candidate.text, aspect)
        if key in self._aspect_coverage_cache:
            return self._aspect_coverage_cache[key]
        data = self._call_json(
            ASPECT_COVERAGE_PROMPT.format(
                query=self.question,
                aspect=aspect,
                rubric_text=candidate.text,
            ),
            temperature=0.0,
        )
        covered = bool(data and data.get("covered") is True)
        self._aspect_coverage_cache[key] = covered
        self.logs.append(
            {
                "step": "aspect_coverage_judge",
                "candidate_id": candidate.id,
                "aspect": aspect,
                "covered": covered,
            }
        )
        return covered

    def coverage_score(self, selected: list[ARTSNode], aspects: list[str]) -> float:
        if not aspects:
            return 1.0 if selected else 0.0
        covered_aspects = set()
        for candidate in selected:
            for aspect in aspects:
                if self.candidate_covers_aspect(candidate, aspect):
                    covered_aspects.add(aspect)
        return len(covered_aspects) / len(aspects)

    def uncovered_aspects(self, selected: list[ARTSNode], aspects: list[str]) -> list[str]:
        if not aspects:
            return []
        uncovered = []
        for aspect in aspects:
            if not any(self.candidate_covers_aspect(candidate, aspect) for candidate in selected):
                uncovered.append(aspect)
        return uncovered

    def redundancy_score(self, selected: list[ARTSNode]) -> float:
        if self.uses_semantic_policy():
            return self.semantic_redundancy_score(selected)
        if len(selected) < 2:
            return 0.0
        best = 0.0
        for index, left in enumerate(selected):
            for right in selected[index + 1 :]:
                best = max(best, max(0.0, spearman_rank_correlation(left.scores, right.scores)))
        return min(1.0, best)

    def semantic_pair_redundant(self, parent: ARTSNode, left: ARTSNode, right: ARTSNode) -> bool:
        key = (parent.id, left.text, right.text)
        reverse_key = (parent.id, right.text, left.text)
        if key in self._semantic_redundancy_cache:
            return self._semantic_redundancy_cache[key]
        if reverse_key in self._semantic_redundancy_cache:
            return self._semantic_redundancy_cache[reverse_key]
        data = self._call_json(
            SEMANTIC_NONREDUNDANCY_PROMPT.format(
                query=self.question,
                parent_text=parent.text,
                left=left.text,
                right=right.text,
            ),
            temperature=0.0,
        )
        redundant = bool(data and data.get("redundant") is True)
        self._semantic_redundancy_cache[key] = redundant
        self.logs.append(
            {
                "step": "semantic_nonredundancy",
                "parent_id": parent.id,
                "left_id": left.id,
                "right_id": right.id,
                "redundant": redundant,
                "reason": data.get("reason", "") if data else "Invalid semantic nonredundancy JSON.",
            }
        )
        return redundant

    def semantic_redundancy_score(self, selected: list[ARTSNode], parent: Optional[ARTSNode] = None) -> float:
        if len(selected) < 2:
            return 0.0
        parent_node = parent or self.nodes.get(selected[0].parent_id or "", None)
        if parent_node is None:
            return max(_text_jaccard(left.text, right.text) for index, left in enumerate(selected) for right in selected[index + 1 :])
        redundant_pairs = 0
        total_pairs = 0
        for index, left in enumerate(selected):
            for right in selected[index + 1 :]:
                total_pairs += 1
                if self.semantic_pair_redundant(parent_node, left, right):
                    redundant_pairs += 1
        return redundant_pairs / total_pairs if total_pairs else 0.0

    def aggregated_distinct(self, selected: list[ARTSNode]) -> float:
        if not selected:
            return 0.0
        return compute_distinct_matrix(aggregate_node_score_matrices(selected, self.config.beta))

    def coverage_constraint_ok(self, selected: list[ARTSNode], aspects: list[str]) -> bool:
        if not self.config.coverage_check:
            return bool(selected)
        if self.uses_semantic_policy() and selected:
            parent = self.nodes.get(selected[0].parent_id or "")
            all_explicit_requirements = all(
                semantic_substantive(child) or semantic_explicit_query_requirement(child)
                for child in selected
            )
            if (
                parent is not None
                and not semantic_format_only(parent)
                and not any(semantic_substantive(child) for child in selected)
                and not all_explicit_requirements
            ):
                return False
            non_explicit_format_count = sum(
                1
                for child in selected
                if semantic_format_only(child) and not semantic_explicit_query_requirement(child)
            )
            if parent is not None and not semantic_format_only(parent) and non_explicit_format_count > 1:
                return False
        return self.coverage_score(selected, aspects) >= self.config.tau_cov

    def selection_value(self, selected: list[ARTSNode], aspects: Optional[list[str]] = None) -> float:
        if not selected:
            return 0.0
        if self.config.decision_policy == "semantic":
            distinct = _mean([semantic_distinctive(child) for child in selected])
        elif self.config.decision_policy == "hybrid":
            # This generic value remains semantic-facing. Hybrid calibration
            # ranking uses hybrid_score_objective after hard validation.
            distinct = _mean([semantic_distinctive(child) for child in selected])
        else:
            distinct = self.aggregated_distinct(selected) if self.config.use_distinct else 1.0
        redundancy = self.redundancy_score(selected)
        size_penalty = self.config.lambda_size * len(selected)
        format_count = (
            sum(
                1
                for child in selected
                if semantic_format_only(child) and not semantic_explicit_query_requirement(child)
            )
            if self.uses_semantic_policy()
            else 0
        )
        substantive_count = sum(1 for child in selected if semantic_substantive(child)) if self.uses_semantic_policy() else len(selected)
        explicit_count = (
            sum(1 for child in selected if semantic_explicit_query_requirement(child))
            if self.uses_semantic_policy()
            else 0
        )
        format_penalty = 0.0
        if self.uses_semantic_policy():
            format_penalty = 0.15 * format_count
            if format_count > 1:
                format_penalty += 0.5 * (format_count - 1)
            if substantive_count == 0 and explicit_count != len(selected):
                format_penalty += 1.0
        return distinct - self.config.lambda_red * redundancy - size_penalty - format_penalty

    def semantic_candidate_record(self, candidate: ARTSNode, aspects: list[str], include_score: bool = False) -> dict[str, Any]:
        return {
            "id": candidate.id,
            "text": candidate.text,
            "semantic_quality": candidate.metadata.get("semantic_quality", {}),
            "substantive": semantic_substantive(candidate),
            "format_only": semantic_format_only(candidate),
            "explicit_query_requirement": semantic_explicit_query_requirement(candidate),
            "covered_aspects": [
                aspect for aspect in aspects if self.candidate_covers_aspect(candidate, aspect)
            ],
            "score": {
                "scorability": candidate.scorability,
                "distinct": node_distinct(candidate) if self.uses_score_policy() else None,
                "low_scorability_retained": bool(candidate.metadata.get("low_scorability_retained", False)),
                "score_confidence_penalty": score_confidence_penalty(candidate),
                "score_selection_priority": hybrid_score_selection_priority(candidate) if self.config.decision_policy == "hybrid" else None,
            }
            if include_score and self.config.decision_policy == "hybrid"
            else {},
        }

    def semantic_coverage_records(self, candidates: list[ARTSNode], aspects: list[str]) -> dict[str, list[str]]:
        return {
            candidate.id: [aspect for aspect in aspects if self.candidate_covers_aspect(candidate, aspect)]
            for candidate in candidates
        }

    def semantic_redundancy_records(self, node: ARTSNode, selected: list[ARTSNode]) -> list[dict[str, Any]]:
        records = []
        for index, left in enumerate(selected):
            for right in selected[index + 1 :]:
                redundant = self.semantic_pair_redundant(node, left, right)
                records.append(
                    {
                        "left_id": left.id,
                        "right_id": right.id,
                        "redundant": redundant,
                    }
                )
        return records

    def hybrid_score_redundancy_records(self, candidates: list[ARTSNode], aspects: list[str]) -> list[dict[str, Any]]:
        if self.config.decision_policy != "hybrid" or len(candidates) < 2:
            return []
        records = []
        coverage = self.semantic_coverage_records(candidates, aspects)
        for index, left in enumerate(candidates):
            for right in candidates[index + 1 :]:
                left_scores = list(left.scores or [])
                right_scores = list(right.scores or [])
                correlation = spearman_rank_correlation(left_scores, right_scores)
                overlap = sorted(set(coverage.get(left.id, [])) & set(coverage.get(right.id, [])))
                warning = correlation >= 0.85 and bool(overlap)
                records.append(
                    {
                        "left_id": left.id,
                        "right_id": right.id,
                        "score_correlation": correlation,
                        "coverage_overlap": overlap,
                        "warning": "high_score_redundancy" if warning else "",
                        "left_score_selection_priority": hybrid_score_selection_priority(left),
                        "right_score_selection_priority": hybrid_score_selection_priority(right),
                    }
                )
        return records

    def semantic_selection_valid(self, node: ARTSNode, selected: list[ARTSNode], aspects: list[str]) -> tuple[bool, str]:
        if not selected:
            return False, "empty_selection"
        required_children = min(max(1, self.config.min_children), max(1, len(aspects)))
        if len(aspects) > 1 and len(selected) < required_children:
            return False, f"too_few_children_for_independent_aspects:{len(selected)}<{required_children}"
        if not self.coverage_constraint_ok(selected, aspects):
            return False, "coverage_or_format_substantive_constraint_failed"
        if self.uses_hard_semantic_constraints():
            if aspects and self.coverage_score(selected, aspects) < 1.0:
                return False, "not_all_independent_aspects_covered"
            for child in selected:
                if child.depth >= self.config.max_depth and not semantic_atomic(child):
                    return False, f"non_atomic_child_at_depth_limit:{child.id}"
        for index, left in enumerate(selected):
            for right in selected[index + 1 :]:
                if self.semantic_pair_redundant(node, left, right):
                    return False, f"redundant_pair:{left.id},{right.id}"
        return True, "ok"

    def _semantic_set_score(self, selected: list[ARTSNode]) -> float:
        return _mean([semantic_candidate_score(child) for child in selected])

    def _choose_semantic_feasible_set(
        self,
        node: ARTSNode,
        feasible_sets: list[tuple[list[ARTSNode], float, float, str]],
        aspects: list[str],
    ) -> tuple[list[ARTSNode], bool, str]:
        """Optimize calibration utility strictly inside the semantic feasible set."""
        validated: list[tuple[list[ARTSNode], float, float, str]] = []
        rejected: list[dict[str, Any]] = []
        seen_sets: set[tuple[str, ...]] = set()
        for selected, semantic_score, confidence, reason in feasible_sets:
            signature = tuple(sorted(child.id for child in selected))
            if not signature or signature in seen_sets:
                continue
            seen_sets.add(signature)
            valid, invalid_reason = self.semantic_selection_valid(node, selected, aspects)
            if not valid:
                rejected.append(
                    {
                        "selected_ids": list(signature),
                        "reason": invalid_reason,
                    }
                )
                continue
            validated.append((selected, semantic_score, confidence, reason))

        if not validated:
            if rejected:
                self.logs.append(
                    {
                        "step": "calibration_semantic_feasible_set_selection",
                        "parent_id": node.id,
                        "candidate_sets": [],
                        "rejected_sets": rejected,
                        "selected_ids": [],
                        "calibration_used": False,
                        "reason": "no_semantically_feasible_set",
                    }
                )
            return [], False, "no_semantically_feasible_set"

        if self.config.decision_policy != "hybrid":
            chosen = max(validated, key=lambda item: item[1])
            return chosen[0], False, "semantic_selected_feasible_set"

        # Every surviving set already satisfies coverage, atomicity, explicit
        # constraints and semantic non-redundancy. Calibration therefore only
        # resolves which valid local structure is empirically most useful.
        ranked = sorted(
            validated,
            key=lambda item: self.hybrid_score_objective(item[0], aspects),
            reverse=True,
        )
        chosen: Optional[tuple[list[ARTSNode], float, float, str]] = None
        for item in ranked:
            valid, invalid_reason = self.semantic_selection_valid(node, item[0], aspects)
            if valid:
                chosen = item
                break
            rejected.append(
                {
                    "selected_ids": [child.id for child in item[0]],
                    "reason": f"post_calibration_revalidation:{invalid_reason}",
                }
            )
        if chosen is None:
            return [], True, "calibration_selection_failed_semantic_revalidation"

        self.logs.append(
            {
                "step": "calibration_semantic_feasible_set_selection",
                "parent_id": node.id,
                "calibration_used": True,
                "calibration_role": "optimize_every_intermediate_node_inside_semantic_feasible_set",
                "candidate_sets": [
                    {
                        "selected_ids": [child.id for child in selected],
                        "semantic_score": semantic_score,
                        "semantic_confidence": confidence,
                        "calibration_objective": self.hybrid_score_objective(selected, aspects),
                        "calibration_metrics": self.calibration_set_metrics(selected, aspects),
                        "reason": reason,
                    }
                    for selected, semantic_score, confidence, reason in ranked
                ],
                "rejected_sets": rejected,
                "selected_ids": [child.id for child in chosen[0]],
            }
        )
        return chosen[0], True, "calibration_selected_semantic_feasible_set"

    def semantic_select_children(
        self,
        node: ARTSNode,
        candidates: list[ARTSNode],
        aspects: list[str],
    ) -> tuple[list[ARTSNode], list[ARTSNode], list[str]]:
        if not candidates:
            return [], [], aspects
        by_id = {candidate.id: candidate for candidate in candidates}
        coverage = self.semantic_coverage_records(candidates, aspects)
        score_redundancy = self.hybrid_score_redundancy_records(candidates, aspects)
        data = self._call_json(
            SEMANTIC_SELECT_CHILDREN_PROMPT.format(
                query=self.question,
                parent_text=node.text,
                aspects_json=json.dumps(aspects, ensure_ascii=False, indent=2),
                candidates_json=json.dumps(
                    [self.semantic_candidate_record(candidate, aspects, include_score=False) for candidate in candidates],
                    ensure_ascii=False,
                    indent=2,
                ),
                coverage_json=json.dumps(coverage, ensure_ascii=False, indent=2),
                score_redundancy_json=json.dumps([], ensure_ascii=False, indent=2),
            ),
            temperature=0.0,
        )
        raw_feasible_sets = data.get("feasible_sets", []) if data else []
        feasible_sets: list[tuple[list[ARTSNode], float, float, str]] = []
        if isinstance(raw_feasible_sets, list):
            for raw_set in raw_feasible_sets[:3]:
                if not isinstance(raw_set, dict):
                    continue
                raw_ids = raw_set.get("selected_ids", [])
                if not isinstance(raw_ids, list):
                    continue
                seen_ids: set[str] = set()
                children = [
                    by_id[item]
                    for item in raw_ids
                    if item in by_id and not (item in seen_ids or seen_ids.add(item))
                ]
                valid, _invalid_reason = self.semantic_selection_valid(node, children, aspects)
                if not valid:
                    continue
                try:
                    semantic_score = _clamp01(float(raw_set.get("semantic_score", self._semantic_set_score(children))))
                except (TypeError, ValueError):
                    semantic_score = self._semantic_set_score(children)
                try:
                    confidence = _clamp01(float(raw_set.get("confidence", 1.0)))
                except (TypeError, ValueError):
                    confidence = 0.0
                feasible_sets.append((children, semantic_score, confidence, str(raw_set.get("reason", ""))))

        # Backward-compatible primary set. It is still validated as a semantic
        # feasible set and receives no calibration influence at this stage.
        selected_ids = data.get("selected_ids", []) if data else []
        fallback = [by_id[item] for item in selected_ids if item in by_id] if isinstance(selected_ids, list) else []
        seen = set()
        fallback = [item for item in fallback if not (item.id in seen or seen.add(item.id))]
        fallback_valid, _fallback_reason = self.semantic_selection_valid(node, fallback, aspects)
        if fallback_valid and not any({child.id for child in item[0]} == {child.id for child in fallback} for item in feasible_sets):
            feasible_sets.append((fallback, self._semantic_set_score(fallback), 1.0, "primary_selected_ids"))

        selected, calibration_used, selection_reason = self._choose_semantic_feasible_set(node, feasible_sets, aspects)
        remaining = [candidate for candidate in candidates if candidate.id not in {item.id for item in selected}]
        raw_missing = data.get("missing_aspects", []) if data else aspects
        missing = [str(item).strip() for item in raw_missing if str(item).strip()] if isinstance(raw_missing, list) else []
        valid, invalid_reason = self.semantic_selection_valid(node, selected, aspects)
        self.logs.append(
            {
                "step": "semantic_select_children",
                "parent_id": node.id,
                "selected_children": [child.id for child in selected],
                "missing_aspects": missing,
                "valid": valid,
                "invalid_reason": invalid_reason,
                "semantic_feasible_set_count": len(feasible_sets),
                "calibration_used": calibration_used,
                "selection_reason": selection_reason,
                "coverage": self.coverage_score(selected, aspects),
                "reason": data.get("reason", "") if data else "Invalid semantic select JSON.",
                "score_used": calibration_used,
                "score_reason": selection_reason,
                "score_redundancy": score_redundancy,
                "rejected": data.get("rejected", []) if data else [],
            }
        )
        return selected, remaining, missing

    def deterministic_semantic_feasible_set(
        self,
        node: ARTSNode,
        candidates: list[ARTSNode],
        aspects: list[str],
    ) -> list[ARTSNode]:
        """Recover from an invalid selector JSON without weakening semantics.

        The model selector is the preferred path, but a formatting error or an
        undersized selected_ids list must not make an otherwise usable sample
        fail. Enumerate a small set of candidate combinations, retain only sets
        that pass the exact same coverage/atomicity/non-redundancy validator,
        and apply the normal constrained calibration chooser.
        """
        unique: list[ARTSNode] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.id in seen:
                continue
            seen.add(candidate.id)
            unique.append(candidate)
        if not unique:
            return []

        if len(aspects) > 1:
            required = min(max(1, self.config.min_children), len(aspects), len(unique))
        elif node.parent_id is None or not semantic_atomic(node):
            required = min(max(2, self.config.min_children), len(unique))
        else:
            required = 1

        # Candidate pools are normally <=10. Limiting the width keeps recovery
        # bounded while still exploring alternative atomic decompositions.
        max_width = min(len(unique), max(required, min(6, required + 2)))
        feasible: list[tuple[list[ARTSNode], float, float, str]] = []
        for width in range(required, max_width + 1):
            for group in combinations(unique, width):
                selected = list(group)
                valid, _reason = self.semantic_selection_valid(node, selected, aspects)
                if not valid:
                    continue
                feasible.append(
                    (
                        selected,
                        self._semantic_set_score(selected),
                        1.0,
                        "deterministic_semantic_feasible_set",
                    )
                )

        # No score-based API call is needed here: every candidate already has
        # diagnostics on the shared response set. Evaluate every semantically
        # valid fallback combination so a semantic pre-ranking cannot hide the
        # calibration-optimal feasible structure.
        selected, calibration_used, reason = self._choose_semantic_feasible_set(node, feasible, aspects)
        self.logs.append(
            {
                "step": "deterministic_semantic_fallback",
                "parent_id": node.id,
                "candidate_count": len(unique),
                "required_children": required,
                "feasible_set_count": len(feasible),
                "selected_children": [child.id for child in selected],
                "calibration_used": calibration_used,
                "reason": reason,
            }
        )
        return selected

    def semantic_apply_improve_action(
        self,
        selected: list[ARTSNode],
        remaining: list[ARTSNode],
        action: str,
        remove_ids: list[str],
        add_ids: list[str],
    ) -> tuple[list[ARTSNode], list[ARTSNode]]:
        selected_by_id = {node.id: node for node in selected}
        remaining_by_id = {node.id: node for node in remaining}
        remove_set = {str(item) for item in remove_ids}
        add_set = {str(item) for item in add_ids}
        new_selected = [node for node in selected if node.id not in remove_set]
        moved_to_remaining = [node for node in selected if node.id in remove_set]
        additions = [remaining_by_id[node_id] for node_id in add_set if node_id in remaining_by_id]
        existing_ids = {node.id for node in new_selected}
        for node in additions:
            if node.id not in existing_ids:
                new_selected.append(node)
                existing_ids.add(node.id)
        new_remaining = [node for node in remaining if node.id not in add_set] + moved_to_remaining
        if action == "prune":
            return new_selected, new_remaining
        if action == "replace":
            return new_selected, new_remaining
        if action == "add":
            return new_selected, new_remaining
        return selected, remaining

    def hybrid_score_gate_semantic_action(
        self,
        action: str,
        selected: list[ARTSNode],
        trial_selected: list[ARTSNode],
        added_ids: list[str],
        aspects: list[str],
    ) -> tuple[bool, str, float, float]:
        if self.config.decision_policy != "hybrid":
            return True, "score_gate_not_applicable", 0.0, 0.0
        before = self.hybrid_score_objective(selected, aspects)
        after = self.hybrid_score_objective(trial_selected, aspects)
        accepted = after + 1e-12 >= before
        return accepted, "calibration_non_degradation_gate_inside_semantic_feasible_set", before, after

    def hybrid_refine_replace_by_score(
        self,
        node: ARTSNode,
        selected: list[ARTSNode],
        remaining: list[ARTSNode],
        remove_ids: list[str],
        add_ids: list[str],
        aspects: list[str],
    ) -> tuple[list[str], dict[str, Any]]:
        if self.config.decision_policy != "hybrid" or not remove_ids:
            return add_ids, {}
        # All alternatives considered below must pass the same semantic
        # validator; calibration may refine only within that feasible set.
        selected_by_id = {child.id: child for child in selected}
        remove_set = {str(item) for item in remove_ids}
        removed = [selected_by_id[node_id] for node_id in remove_set if node_id in selected_by_id]
        if not removed:
            return add_ids, {}
        coverage = self.semantic_coverage_records(selected + remaining, aspects)
        base_selected = [child for child in selected if child.id not in remove_set]
        best_ids = list(add_ids)
        best_selected, _best_remaining = self.semantic_apply_improve_action(selected, remaining, "replace", remove_ids, best_ids)
        best_valid, _reason = self.semantic_selection_valid(node, best_selected, aspects)
        best_score = self.hybrid_score_objective(best_selected, aspects) if best_valid else float("-inf")
        best_payload: dict[str, Any] = {
            "score_refine_considered": 0,
            "score_refine_original_add_ids": list(add_ids),
        }

        for candidate in remaining:
            if not self.hybrid_semantic_passed(candidate):
                continue
            candidate_aspects = set(coverage.get(candidate.id, []))
            replaceable = False
            for existing in removed:
                existing_aspects = set(coverage.get(existing.id, []))
                same_aspects = bool(existing_aspects) and existing_aspects == candidate_aspects
                semantically_redundant = self.semantic_pair_redundant(node, existing, candidate)
                if same_aspects or semantically_redundant:
                    replaceable = True
                    break
            if not replaceable:
                continue
            trial_selected = base_selected + [candidate]
            valid, _invalid_reason = self.semantic_selection_valid(node, trial_selected, aspects)
            if not valid:
                continue
            score = self.hybrid_score_objective(trial_selected, aspects)
            best_payload["score_refine_considered"] += 1
            if score > best_score:
                best_score = score
                best_ids = [candidate.id]
                best_payload.update(
                    {
                        "score_refine_selected_add_ids": best_ids,
                        "score_refine_objective": score,
                        "score_refine_selected_priority": hybrid_score_selection_priority(candidate),
                    }
                )
        if best_ids != list(add_ids):
            best_payload["score_refine_changed"] = True
            return best_ids, best_payload
        best_payload["score_refine_changed"] = False
        return add_ids, best_payload

    def semantic_improve_selected_set(
        self,
        node: ARTSNode,
        selected: list[ARTSNode],
        remaining: list[ARTSNode],
        aspects: list[str],
        max_rounds: int = 4,
    ) -> list[ARTSNode]:
        selected = list(selected)
        remaining = list(remaining)
        for round_index in range(max_rounds):
            all_nodes = selected + remaining
            coverage = self.semantic_coverage_records(all_nodes, aspects)
            redundancy = self.semantic_redundancy_records(node, selected)
            score_redundancy = self.hybrid_score_redundancy_records(all_nodes, aspects)
            data = self._call_json(
                SEMANTIC_IMPROVE_SELECTED_PROMPT.format(
                    query=self.question,
                    parent_text=node.text,
                    aspects_json=json.dumps(aspects, ensure_ascii=False, indent=2),
                    selected_json=json.dumps(
                        [self.semantic_candidate_record(candidate, aspects, include_score=False) for candidate in selected],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    remaining_json=json.dumps(
                        [self.semantic_candidate_record(candidate, aspects, include_score=False) for candidate in remaining],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    coverage_json=json.dumps(coverage, ensure_ascii=False, indent=2),
                    redundancy_json=json.dumps(redundancy, ensure_ascii=False, indent=2),
                    score_redundancy_json=json.dumps([], ensure_ascii=False, indent=2),
                ),
                temperature=0.0,
            )
            action = str(data.get("action", "keep")).strip().lower() if data else "keep"
            remove_ids = data.get("remove_ids", []) if data else []
            add_ids = data.get("add_ids", []) if data else []
            remove_ids = [str(item) for item in remove_ids] if isinstance(remove_ids, list) else []
            add_ids = [str(item) for item in add_ids] if isinstance(add_ids, list) else []
            valid_before, reason_before = self.semantic_selection_valid(node, selected, aspects)
            if action == "keep":
                accepted = valid_before and not bool(data and data.get("redundancy_found") is True)
                self.logs.append(
                    {
                        "step": "semantic_improve_selected",
                        "parent_id": node.id,
                        "round": round_index + 1,
                        "action": action,
                        "accepted": accepted,
                        "valid_before": valid_before,
                        "invalid_reason": reason_before,
                        "reason": data.get("reason", "") if data else "Invalid semantic improve JSON.",
                        "score_used": bool(data.get("score_used")) if data else False,
                        "score_reason": data.get("score_reason", "") if data else "",
                        "score_redundancy": score_redundancy,
                    }
                )
                break
            if action not in {"prune", "replace", "add"}:
                break
            score_refine_payload: dict[str, Any] = {}
            if action == "replace":
                add_ids, score_refine_payload = self.hybrid_refine_replace_by_score(
                    node,
                    selected,
                    remaining,
                    remove_ids,
                    add_ids,
                    aspects,
                )
            trial_selected, trial_remaining = self.semantic_apply_improve_action(selected, remaining, action, remove_ids, add_ids)
            valid_after, reason_after = self.semantic_selection_valid(node, trial_selected, aspects)
            score_ok, score_gate_reason, score_before, score_after = self.hybrid_score_gate_semantic_action(
                action,
                selected,
                trial_selected,
                add_ids,
                aspects,
            )
            accepted = valid_after and score_ok
            self.logs.append(
                {
                    "step": "semantic_improve_selected",
                    "parent_id": node.id,
                    "round": round_index + 1,
                    "action": action,
                    "remove_ids": remove_ids,
                    "add_ids": add_ids,
                    "accepted": accepted,
                    "valid_after": valid_after,
                    "invalid_reason": reason_after,
                    "reason": data.get("reason", "") if data else "Invalid semantic improve JSON.",
                    "score_used": bool(data.get("score_used")) if data else False,
                    "score_reason": data.get("score_reason", "") if data else "",
                    "score_gate_reason": score_gate_reason,
                    "score_objective_before": score_before,
                    "score_objective_after": score_after,
                    "score_redundancy": score_redundancy,
                    "selected_before": [child.id for child in selected],
                    "selected_after": [child.id for child in trial_selected],
                    **score_refine_payload,
                }
            )
            if not accepted:
                break
            selected, remaining = trial_selected, trial_remaining
        return selected

    def hybrid_semantic_passed(self, node: ARTSNode) -> bool:
        quality = node.metadata.get("semantic_quality", {})
        return isinstance(quality, dict) and bool(quality.get("pass") is True)

    def calibration_set_metrics(self, selected: list[ARTSNode], aspects: list[str]) -> dict[str, float]:
        """Return empirical metrics for one already-semantic-valid local set."""
        if not selected:
            return {
                "coverage": 0.0,
                "judge_agreement": 0.0,
                "response_discrimination": 0.0,
                "judge_stability": 0.0,
                "score_redundancy": 0.0,
                "size_penalty": 0.0,
                "objective": float("-inf"),
            }

        aggregate_matrix = aggregate_node_score_matrices(selected, self.config.beta)
        response_discrimination = (
            compute_distinct_matrix(aggregate_matrix) if self.config.use_distinct else 1.0
        )
        judge_agreement = _mean(
            [float(child.scorability or 0.0) * score_confidence_penalty(child) for child in selected]
        )
        per_judge_variance = judge_response_variances(aggregate_matrix)
        judge_stability = 1.0 / (1.0 + _variance(per_judge_variance))
        score_redundancy = 0.0
        if len(selected) >= 2:
            score_redundancy = max(
                max(0.0, spearman_rank_correlation(left.scores, right.scores))
                for index, left in enumerate(selected)
                for right in selected[index + 1 :]
            )
        coverage = self.coverage_score(selected, aspects)
        size_penalty = self.config.lambda_size * len(selected)
        objective = (
            0.45 * response_discrimination
            + 0.30 * judge_agreement
            + 0.25 * judge_stability
            - self.config.lambda_red * score_redundancy
            - size_penalty
        )
        return {
            "coverage": coverage,
            "judge_agreement": judge_agreement,
            "response_discrimination": response_discrimination,
            "judge_stability": judge_stability,
            "score_redundancy": score_redundancy,
            "size_penalty": size_penalty,
            "objective": objective,
        }

    def hybrid_score_objective(self, selected: list[ARTSNode], aspects: list[str]) -> float:
        return self.calibration_set_metrics(selected, aspects)["objective"]

    def hybrid_score_add_acceptable(self, candidate: ARTSNode) -> bool:
        return (
            float(candidate.scorability or 0.0) >= self.config.tau_scor
            and node_distinct(candidate) >= self.config.low_distinct_threshold
        )

    def hybrid_score_assisted_improve(
        self,
        node: ARTSNode,
        selected: list[ARTSNode],
        remaining: list[ARTSNode],
        aspects: list[str],
    ) -> tuple[list[ARTSNode], list[ARTSNode]]:
        if self.config.decision_policy != "hybrid" or not selected or not remaining:
            return selected, remaining

        selected = list(selected)
        remaining = list(remaining)
        max_rounds = max(1, (len(selected) + 1) * (len(remaining) + 1))
        for round_index in range(max_rounds):
            all_nodes = selected + remaining
            coverage = self.semantic_coverage_records(all_nodes, aspects)
            current_objective = self.hybrid_score_objective(selected, aspects)
            best_action = ""
            best_gain = 0.0
            best_selected: Optional[list[ARTSNode]] = None
            best_remaining: Optional[list[ARTSNode]] = None
            best_payload: dict[str, Any] = {}

            for existing in selected:
                if not self.hybrid_semantic_passed(existing):
                    continue
                existing_aspects = set(coverage.get(existing.id, []))
                if not existing_aspects:
                    continue
                existing_priority = hybrid_score_selection_priority(existing)
                for candidate in remaining:
                    if not self.hybrid_semantic_passed(candidate):
                        continue
                    candidate_aspects = set(coverage.get(candidate.id, []))
                    same_aspects = bool(existing_aspects) and existing_aspects == candidate_aspects
                    semantically_redundant = self.semantic_pair_redundant(node, existing, candidate)
                    if not same_aspects and not semantically_redundant:
                        continue
                    candidate_priority = hybrid_score_selection_priority(candidate)
                    trial_selected = [child for child in selected if child.id != existing.id] + [candidate]
                    valid, invalid_reason = self.semantic_selection_valid(node, trial_selected, aspects)
                    if not valid:
                        continue
                    trial_remaining = [child for child in remaining if child.id != candidate.id] + [existing]
                    gain = self.hybrid_score_objective(trial_selected, aspects) - current_objective
                    if gain > best_gain:
                        best_action = "replace"
                        best_gain = gain
                        best_selected = trial_selected
                        best_remaining = trial_remaining
                        best_payload = {
                            "replaced_id": existing.id,
                            "replacement_id": candidate.id,
                            "shared_aspects": sorted(existing_aspects & candidate_aspects),
                            "same_aspects": same_aspects,
                            "semantically_redundant": semantically_redundant,
                            "replaced_score_selection_priority": existing_priority,
                            "replacement_score_selection_priority": candidate_priority,
                            "invalid_reason": invalid_reason,
                            "reason": (
                                "Replaced among semantically redundant or exact-same-aspect candidates using score objective "
                                "while preserving semantic constraints."
                            ),
                        }

            if not self.coverage_constraint_ok(selected, aspects):
                for candidate in remaining:
                    if not self.hybrid_semantic_passed(candidate) or not self.hybrid_score_add_acceptable(candidate):
                        continue
                    trial_selected = selected + [candidate]
                    valid, invalid_reason = self.semantic_selection_valid(node, trial_selected, aspects)
                    if not valid:
                        continue
                    gain = self.hybrid_score_objective(trial_selected, aspects) - current_objective
                    if gain > best_gain:
                        best_action = "add"
                        best_gain = gain
                        best_selected = trial_selected
                        best_remaining = [child for child in remaining if child.id != candidate.id]
                        best_payload = {
                            "added_id": candidate.id,
                            "added_score_selection_priority": hybrid_score_selection_priority(candidate),
                            "scorability": candidate.scorability,
                            "distinct": node_distinct(candidate),
                            "invalid_reason": invalid_reason,
                            "reason": "Added a semantically valid coverage-repair child with acceptable score diagnostics.",
                        }

            if not best_action or best_gain <= self.config.greedy_margin or best_selected is None or best_remaining is None:
                break
            selected, remaining = best_selected, best_remaining
            self.logs.append(
                {
                    "step": "hybrid_score_assisted_improve",
                    "parent_id": node.id,
                    "round": round_index + 1,
                    "action": best_action,
                    "gain": best_gain,
                    "objective": self.hybrid_score_objective(selected, aspects),
                    "coverage": self.coverage_score(selected, aspects),
                    "selected_children": [child.id for child in selected],
                    **best_payload,
                }
            )
        return selected, remaining

    def calibration_resolve_equivalent_candidates(
        self,
        node: ARTSNode,
        selected: list[ARTSNode],
        remaining: list[ARTSNode],
        aspects: list[str],
    ) -> tuple[list[ARTSNode], list[ARTSNode]]:
        """Use calibration only for semantically equivalent near-ties.

        Alternatives must cover exactly the same aspects, be judged semantic
        paraphrases/subsumptions, and fall within the semantic-score epsilon.
        This operation cannot add, prune, merge, or change coverage structure.
        """
        if self.config.decision_policy != "hybrid" or not selected or not remaining:
            return selected, remaining
        epsilon = max(0.0, float(self.config.semantic_score_epsilon))
        selected = list(selected)
        remaining = list(remaining)
        coverage = self.semantic_coverage_records(selected + remaining, aspects)
        for index, existing in enumerate(list(selected)):
            existing_aspects = set(coverage.get(existing.id, []))
            if not existing_aspects:
                continue
            alternatives = [existing]
            for candidate in remaining:
                candidate_aspects = set(coverage.get(candidate.id, []))
                if candidate_aspects != existing_aspects:
                    continue
                if not self.semantic_pair_redundant(node, existing, candidate):
                    continue
                if abs(semantic_candidate_score(existing) - semantic_candidate_score(candidate)) > epsilon:
                    continue
                alternatives.append(candidate)
            if len(alternatives) == 1:
                continue
            winner = max(alternatives, key=hybrid_score_selection_priority)
            if winner.id == existing.id:
                self.logs.append(
                    {
                        "step": "calibration_candidate_tiebreak",
                        "parent_id": node.id,
                        "candidate_ids": [child.id for child in alternatives],
                        "winner_id": existing.id,
                        "changed": False,
                        "semantic_epsilon": epsilon,
                    }
                )
                continue
            trial = list(selected)
            trial[index] = winner
            valid, invalid_reason = self.semantic_selection_valid(node, trial, aspects)
            if not valid:
                self.logs.append(
                    {
                        "step": "calibration_candidate_tiebreak",
                        "parent_id": node.id,
                        "candidate_ids": [child.id for child in alternatives],
                        "winner_id": winner.id,
                        "changed": False,
                        "reason": f"semantic_validation_rejected:{invalid_reason}",
                    }
                )
                continue
            selected[index] = winner
            remaining = [child for child in remaining if child.id != winner.id] + [existing]
            self.logs.append(
                {
                    "step": "calibration_candidate_tiebreak",
                    "parent_id": node.id,
                    "candidate_ids": [child.id for child in alternatives],
                    "winner_id": winner.id,
                    "replaced_id": existing.id,
                    "changed": True,
                    "semantic_epsilon": epsilon,
                    "calibration_priorities": {
                        child.id: hybrid_score_selection_priority(child) for child in alternatives
                    },
                }
            )
        return selected, remaining

    def semantic_greedy_select(self, node: ARTSNode, candidates: list[ARTSNode], aspects: list[str]) -> list[ARTSNode]:
        candidate_pool = list(candidates)
        selected, remaining, missing = self.semantic_select_children(node, candidates, aspects)
        valid, _reason = self.semantic_selection_valid(node, selected, aspects)
        if not valid:
            all_candidates = selected + remaining
            repair_aspects = missing or self.uncovered_aspects(selected, aspects)
            for missing_aspect in repair_aspects:
                candidate = self.generate_local_coverage_repair_candidate(node, missing_aspect, all_candidates)
                if candidate is not None:
                    all_candidates.append(candidate)
            candidate_pool = list(all_candidates)
            selected, remaining, _missing = self.semantic_select_children(node, all_candidates, aspects)
        valid, _reason = self.semantic_selection_valid(node, selected, aspects)
        if not valid:
            selected = self.deterministic_semantic_feasible_set(node, candidate_pool, aspects)
            selected_ids = {child.id for child in selected}
            remaining = [candidate for candidate in candidate_pool if candidate.id not in selected_ids]
        if self.config.decision_policy == "hybrid":
            # The selected set is already semantically feasible and ranked by
            # calibration. Explore only semantic-preserving replacements and
            # keep a change only when the calibration objective improves.
            selected_ids = {child.id for child in selected}
            remaining = [candidate for candidate in candidate_pool if candidate.id not in selected_ids]
            selected, remaining = self.hybrid_score_assisted_improve(
                node,
                selected,
                remaining,
                aspects,
            )
        else:
            selected = self.semantic_improve_selected_set(node, selected, remaining, aspects)
        valid, invalid_reason = self.semantic_selection_valid(node, selected, aspects)
        if not valid:
            self.logs.append(
                {
                    "step": "semantic_selection_failed",
                    "parent_id": node.id,
                    "selected_children": [child.id for child in selected],
                    "invalid_reason": invalid_reason,
                    "coverage": self.coverage_score(selected, aspects),
                }
            )
            return []
        return selected

    def construct_coverage_set(
        self,
        node: ARTSNode,
        candidates: list[ARTSNode],
        aspects: list[str],
    ) -> tuple[list[ARTSNode], list[ARTSNode]]:
        selected: list[ARTSNode] = []
        remaining = list(candidates)
        if not self.config.coverage_check:
            return selected, remaining
        while remaining and self.coverage_score(selected, aspects) < self.config.tau_cov:
            current_coverage = self.coverage_score(selected, aspects)
            best_candidate = None
            best_key = None
            for candidate in remaining:
                if (
                    self.uses_semantic_policy()
                    and not semantic_format_only(node)
                    and semantic_format_only(candidate)
                    and any(semantic_format_only(child) for child in selected)
                ):
                    continue
                trial = selected + [candidate]
                trial_coverage = self.coverage_score(trial, aspects)
                coverage_gain = trial_coverage - current_coverage
                tiebreak = self.hybrid_score_tiebreak(candidate) if self.config.decision_policy == "hybrid" else self.value_as_leaf(candidate)
                key = (coverage_gain, self.selection_value(trial, aspects), tiebreak)
                if best_key is None or key > best_key:
                    best_candidate = candidate
                    best_key = key
            if best_candidate is None or best_key is None or best_key[0] <= 0.0:
                break
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            self.logs.append(
                {
                    "step": "coverage_construct",
                    "parent_id": node.id,
                    "selected_id": best_candidate.id,
                    "coverage": self.coverage_score(selected, aspects),
                    "coverage_gain": best_key[0],
                    "objective": self.selection_value(selected, aspects),
                    "distinct": self.aggregated_distinct(selected),
                    "redundancy": self.redundancy_score(selected),
                    "tau_cov": self.config.tau_cov,
                }
            )
        return selected, remaining

    def generate_local_coverage_repair_candidate(
        self,
        node: ARTSNode,
        missing_aspect: str,
        candidates: list[ARTSNode],
    ) -> Optional[ARTSNode]:
        data = self._call_json(
            LOCAL_COVERAGE_REPAIR_PROMPT.format(
                query=self.question,
                parent_text=node.text,
                missing_aspect=missing_aspect,
                candidates_json=json.dumps([candidate.text for candidate in candidates], ensure_ascii=False, indent=2),
            )
        )
        text = str(data.get("rubric", "")).strip() if data else ""
        if not text:
            self.logs.append(
                {
                    "step": "local_coverage_repair_generate_failed",
                    "parent_id": node.id,
                    "missing_aspect": missing_aspect,
                }
            )
            return None
        generator_weight = _positive_float(data.get("weight") if data else None, default=3.0)
        importance_score = self.probe_importance(text, fallback=generator_weight)
        candidate = ARTSNode(
            id=self._new_id(),
            text=text,
            depth=node.depth + 1,
            parent_id=node.id,
            importance=importance_score,
            local_weight=importance_score,
            metadata={
                "generated_for_local_coverage_aspect": missing_aspect,
                "llm_weight": importance_score,
                "importance_score": importance_score,
                "generator_weight": generator_weight,
                "weight_basis": "direct_independent_importance_1_to_5",
            },
        )
        passed = self.filter_candidate(candidate)
        if passed is None:
            self.logs.append(
                {
                    "step": "local_coverage_repair_rejected",
                    "parent_id": node.id,
                    "missing_aspect": missing_aspect,
                    "candidate_text": text,
                    "reason": "scorability_filter_failed",
                }
            )
        return passed

    def repair_coverage_candidates(
        self,
        node: ARTSNode,
        selected: list[ARTSNode],
        remaining: list[ARTSNode],
        aspects: list[str],
    ) -> tuple[list[ARTSNode], list[ARTSNode]]:
        if not self.config.coverage_check or self.coverage_constraint_ok(selected, aspects):
            return selected, remaining
        all_candidates = selected + remaining
        for missing_aspect in self.uncovered_aspects(selected, aspects):
            candidate = self.generate_local_coverage_repair_candidate(node, missing_aspect, all_candidates)
            if candidate is None:
                continue
            remaining.append(candidate)
            all_candidates.append(candidate)
            self.logs.append(
                {
                    "step": "local_coverage_repair_add_candidate",
                    "parent_id": node.id,
                    "candidate_id": candidate.id,
                    "missing_aspect": missing_aspect,
                    "candidate_text": candidate.text,
                    "coverage_before": self.coverage_score(selected, aspects),
                }
            )
            selected, remaining = self.construct_coverage_set(node, selected + remaining, aspects)
            if self.coverage_constraint_ok(selected, aspects):
                break
            all_candidates = selected + remaining
        return selected, remaining

    def improve_selected_set(
        self,
        node: ARTSNode,
        selected: list[ARTSNode],
        remaining: list[ARTSNode],
        aspects: list[str],
    ) -> list[ARTSNode]:
        selected = list(selected)
        remaining = list(remaining)
        while selected or remaining:
            current = self.selection_value(selected, aspects)
            best_action = ""
            best_gain = 0.0
            best_selected: Optional[list[ARTSNode]] = None
            best_remaining: Optional[list[ARTSNode]] = None
            best_candidate_id = None
            best_replaced_id = None

            # Prefer simplifying first: prune, then replace, then optionally add.
            for existing in selected:
                trial_selected = [item for item in selected if item.id != existing.id]
                if not trial_selected or not self.coverage_constraint_ok(trial_selected, aspects):
                    continue
                gain = self.selection_value(trial_selected, aspects) - current
                if gain > best_gain:
                    best_action = "prune"
                    best_gain = gain
                    best_selected = trial_selected
                    best_remaining = remaining + [existing]
                    best_candidate_id = None
                    best_replaced_id = existing.id

            if best_action == "":
                for existing in selected:
                    for candidate in remaining:
                        trial_selected = [item for item in selected if item.id != existing.id] + [candidate]
                        if not self.coverage_constraint_ok(trial_selected, aspects):
                            continue
                        gain = self.selection_value(trial_selected, aspects) - current
                        if gain > best_gain:
                            best_action = "replace"
                            best_gain = gain
                            best_selected = trial_selected
                            best_remaining = [item for item in remaining if item.id != candidate.id] + [existing]
                            best_candidate_id = candidate.id
                            best_replaced_id = existing.id

            if best_action == "":
                for candidate in remaining:
                    trial_selected = selected + [candidate]
                    if not self.coverage_constraint_ok(trial_selected, aspects):
                        continue
                    gain = self.selection_value(trial_selected, aspects) - current
                    if gain > best_gain:
                        best_action = "add"
                        best_gain = gain
                        best_selected = trial_selected
                        best_remaining = [item for item in remaining if item.id != candidate.id]
                        best_candidate_id = candidate.id
                        best_replaced_id = None

            if not best_action or best_gain <= self.config.greedy_margin or best_selected is None or best_remaining is None:
                break
            selected = best_selected
            remaining = best_remaining
            self.logs.append(
                {
                    "step": "constrained_select",
                    "parent_id": node.id,
                    "action": best_action,
                    "candidate_id": best_candidate_id,
                    "replaced_id": best_replaced_id,
                    "gain": best_gain,
                    "objective": self.selection_value(selected, aspects),
                    "distinct": self.aggregated_distinct(selected),
                    "coverage": self.coverage_score(selected, aspects),
                    "redundancy": self.redundancy_score(selected),
                    "size": len(selected),
                    "tau_cov": self.config.tau_cov,
                    "lambda_red": self.config.lambda_red,
                    "lambda_size": self.config.lambda_size,
                }
            )
        return selected

    def greedy_select(self, node: ARTSNode, candidates: list[ARTSNode], aspects: list[str]) -> list[ARTSNode]:
        if self.uses_semantic_policy():
            return self.semantic_greedy_select(node, candidates, aspects)
        selected, remaining = self.construct_coverage_set(node, candidates, aspects)
        if self.config.coverage_check and not self.coverage_constraint_ok(selected, aspects):
            selected, remaining = self.repair_coverage_candidates(node, selected, remaining, aspects)
        if self.config.coverage_check and not self.coverage_constraint_ok(selected, aspects):
            self.logs.append(
                {
                    "step": "coverage_constraint_failed",
                    "parent_id": node.id,
                    "coverage": self.coverage_score(selected, aspects),
                    "tau_cov": self.config.tau_cov,
                    "selected_children": [child.id for child in selected],
                }
            )
            return []
        return self.improve_selected_set(node, selected, remaining, aspects)

    def value(self, children: list[ARTSNode], aspects: Optional[list[str]] = None) -> float:
        if not children:
            return 0.0
        return self.selection_value(children, aspects)

    def value_as_leaf(self, node: ARTSNode) -> float:
        if self.config.decision_policy == "semantic":
            return 0.5 * semantic_distinctive(node) + 0.5 * semantic_leaf_quality(node)
        if self.config.decision_policy == "hybrid":
            return 0.5 * semantic_distinctive(node) + 0.5 * semantic_leaf_quality(node)
        return node_distinct(node)

    def hybrid_score_tiebreak(self, node: ARTSNode) -> float:
        if self.config.decision_policy != "hybrid":
            return self.value_as_leaf(node)
        return hybrid_score_selection_priority(node)

    def semantic_should_decompose(self, node: ARTSNode, selected: list[ARTSNode], aspects: list[str]) -> tuple[bool, float, str]:
        data = self._call_json(
            SEMANTIC_DECOMPOSE_PROMPT.format(
                query=self.question,
                parent_text=node.text,
                aspects_json=json.dumps(aspects, ensure_ascii=False, indent=2),
                children_json=json.dumps(
                    [
                        {
                            "id": child.id,
                            "text": child.text,
                            "semantic_quality": child.metadata.get("semantic_quality", {}),
                            "substantive": semantic_substantive(child),
                            "format_only": semantic_format_only(child),
                        }
                        for child in selected
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
            ),
            temperature=0.0,
        )
        decompose = bool(data and data.get("decompose") is True)
        confidence = _clamp01(float(data.get("confidence", 0.0) or 0.0)) if data else 0.0
        reason = str(data.get("reason", "")) if data else "Invalid semantic decompose JSON."
        self.logs.append(
            {
                "step": "semantic_decompose",
                "node_id": node.id,
                "selected_children": [child.id for child in selected],
                "decompose": decompose,
                "confidence": confidence,
                "reason": reason,
            }
        )
        return decompose, confidence, reason

    def hybrid_atomic_leaf_ok(self, node: ARTSNode) -> bool:
        # Kept under the historical name for compatibility with existing
        # checkpoints/tests.  The semantic-only ablation must use this same
        # semantic leaf gate; only calibration is removed from that cell.
        if not self.uses_hard_semantic_constraints() or node.parent_id is None:
            return False
        explicit_requirement = semantic_explicit_query_requirement(node)
        leaf_quality_floor = 0.6 if explicit_requirement else max(0.75, self.config.tau_scor)
        is_semantic_atomic = (
            semantic_atomic(node)
            and int(node.metadata.get("independent_requirement_count", 1) or 1) <= 1
            and
            semantic_leaf_quality(node) >= leaf_quality_floor
            and (semantic_substantive(node) or explicit_requirement)
            and (not semantic_format_only(node) or explicit_requirement)
        )
        score_distinct = node_distinct(node)
        score_supported = float(node.scorability or 0.0) >= self.config.tau_scor
        # Calibration cannot decide whether a node is semantically atomic.
        ok = is_semantic_atomic
        self.logs.append(
            {
                "step": "hybrid_atomic_leaf_check",
                "node_id": node.id,
                "text": node.text,
                "semantic_leaf_quality": semantic_leaf_quality(node),
                "semantic_atomic": is_semantic_atomic,
                "explicit_query_requirement": explicit_requirement,
                "leaf_quality_floor": leaf_quality_floor,
                "scorability": node.scorability,
                "score_distinct": score_distinct,
                "score_supported": score_supported,
                "calibration_role": "diagnostic_only_not_leaf_gate",
                "decision": "LEAF" if ok else "TRY_DECOMPOSE",
            }
        )
        return ok

    def decomposer(self, node: ARTSNode) -> tuple[str, list[ARTSNode]]:
        if self.hybrid_atomic_leaf_ok(node):
            return "LEAF", []
        candidates = self.generator(node)
        if self.uses_semantic_aspect_routing():
            aspects = self.precompute_aspects(node)
        else:
            aspects = []
            self.logs.append(
                {
                    "step": "precompute_aspects_skipped",
                    "node_id": node.id,
                    "node_text": node.text,
                    "reason": "strict_no_semantic_calibration_only_policy",
                }
            )
        selected = self.greedy_select(node, candidates, aspects)
        decision = "LEAF"
        reason = ""
        if not selected:
            reason = "no_selected_children"
            self.logs.append(
                {
                    "step": "decomposer",
                    "node_id": node.id,
                    "node_text": node.text,
                    "decision": decision,
                    "reason": reason,
                    "candidate_count": len(candidates),
                    "selected_count": len(selected),
                    "selected_children": [child.id for child in selected],
                    "value_selected": self.value(selected, aspects),
                    "value_as_leaf": self.value_as_leaf(node),
                    "aspects": aspects,
                }
            )
            return "LEAF", []
        if not self.coverage_constraint_ok(selected, aspects):
            reason = "coverage_constraint_failed"
            self.logs.append(
                {
                    "step": "decomposer",
                    "node_id": node.id,
                    "node_text": node.text,
                    "decision": decision,
                    "reason": reason,
                    "candidate_count": len(candidates),
                    "selected_count": len(selected),
                    "selected_children": [child.id for child in selected],
                    "value_selected": self.selection_value(selected, aspects),
                    "distinct_selected": self.aggregated_distinct(selected),
                    "parent_distinct": self.value_as_leaf(node),
                    "value_as_leaf": self.value_as_leaf(node),
                    "coverage": self.coverage_score(selected, aspects),
                    "tau_cov": self.config.tau_cov,
                    "aspects": aspects,
                }
            )
            return "LEAF", []
        objective_selected = self.selection_value(selected, aspects)
        distinct_selected = self.aggregated_distinct(selected)
        parent_distinct = self.value_as_leaf(node)
        if not self.config.use_distinct and not self.uses_semantic_policy():
            decision = "DECOMPOSE"
            reason = "distinct_disabled_coverage_satisfied"
            self.logs.append(
                {
                    "step": "decomposer",
                    "node_id": node.id,
                    "node_text": node.text,
                    "decision": decision,
                    "reason": reason,
                    "candidate_count": len(candidates),
                    "selected_count": len(selected),
                    "selected_children": [child.id for child in selected],
                    "value_selected": objective_selected,
                    "distinct_selected": distinct_selected,
                    "parent_distinct": parent_distinct,
                    "value_as_leaf": parent_distinct,
                    "coverage": self.coverage_score(selected, aspects),
                    "tau_cov": self.config.tau_cov,
                    "aspects": aspects,
                }
            )
            return "DECOMPOSE", selected
        if self.uses_semantic_policy():
            semantic_decompose, semantic_confidence, semantic_reason = self.semantic_should_decompose(node, selected, aspects)
            score_parent_distinct = node_distinct(node)
            confidence_threshold = _clamp01(float(self.config.semantic_confidence_threshold))
            hard_requires_decompose = node.parent_id is None or not semantic_atomic(node)
            children_calibration_metrics = self.calibration_set_metrics(selected, aspects)
            parent_calibration_metrics = self.calibration_set_metrics([node], [])
            children_calibration = children_calibration_metrics["objective"]
            parent_calibration = parent_calibration_metrics["objective"]

            if self.config.decision_policy == "hybrid":
                # Every real intermediate node reaches this constrained
                # calibration decision. Semantics still has veto power: a root
                # or non-atomic node cannot be collapsed into a leaf merely
                # because the empirical score happens to prefer it.
                calibration_used = True
                if hard_requires_decompose:
                    decision = "DECOMPOSE"
                    reason = "semantic_hard_atomicity_requires_decompose_calibration_ranked_children"
                else:
                    decision = (
                        "DECOMPOSE"
                        if children_calibration > parent_calibration * (1.0 + self.config.margin)
                        else "LEAF"
                    )
                    reason = "calibration_selected_decomposition_inside_semantic_feasible_actions"
            else:
                calibration_used = False
                decision = "DECOMPOSE" if semantic_decompose else "LEAF"
                reason = "semantic_decomposition_policy"

            self.logs.append(
                {
                    "step": "decomposer",
                    "node_id": node.id,
                    "node_text": node.text,
                    "decision": decision,
                    "reason": reason,
                    "candidate_count": len(candidates),
                    "selected_count": len(selected),
                    "selected_children": [child.id for child in selected],
                    "value_selected": objective_selected,
                    "distinct_selected": distinct_selected,
                    "parent_distinct": parent_distinct,
                    "score_parent_distinct": score_parent_distinct,
                    "value_as_leaf": parent_distinct,
                    "coverage": self.coverage_score(selected, aspects),
                    "tau_cov": self.config.tau_cov,
                    "aspects": aspects,
                    "semantic_confidence": semantic_confidence,
                    "semantic_reason": semantic_reason,
                    "semantic_confidence_threshold": confidence_threshold,
                    "hard_requires_decompose": hard_requires_decompose,
                    "calibration_used": calibration_used,
                    "calibration_role": (
                        "optimize_every_intermediate_node_inside_semantic_feasible_actions"
                        if calibration_used
                        else "unused"
                    ),
                    "children_calibration_objective": children_calibration,
                    "parent_calibration_objective": parent_calibration,
                    "children_calibration_metrics": children_calibration_metrics,
                    "parent_calibration_metrics": parent_calibration_metrics,
                }
            )
            return ("DECOMPOSE", selected) if decision == "DECOMPOSE" else ("LEAF", [])
        if distinct_selected > parent_distinct * (1.0 + self.config.margin):
            decision = "DECOMPOSE"
            reason = "distinct_improved_under_coverage_constraint"
            self.logs.append(
                {
                    "step": "decomposer",
                    "node_id": node.id,
                    "node_text": node.text,
                    "decision": decision,
                    "reason": reason,
                    "candidate_count": len(candidates),
                    "selected_count": len(selected),
                    "selected_children": [child.id for child in selected],
                    "value_selected": objective_selected,
                    "distinct_selected": distinct_selected,
                    "parent_distinct": parent_distinct,
                    "value_as_leaf": parent_distinct,
                    "coverage": self.coverage_score(selected, aspects),
                    "tau_cov": self.config.tau_cov,
                    "aspects": aspects,
                }
            )
            return "DECOMPOSE", selected
        reason = "distinct_not_improved"
        self.logs.append(
            {
                "step": "decomposer",
                "node_id": node.id,
                "node_text": node.text,
                "decision": decision,
                "reason": reason,
                "candidate_count": len(candidates),
                "selected_count": len(selected),
                "selected_children": [child.id for child in selected],
                "value_selected": objective_selected,
                "distinct_selected": distinct_selected,
                "parent_distinct": parent_distinct,
                "value_as_leaf": parent_distinct,
                "coverage": self.coverage_score(selected, aspects),
                "tau_cov": self.config.tau_cov,
                "aspects": aspects,
            }
        )
        return "LEAF", []

    def mark_leaf(self, node: ARTSNode, reason: str) -> None:
        node.status = "leaf"
        node.metadata["leaf_reason"] = reason
        self.leaf_scoring_points[node.id] = node

    def apply_decomposition(self, node: ARTSNode, children: list[ARTSNode]) -> None:
        node.status = "internal"
        self.internal_nodes[node.id] = node
        self.assign_child_weights(node, children)
        for child in children:
            self.nodes[child.id] = child
            node.children.append(child.id)
            self.push_frontier(child)

    def forced_split_jointly_sufficient(self, node: ARTSNode, children: list[ARTSNode]) -> tuple[bool, str]:
        if len(children) < 2 or any(not semantic_atomic(child) for child in children):
            return False, "fewer_than_two_or_non_atomic_children"
        # Use one coherent set-level verdict.  A separate pairwise call can
        # contradict the full-query split (for example, mistaking "gibberish"
        # and "same character count" for duplicates) and would prevent the
        # sufficiency judge from considering coverage and non-redundancy
        # together.  The prompt below explicitly enforces both properties.
        data = self._call_json(
            FORCED_SPLIT_SUFFICIENCY_PROMPT.format(
                query=self.question,
                rubric_text=node.text,
                children_json=json.dumps(
                    [{"id": child.id, "rubric": child.text} for child in children],
                    ensure_ascii=False,
                    indent=2,
                ),
            ),
            temperature=0.0,
        )
        sufficient = bool(data and data.get("sufficient") is True)
        reason = str(data.get("reason", "")) if data else "Invalid split-sufficiency JSON."
        self.logs.append(
            {
                "step": "forced_split_joint_sufficiency",
                "node_id": node.id,
                "child_ids": [child.id for child in children],
                "child_texts": [child.text for child in children],
                "sufficient": sufficient,
                "reason": reason,
            }
        )
        return sufficient, reason

    def forced_atomic_split(self, node: ARTSNode, max_children: int) -> list[ARTSNode]:
        """Generate and strictly validate an atomic split for a compound node.

        This is a repair path, not a relaxed admission path.  Every generated
        child still goes through the normal semantic and calibration filters,
        and the complete set must cover the source node without redundancy.
        """
        aspects = self.precompute_aspects(node)
        # The ordinary decomposition may have estimated only two requirements
        # while the independent-aspect extractor (and MIN_CHILDREN) requires
        # more.  Size the repair prompt from the stricter evidence.
        max_children = max(
            2,
            min(
                self.config.budget,
                max(int(max_children), self.config.min_children, len(aspects)),
            ),
        )
        quality = node.metadata.get("semantic_quality", {})
        atomicity_reason = str(
            quality.get("atomicity_reason", node.metadata.get("atomicity_reason", ""))
            if isinstance(quality, dict)
            else node.metadata.get("atomicity_reason", "")
        )
        data = self._call_json(
            FORCED_ATOMIC_SPLIT_PROMPT.format(
                query=self.question,
                rubric_text=node.text,
                atomicity_reason=atomicity_reason or "The criterion contains multiple independently fail-able requirements.",
                max_children=max_children,
            ),
            temperature=0.0,
        )
        raw_items = data.get("atomic_leaves", []) if data else []
        if not isinstance(raw_items, list):
            raw_items = []

        def build_candidates(items: list[Any], repair_mode: str) -> list[ARTSNode]:
            candidates: list[ARTSNode] = []
            seen: set[str] = set()
            for item in items[:max_children]:
                if isinstance(item, dict):
                    text = str(item.get("rubric") or item.get("text") or item.get("criterion") or "").strip()
                    generator_weight = _positive_float(item.get("weight"), default=3.0)
                else:
                    text = str(item).strip()
                    generator_weight = 3.0
                normalized = " ".join(text.casefold().split())
                if not text or normalized in seen:
                    continue
                seen.add(normalized)
                importance_score = self.probe_importance(text, fallback=generator_weight)
                candidate = ARTSNode(
                    id=self._new_id(),
                    text=text,
                    depth=min(node.depth + 1, self.config.max_depth),
                    parent_id=node.id,
                    importance=importance_score,
                    local_weight=importance_score,
                    metadata={
                        "forced_atomic_split_from": node.id,
                        "forced_atomic_split_mode": repair_mode,
                        "llm_weight": importance_score,
                        "importance_score": importance_score,
                        "generator_weight": generator_weight,
                        "weight_basis": "direct_independent_importance_1_to_5",
                    },
                )
                filtered = self.filter_candidate(candidate)
                if filtered is None or not semantic_atomic(filtered):
                    continue
                candidates.append(filtered)
            return candidates

        candidates = build_candidates(raw_items, "joint_split")
        if candidates:
            self._record_semantic_fallback_candidates(node.id, candidates)
        valid, validation_reason = self.semantic_selection_valid(node, candidates, aspects)
        if len(candidates) < 2:
            valid = False
            validation_reason = f"forced_split_returned_fewer_than_two_atomic_leaves:{len(candidates)}"

        if not valid and len(candidates) >= 2:
            sufficient, sufficiency_reason = self.forced_split_jointly_sufficient(node, candidates)
            if sufficient:
                valid = True
                validation_reason = "joint_semantic_sufficiency:" + sufficiency_reason
            else:
                validation_reason = (
                    f"aspect_validation={validation_reason}; "
                    f"joint_semantic_sufficiency={sufficiency_reason}"
                )

        # If a single multi-item generation omitted an aspect or produced a
        # compound child, generate each previously extracted requirement in an
        # independent request.  Calibration may rank valid alternatives, but
        # it still cannot admit a non-atomic or uncovered set.
        aspect_fallback_used = False
        if not valid and len(aspects) >= 2 and len(aspects) <= self.config.budget:
            aspect_fallback_used = True
            aspect_items: list[dict[str, Any]] = []
            for aspect in aspects:
                aspect_data = self._call_json(
                    ATOMIC_LEAF_FOR_ASPECT_PROMPT.format(
                        query=self.question,
                        rubric_text=node.text,
                        aspect=aspect,
                    ),
                    temperature=0.0,
                )
                if aspect_data:
                    aspect_items.append(aspect_data)
            fallback_candidates = build_candidates(aspect_items, "one_request_per_aspect")
            if fallback_candidates:
                self._record_semantic_fallback_candidates(node.id, fallback_candidates)
            fallback_valid, fallback_reason = self.semantic_selection_valid(
                node,
                fallback_candidates,
                aspects,
            )
            if len(fallback_candidates) < 2:
                fallback_valid = False
                fallback_reason = (
                    "aspect_fallback_returned_fewer_than_two_atomic_leaves:"
                    f"{len(fallback_candidates)}"
                )
            if fallback_valid:
                candidates = fallback_candidates
                valid = True
                validation_reason = "one_request_per_aspect:" + fallback_reason
            else:
                validation_reason = (
                    f"joint={validation_reason}; aspect_fallback={fallback_reason}"
                )

        self.logs.append(
            {
                "step": "forced_atomic_split",
                "node_id": node.id,
                "node_text": node.text,
                "max_children": max_children,
                "raw_count": len(raw_items),
                "accepted_count": len(candidates),
                "accepted_ids": [candidate.id for candidate in candidates],
                "valid": valid,
                "validation_reason": validation_reason,
                "aspect_count": len(aspects),
                "aspect_fallback_used": aspect_fallback_used,
                "generator_reason": data.get("reason", "") if data else "Invalid forced split JSON.",
            }
        )
        return candidates if valid else []

    def _terminal_eviction_candidates(self, exclude_ids: set[str]) -> list[ARTSNode]:
        candidates: list[ARTSNode] = []
        terminal_nodes = list(self.leaf_scoring_points.values()) + [
            node
            for node in self.nodes.values()
            if node.status == "frontier"
        ]
        seen: set[str] = set()
        for candidate in terminal_nodes:
            if candidate.id in seen or candidate.id in exclude_ids or candidate.id == self.root_id:
                continue
            seen.add(candidate.id)
            parent = self.nodes.get(candidate.parent_id or "")
            if parent is None or len(parent.children) <= 1:
                continue
            candidates.append(candidate)
        return sorted(
            candidates,
            key=lambda candidate: (
                1 if semantic_explicit_query_requirement(candidate) else 0,
                semantic_candidate_score(candidate),
                float(candidate.importance),
            ),
        )

    def ensure_terminal_capacity(self, add_count: int, exclude_ids: Optional[set[str]] = None) -> bool:
        """Free low-value terminal slots before an obligatory atomic split."""
        overflow = self.active_terminal_count() + int(add_count) - self.config.budget
        if overflow <= 0:
            return True
        candidates = self._terminal_eviction_candidates(exclude_ids or set())
        if len(candidates) < overflow:
            self.logs.append(
                {
                    "step": "forced_atomic_split_capacity_failed",
                    "required_slots": overflow,
                    "eligible_terminal_count": len(candidates),
                }
            )
            return False
        for candidate in candidates[:overflow]:
            self.leaf_scoring_points.pop(candidate.id, None)
            candidate.status = "removed"
            parent = self.nodes.get(candidate.parent_id or "")
            if parent is not None:
                parent.children = [child_id for child_id in parent.children if child_id != candidate.id]
            self.logs.append(
                {
                    "step": "forced_atomic_split_capacity_evict",
                    "node_id": candidate.id,
                    "node_text": candidate.text,
                    "semantic_score": semantic_candidate_score(candidate),
                    "importance": candidate.importance,
                }
            )
        return True

    def apply_forced_atomic_split(self, node: ARTSNode, children: list[ARTSNode]) -> None:
        """Apply a validated split without exceeding the configured depth."""
        if node.depth < self.config.max_depth:
            self.apply_decomposition(node, children)
            return
        parent = self.nodes.get(node.parent_id or "")
        if parent is None:
            raise ValueError("Cannot replace a depth-limited root with atomic leaves.")
        parent.children = [child_id for child_id in parent.children if child_id != node.id]
        node.status = "removed"
        self.internal_nodes.pop(node.id, None)
        self.leaf_scoring_points.pop(node.id, None)
        self.assign_child_weights(parent, children)
        for child in children:
            child.parent_id = parent.id
            child.depth = node.depth
            child.status = "leaf"
            self.nodes[child.id] = child
            parent.children.append(child.id)
            self.leaf_scoring_points[child.id] = child
        self.logs.append(
            {
                "step": "forced_atomic_split_depth_preserving_replace",
                "node_id": node.id,
                "parent_id": parent.id,
                "replacement_ids": [child.id for child in children],
                "depth": node.depth,
            }
        )

    def assign_child_weights(self, node: ARTSNode, children: list[ARTSNode]) -> None:
        """Keep independent candidate importance; do not propagate parent weight."""
        raw_weights = [
            _positive_float(
                child.metadata.get("importance_score", child.metadata.get("llm_weight")),
                default=1.0,
            )
            for child in children
        ]
        assigned = []
        for child, raw_weight in zip(children, raw_weights):
            child.local_weight = raw_weight
            child.importance = raw_weight
            child.metadata["weight_basis"] = "direct_independent_importance_1_to_5"
            child.metadata["direct_importance_raw"] = raw_weight
            child.metadata.pop("parent_importance", None)
            assigned.append(
                {
                    "child_id": child.id,
                    "direct_importance": raw_weight,
                    "semantic_format_only": semantic_format_only(child),
                    "semantic_substantive": semantic_substantive(child),
                    "score_confidence_penalty": score_confidence_penalty(child),
                    "low_scorability_retained": bool(child.metadata.get("low_scorability_retained", False)),
                }
            )
        self.logs.append(
            {
                "step": "assign_candidate_direct_importance",
                "parent_id": node.id,
                "basis": "independent_importance_no_parent_propagation",
                "children": assigned,
            }
        )

    def global_coverage_check(self) -> tuple[bool, list[str], str]:
        if not self.config.coverage_check:
            return True, [], "Coverage check disabled."
        leaves = [
            {"id": node.id, "text": node.text, "weight": node.importance}
            for node in self.leaf_scoring_points.values()
        ]
        data = self._call_json(
            GLOBAL_COVERAGE_PROMPT.format(
                query=self.question,
                root_text=self.config.root_text,
                leaves_json=json.dumps(leaves, ensure_ascii=False, indent=2),
            ),
            temperature=0.0,
        )
        raw_atomicity = data.get("atomicity_violations", []) if data else []
        atomicity_violations = raw_atomicity if isinstance(raw_atomicity, list) else []
        raw_redundant = data.get("redundant_pairs", []) if data else []
        redundant_pairs = raw_redundant if isinstance(raw_redundant, list) else []
        local_non_atomic = [
            {"leaf_id": node.id, "reason": node.metadata.get("atomicity_reason", "semantic_atomic=false")}
            for node in self.leaf_scoring_points.values()
            if self.uses_hard_semantic_constraints()
            and not semantic_atomic(node)
        ]
        atomicity_violations = atomicity_violations + local_non_atomic
        covered = bool(
            data
            and data.get("covered") is True
            and not atomicity_violations
            and not redundant_pairs
        )
        raw_missing = data.get("missing_aspects", []) if data else []
        missing = [str(item).strip() for item in raw_missing if str(item).strip()] if isinstance(raw_missing, list) else []
        reason = str(data.get("reason", "")) if data else "Invalid global coverage JSON."
        self._last_global_coverage_diagnostics = {
            "covered": covered,
            "reason": reason,
            "missing_aspects": missing,
            "atomicity_violations": atomicity_violations,
            "redundant_pairs": redundant_pairs,
        }
        self.logs.append(
            {
                "step": "global_coverage_check",
                "covered": covered,
                "reason": reason,
                "missing_aspects": missing,
                "atomicity_violations": atomicity_violations,
                "redundant_pairs": redundant_pairs,
                "leaf_ids": list(self.leaf_scoring_points.keys()),
            }
        )
        return covered, missing, reason

    def generate_missing_aspect_leaf(self, missing_aspect: str) -> Optional[ARTSNode]:
        leaves = [
            {"id": node.id, "text": node.text}
            for node in self.leaf_scoring_points.values()
        ]
        data = self._call_json(
            MISSING_ASPECT_LEAF_PROMPT.format(
                query=self.question,
                root_text=self.config.root_text,
                leaves_json=json.dumps(leaves, ensure_ascii=False, indent=2),
                missing_aspect=missing_aspect,
            )
        )
        text = str(data.get("rubric", "")).strip() if data else ""
        if not text:
            self.logs.append({"step": "final_repair_generate_failed", "missing_aspect": missing_aspect})
            return None
        generator_weight = _positive_float(data.get("weight") if data else None, default=3.0)
        importance_score = self.probe_importance(text, fallback=generator_weight)
        candidate = ARTSNode(
            id=self._new_id(),
            text=text,
            depth=1,
            parent_id=None,
            importance=importance_score,
            local_weight=importance_score,
            metadata={
                "generated_for_missing_aspect": missing_aspect,
                "llm_weight": importance_score,
                "importance_score": importance_score,
                "generator_weight": generator_weight,
                "weight_basis": "direct_independent_importance_1_to_5",
            },
        )
        candidate = self.filter_candidate(candidate)
        if (
            candidate is not None
            and self.uses_hard_semantic_constraints()
            and not semantic_atomic(candidate)
        ):
            self.logs.append(
                {
                    "step": "final_repair_generate_failed",
                    "missing_aspect": missing_aspect,
                    "candidate_id": candidate.id,
                    "reason": "generated_missing_aspect_leaf_was_not_atomic",
                }
            )
            return None
        return candidate

    def repair_parent_candidates(self) -> list[dict[str, Any]]:
        candidates = []
        for node_id, node in self.nodes.items():
            if node_id == self.root_id or node.children:
                candidates.append(
                    {
                        "id": node_id,
                        "text": node.text,
                        "depth": node.depth,
                        "children": [self.nodes[child_id].text for child_id in node.children if child_id in self.nodes],
                    }
                )
        return sorted(candidates, key=lambda item: int(item["depth"]))

    def choose_repair_parent(self, missing_aspect: str, candidate: ARTSNode) -> str:
        parents = self.repair_parent_candidates()
        valid_parent_ids = {str(parent["id"]) for parent in parents}
        if valid_parent_ids == {self.root_id}:
            return self.root_id
        data = self._call_json(
            REPAIR_PARENT_PROMPT.format(
                query=self.question,
                missing_aspect=missing_aspect,
                new_leaf=candidate.text,
                parents_json=json.dumps(parents, ensure_ascii=False, indent=2),
            ),
            temperature=0.0,
        )
        parent_id = str(data.get("parent_id", "")).strip() if data else ""
        if parent_id not in valid_parent_ids:
            parent_id = self.root_id
        self.logs.append(
            {
                "step": "global_coverage_repair_parent",
                "missing_aspect": missing_aspect,
                "candidate_id": candidate.id,
                "parent_id": parent_id,
                "reason": data.get("reason", "") if data else "Invalid parent selection JSON.",
            }
        )
        return parent_id

    def leaf_redundancy(self, candidate: ARTSNode) -> float:
        if not self.leaf_scoring_points:
            return 0.0
        return max(_text_jaccard(candidate.text, leaf.text) for leaf in self.leaf_scoring_points.values())

    def low_value_leaf_id(self) -> Optional[str]:
        if not self.leaf_scoring_points:
            return None
        return min(
            self.leaf_scoring_points,
            key=lambda leaf_id: float(self.leaf_scoring_points[leaf_id].scorability or 0.0)
            * self.value_as_leaf(self.leaf_scoring_points[leaf_id]),
        )

    def remove_leaf(self, leaf_id: str) -> None:
        leaf = self.leaf_scoring_points.pop(leaf_id, None)
        if leaf is None:
            return
        leaf.status = "removed"
        if leaf.parent_id and leaf.parent_id in self.nodes:
            parent = self.nodes[leaf.parent_id]
            parent.children = [child_id for child_id in parent.children if child_id != leaf_id]
            if not parent.children and parent.id != self.root_id:
                parent.status = "leaf"
                self.internal_nodes.pop(parent.id, None)
                self.leaf_scoring_points[parent.id] = parent

    def add_final_leaf(self, leaf: ARTSNode, parent_id: Optional[str] = None) -> None:
        parent = self.nodes.get(parent_id or self.root_id, self.nodes[self.root_id])
        if self.root_id in self.leaf_scoring_points:
            self.leaf_scoring_points.pop(self.root_id, None)
        if parent.id in self.leaf_scoring_points:
            self.leaf_scoring_points.pop(parent.id, None)
        parent.status = "internal"
        self.internal_nodes[parent.id] = parent
        leaf.parent_id = parent.id
        leaf.depth = parent.depth + 1
        leaf.status = "leaf"
        self.nodes[leaf.id] = leaf
        if leaf.id not in parent.children:
            parent.children.append(leaf.id)
        self.leaf_scoring_points[leaf.id] = leaf

    def detach_final_leaf(self, leaf_id: str) -> None:
        """Detach a final leaf without turning an empty structural parent into a leaf."""
        leaf = self.leaf_scoring_points.pop(leaf_id, None)
        if leaf is None:
            return
        leaf.status = "removed"
        parent = self.nodes.get(leaf.parent_id or "")
        if parent is not None:
            parent.children = [child_id for child_id in parent.children if child_id != leaf_id]

    def prune_empty_internal_nodes(self) -> None:
        changed = True
        while changed:
            changed = False
            for node_id, node in list(self.internal_nodes.items()):
                if node_id == self.root_id or node.children:
                    continue
                parent = self.nodes.get(node.parent_id or "")
                if parent is not None:
                    parent.children = [child_id for child_id in parent.children if child_id != node_id]
                node.status = "removed"
                self.internal_nodes.pop(node_id, None)
                changed = True

    def repair_redundant_pair(self, record: dict[str, Any]) -> bool:
        left_id = str(record.get("left_id", "")).strip()
        right_id = str(record.get("right_id", "")).strip()
        left = self.leaf_scoring_points.get(left_id)
        right = self.leaf_scoring_points.get(right_id)
        if left is None or right is None or left.id == right.id:
            return False

        def retain_value(node: ARTSNode) -> tuple[int, float, float, float]:
            return (
                1 if semantic_explicit_query_requirement(node) else 0,
                semantic_candidate_score(node),
                float(node.scorability or 0.0),
                float(node.importance),
            )

        keep, remove = (left, right) if retain_value(left) >= retain_value(right) else (right, left)
        self.detach_final_leaf(remove.id)
        self.prune_empty_internal_nodes()
        self.logs.append(
            {
                "step": "global_coverage_repair_redundancy",
                "kept_leaf_id": keep.id,
                "removed_leaf_id": remove.id,
                "reason": str(record.get("reason", "")),
            }
        )
        return True

    def repair_atomicity_violation(self, record: dict[str, Any]) -> bool:
        leaf_id = str(record.get("leaf_id", "")).strip()
        leaf = self.leaf_scoring_points.get(leaf_id)
        if leaf is None:
            return False
        replacement_capacity = max(2, min(4, self.config.budget))
        children = self.forced_atomic_split(leaf, replacement_capacity)
        if not children:
            return False
        if not self.ensure_terminal_capacity(len(children) - 1, exclude_ids={leaf.id}):
            return False
        parent_id = leaf.parent_id or self.root_id
        self.detach_final_leaf(leaf.id)
        for child in children:
            self.add_final_leaf(child, parent_id=parent_id)
        self.prune_empty_internal_nodes()
        self.logs.append(
            {
                "step": "global_coverage_repair_atomicity",
                "replaced_leaf_id": leaf.id,
                "replacement_ids": [child.id for child in children],
                "reason": str(record.get("reason", "")),
            }
        )
        return True

    def rewrite_final_leaf_set(self, diagnostics: dict[str, Any]) -> bool:
        """Holistic fallback when validation reports no actionable local edit."""
        old_leaves = list(self.leaf_scoring_points.values())
        leaves_json = json.dumps(
            [{"id": leaf.id, "text": leaf.text} for leaf in old_leaves],
            ensure_ascii=False,
            indent=2,
        )
        data = self._call_json(
            FINAL_SEMANTIC_REWRITE_PROMPT.format(
                query=self.question,
                root_text=self.config.root_text,
                leaves_json=leaves_json,
                diagnostics_json=json.dumps(diagnostics, ensure_ascii=False, indent=2),
                budget=self.config.budget,
            ),
            temperature=0.0,
        )
        raw_items = data.get("rubrics", []) if data else []
        if not isinstance(raw_items, list) or not 1 <= len(raw_items) <= self.config.budget:
            self.logs.append(
                {
                    "step": "global_coverage_repair_rewrite_rejected",
                    "reason": "invalid_replacement_count",
                    "replacement_count": len(raw_items) if isinstance(raw_items, list) else 0,
                }
            )
            return False

        old_leaf_ids = {leaf.id for leaf in old_leaves}
        old_parent_ids = {leaf.id: (leaf.parent_id or self.root_id) for leaf in old_leaves}
        replacements: list[tuple[ARTSNode, list[str]]] = []
        seen: set[str] = set()
        for item in raw_items:
            if not isinstance(item, dict):
                return False
            text = str(item.get("rubric") or item.get("text") or "").strip()
            normalized = " ".join(text.casefold().split())
            if not text or normalized in seen:
                return False
            seen.add(normalized)
            source_ids_raw = item.get("source_leaf_ids", [])
            source_ids = [
                str(source_id)
                for source_id in source_ids_raw
                if str(source_id) in old_leaf_ids
            ] if isinstance(source_ids_raw, list) else []
            generator_weight = _positive_float(item.get("weight"), default=3.0)
            importance_score = self.probe_importance(text, fallback=generator_weight)
            candidate = ARTSNode(
                id=self._new_id(),
                text=text,
                depth=1,
                parent_id=self.root_id,
                importance=importance_score,
                local_weight=importance_score,
                metadata={
                    "final_semantic_rewrite": True,
                    "source_leaf_ids": source_ids,
                    "llm_weight": importance_score,
                    "importance_score": importance_score,
                    "generator_weight": generator_weight,
                    "weight_basis": "direct_independent_importance_1_to_5",
                },
            )
            filtered = self.filter_candidate(candidate)
            if filtered is None or not semantic_atomic(filtered):
                self.logs.append(
                    {
                        "step": "global_coverage_repair_rewrite_rejected",
                        "reason": "replacement_failed_semantic_or_atomic_gate",
                        "rubric": text,
                    }
                )
                return False
            replacements.append((filtered, source_ids))

        for leaf in old_leaves:
            self.detach_final_leaf(leaf.id)
        for candidate, source_ids in replacements:
            parent_id = next(
                (
                    old_parent_ids[source_id]
                    for source_id in source_ids
                    if old_parent_ids.get(source_id) in self.nodes
                ),
                self.root_id,
            )
            self.add_final_leaf(candidate, parent_id=parent_id)
        self.prune_empty_internal_nodes()
        self.logs.append(
            {
                "step": "global_coverage_repair_rewrite_apply",
                "old_leaf_count": len(old_leaves),
                "new_leaf_count": len(replacements),
                "new_leaf_ids": [candidate.id for candidate, _source_ids in replacements],
                "generator_reason": data.get("reason", "") if data else "",
            }
        )
        return True

    def normalize_tree_weights(self) -> None:
        """Normalize final leaves once, directly from their own importance."""
        leaves = list(self.leaf_scoring_points.values())
        raw_weights = {
            leaf.id: _positive_float(
                leaf.metadata.get(
                    "importance_score",
                    leaf.metadata.get("llm_weight", leaf.importance),
                ),
                default=1.0,
            )
            for leaf in leaves
        }
        total = sum(raw_weights.values())
        if total <= 0.0:
            raw_weights = {leaf.id: 1.0 for leaf in leaves}
            total = float(len(leaves))

        for leaf in leaves:
            raw_weight = raw_weights[leaf.id]
            normalized_weight = raw_weight / total
            leaf.importance = normalized_weight
            leaf.local_weight = normalized_weight
            leaf.metadata["direct_importance_raw"] = raw_weight
            leaf.metadata["weight_basis"] = "direct_leaf_global_normalization"
            leaf.metadata["final_leaf_weight"] = normalized_weight
            leaf.metadata.pop("parent_importance", None)

        leaf_ids = set(self.leaf_scoring_points)
        for node_id, node in self.nodes.items():
            if node_id in leaf_ids:
                continue
            node.metadata["search_importance"] = node.importance
            node.metadata["weight_basis"] = "structure_only_no_final_weight"
            node.importance = 0.0
            node.local_weight = 0.0

        self.logs.append(
            {
                "step": "final_leaf_weight_normalize",
                "basis": "direct_leaf_global_normalization",
                "leaf_count": len(self.leaf_scoring_points),
                "weight_sum_before": total,
                "weight_sum_after": sum(leaf.importance for leaf in self.leaf_scoring_points.values()),
                "raw_leaf_importance": raw_weights,
                "leaf_weights": {leaf.id: leaf.importance for leaf in self.leaf_scoring_points.values()},
            }
        )

    def final_coverage_repair(self) -> bool:
        if not self.config.coverage_check:
            return True
        for attempt in range(self.config.max_final_repair_attempts + 1):
            covered, missing_aspects, reason = self.global_coverage_check()
            if covered:
                self.logs.append({"step": "global_coverage_repair_done", "attempt": attempt, "covered": True})
                return True
            if attempt >= self.config.max_final_repair_attempts:
                self.logs.append(
                    {
                        "step": "global_coverage_repair_done",
                        "attempt": attempt,
                        "covered": False,
                        "reason": reason,
                    }
                )
                return False
            diagnostics = dict(self._last_global_coverage_diagnostics)
            changed = False
            reserve_for_holistic_rewrite = attempt == self.config.max_final_repair_attempts - 1

            # Remove confirmed duplicates first.  Besides repairing semantic
            # redundancy, this can free a slot needed to split a compound leaf.
            redundant_pairs = diagnostics.get("redundant_pairs", [])
            if not reserve_for_holistic_rewrite and isinstance(redundant_pairs, list):
                for record in redundant_pairs:
                    if isinstance(record, dict):
                        changed = self.repair_redundant_pair(record) or changed

            atomicity_violations = diagnostics.get("atomicity_violations", [])
            if not reserve_for_holistic_rewrite and isinstance(atomicity_violations, list):
                seen_atomicity_ids: set[str] = set()
                for record in atomicity_violations:
                    if not isinstance(record, dict):
                        continue
                    leaf_id = str(record.get("leaf_id", "")).strip()
                    if not leaf_id or leaf_id in seen_atomicity_ids:
                        continue
                    seen_atomicity_ids.add(leaf_id)
                    changed = self.repair_atomicity_violation(record) or changed

            for missing_aspect in ([] if reserve_for_holistic_rewrite else missing_aspects):
                candidate = self.generate_missing_aspect_leaf(missing_aspect)
                if candidate is None:
                    continue
                redundancy = self.leaf_redundancy(candidate)
                distinct = self.value_as_leaf(candidate)
                if redundancy >= self.config.redundancy_threshold or (
                    self.config.use_distinct and distinct <= self.config.low_distinct_threshold
                ):
                    self.logs.append(
                        {
                            "step": "global_coverage_repair_skip",
                            "missing_aspect": missing_aspect,
                            "candidate_id": candidate.id,
                            "reason": "redundant_or_low_distinct",
                            "redundancy": redundancy,
                            "distinct": distinct,
                        }
                    )
                    continue
                action = "add"
                replaced_leaf_id = None
                if len(self.leaf_scoring_points) >= self.config.budget:
                    replaced_leaf_id = self.low_value_leaf_id()
                    if replaced_leaf_id is None:
                        continue
                    action = "replace"
                    self.detach_final_leaf(replaced_leaf_id)
                    self.prune_empty_internal_nodes()
                parent_id = self.choose_repair_parent(missing_aspect, candidate)
                self.add_final_leaf(candidate, parent_id=parent_id)
                changed = True
                self.logs.append(
                    {
                        "step": "global_coverage_repair_apply",
                        "action": action,
                        "missing_aspect": missing_aspect,
                        "candidate_id": candidate.id,
                        "candidate_text": candidate.text,
                        "parent_id": parent_id,
                        "replaced_leaf_id": replaced_leaf_id,
                        "redundancy": redundancy,
                        "distinct": distinct,
                        "scorability": candidate.scorability,
                    }
                )

            # Some global validators report only a prose reason.  In that case
            # request a complete corrected set, then subject it to the same
            # strict filter and the next loop's independent global validation.
            if not changed:
                changed = self.rewrite_final_leaf_set(diagnostics)
            if changed:
                self.normalize_tree_weights()
                continue
            self.logs.append({"step": "global_coverage_repair_no_change", "attempt": attempt})
            return False
        return False

    def accept_best_available_semantic_leaf_set(self, trigger_reason: str) -> bool:
        """Keep individually feasible leaves when strict set validation exhausts repairs.

        This is deliberately a last resort.  It never admits a non-atomic or
        semantically rejected leaf; it only relaxes the set-level requirement
        that every aspect/redundancy judgment must pass simultaneously.
        """
        if not (
            self.config.semantic_failure_fallback
            and self.uses_hard_semantic_constraints()
        ):
            return False
        retained = [
            leaf
            for leaf in self.leaf_scoring_points.values()
            if self._semantic_fallback_leaf_ok(leaf)
        ]
        if not retained:
            self.logs.append(
                {
                    "step": "semantic_final_best_feasible_fallback_unavailable",
                    "reason": trigger_reason,
                }
            )
            return False
        retained_ids = {leaf.id for leaf in retained}
        removed_ids = [
            leaf_id for leaf_id in self.leaf_scoring_points if leaf_id not in retained_ids
        ]
        for leaf_id in removed_ids:
            self.detach_final_leaf(leaf_id)
        self.prune_empty_internal_nodes()
        for leaf in retained:
            leaf.metadata["semantic_failure_fallback"] = True
            leaf.metadata["semantic_failure_fallback_stage"] = "final_set_validation"
        self.logs.append(
            {
                "step": "semantic_final_best_feasible_fallback",
                "retained_leaf_ids": [leaf.id for leaf in retained],
                "removed_leaf_ids": removed_ids,
                "retained_count": len(retained),
                "reason": trigger_reason,
                "relaxation": "set_level_coverage_or_redundancy_only",
            }
        )
        return True

    def active_terminal_count(self) -> int:
        return len(self.leaf_scoring_points) + sum(1 for _, _, node_id in self.frontier if self.nodes.get(node_id, None) and self.nodes[node_id].status == "frontier")

    def run_search(self) -> bool:
        steps = 0
        while self.frontier and len(self.leaf_scoring_points) < self.config.budget and steps < self.config.max_steps:
            node = self.pop_frontier()
            if node is None:
                break
            if not self.gate(node, self.config):
                if not (
                    self.uses_hard_semantic_constraints()
                    and not semantic_atomic(node)
                ):
                    self.mark_leaf(node, "gate_terminate")
                    steps += 1
                    continue
                self.logs.append(
                    {
                        "step": "semantic_hard_constraint_gate_override",
                        "node_id": node.id,
                        "depth": node.depth,
                        "max_depth": self.config.max_depth,
                        "reason": "non_atomic_nodes_cannot_be_final_leaves",
                    }
                )

            action, selected = self.decomposer(node)
            would_active_count = self.active_terminal_count() + len(selected)
            if action == "DECOMPOSE" and would_active_count <= self.config.budget:
                self.apply_decomposition(node, selected)
            else:
                if (
                    self.uses_hard_semantic_constraints()
                    and not semantic_atomic(node)
                ):
                    independent_count = int(node.metadata.get("independent_requirement_count", 2) or 2)
                    forced_children = self.forced_atomic_split(
                        node,
                        max_children=max(2, min(4, independent_count)),
                    )
                    if forced_children and self.ensure_terminal_capacity(
                        len(forced_children),
                        exclude_ids={node.id},
                    ):
                        self.apply_forced_atomic_split(node, forced_children)
                        action = "FORCED_ATOMIC_DECOMPOSE"
                        selected = forced_children
                    else:
                        stage = "semantic_atomicity"
                        reason = (
                            "Non-atomic node could not be decomposed within the semantic/budget constraints "
                            "after targeted atomic repair: "
                            f"node={node.id} action={action} active_if_decomposed={would_active_count} "
                            f"budget={self.config.budget}"
                        )
                        fallback = self.apply_best_semantic_fallback(node, reason)
                        if fallback is None:
                            self._set_failure(stage, reason)
                            return False
                        action = "BEST_SEMANTIC_FEASIBLE_FALLBACK"
                        selected = [fallback]
                else:
                    reason = "value_or_coverage_terminate" if action != "DECOMPOSE" else "budget_terminate"
                    self.mark_leaf(node, reason)
            steps += 1
            self.logs.append(
                {
                    "step": "search_step",
                    "step_index": steps,
                    "node_id": node.id,
                    "action": action,
                    "selected_children": [child.id for child in selected],
                    "used_budget": len(self.leaf_scoring_points),
                    "active_terminal_count": self.active_terminal_count(),
                }
            )

        while self.frontier and len(self.leaf_scoring_points) < self.config.budget:
            node = self.pop_frontier()
            if node is not None:
                if (
                    self.uses_hard_semantic_constraints()
                    and not semantic_atomic(node)
                ):
                    independent_count = int(node.metadata.get("independent_requirement_count", 2) or 2)
                    forced_children = self.forced_atomic_split(
                        node,
                        max_children=max(2, min(4, independent_count)),
                    )
                    if not forced_children or not self.ensure_terminal_capacity(
                        len(forced_children),
                        exclude_ids={node.id},
                    ):
                        reason = (
                            "Non-atomic frontier node could not be repaired into final atomic leaves: "
                            f"node={node.id}"
                        )
                        fallback = self.apply_best_semantic_fallback(node, reason)
                        if fallback is None:
                            self._set_failure("semantic_atomicity", reason)
                            return False
                        continue
                    self.apply_forced_atomic_split(node, forced_children)
                    continue
                self.mark_leaf(node, "final_frontier_flush")

        if not self.leaf_scoring_points:
            self._set_failure("search", "ARTS produced no leaf scoring points.")
            return False
        return True

    def to_rubric_tree(self) -> RubricTree:
        def convert(node_id: str) -> RubricNode:
            node = self.nodes[node_id]
            return RubricNode(
                node_id=node.id,
                text=node.text,
                children=[convert(child_id) for child_id in node.children],
                weight=node.importance if not node.children else 0.0,
            )

        return RubricTree(root=convert(self.root_id))

    def run(self) -> Optional[ARTSResult]:
        if not self.initialize():
            return None
        if not self.run_search():
            return None
        # The score-only/no-semantic ablation must not be rejected by the
        # semantic final validator it is intended to remove.  Semantic-only
        # and hybrid runs share exactly the same final feasible-set check.
        if self.uses_semantic_policy() and not self.final_coverage_repair():
            reason = "Final leaves failed coverage, atomicity, or semantic non-redundancy validation."
            if not self.accept_best_available_semantic_leaf_set(reason):
                self._set_failure("semantic_final_validation", reason)
                return None
        self.normalize_tree_weights()
        return ARTSResult(
            tree=self.to_rubric_tree(),
            internal_nodes={node_id: asdict(node) for node_id, node in self.internal_nodes.items()},
            leaf_scoring_points={node_id: asdict(node) for node_id, node in self.leaf_scoring_points.items()},
            raw_nodes={node_id: asdict(node) for node_id, node in self.nodes.items()},
            calibration_responses=self.calibration_responses,
            logs=self.logs,
            budget=self.config.budget,
            max_depth=self.config.max_depth,
            leaf_weighting=self.config.leaf_weighting,
        )


def generate_arts_tree(
    question: str,
    config: ARTSConfig,
    calibration_responses: Optional[list[str]] = None,
    gate: GateFn = default_gate,
    log_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> Optional[ARTSResult]:
    return ARTSRunner(
        question=question,
        config=config,
        calibration_responses=calibration_responses,
        gate=gate,
        log_callback=log_callback,
    ).run()


def generate_arts_tree_with_diagnostics(
    question: str,
    config: ARTSConfig,
    calibration_responses: Optional[list[str]] = None,
    gate: GateFn = default_gate,
    log_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> ARTSGenerationOutcome:
    runner = ARTSRunner(
        question=question,
        config=config,
        calibration_responses=calibration_responses,
        gate=gate,
        log_callback=log_callback,
    )
    try:
        result = runner.run()
    except Exception as exc:
        runner._set_failure("exception", f"{type(exc).__name__}: {str(exc)[:300]}")
        result = None
    return ARTSGenerationOutcome(
        result=result,
        failure_stage=runner.failure_stage,
        failure_reason=runner.failure_reason,
        logs=runner.logs,
    )


def aggregate(score_vectors: list[list[float]], beta: float = 8.0) -> list[float]:
    if not score_vectors:
        return []
    width = min(len(scores) for scores in score_vectors)
    if width <= 0:
        return []
    columns = [[_clamp01(scores[index]) for scores in score_vectors] for index in range(width)]
    if len(score_vectors) == 1:
        return list(columns[index][0] for index in range(width))
    if beta <= 0:
        return [min(column) for column in columns]
    epsilon = 1e-6
    return [(_mean([max(value, epsilon) ** (-beta) for value in column])) ** (-1.0 / beta) for column in columns]


def aggregate_node_score_matrices(nodes: list[ARTSNode], beta: float = 8.0) -> list[list[float]]:
    matrices = [node_score_matrix(node) for node in nodes]
    matrices = [matrix for matrix in matrices if matrix and matrix[0]]
    if not matrices:
        return []
    n_responses = min(len(matrix) for matrix in matrices)
    n_judges = min(len(row) for matrix in matrices for row in matrix[:n_responses])
    if n_responses <= 0 or n_judges <= 0:
        return []
    if len(matrices) == 1:
        return [
            [_clamp01(matrices[0][response_index][judge_index]) for judge_index in range(n_judges)]
            for response_index in range(n_responses)
        ]
    aggregated = []
    for response_index in range(n_responses):
        row = []
        for judge_index in range(n_judges):
            values = [_clamp01(matrix[response_index][judge_index]) for matrix in matrices]
            if beta <= 0:
                row.append(min(values))
            else:
                epsilon = 1e-6
                row.append((_mean([max(value, epsilon) ** (-beta) for value in values])) ** (-1.0 / beta))
        aggregated.append(row)
    return aggregated


def node_score_matrix(node: ARTSNode) -> list[list[float]]:
    matrix = node.metadata.get("judge_scores")
    if isinstance(matrix, list) and matrix and all(isinstance(row, list) for row in matrix):
        return [[_clamp01(value) for value in row] for row in matrix]
    return [[_clamp01(score)] for score in node.scores]


def judge_response_variances(score_matrix: list[list[float]]) -> list[float]:
    if not score_matrix:
        return []
    n_judges = min((len(row) for row in score_matrix), default=0)
    return [
        _variance([_clamp01(row[judge_index]) for row in score_matrix if len(row) > judge_index])
        for judge_index in range(n_judges)
    ]


def compute_distinct_matrix(score_matrix: list[list[float]]) -> float:
    return _mean(judge_response_variances(score_matrix))


def node_distinct(node: ARTSNode) -> float:
    return compute_distinct_matrix(node_score_matrix(node))


def semantic_distinctive(node: ARTSNode) -> float:
    quality = node.metadata.get("semantic_quality", {})
    if isinstance(quality, dict):
        return _clamp01(float(quality.get("distinctive", node.metadata.get("semantic_distinctive", 0.0)) or 0.0))
    return _clamp01(float(node.metadata.get("semantic_distinctive", 0.0) or 0.0))


def semantic_leaf_quality(node: ARTSNode) -> float:
    quality = node.metadata.get("semantic_quality", {})
    if isinstance(quality, dict):
        return _clamp01(float(quality.get("leaf_quality", node.metadata.get("semantic_leaf_quality", 0.0)) or 0.0))
    return _clamp01(float(node.metadata.get("semantic_leaf_quality", 0.0) or 0.0))


def semantic_atomic(node: ARTSNode) -> bool:
    quality = node.metadata.get("semantic_quality", {})
    if isinstance(quality, dict) and "atomic" in quality:
        return bool(quality.get("atomic"))
    return bool(node.metadata.get("semantic_atomic", False))


def semantic_candidate_score(node: ARTSNode) -> float:
    """Semantic-only quality used before calibration is legally consulted."""
    quality = node.metadata.get("semantic_quality", {})
    if not isinstance(quality, dict):
        return 0.0
    values = []
    for key in (
        "concrete",
        "verifiable",
        "unambiguous",
        "task_specific",
        "distinctive",
        "leaf_quality",
    ):
        try:
            values.append(_clamp01(float(quality.get(key, 0.0) or 0.0)))
        except (TypeError, ValueError):
            values.append(0.0)
    return _mean(values)


def semantic_substantive(node: ARTSNode) -> bool:
    quality = node.metadata.get("semantic_quality", {})
    if isinstance(quality, dict) and "substantive" in quality:
        return bool(quality.get("substantive"))
    return bool(node.metadata.get("semantic_substantive", False))


def semantic_format_only(node: ARTSNode) -> bool:
    quality = node.metadata.get("semantic_quality", {})
    if isinstance(quality, dict) and "format_only" in quality:
        return bool(quality.get("format_only"))
    return bool(node.metadata.get("semantic_format_only", False))


def semantic_explicit_query_requirement(node: ARTSNode) -> bool:
    quality = node.metadata.get("semantic_quality", {})
    if isinstance(quality, dict) and "explicit_query_requirement" in quality:
        return bool(quality.get("explicit_query_requirement"))
    return bool(node.metadata.get("semantic_explicit_query_requirement", False))


def score_confidence_penalty(node: ARTSNode) -> float:
    return _clamp01(float(node.metadata.get("score_confidence_penalty", 1.0) or 1.0))


def hybrid_score_selection_priority(node: ARTSNode) -> float:
    return float(node.scorability or 0.0) * node_distinct(node) * score_confidence_penalty(node)


def compute_distinct(scores: list[float]) -> float:
    return _variance(scores)


def spearman_rank_correlation(left: list[float], right: list[float]) -> float:
    width = min(len(left), len(right))
    if width < 2:
        return 0.0
    left_ranks = _ranks([float(value) for value in left[:width]])
    right_ranks = _ranks([float(value) for value in right[:width]])
    return _pearson(left_ranks, right_ranks)


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for rank_index in range(index, end):
            ranks[indexed[rank_index][0]] = average_rank
        index = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(left_var * right_var)
    if denominator <= 1e-12:
        return 0.0
    return numerator / denominator


def _positive_float(value: Any, default: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if math.isnan(parsed) or parsed <= 0.0:
        return default
    return parsed


def _text_jaccard(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-zA-Z0-9]+", left.lower()))
    right_tokens = set(re.findall(r"[a-zA-Z0-9]+", right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _json_object(text: str) -> Optional[dict[str, Any]]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        data = loads_json_with_escape_repair(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = loads_json_with_escape_repair(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def _plain_text_response(text: str) -> str:
    cleaned = re.sub(r"^```(?:json|text)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    if not cleaned:
        return ""
    parsed = _json_object(cleaned)
    if parsed is not None:
        return str(parsed.get("response", "")).strip()
    return cleaned


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _clamp01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _score_cache_key(question: str, response: str, criterion: str, judge_config: dict[str, Any]) -> str:
    payload = {
        "question": question,
        "response": response,
        "criterion": criterion,
        "base_url": judge_config.get("base_url"),
        "model": judge_config.get("model"),
        "temperature": judge_config.get("temperature", 0.0),
        "top_logprobs": judge_config.get("top_logprobs", 20),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _yes_no_probability(completion: Any) -> float:
    logprobs = completion.choices[0].logprobs
    if logprobs is None or not logprobs.content:
        return _yes_no_from_text(completion.choices[0].message.content or "")
    top = logprobs.content[0].top_logprobs
    p_yes = 0.0
    p_no = 0.0
    for entry in top:
        token = entry.token.strip().upper()
        if token in {"YES", "Y"}:
            p_yes += math.exp(entry.logprob)
        elif token in {"NO", "N"}:
            p_no += math.exp(entry.logprob)
    return 0.5 if p_yes + p_no < 1e-6 else p_yes / (p_yes + p_no)


def _yes_no_from_text(text: str) -> float:
    answer = text.strip().upper()
    if answer.startswith("YES"):
        return 1.0
    if answer.startswith("NO"):
        return 0.0
    return 0.5
