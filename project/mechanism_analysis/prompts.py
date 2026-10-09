JSON_SYSTEM = """Return only one valid JSON object, without Markdown or hidden reasoning. Follow the requested schema exactly. Use concise observable reasons."""

TREE_ROOT_PROMPT = """Create the first level of an adaptive evaluation-rubric tree using ONLY the query below.
QUERY:
{query}

The root is Overall response quality. Split it into 2-8 distinct, useful semantic branches. A branch criterion describes what a good response must satisfy, without revealing an answer. Do not inspect or speculate about candidate responses. Return:
{{"children":[{{"name":"...","criterion":"..."}}]}}
"""

TREE_SPLIT_PROMPT = """Decide whether this rubric node should remain atomic or be split, using ONLY the query and node path.
QUERY:
{query}
NODE PATH: {path}
CURRENT CRITERION: {criterion}
CURRENT DEPTH: {depth}; MAX DEPTH: {max_depth}

Keep it atomic when it is independently judgeable and covers one requirement/failure mode. Split only when the criterion contains multiple non-overlapping requirements that are more diagnostic separately. Avoid unary splits and cosmetic restatements. Return one of:
{{"decision":"atomic","reason":"..."}}
or
{{"decision":"split","reason":"...","children":[{{"name":"...","criterion":"..."}}, ...]}}
Split must contain 2-6 children.
"""

LOCAL_SPLIT_CHECK_PROMPT = """Check whether the proposed DIRECT CHILDREN of one rubric parent are mutually exclusive.
Use ONLY the query, parent path, parent criterion, and proposed children. Do not use Gold rubrics or candidate responses.

QUERY:
{query}
PARENT PATH: {path}
PARENT CRITERION: {criterion}
PROPOSED DIRECT CHILDREN:
{children}

Flag a pair only when at least one condition holds:
- duplicate: they judge substantially the same response error or are paraphrases;
- subsumes: one criterion contains the other's full decision plus no clean independent boundary;
- partially_overlapping: they require the same response evidence and their pass/fail boundaries materially overlap.
Do not flag a pair merely because both refine the same parent. Return only overlapping pairs; use the exact child IDs. Return:
{{"valid":true,"overlaps":[]}}
or
{{"valid":false,"overlaps":[{{"left_id":"C1","right_id":"C2","relation":"duplicate|subsumes|partially_overlapping","reason":"observable overlap"}}]}}
"""

LOCAL_SPLIT_REPAIR_PROMPT = """Rewrite a proposed sibling split so all direct children are mutually exclusive while jointly preserving the parent's useful scope.
Use ONLY the query and parent information. Do not use Gold rubrics or candidate responses.

QUERY:
{query}
PARENT PATH: {path}
PARENT CRITERION: {criterion}
CURRENT CHILDREN:
{children}
OVERLAP FINDINGS:
{overlaps}

Return 2-6 complete replacement children. Each child must have a distinct pass/fail boundary and should rely on different response evidence where possible. Do not merely rename the old children. Return:
{{"children":[{{"name":"...","criterion":"..."}}, ...]}}
"""

FLAT_GENERATION_PROMPT = """Using ONLY the query, directly create a flat set of atomic evaluation rubrics.
QUERY:
{query}

Each item must be independently judgeable, relevant, non-overlapping, and describe what a good response must satisfy. Do not create a hierarchy; do not inspect or speculate about candidate responses; do not reveal an answer. Return:
{{"rubrics":[{{"criterion":"...","category":"..."}}, ...]}}
Generate 3-{max_leaves} items, stopping when the query is adequately covered.
"""

COVERAGE_PROMPT = """Compare generated rubrics with the immutable Gold rubrics for the query.
QUERY:
{query}
GOLD RUBRICS (ID, path, criterion, desired state):
{gold}
GENERATED RUBRICS (ID, path, criterion):
{generated}

Semantic equivalence counts even when wording differs. A Gold item is covered only if generated items jointly contain its full requirement. Classify each generated item as exactly one of:
- matched: substantively covers one or more Gold requirements;
- novel_valid: a useful, query-grounded evaluation category absent from all Gold items (not merely more detail or rephrasing);
- redundant: duplicates another generated item without adding a Gold requirement;
- off_task: vague, invalid, answer-revealing, or unsupported by the query.
For novel_valid only, novel_category must be exactly one of safety_risk, reasoning_numerical, context_personalization, instruction_compliance, factual_technical, evidence_attribution, completeness_relevance, communication_style, or other_valid. For all other classes it must be null.
Return every ID exactly once:
{{"gold_coverage":[{{"gold_id":"...","covered":true,"generated_ids":["..."],"reason":"..."}}],"generated_classification":[{{"generated_id":"...","class":"matched|novel_valid|redundant|off_task","gold_ids":["..."],"novel_category":null,"reason":"..."}}]}}
"""

FLAT_LOCALIZATION_PROMPT = """Judge a response against every Gold rubric. For desired_state=criterion_met, pass means the response satisfies the criterion. For desired_state=criterion_not_met, pass means the undesirable criterion is absent. Use only observable response content. Return every leaf exactly once:
QUERY:
{query}
RESPONSE:
{response}
GOLD LEAVES:
{leaves}
Return {{"leaf_results":[{{"leaf_id":"...","passes":true,"evidence":"..."}}]}}
"""

FLAT_SINGLE_LEAF_PROMPT = """Judge ONE Gold rubric leaf without any Tree-path context. Make the final pass/fail decision only for this leaf criterion. For desired_state=criterion_met, pass means the response satisfies the criterion. For desired_state=criterion_not_met, pass means the undesirable criterion is absent. Use only observable response content.

QUERY:
{query}
RESPONSE:
{response}
LEAF ID: {leaf_id}
LEAF CRITERION: {criterion}
DESIRED STATE: {desired_state}

Return {{"passes":true,"evidence":"brief observable evidence tied to this leaf"}}
"""

TREE_ROUTE_PROMPT = """Route a top-down Gold-tree evaluation. Decide which direct branches may contain at least one failed desired state. Be recall-oriented but do not flag a branch without observable evidence. Direct leaf branches are judged here; internal branches are only routed.
QUERY:
{query}
RESPONSE:
{response}
CURRENT PATH: {path}
DIRECT BRANCHES:
{branches}
Return every branch exactly once:
{{"branch_results":[{{"branch_id":"...","kind":"leaf|internal","possible_failure":true,"leaf_passes":false,"evidence":"..."}}]}}
For internal branches leaf_passes must be null. For leaves, possible_failure must equal not leaf_passes.
"""

TREE_NODE_GATE_PROMPT = """Judge ONE Gold-tree node as a top-down routing gate. Determine whether the response has any failure within this node's semantic scope. Do not judge or enumerate sibling nodes.

QUERY:
{query}
RESPONSE:
{response}
NODE PATH: {path}
NODE NAME: {name}
NODE DESCRIPTION: {description}
NODE ASSIGNMENT SUMMARY: {assignment_summary}

Return fail when there is observable evidence of an error in this scope. Return pass only when the scope is clearly satisfied. Return uncertain when evidence is incomplete or the parent description is broader than what can be safely verified; uncertain will descend to children to protect recall.
Return {{"decision":"pass|fail|uncertain","evidence":"brief observable evidence"}}
"""

TREE_SINGLE_LEAF_PROMPT = """Judge ONE Gold rubric leaf. For desired_state=criterion_met, pass means the response satisfies the criterion. For desired_state=criterion_not_met, pass means the undesirable criterion is absent. Use only observable response content.

QUERY:
{query}
RESPONSE:
{response}
LEAF ID: {leaf_id}
CRITERION: {criterion}
DESIRED STATE: {desired_state}

Return {{"passes":true,"evidence":"brief observable evidence"}}
"""

TREE_PATH_LEAF_PROMPT = """You are given ONE Root-to-Leaf semantic path from a larger Gold rubric tree. This path is not the whole tree. The final item in the path is the only leaf being judged.

Use the parent nodes on this path to understand the leaf's intended semantic scope, terminology, and required evidence. The parent nodes are NOT gates: they are context, are not scored separately, and must not decide whether the leaf is inspected. Determine whether the RESPONSE violates the final leaf standard.

- If desired_state=criterion_met, the leaf is violated when the response does not satisfy the leaf criterion.
- If desired_state=criterion_not_met, the leaf is violated when the response exhibits the undesirable criterion.

Return passes=false exactly when the leaf is violated; otherwise return passes=true. Base the decision on observable response content interpreted under this path.

QUERY:
{query}
RESPONSE:
{response}
FULL SEMANTIC PATH FROM ROOT TO LEAF:
{semantic_path}
LEAF ID: {leaf_id}
LEAF CRITERION: {criterion}
DESIRED STATE: {desired_state}

Return {{"passes":true,"evidence":"brief observable evidence tied to this leaf"}}
"""

TREE_SOFT_ROUTER_PROMPT = """You are performing ERROR ROUTING, not final scoring of a parent node.

Judge whether the response may violate ANY descendant standard under the current node. Do not conclude that all descendants pass merely because the response performs well overall in this dimension. Any local omission, incorrect condition, factual conflict, numerical/relational error, or safety risk is enough to raise inspection priority.

QUERY:
{query}
RESPONSE:
{response}
CURRENT NODE PATH: {path}
CURRENT NODE NAME: {name}
CURRENT NODE DESCRIPTION: {description}
CURRENT NODE ASSIGNMENT SUMMARY: {assignment_summary}
DIRECT CHILD BRANCHES AVAILABLE FOR ROUTING:
{branches}

This stage MUST NOT make final leaf pass/fail decisions. Estimate routing probabilities only. Candidate children may be a subset, but every returned child_id must exactly match an available branch ID. Include the most suspicious children even when probability is below 0.30 if uncertainty remains.

Return:
{{
  "node_id":"{node_id}",
  "descendant_violation_probability":0.0,
  "routing_decision":"expand|deprioritize",
  "candidate_children":[{{"child_id":"...","probability":0.0}}],
  "evidence":["brief suspicious evidence"],
  "uncertainty":"low|medium|high"
}}
"""

WEIGHT_PROMPT = """Allocate relative importance among these sibling Gold-rubric branches for the query. Use only Query and Gold rubric structure; never use candidate answers or preference labels. Every listed Gold branch is applicable and must retain positive influence. Assign each child an INTEGER importance value from 1 to 5: 1 means low importance and 5 means core importance. Never return 0, a negative number, a fraction, or a value above 5. For an internal child, value represents the total importance of its entire descendant subtree. Values will be normalized across this sibling group and then multiplied by the parent's global weight.
QUERY:
{query}
PARENT PATH: {path}
CHILDREN:
{children}
Return every child exactly once: {{"weights":[{{"id":"...","value":1,"reason":"..."}}]}}
"""

FLAT_WEIGHT_PROMPT = """Assign direct relative importance to all atomic Gold rubrics for the query. Use only Query and rubric text; never use candidate answers or preference labels. Every listed Gold rubric is applicable and must retain positive influence. Assign each leaf an INTEGER importance value from 1 to 5: 1 means low importance and 5 means core importance. Never return 0, a negative number, a fraction, or a value above 5. Values will be normalized across all Gold leaves.
QUERY:
{query}
GOLD LEAVES:
{leaves}
Return every leaf exactly once: {{"weights":[{{"id":"...","value":1,"reason":"..."}}]}}
"""

FLAT_SINGLE_LEAF_WEIGHT_PROMPT = """Assign importance to ONE atomic Gold rubric for the query. You do not see the Tree path or any other rubric. Judge this rubric's intrinsic normative importance using only the Query and rubric text; never use candidate answers or preference labels.

Assign an INTEGER importance value from 1 to 5. The independently assigned values for all rubrics will later be globally normalized to sum to 1. Never return 0, a negative number, a fraction, or a value above 5.

QUERY:
{query}
RUBRIC ID: {leaf_id}
RUBRIC: {criterion}
DESIRED STATE: {desired_state}

Return exactly: {{"id":"{leaf_id}","value":1,"reason":"brief importance rationale"}}
"""

PATH_LEAF_WEIGHT_PROMPT = """Assign importance to ONE atomic Gold-rubric leaf for the query. The complete semantic path from the Tree root to this leaf is context that disambiguates the criterion. Judge the leaf's intrinsic normative importance for answering the query correctly; do not score path depth and do not allocate a parent budget. Use only Query and Gold-rubric path; never use candidate answers or preference labels.

Assign an INTEGER importance value from 1 to 5:
- 1: low importance;
- 3: meaningful but not decisive;
- 5: essential/core requirement.
Never return 0, a negative number, a fraction, or a value above 5. The independently assigned values for all leaves will later be globally normalized to sum to 1.

QUERY:
{query}
FULL SEMANTIC PATH FROM ROOT TO LEAF:
{semantic_path}
LEAF ID: {leaf_id}
LEAF CRITERION: {criterion}
DESIRED STATE: {desired_state}

Return exactly: {{"id":"{leaf_id}","value":1,"reason":"brief importance rationale"}}
"""

FULL_TREE_TOP_DOWN_WEIGHT_PROMPT = """Allocate relative importance among the CURRENT sibling Gold-rubric branches. First inspect the complete Gold Tree to understand how the current group fits into the global rubric structure. Then assign weights only to CURRENT CHILDREN. Use only Query and Gold rubric structure; never use candidate answers or preference labels. Every current child is applicable and must retain positive influence. Assign each current child an INTEGER importance value from 1 to 5. For an internal child, its value represents the total importance of its descendant subtree. Current sibling values will be normalized, multiplied by the parent's global weight, and recursively propagated.

QUERY:
{query}
COMPLETE GOLD TREE:
{full_tree}
CURRENT PARENT PATH:
{path}
CURRENT CHILDREN TO WEIGHT:
{children}

Return every CURRENT child exactly once and no other node: {{"weights":[{{"id":"...","value":1,"reason":"..."}}]}}
"""

FULL_TREE_PATH_LEAF_WEIGHT_PROMPT = """Assign importance to ONE atomic Gold-rubric leaf. First inspect the complete Gold Tree to calibrate this leaf against all other branches and leaves. Then use the highlighted Root-to-Leaf semantic path to interpret the target leaf precisely. Use only Query and Gold rubric structure; never use candidate answers or preference labels. Do not allocate a parent budget.

Assign an INTEGER importance value from 1 to 5. The independently assigned values for all leaves will later be globally normalized to sum to 1. Never return 0, a negative number, a fraction, or a value above 5.

QUERY:
{query}
COMPLETE GOLD TREE:
{full_tree}
TARGET ROOT-TO-LEAF PATH:
{semantic_path}
LEAF ID: {leaf_id}
LEAF CRITERION: {criterion}
DESIRED STATE: {desired_state}

Return exactly: {{"id":"{leaf_id}","value":1,"reason":"brief importance rationale"}}
"""

RESPONSE_CONDITIONED_TREE_WEIGHT_PROMPT = """Allocate relative importance among these sibling Gold-rubric branches for comparing Candidate A and Candidate B on the query. You may inspect both candidates, but you are assigning criterion importance—not selecting a winner. Never use or guess a preference label. Every listed Gold branch is applicable and must retain positive influence. Give higher importance to a branch when it is intrinsically important for the query and/or materially distinguishes the two candidates. Do not lower an important safety or hard-constraint branch merely because both candidates appear similar on it. Assign each child an INTEGER importance value from 1 to 5: 1 means low importance and 5 means core importance. Never return 0, a negative number, a fraction, or a value above 5. For an internal child, value represents the total importance of its entire descendant subtree. Values will be normalized across this sibling group and multiplied by the parent's global weight.
QUERY:
{query}
CANDIDATE A:
{response_a}
CANDIDATE B:
{response_b}
PARENT PATH: {path}
CHILDREN:
{children}
Return every child exactly once: {{"weights":[{{"id":"...","value":1,"reason":"..."}}]}}
"""

RESPONSE_CONDITIONED_FLAT_WEIGHT_PROMPT = """Assign direct relative importance to all atomic Gold rubrics for comparing Candidate A and Candidate B on the query. You may inspect both candidates, but you are assigning criterion importance—not selecting a winner. Never use or guess a preference label. Every listed Gold rubric is applicable and must retain positive influence. Give higher importance to a criterion when it is intrinsically important for the query and/or materially distinguishes the two candidates. Do not lower an important safety or hard-constraint criterion merely because both candidates appear similar on it. Assign each leaf an INTEGER importance value from 1 to 5: 1 means low importance and 5 means core importance. Never return 0, a negative number, a fraction, or a value above 5. Values will be normalized across all Gold leaves.
QUERY:
{query}
CANDIDATE A:
{response_a}
CANDIDATE B:
{response_b}
GOLD LEAVES:
{leaves}
Return every leaf exactly once: {{"weights":[{{"id":"...","value":1,"reason":"..."}}]}}
"""
