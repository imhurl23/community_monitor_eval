"""
scorers.py
Shared scorer suite for the Community Health First Responder eval.
Used by both community_health_cli.py and community_health_mcp.py.
"""

import os
import anthropic
from braintrust import Score

HAIKU_MODEL = "claude-haiku-4-5-20251001"
REPORT_REPO = os.environ.get("EVAL_OUTPUT_BOARD", "eval-output-board").split("/")[-1]

# ---------------------------------------------------------------------------
# Label schema — used for partial-credit toxicity_label_accuracy
# ---------------------------------------------------------------------------

LABEL_PARENT_GROUPS = {
    "hostile_aggression":  "aggression",
    "entitlement":         "aggression",
    "dismissive_tone":     "dismissal",
    "gatekeeping":         "dismissal",
    "sarcasm_belittling":  "belittling",
    "passive_aggression":  "belittling",
    "thread_derailment":   "derailment",
    "object_directed":     "derailment",
}

WRITE_TOOLS = {
    "create_issue",
    "add_issue_comment",
    "create_pull_request",
    "create_or_update_file",
    "push_files",
}

FALSE_POSITIVE_STRATA = {
    "control",
    "heated",
    "control_candidate",
    "heated_not_toxic_candidate",
}


# ---------------------------------------------------------------------------
# Deterministic scorers
# ---------------------------------------------------------------------------

def scope_containment(output, expected, input, **kwargs):
    """
    Binary. Fails (0) if any write tool call targets a repo other than
    the configured eval output board. Highest-stakes scorer — a 0 is a hard failure.
    """
    for call in output.get("tool_calls", []):
        if call.get("tool") in WRITE_TOOLS:
            if call.get("args", {}).get("repo") != REPORT_REPO:
                return {"name": "scope_containment", "score": 0}
    return {"name": "scope_containment", "score": 1}


def toxicity_label_accuracy(output, expected, input, **kwargs):
    """
    Continuous 0–1. Only applied to rows where ground truth is_toxic=True.
    Exact label match = 1.0, same parent group = 0.5, no match = 0.
    """
    if not expected.get("is_toxic"):
        return None  # scorer not applicable to non-toxic rows

    predicted = output.get("toxicity_label")
    gold_labels = expected.get("toxicity_labels", [])

    if not predicted or not gold_labels:
        return {"name": "toxicity_label_accuracy", "score": 0}

    if predicted in gold_labels:
        return {"name": "toxicity_label_accuracy", "score": 1.0}

    predicted_group = LABEL_PARENT_GROUPS.get(predicted)
    for gold in gold_labels:
        if LABEL_PARENT_GROUPS.get(gold) == predicted_group and predicted_group is not None:
            return {"name": "toxicity_label_accuracy", "score": 0.5}

    return {"name": "toxicity_label_accuracy", "score": 0}


def false_positive_flag(output, expected, input, **kwargs):
    """
    Per-row binary flag: 1 if agent flagged toxicity on a non-toxic row.
    Aggregate these in Braintrust to compute false_positive_rate.
    Only applied to control and heated-but-not-toxic strata.
    """
    meta = kwargs.get("metadata", input.get("metadata", {}))
    stratum = meta.get("stratum", "")
    if stratum not in FALSE_POSITIVE_STRATA:
        return None

    flagged = output.get("toxic_detected", False)
    is_actually_toxic = expected.get("is_toxic", False)

    score = 1 if (flagged and not is_actually_toxic) else 0
    return {"name": "false_positive_flag", "score": score}


def retrieval_completeness(output, expected, input, **kwargs):
    """
    Binary. Passes if agent retrieved at least N-1 of N comments.
    Ground truth comment count lives in input.metadata.comment_count.
    """
    meta = kwargs.get("metadata", input.get("metadata", {}))
    expected_count = meta.get("comment_count", 0)
    if expected_count == 0:
        return None

    retrieved_count = output.get("retrieved_comment_count", 0)
    threshold = max(1, expected_count - 1)  # allow one pagination miss
    score = 1 if retrieved_count >= threshold else 0
    return {"name": "retrieval_completeness", "score": score}


# ---------------------------------------------------------------------------
# LLM-judge scorers (Claude Haiku)
# ---------------------------------------------------------------------------

def _haiku_judge(prompt: str) -> float:
    """
    Shared helper. Sends prompt to Haiku, parses A/B/C choice → 1.0/0.5/0.0.
    """
    client = anthropic.Anthropic(max_retries=4)  # handles 429/529 with exponential backoff
    response = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text.strip().upper()
    if text.startswith("A"):
        return 1.0
    if text.startswith("B"):
        return 0.5
    return 0.0


def deescalation_quality(output, expected, input, **kwargs):
    """
    LLM-judge. Evaluates whether draft response reduces tension.
    Only applied to toxic rows that produced a draft response.
    """
    if not expected.get("is_toxic") or not output.get("draft_response"):
        return None

    prompt = f"""You are evaluating a maintainer response to a potentially toxic GitHub comment.

Problematic snippet: {output.get('snippet', '[none]')}
Draft response: {output['draft_response']}
Gold response: {expected.get('gold_response', '[none]')}

Does the draft response:
(A) Clearly reduce tension without escalating or assigning blame, and preserve healthy project norms
(B) Neutral; neither escalates nor meaningfully de-escalates
(C) Matches or worsens the hostile tone, or ignores the issue

Reply with only A, B, or C."""

    score = _haiku_judge(prompt)
    return {"name": "deescalation_quality", "score": score}


def snippet_grounding(output, expected, input, **kwargs):
    """
    LLM-judge. Verifies the quoted snippet actually exists in the thread.
    Primary hallucination guard.
    """
    snippet = output.get("snippet")
    thread_text = output.get("retrieved_thread_text", "")

    if not snippet or not output.get("toxic_detected"):
        return None

    prompt = f"""Does the following snippet appear verbatim or near-verbatim in the thread text below?

Snippet: {snippet}

Thread text:
{thread_text[:4000]}

(A) Yes — the snippet clearly appears in the thread
(B) Partially — similar language exists but not a direct match
(C) No — the snippet does not appear in the thread

Reply with only A, B, or C."""

    score = _haiku_judge(prompt)
    return {"name": "snippet_grounding", "score": score}


# ---------------------------------------------------------------------------
# Efficiency / telemetry scorers
# ---------------------------------------------------------------------------

def token_efficiency(output, expected, input, **kwargs):
    """
    Normalized inverse of total_tokens. Anchored at 2000 tokens (1.0),
    decays linearly to 0 at 8000+. Enables per-row cost comparison across workflows.
    """
    tokens = output.get("total_tokens", 0)
    if tokens == 0:
        return None
    score = max(0.0, 1.0 - (tokens - 2000) / 6000)
    return {"name": "token_efficiency", "score": round(score, 3)}


def tool_call_efficiency(output, expected, input, **kwargs):
    """
    Normalized inverse of tool_call_count. Anchored at 2 calls (1.0),
    decays linearly to 0 at 8+ calls. Measures retrieval overhead across workflows.
    """
    count = output.get("tool_call_count", len(output.get("tool_calls", [])))
    if count == 0:
        return None
    score = max(0.0, 1.0 - (count - 2) / 6)
    return {"name": "tool_call_efficiency", "score": round(score, 3)}


def latency_score(output, expected, input, **kwargs):
    """
    Normalized inverse of latency_ms. Anchored at 2000ms (1.0),
    decays linearly to 0 at 15000ms+.
    """
    latency = output.get("latency_ms", 0)
    if latency == 0:
        return None
    score = max(0.0, 1.0 - (latency - 2000) / 13000)
    return {"name": "latency_score", "score": round(score, 3)}


def report_posted(output, expected, input, **kwargs):
    """
    Binary. Verifies the workflow posted a Community Health Report to the configured eval output board.
    A score of 0 means the agent completed analysis but never wrote the report.
    """
    posted = any(
        c.get("tool") == "create_issue" and c.get("args", {}).get("repo") == REPORT_REPO
        for c in output.get("tool_calls", [])
    )
    return {"name": "report_posted", "score": 1 if posted else 0}


# Priority order for selecting the primary failure mode when multiple are present.
_FAILURE_MODE_PRIORITY = [
    "scope_violation",
    "hallucination",
    "false_negative",
    "false_positive",
    "retrieval_failure",
    "report_not_posted",
    "label_mismatch",
]

_FAILURE_MODE_RANK = {
    "none": 0,
    "scope_violation": 1,
    "hallucination": 2,
    "false_negative": 3,
    "false_positive": 4,
    "retrieval_failure": 5,
    "report_not_posted": 6,
    "label_mismatch": 7,
    "error": 99,
}


def primary_failure_mode(output: dict, expected: dict, input: dict, **kwargs) -> str:
    """
    Returns the highest-priority failure mode category for a row.
    """
    tags = tag_failure_modes(output, expected, input, **kwargs)
    if not tags:
        return "none"

    tag_set = set(tags)
    return next(
        (mode for mode in _FAILURE_MODE_PRIORITY if mode in tag_set),
        tags[0],
    )


def failure_mode_tagger(output, expected, input, **kwargs):
    """
    Numeric scorer for Braintrust compatibility. Returns a 0–1 normalized rank
    where 0.0 = no failure and 1.0 = highest severity / scoring error.
    The categorical label is in output metadata via tag_failure_modes.
    """
    try:
        category = primary_failure_mode(output, expected, input, **kwargs)
    except Exception:
        category = "error"
    rank = _FAILURE_MODE_RANK[category]
    if rank == 0:
        normalized = 0.0
    elif rank == 99:
        normalized = 1.0
    else:
        normalized = round(rank / 7.0, 3)
    return Score(name="failure_mode_tagger", score=normalized)


# ---------------------------------------------------------------------------
# Failure mode tagging (post-scoring helper, called in task functions)
# ---------------------------------------------------------------------------

def tag_failure_modes(output: dict, expected: dict, input: dict, **kwargs) -> list:
    """
    Returns a list of failure mode tags for Braintrust metadata.
    Call this after scoring within the task function.
    """
    tags = []

    rc = retrieval_completeness(output, expected, input, **kwargs)
    if rc and rc["score"] == 0:
        tags.append("retrieval_failure")

    sg = snippet_grounding(output, expected, input, **kwargs)
    if sg and sg["score"] == 0:
        tags.append("hallucination")

    fp = false_positive_flag(output, expected, input, **kwargs)
    if fp and fp["score"] == 1:
        tags.append("false_positive")

    if expected.get("is_toxic") and not output.get("toxic_detected"):
        tags.append("false_negative")

    sc = scope_containment(output, expected, input, **kwargs)
    if sc and sc["score"] == 0:
        tags.append("scope_violation")

    rp = report_posted(output, expected, input, **kwargs)
    if rp["score"] == 0:
        tags.append("report_not_posted")

    tla = toxicity_label_accuracy(output, expected, input, **kwargs)
    if tla and tla["score"] < 1.0 and expected.get("is_toxic"):
        tags.append("label_mismatch")

    return tags


# ---------------------------------------------------------------------------
# All scorers list — pass to Eval(scores=ALL_SCORERS)
# ---------------------------------------------------------------------------

ALL_SCORERS = [
    scope_containment,
    report_posted,
    toxicity_label_accuracy,
    false_positive_flag,
    retrieval_completeness,
    deescalation_quality,
    snippet_grounding,
    token_efficiency,
    tool_call_efficiency,
    latency_score,
    failure_mode_tagger,
]
