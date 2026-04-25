"""
community_health_cli.py
CLI workflow eval for the Community Health First Responder experiment.
Uses `gh` CLI subprocess calls for GitHub retrieval.

Run:
    bt eval community_health_cli.py --project "community-health-eval"

Set RUN_NUMBER env var to name experiments:
    RUN_NUMBER=1 bt eval community_health_cli.py --project "community-health-eval"
"""

import json
import os
import subprocess
import time

import anthropic
import braintrust
from braintrust import Eval, traced

from scorers import ALL_SCORERS, tag_failure_modes

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT = "community-health-eval"
DATASET_NAME = "community-health-v1"
RUN_NUMBER = os.environ.get("RUN_NUMBER", "1")
EXPERIMENT_NAME = f"cli-baseline-run-{RUN_NUMBER}"
MODEL = "claude-sonnet-4-20250514"

TOXICITY_LABELS = [
    "hostile_aggression", "entitlement", "dismissive_tone",
    "sarcasm_belittling", "passive_aggression", "gatekeeping",
    "thread_derailment", "object_directed",
]

# ---------------------------------------------------------------------------
# GitHub retrieval helpers
# ---------------------------------------------------------------------------

def _run_gh(args: list) -> dict | list:
    result = subprocess.run(
        ["gh"] + args,
        capture_output=True,
        text=True,
        env={**os.environ, "GH_TOKEN": os.environ.get("GITHUB_AGENT_TOKEN", "")}
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh command failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def fetch_issue(repo: str, number: int) -> dict:
    return _run_gh([
        "issue", "view", str(number),
        "--repo", repo,
        "--json", "title,body,comments,author,number"
    ])


def fetch_pr(repo: str, number: int) -> dict:
    return _run_gh([
        "pr", "view", str(number),
        "--repo", repo,
        "--json", "title,body,comments,reviews,number"
    ])


def fetch_issue_comments_paginated(repo: str, number: int) -> list:
    """Fallback for threads with >30 comments."""
    return _run_gh([
        "api",
        f"/repos/{repo}/issues/{number}/comments",
        "--paginate"
    ])


def fetch_pr_review_comments_paginated(repo: str, number: int) -> list:
    """Fallback for PR inline review comments — known gap in gh pr view."""
    return _run_gh([
        "api",
        f"/repos/{repo}/pulls/{number}/comments",
        "--paginate"
    ])


def build_thread_text(thread_data: dict, extra_comments: list | None = None) -> str:
    """Flatten thread data into a single string for LLM context."""
    lines = []
    lines.append(f"Title: {thread_data.get('title', '')}")
    lines.append(f"Body: {thread_data.get('body', '')}")
    lines.append("---")

    comments = thread_data.get("comments", [])
    for c in comments:
        author = c.get("author", {}).get("login", "unknown") if isinstance(c.get("author"), dict) else c.get("author", "unknown")
        lines.append(f"[{author}]: {c.get('body', '')}")

    if extra_comments:
        for c in extra_comments:
            lines.append(f"[{c.get('user', {}).get('login', 'unknown')}]: {c.get('body', '')}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task function
# ---------------------------------------------------------------------------

def cli_agent_task(input: dict) -> dict:
    repo = input["repo"]
    number = input["discussion_number"]
    dtype = input["discussion_type"]
    start = time.time()
    tool_calls = []
    extra_comments = None

    # --- Retrieval ---
    with traced("gh-retrieve"):
        if dtype == "issue":
            thread_data = fetch_issue(repo, number)
            tool_calls.append({"tool": "gh_issue_view", "args": {"repo": repo, "number": number}})

            # Use paginated fallback if comment count suggests truncation
            expected_count = input.get("metadata", {}).get("comment_count", 0)
            if expected_count > 28:
                extra_comments = fetch_issue_comments_paginated(repo, number)
                tool_calls.append({"tool": "gh_api_paginate", "args": {"repo": repo, "number": number}})
        else:
            thread_data = fetch_pr(repo, number)
            tool_calls.append({"tool": "gh_pr_view", "args": {"repo": repo, "number": number}})

            # Always fetch PR inline review comments via API fallback
            extra_comments = fetch_pr_review_comments_paginated(repo, number)
            tool_calls.append({"tool": "gh_api_pr_comments", "args": {"repo": repo, "number": number}})

    thread_text = build_thread_text(thread_data, extra_comments)
    retrieved_comment_count = len(thread_data.get("comments", [])) + (len(extra_comments) if extra_comments else 0)

    # --- LLM analysis ---
    client = anthropic.Anthropic()
    with traced("llm-analyze"):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": f"""You are a community health analyst for open source projects.
Analyze the following GitHub {dtype} thread for toxic or discouraging content.

Use this toxicity label schema (OSS-specific):
- hostile_aggression: explicit threats, insults, name-calling
- entitlement: demanding tone ("fix this NOW", "why hasn't this been done")
- dismissive_tone: curt rejection, closing without engagement
- sarcasm_belittling: ironic minimization, mocking effort or skill
- passive_aggression: technically civil but subtly hostile framing
- gatekeeping: condescension toward perceived skill level
- thread_derailment: personal attacks displacing technical discussion
- object_directed: hostility directed at code/project ("this codebase is trash")

Thread:
{thread_text}

Respond ONLY with a JSON object — no markdown, no preamble:
{{
  "toxic_detected": true or false,
  "snippet": "exact quoted text from the thread that is problematic, or null",
  "toxicity_label": "one label from the schema above, or null",
  "severity": "low, medium, or high, or null",
  "draft_response": "a de-escalating maintainer response, or null if not toxic"
}}"""
            }]
        )

    raw = response.content[0].text.strip()
    # Strip markdown fences if model wraps output despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    output = json.loads(raw.strip())

    latency_ms = int((time.time() - start) * 1000)

    output["tool_calls"] = tool_calls
    output["latency_ms"] = latency_ms
    output["total_tokens"] = response.usage.input_tokens + response.usage.output_tokens
    output["prompt_tokens"] = response.usage.input_tokens
    output["completion_tokens"] = response.usage.output_tokens
    output["discussion_id"] = f"{repo}-{dtype}-{number}"
    output["retrieved_comment_count"] = retrieved_comment_count
    output["retrieved_thread_text"] = thread_text

    return output


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset():
    bt = braintrust.login(api_key=os.environ["BRAINTRUST_API_KEY"])
    dataset = bt.get_dataset(PROJECT, DATASET_NAME)
    return list(dataset)


# ---------------------------------------------------------------------------
# Eval entry point
# ---------------------------------------------------------------------------

Eval(
    PROJECT,
    experiment_name=EXPERIMENT_NAME,
    data=load_dataset,
    task=cli_agent_task,
    scores=ALL_SCORERS,
    metadata={"workflow": "cli", "model": MODEL},
)
