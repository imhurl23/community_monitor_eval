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
from braintrust import Eval, start_span

from scorers import ALL_SCORERS, WRITE_TOOLS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT = "community-health-eval"
DATASET_NAME = "pd_community_health_labeled"
RUN_NUMBER = os.environ.get("RUN_NUMBER", "1")
EXPERIMENT_NAME = f"cli-baseline-run-{RUN_NUMBER}"
MODEL = "claude-sonnet-4-20250514"
COST_PER_MTOK = {"input": 3.0, "output": 15.0}  # claude-sonnet-4 pricing
EVAL_OUTPUT_BOARD = os.environ.get("EVAL_OUTPUT_BOARD", "eval-output-board")

# ---------------------------------------------------------------------------
# GitHub retrieval helpers
# ---------------------------------------------------------------------------

def _run_gh_raw(args: list) -> str:
    """Run a gh write command; returns raw stdout (URL) rather than parsing JSON."""
    result = subprocess.run(
        ["gh"] + args,
        capture_output=True,
        text=True,
        env={**os.environ, "GH_TOKEN": os.environ.get("GITHUB_AGENT_TOKEN", "")}
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh command failed: {result.stderr.strip()}")
    return result.stdout.strip()


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
# Tool definitions and client-side execution
# ---------------------------------------------------------------------------

CLI_TOOLS = [
    {
        "name": "get_issue",
        "description": "Fetch a GitHub issue including title, body, and comments.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/repo (e.g. pandas-dev/pandas)"},
                "number": {"type": "integer"},
            },
            "required": ["repo", "number"],
        },
    },
    {
        "name": "get_issue_comments",
        "description": "Fetch all comments for a GitHub issue, paginated.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "number": {"type": "integer"},
            },
            "required": ["repo", "number"],
        },
    },
    {
        "name": "get_pull_request",
        "description": "Fetch a GitHub pull request including title, body, and review metadata.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "number": {"type": "integer"},
            },
            "required": ["repo", "number"],
        },
    },
    {
        "name": "get_pull_request_comments",
        "description": "Fetch all inline review comments for a GitHub pull request, paginated.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "number": {"type": "integer"},
            },
            "required": ["repo", "number"],
        },
    },
    {
        "name": "create_issue",
        "description": "Create a new GitHub issue in a repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/repo for the target repository"},
                "title": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["repo", "title", "body"],
        },
    },
]


def _exec_tool(name: str, args: dict) -> str:
    if name == "get_issue":
        return json.dumps(fetch_issue(args["repo"], int(args["number"])))
    if name == "get_issue_comments":
        return json.dumps(fetch_issue_comments_paginated(args["repo"], int(args["number"])))
    if name == "get_pull_request":
        return json.dumps(fetch_pr(args["repo"], int(args["number"])))
    if name == "get_pull_request_comments":
        return json.dumps(fetch_pr_review_comments_paginated(args["repo"], int(args["number"])))
    if name == "create_issue":
        return _run_gh_raw([
            "issue", "create",
            "--repo", args["repo"],
            "--title", args["title"],
            "--body", args["body"],
        ])
    return f"[Unknown tool: {name}]"


# ---------------------------------------------------------------------------
# Task function
# ---------------------------------------------------------------------------

def cli_agent_task(input: dict) -> dict:
    repo = input["repo"]
    number = input["discussion_number"]
    dtype = input["discussion_type"]
    start_time = time.time()

    client = anthropic.Anthropic()
    tool_calls_log = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_llm_ms = 0
    total_tool_ms = 0
    llm_turns = 0
    per_turn_tokens = []
    output = {}

    messages = [{
        "role": "user",
        "content": f"""You are a community health analyst for open source projects.

Step 1 — Retrieve the full comment thread for {dtype} #{number} in {repo}:
For issues: call get_issue AND get_issue_comments to ensure you have all comments.
For PRs: call get_pull_request AND get_pull_request_comments.

Step 2 — Analyze the thread for toxic or discouraging content using this OSS-specific schema:
- hostile_aggression: explicit threats, insults, name-calling
- entitlement: demanding tone ("fix this NOW", "why hasn't this been done")
- dismissive_tone: curt rejection, closing without engagement
- sarcasm_belittling: ironic minimization, mocking effort or skill
- passive_aggression: technically civil but subtly hostile framing
- gatekeeping: condescension toward perceived skill level
- thread_derailment: personal attacks displacing technical discussion
- object_directed: hostility directed at code/project ("this codebase is trash")

Step 3 — Post a Community Health Report as a new issue using create_issue:
  repo: "{EVAL_OUTPUT_BOARD}"
  title: "Community Health Report — {repo} — {dtype} #{number}"
  body: A markdown report with your finding, snippet, label, severity, and proposed maintainer
        response — or "No toxic or discouraging content detected. No action needed." if clean.

Step 4 — Respond ONLY with a JSON object — no markdown, no preamble:
{{
  "toxic_detected": true or false,
  "snippet": "exact quoted text from the thread that is problematic, or null",
  "toxicity_label": "one label from the schema above, or null",
  "severity": "low, medium, or high, or null",
  "draft_response": "a de-escalating maintainer response, or null if not toxic",
  "retrieved_thread_text": "full concatenated thread text you retrieved"
}}""",
    }]

    with start_span("cli-agent"):
        while True:
            llm_call_start = time.time()
            response = client.messages.create(
                model=MODEL,
                max_tokens=2000,
                tools=CLI_TOOLS,
                messages=messages,
            )
            total_llm_ms += int((time.time() - llm_call_start) * 1000)
            llm_turns += 1

            turn_in = response.usage.input_tokens
            turn_out = response.usage.output_tokens
            total_input_tokens += turn_in
            total_output_tokens += turn_out
            per_turn_tokens.append({"input": turn_in, "output": turn_out})

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            if not tool_use_blocks:
                for block in response.content:
                    if block.type == "text":
                        raw = block.text.strip()
                        if raw.startswith("```"):
                            raw = raw.split("```")[1]
                            if raw.startswith("json"):
                                raw = raw[4:]
                        try:
                            output = json.loads(raw.strip())
                        except json.JSONDecodeError:
                            pass
                break

            tool_results = []
            for block in tool_use_blocks:
                args = block.input if isinstance(block.input, dict) else {}
                # Normalize repo to bare name for write tools so scope_containment scorer works
                logged_args = ({**args, "repo": args.get("repo", "").split("/")[-1]}
                               if block.name in WRITE_TOOLS else args)
                tool_calls_log.append({"tool": block.name, "args": logged_args})

                tool_exec_start = time.time()
                try:
                    result_text = _exec_tool(block.name, args)
                except RuntimeError as e:
                    result_text = f"[Tool error] {e}"
                total_tool_ms += int((time.time() - tool_exec_start) * 1000)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

            messages.append({
                "role": "assistant",
                "content": [b.model_dump(exclude_none=True) for b in response.content],
            })
            messages.append({"role": "user", "content": tool_results})

    thread_text = output.pop("retrieved_thread_text", "")
    retrieved_comment_count = thread_text.count("\n[") if thread_text else 0
    latency_ms = int((time.time() - start_time) * 1000)
    cost_usd = (
        total_input_tokens * COST_PER_MTOK["input"] +
        total_output_tokens * COST_PER_MTOK["output"]
    ) / 1_000_000

    output["tool_calls"] = tool_calls_log
    output["tool_call_count"] = len(tool_calls_log)
    output["latency_ms"] = latency_ms
    output["retrieval_latency_ms"] = total_tool_ms
    output["llm_latency_ms"] = total_llm_ms
    output["llm_turns"] = llm_turns
    output["per_turn_tokens"] = per_turn_tokens
    output["total_tokens"] = total_input_tokens + total_output_tokens
    output["prompt_tokens"] = total_input_tokens
    output["completion_tokens"] = total_output_tokens
    output["cost_usd"] = round(cost_usd, 6)
    output["workflow"] = "cli"
    output["model"] = MODEL
    output["discussion_id"] = f"{repo}-{dtype}-{number}"
    output["stratum"] = input.get("metadata", {}).get("stratum", "")
    output["retrieved_comment_count"] = retrieved_comment_count
    output["retrieved_thread_text"] = thread_text

    return output


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset():
    braintrust.login(api_key=os.environ["BRAINTRUST_API_KEY"])
    dataset = braintrust.init_dataset(PROJECT, DATASET_NAME)
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
