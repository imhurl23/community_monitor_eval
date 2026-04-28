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
import ssl
import subprocess
import time
import urllib.error
import urllib.request
import anthropic
import braintrust
import certifi
from braintrust import Eval, start_span, wrap_anthropic

from scorers import ALL_SCORERS, WRITE_TOOLS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT = "community-health-eval"
DATASET_NAME = "community_monitor_pandas"
RUN_NUMBER = os.environ.get("RUN_NUMBER", "1")
EXPERIMENT_PREFIX = os.environ.get("CLI_EXPERIMENT_PREFIX", "cli-improved")
EXPERIMENT_NAME = f"{EXPERIMENT_PREFIX}-run-{RUN_NUMBER}"
# Cost optimisation — retrieval and analysis both default to Haiku now.
# Override ANALYSIS_MODEL/RETRIEVAL_MODEL if you want to spend more for quality.
MODEL_PRICING = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
}
MODEL = os.environ.get("ANALYSIS_MODEL", "claude-haiku-4-5-20251001")
COST_PER_MTOK = MODEL_PRICING[MODEL]
RETRIEVAL_MODEL = os.environ.get("RETRIEVAL_MODEL", "claude-haiku-4-5-20251001")
RETRIEVAL_COST_PER_MTOK = MODEL_PRICING[RETRIEVAL_MODEL]
EVAL_OUTPUT_BOARD = os.environ.get("EVAL_OUTPUT_BOARD", "eval-output-board")
LABEL_CLASSIFIER_MODEL = os.environ.get("LABEL_CLASSIFIER_MODEL", "claude-sonnet-4-20250514")

# Throttle limits — override via env vars to prevent runaway spend
MAX_AGENT_TURNS = int(os.environ.get("MAX_AGENT_TURNS", "10"))
MAX_TOTAL_TOKENS = int(os.environ.get("MAX_TOTAL_TOKENS", "80000"))
MAX_TOOL_CALLS = int(os.environ.get("MAX_TOOL_CALLS", "6"))        # total tool calls per row
MAX_LATENCY_MS = int(os.environ.get("MAX_LATENCY_MS", "120000"))   # 2-min wall-clock limit per row
MAX_TOOL_EXEC_SECONDS = float(
    os.environ.get("CLI_TOOL_EXEC_SECONDS", os.environ.get("MAX_TOOL_EXEC_SECONDS", "30"))
)
# Cost optimisation — tip 2: 8 K chars ≈ 2 K tokens, enough for any real thread.
# Paginated responses can exceed 100 K chars and inflate context on every re-sent turn.
MAX_TOOL_RESULT_CHARS = int(os.environ.get("MAX_TOOL_RESULT_CHARS", "8000"))
MAX_TOOL_ERRORS = int(os.environ.get("MAX_TOOL_ERRORS", "3"))       # abort after N consecutive tool errors

# ---------------------------------------------------------------------------
# GitHub retrieval helpers
# ---------------------------------------------------------------------------

def _run_gh_raw(args: list) -> str:
    """Run a gh write command; returns raw stdout (URL) rather than parsing JSON."""
    try:
        result = subprocess.run(
            ["gh"] + args,
            capture_output=True,
            text=True,
            timeout=MAX_TOOL_EXEC_SECONDS,
            env={**os.environ, "GH_TOKEN": os.environ.get("GITHUB_AGENT_TOKEN", "")}
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"gh command timed out after {MAX_TOOL_EXEC_SECONDS}s") from exc
    if result.returncode != 0:
        raise RuntimeError(f"gh command failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _run_gh(args: list) -> dict | list:
    try:
        result = subprocess.run(
            ["gh"] + args,
            capture_output=True,
            text=True,
            timeout=MAX_TOOL_EXEC_SECONDS,
            env={**os.environ, "GH_TOKEN": os.environ.get("GITHUB_AGENT_TOKEN", "")}
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"gh command timed out after {MAX_TOOL_EXEC_SECONDS}s") from exc
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
    return f"[Unknown tool: {name}]"


def _build_report_body(repo: str, dtype: str, number: int, output: dict) -> str:
    if output.get("task_error"):
        return "\n".join([
            f"# Community Health Report for {repo} {dtype} #{number}",
            "",
            "## Status",
            "Analysis incomplete due to retrieval/tool failure.",
            "",
            "## Error",
            output["task_error"],
        ])

    if output.get("toxic_detected"):
        snippet = output.get("snippet") or "[none]"
        label = output.get("toxicity_label") or "[none]"
        severity = output.get("severity") or "[none]"
        response = output.get("draft_response") or "[none]"
        return "\n".join([
            f"# Community Health Report for {repo} {dtype} #{number}",
            "",
            "## Finding",
            "Toxic or discouraging content detected.",
            "",
            "## Snippet",
            f"> {snippet}",
            "",
            "## Label",
            label,
            "",
            "## Severity",
            severity,
            "",
            "## Proposed maintainer response",
            response,
        ])

    return "\n".join([
        f"# Community Health Report for {repo} {dtype} #{number}",
        "",
        "No toxic or discouraging content detected. No action needed.",
    ])


def _post_report_issue(board_repo: str, title: str, body: str) -> str:
    owner_repo = board_repo if "/" in board_repo else board_repo
    api_url = f"https://api.github.com/repos/{owner_repo}/issues"
    payload = json.dumps({"title": title, "body": body}).encode("utf-8")
    token = os.environ.get("GITHUB_AGENT_TOKEN", "")
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    request = urllib.request.Request(
        api_url,
        data=payload,
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30, context=ssl_context) as response:
            created = json.loads(response.read().decode("utf-8"))
            return created.get("html_url", "")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub issue create failed: HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GitHub issue create failed: {exc.reason}") from exc


def _classify_toxicity_label(client: anthropic.Anthropic, thread_text: str, snippet: str | None) -> str | None:
    prompt = f"""Choose exactly one toxicity label for the GitHub thread below.

Valid labels:
- hostile_aggression
- entitlement
- dismissive_tone
- sarcasm_belittling
- passive_aggression
- gatekeeping
- thread_derailment
- object_directed

Return only the label text, with no explanation.

Problematic snippet:
{snippet or '[none]'}

Thread text:
{thread_text[:6000]}"""
    response = client.messages.create(
        model=LABEL_CLASSIFIER_MODEL,
        max_tokens=50,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    label = response.content[0].text.strip()
    allowed = {
        "hostile_aggression",
        "entitlement",
        "dismissive_tone",
        "sarcasm_belittling",
        "passive_aggression",
        "gatekeeping",
        "thread_derailment",
        "object_directed",
    }
    return label if label in allowed else None


def _normalize_analysis_output(output: dict | None) -> dict:
    if not isinstance(output, dict):
        output = {}

    toxic_detected = output.get("toxic_detected")
    if isinstance(toxic_detected, bool):
        normalized_detected = toxic_detected
    else:
        normalized_detected = any(
            output.get(field) not in (None, "")
            for field in ("snippet", "toxicity_label", "severity", "draft_response")
        )

    output["toxic_detected"] = normalized_detected
    if not normalized_detected:
        output["snippet"] = None
        output["toxicity_label"] = None
        output["severity"] = None
        output["draft_response"] = None
        return output

    output["snippet"] = output.get("snippet") or None
    output["toxicity_label"] = output.get("toxicity_label") or None
    severity = output.get("severity")
    output["severity"] = severity if severity in {"low", "medium", "high"} else None
    output["draft_response"] = output.get("draft_response") or None
    return output


# Cost optimisation — tip 3: strip GitHub metadata from read-tool JSON responses.
# Raw get_issue / get_issue_comments results include timestamps, labels, milestones,
# assignees, etc. that are irrelevant to toxicity analysis. Compressing to just
# title + body + comment texts cuts each tool_result by ~60%, reducing the context
# re-sent to the LLM on every subsequent turn.
_READ_TOOLS = {"get_issue", "get_issue_comments", "get_pull_request", "get_pull_request_comments"}


def _count_retrieved_comments(tool_name: str, result_text: str) -> int:
    if tool_name not in _READ_TOOLS:
        return 0

    try:
        data = json.loads(result_text)
    except (json.JSONDecodeError, ValueError):
        return 0

    if tool_name == "get_issue":
        return 0

    if tool_name == "get_pull_request":
        comments = data.get("comments", []) if isinstance(data, dict) else []
        return len(comments) if isinstance(comments, list) else 0

    if tool_name in ("get_issue_comments", "get_pull_request_comments") and isinstance(data, list):
        return len(data)

    return 0


def _compress_tool_result(tool_name: str, result_text: str) -> str:
    """Strip GitHub metadata, retaining only content fields needed for toxicity analysis."""
    if tool_name not in _READ_TOOLS:
        return result_text
    try:
        data = json.loads(result_text)
    except (json.JSONDecodeError, ValueError):
        return result_text  # plain-text response — return as-is
    if tool_name == "get_issue":
        return json.dumps({
            "title": data.get("title", ""),
            "body": data.get("body", ""),
        })

    if tool_name == "get_pull_request":
        return json.dumps({
            "title": data.get("title", ""),
            "body": data.get("body", ""),
            "comments": [
                {
                    "author": c.get("author", {}).get("login", "") if isinstance(c.get("author"), dict) else c.get("login", ""),
                    "body": c.get("body", ""),
                }
                for c in data.get("comments", [])
            ],
        })
    if tool_name in ("get_issue_comments", "get_pull_request_comments"):
        if isinstance(data, list):
            return json.dumps([
                {"author": c.get("user", {}).get("login", ""), "body": c.get("body", "")}
                for c in data
            ])
    return result_text


# ---------------------------------------------------------------------------
# Task function
# ---------------------------------------------------------------------------

def cli_agent_task(input: dict) -> dict:
    repo = input["repo"]
    number = input["discussion_number"]
    dtype = input["discussion_type"]
    start_time = time.time()

    client = wrap_anthropic(anthropic.Anthropic(max_retries=4))  # handles 429/529 with exponential backoff
    tool_calls_log = []
    total_input_tokens = 0
    total_output_tokens = 0
    retrieval_input_tokens = 0   # tip 1: Haiku turns only
    retrieval_output_tokens = 0
    total_llm_ms = 0
    total_tool_ms = 0
    llm_turns = 0
    per_turn_tokens = []
    tool_error_count = 0
    tool_timeout_count = 0
    tool_error_messages = []
    task_error = None
    retrieved_thread_parts = []
    retrieved_comment_count = 0
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

Step 3 — Respond ONLY with a JSON object — no markdown, no preamble:
{{
  "toxic_detected": true or false,
  "snippet": "exact quoted text from the thread that is problematic, or null",
    "toxicity_label": "if toxic_detected is true, choose exactly one label from the schema above",
  "severity": "low, medium, or high, or null",
    "draft_response": "a de-escalating maintainer response, or null if not toxic"
}}""",
    }]

    with start_span("cli-agent") as span:
        while True:
            if llm_turns >= MAX_AGENT_TURNS:
                task_error = (
                    f"Agent loop aborted: reached MAX_AGENT_TURNS={MAX_AGENT_TURNS}. "
                    "Increase MAX_AGENT_TURNS env var if needed."
                )
                break
            if total_input_tokens + total_output_tokens >= MAX_TOTAL_TOKENS:
                task_error = (
                    f"Agent loop aborted: reached MAX_TOTAL_TOKENS={MAX_TOTAL_TOKENS} "
                    f"({total_input_tokens + total_output_tokens} tokens used). "
                    "Increase MAX_TOTAL_TOKENS env var if needed."
                )
                break
            if len(tool_calls_log) >= MAX_TOOL_CALLS:
                task_error = (
                    f"Agent loop aborted: reached MAX_TOOL_CALLS={MAX_TOOL_CALLS}. "
                    "Increase MAX_TOOL_CALLS env var if needed."
                )
                break
            elapsed_ms = int((time.time() - start_time) * 1000)
            if elapsed_ms >= MAX_LATENCY_MS:
                task_error = (
                    f"Agent loop aborted: exceeded MAX_LATENCY_MS={MAX_LATENCY_MS} "
                    f"({elapsed_ms}ms elapsed). Increase MAX_LATENCY_MS env var if needed."
                )
                break

            # Cost optimisation — tip 1: first turn is pure retrieval (the model only
            # decides which tools to call), so Haiku is sufficient. Later turns use
            # the cheaper ANALYSIS_MODEL by default, which is also Haiku unless overridden.
            # Cost optimisation — tip 4: cap max_tokens per turn to what it actually
            # produces. Retrieval turns emit only tool_use JSON (~50–150 tokens);
            # analysis turns emit the report body + final JSON (~600–900 tokens).
            is_retrieval_turn = len(messages) == 1
            turn_model = RETRIEVAL_MODEL if is_retrieval_turn else MODEL
            turn_max_tokens = 500 if is_retrieval_turn else 1200

            llm_call_start = time.time()
            response = client.messages.create(
                model=turn_model,
                max_tokens=turn_max_tokens,
                temperature=0,
                tools=CLI_TOOLS,
                messages=messages,
            )
            total_llm_ms += int((time.time() - llm_call_start) * 1000)
            llm_turns += 1

            turn_in = response.usage.input_tokens
            turn_out = response.usage.output_tokens
            total_input_tokens += turn_in
            total_output_tokens += turn_out
            if is_retrieval_turn:
                retrieval_input_tokens += turn_in
                retrieval_output_tokens += turn_out
            per_turn_tokens.append({"model": turn_model, "input": turn_in, "output": turn_out})

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
                    tool_error_count = 0  # reset on success
                    tool_calls_log[-1]["result"] = "ok"
                except RuntimeError as e:
                    tool_error_count += 1
                    result_text = f"[Tool error] {e}"
                    tool_error_messages.append(result_text)
                    tool_calls_log[-1]["error"] = result_text
                    if "timed out" in str(e).lower():
                        tool_timeout_count += 1
                    if tool_error_count >= MAX_TOOL_ERRORS:
                        task_error = (
                            f"Agent loop aborted after {tool_error_count} consecutive tool errors; "
                            f"last error: {result_text}"
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        })
                        break
                total_tool_ms += int((time.time() - tool_exec_start) * 1000)
                retrieved_comment_count += _count_retrieved_comments(block.name, result_text)
                # Cost optimisation — tip 3: strip metadata before char-truncation
                # so the character budget is spent on content, not timestamps/labels.
                result_text = _compress_tool_result(block.name, result_text)
                result_text = result_text[:MAX_TOOL_RESULT_CHARS]
                if block.name in _READ_TOOLS:
                    retrieved_thread_parts.append(result_text)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

            if task_error is not None:
                break

            messages.append({
                "role": "assistant",
                "content": [b.model_dump(exclude_none=True) for b in response.content],
            })
            messages.append({"role": "user", "content": tool_results})

            if task_error is not None:
                break

    output = _normalize_analysis_output(output)
    thread_text = "\n".join(retrieved_thread_parts)
    if task_error is not None:
        output["task_error"] = task_error
    output["tool_error_messages"] = tool_error_messages
    output["tool_timeout_count"] = tool_timeout_count
    output["analysis_status"] = "incomplete" if task_error else "completed"
    if output.get("toxic_detected"):
        output["toxicity_label"] = _classify_toxicity_label(
            client,
            thread_text,
            output.get("snippet"),
        ) or output.get("toxicity_label")

    report_title = f"Community Health Report — {repo} — {dtype} #{number}"
    report_body = _build_report_body(repo, dtype, number, output)
    try:
        report_url = _post_report_issue(EVAL_OUTPUT_BOARD, report_title, report_body)
    except RuntimeError as exc:
        output["report_post_error"] = str(exc)
    else:
        tool_calls_log.append({
            "tool": "create_issue",
            "args": {"repo": EVAL_OUTPUT_BOARD.split("/")[-1], "title": report_title, "body": report_body},
            "result": report_url,
        })

    latency_ms = int((time.time() - start_time) * 1000)
    analysis_input_tokens = total_input_tokens - retrieval_input_tokens
    analysis_output_tokens = total_output_tokens - retrieval_output_tokens
    # Cost optimisation — tip 1: apply per-model pricing so cost_usd is accurate.
    cost_usd = (
        retrieval_input_tokens * RETRIEVAL_COST_PER_MTOK["input"] +
        retrieval_output_tokens * RETRIEVAL_COST_PER_MTOK["output"] +
        analysis_input_tokens * COST_PER_MTOK["input"] +
        analysis_output_tokens * COST_PER_MTOK["output"]
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
    output["retrieval_model"] = RETRIEVAL_MODEL
    output["analysis_model"] = MODEL
    output["retrieval_tokens"] = retrieval_input_tokens + retrieval_output_tokens
    output["analysis_tokens"] = analysis_input_tokens + analysis_output_tokens
    output["workflow"] = "cli"
    output["model"] = MODEL
    output["discussion_id"] = f"{repo}-{dtype}-{number}"
    output["stratum"] = input.get("metadata", {}).get("stratum", "")
    output["retrieved_comment_count"] = retrieved_comment_count
    output["retrieved_thread_text"] = thread_text

    span.log(
        metadata={
            "workflow": "cli",
            "repo": repo,
            "discussion_type": dtype,
            "stratum": output["stratum"],
            "discussion_id": output["discussion_id"],
            "retrieval_model": RETRIEVAL_MODEL,
            "analysis_model": MODEL,
        },
        metrics={
            "latency_ms": latency_ms,
            "retrieval_latency_ms": total_tool_ms,
            "llm_latency_ms": total_llm_ms,
            "tool_calls": len(tool_calls_log),
            "tokens": output["total_tokens"],
            "prompt_tokens": total_input_tokens,
            "completion_tokens": total_output_tokens,
            "cost_usd": output["cost_usd"],
            "retrieval_tokens": output["retrieval_tokens"],
            "analysis_tokens": output["analysis_tokens"],
        },
    )

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
