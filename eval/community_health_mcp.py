"""
community_health_mcp.py
MCP workflow eval for the Community Health First Responder experiment.
Uses the GitHub MCP server via mcp-proxy SSE bridge for retrieval,
with client-side tool execution so the server can run locally.

Prerequisites:
    mcp-proxy running on port 8080:
    mcp-proxy --port 8080 -- docker run -i --rm \\
        -e GITHUB_PERSONAL_ACCESS_TOKEN="$GITHUB_AGENT_TOKEN" \\
        ghcr.io/github/github-mcp-server

Run:
    bt eval community_health_mcp.py --project "community-health-eval"

Set RUN_NUMBER env var to name experiments:
    RUN_NUMBER=1 bt eval community_health_mcp.py --project "community-health-eval"
"""

import asyncio
import json
import os
import time

import anthropic
import braintrust
from braintrust import Eval, start_span
from mcp import ClientSession
from mcp.client.sse import sse_client

from scorers import ALL_SCORERS, WRITE_TOOLS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT = "community-health-eval"
DATASET_NAME = "community_monitor_pandas"
RUN_NUMBER = os.environ.get("RUN_NUMBER", "1")
EXPERIMENT_NAME = f"mcp-baseline-run-{RUN_NUMBER}"
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
MCP_SSE_URL = os.environ.get("MCP_SSE_URL", "http://localhost:8080/sse")
EVAL_OUTPUT_BOARD = os.environ.get("EVAL_OUTPUT_BOARD", "eval-output-board")

# Throttle limits — override via env vars to prevent runaway spend
MAX_AGENT_TURNS = int(os.environ.get("MAX_AGENT_TURNS", "10"))
MAX_TOTAL_TOKENS = int(os.environ.get("MAX_TOTAL_TOKENS", "50000"))
MAX_TOOL_CALLS = int(os.environ.get("MAX_TOOL_CALLS", "6"))        # total tool calls per row
MAX_LATENCY_MS = int(os.environ.get("MAX_LATENCY_MS", "120000"))   # 2-min wall-clock limit per row
# Cost optimisation — tip 2: 8 K chars ≈ 2 K tokens, enough for any real thread.
# Paginated responses can exceed 100 K chars and inflate context on every re-sent turn.
MAX_TOOL_RESULT_CHARS = int(os.environ.get("MAX_TOOL_RESULT_CHARS", "8000"))
MAX_TOOL_ERRORS = int(os.environ.get("MAX_TOOL_ERRORS", "3"))       # abort after N consecutive tool errors
_board_parts = EVAL_OUTPUT_BOARD.rsplit("/", 1)
BOARD_OWNER = _board_parts[0] if len(_board_parts) == 2 else ""
BOARD_REPO = _board_parts[-1]

# Cache the MCP tool list — it's static for the lifetime of the process,
# so we call list_tools() once instead of once per row (69 × saved round-trip).
_CACHED_ANTHROPIC_TOOLS: list | None = None

# Braintrust runs ~10 rows concurrently by default. The mcp-proxy bridges to a
# single Docker stdio process, which can't handle that many simultaneous SSE
# sessions — connections get dropped (httpx.ReadError). This semaphore caps
# concurrent MCP sessions. Override MCP_MAX_CONCURRENT env var if needed.
MCP_MAX_CONCURRENT = int(os.environ.get("MCP_MAX_CONCURRENT", "3"))
_MCP_SEMAPHORE: asyncio.Semaphore | None = None


def _get_mcp_semaphore() -> asyncio.Semaphore:
    """Lazily create the semaphore on the running event loop."""
    global _MCP_SEMAPHORE
    if _MCP_SEMAPHORE is None:
        _MCP_SEMAPHORE = asyncio.Semaphore(MCP_MAX_CONCURRENT)
    return _MCP_SEMAPHORE


# Cost optimisation — tip 3: strip GitHub metadata from read-tool JSON responses.
# Raw get_issue / get_issue_comments results include timestamps, labels, milestones,
# assignees, etc. that are irrelevant to toxicity analysis. Compressing to just
# title + body + comment texts cuts each tool_result by ~60%, reducing the context
# re-sent to the LLM on every subsequent turn.
_READ_TOOLS = {"get_issue", "get_issue_comments", "get_pull_request", "get_pull_request_comments"}


def _compress_tool_result(tool_name: str, result_text: str) -> str:
    """Strip GitHub metadata, retaining only content fields needed for toxicity analysis."""
    if tool_name not in _READ_TOOLS:
        return result_text
    try:
        data = json.loads(result_text)
    except (json.JSONDecodeError, ValueError):
        return result_text  # plain-text response — return as-is
    if tool_name in ("get_issue", "get_pull_request"):
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
# Task function (async — Braintrust framework handles awaiting it)
# ---------------------------------------------------------------------------

async def mcp_agent_task(input: dict) -> dict:
    repo = input["repo"]
    number = input["discussion_number"]
    dtype = input["discussion_type"]
    start_time = time.time()

    client = anthropic.Anthropic(max_retries=4)  # handles 429/529 with exponential backoff
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
    output = {}

    # Measure queue wait separately so it doesn't inflate latency_ms or eat
    # into the MAX_LATENCY_MS budget — start_time is reset after semaphore acquire.
    enqueue_time = time.time()
    with start_span("mcp-retrieve-and-analyze"):
        async with _get_mcp_semaphore():
            queue_wait_ms = int((time.time() - enqueue_time) * 1000)
            start_time = time.time()  # task clock starts here, after waiting for a slot
            async with sse_client(MCP_SSE_URL) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Discover tools from the local MCP server — cached after first row
                    global _CACHED_ANTHROPIC_TOOLS
                    if _CACHED_ANTHROPIC_TOOLS is None:
                        tools_result = await session.list_tools()
                        _CACHED_ANTHROPIC_TOOLS = [
                            {
                                "name": t.name,
                                "description": t.description or "",
                                "input_schema": t.inputSchema,
                            }
                            for t in tools_result.tools
                        ]
                    anthropic_tools = _CACHED_ANTHROPIC_TOOLS

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
  owner: "{BOARD_OWNER}"
  repo: "{BOARD_REPO}"
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

                    # Agentic tool-use loop
                    while True:
                        if llm_turns >= MAX_AGENT_TURNS:
                            raise RuntimeError(
                                f"Agent loop aborted: reached MAX_AGENT_TURNS={MAX_AGENT_TURNS}. "
                                "Increase MAX_AGENT_TURNS env var if needed."
                            )
                        if total_input_tokens + total_output_tokens >= MAX_TOTAL_TOKENS:
                            raise RuntimeError(
                                f"Agent loop aborted: reached MAX_TOTAL_TOKENS={MAX_TOTAL_TOKENS} "
                                f"({total_input_tokens + total_output_tokens} tokens used). "
                                "Increase MAX_TOTAL_TOKENS env var if needed."
                            )
                        if len(tool_calls_log) >= MAX_TOOL_CALLS:
                            raise RuntimeError(
                                f"Agent loop aborted: reached MAX_TOOL_CALLS={MAX_TOOL_CALLS}. "
                                "Increase MAX_TOOL_CALLS env var if needed."
                            )
                        elapsed_ms = int((time.time() - start_time) * 1000)
                        if elapsed_ms >= MAX_LATENCY_MS:
                            raise RuntimeError(
                                f"Agent loop aborted: exceeded MAX_LATENCY_MS={MAX_LATENCY_MS} "
                                f"({elapsed_ms}ms elapsed). Increase MAX_LATENCY_MS env var if needed."
                            )

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
                            tools=anthropic_tools,
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

                        # Collect any tool_use blocks from this turn
                        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

                        if not tool_use_blocks:
                            # No more tool calls — extract final JSON from text
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

                        # Execute each tool call against the local MCP server
                        tool_results = []
                        for block in tool_use_blocks:
                            args = block.input if isinstance(block.input, dict) else {}
                            # Normalize repo to bare name for write tools so scope_containment scorer works
                            logged_args = ({**args, "repo": args.get("repo", "").split("/")[-1]}
                                           if block.name in WRITE_TOOLS else args)
                            tool_calls_log.append({"tool": block.name, "args": logged_args})

                            tool_exec_start = time.time()
                            result = await session.call_tool(block.name, args)
                            total_tool_ms += int((time.time() - tool_exec_start) * 1000)

                            result_text = "\n".join(
                                c.text for c in result.content if hasattr(c, "text")
                            )
                            if result.isError:
                                tool_error_count += 1
                                result_text = f"[Tool error] {result_text}"
                                if tool_error_count >= MAX_TOOL_ERRORS:
                                    raise RuntimeError(
                                        f"Agent loop aborted: {tool_error_count} consecutive tool errors. "
                                        "Increase MAX_TOOL_ERRORS env var if needed."
                                    )
                            else:
                                tool_error_count = 0  # reset on success
                            # Cost optimisation — tip 3: strip metadata before char-truncation
                            # so the character budget is spent on content, not timestamps/labels.
                            result_text = _compress_tool_result(block.name, result_text)
                            result_text = result_text[:MAX_TOOL_RESULT_CHARS]

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result_text,
                            })

                        # Append assistant turn and tool results, then continue
                        messages.append({
                            "role": "assistant",
                            "content": [b.model_dump(exclude_none=True) for b in response.content],
                        })
                        messages.append({"role": "user", "content": tool_results})

    thread_text = output.pop("retrieved_thread_text", "")
    retrieved_comment_count = thread_text.count("\n[") if thread_text else 0
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
    output["queue_wait_ms"] = queue_wait_ms  # time spent waiting for MCP semaphore slot
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
    output["workflow"] = "mcp"
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
    task=mcp_agent_task,
    scores=ALL_SCORERS,
    metadata={"workflow": "mcp", "model": MODEL},
)
