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

import json
import os
import time

import anthropic
import braintrust
from braintrust import Eval, start_span
from mcp import ClientSession
from mcp.client.sse import sse_client

from scorers import ALL_SCORERS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT = "community-health-eval"
DATASET_NAME = "pd_community_health_labeled"
RUN_NUMBER = os.environ.get("RUN_NUMBER", "1")
EXPERIMENT_NAME = f"mcp-baseline-run-{RUN_NUMBER}"
MODEL = "claude-sonnet-4-20250514"
MCP_SSE_URL = os.environ.get("MCP_SSE_URL", "http://localhost:8080/sse")
COST_PER_MTOK = {"input": 3.0, "output": 15.0}  # claude-sonnet-4 pricing
EVAL_OUTPUT_BOARD = os.environ.get("EVAL_OUTPUT_BOARD", "eval-output-board")
_board_parts = EVAL_OUTPUT_BOARD.rsplit("/", 1)
BOARD_OWNER = _board_parts[0] if len(_board_parts) == 2 else ""
BOARD_REPO = _board_parts[-1]


# ---------------------------------------------------------------------------
# Task function (async — Braintrust framework handles awaiting it)
# ---------------------------------------------------------------------------

async def mcp_agent_task(input: dict) -> dict:
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

    with start_span("mcp-retrieve-and-analyze"):
        async with sse_client(MCP_SSE_URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Discover tools from the local MCP server
                tools_result = await session.list_tools()
                anthropic_tools = [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "input_schema": t.inputSchema,
                    }
                    for t in tools_result.tools
                ]

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
                    llm_call_start = time.time()
                    response = client.messages.create(
                        model=MODEL,
                        max_tokens=2000,
                        tools=anthropic_tools,
                        messages=messages,
                    )
                    total_llm_ms += int((time.time() - llm_call_start) * 1000)
                    llm_turns += 1

                    turn_in = response.usage.input_tokens
                    turn_out = response.usage.output_tokens
                    total_input_tokens += turn_in
                    total_output_tokens += turn_out
                    per_turn_tokens.append({"input": turn_in, "output": turn_out})

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
                        tool_calls_log.append({"tool": block.name, "args": args})

                        tool_exec_start = time.time()
                        result = await session.call_tool(block.name, args)
                        total_tool_ms += int((time.time() - tool_exec_start) * 1000)

                        result_text = "\n".join(
                            c.text for c in result.content if hasattr(c, "text")
                        )
                        if result.isError:
                            result_text = f"[Tool error] {result_text}"

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
