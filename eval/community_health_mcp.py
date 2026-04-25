"""
community_health_mcp.py
MCP workflow eval for the Community Health First Responder experiment.
Uses the GitHub MCP server via mcp-proxy SSE bridge for retrieval.

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
from braintrust import Eval, traced

from scorers import ALL_SCORERS, tag_failure_modes

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT = "community-health-eval"
DATASET_NAME = "community-health-v1"
RUN_NUMBER = os.environ.get("RUN_NUMBER", "1")
EXPERIMENT_NAME = f"mcp-baseline-run-{RUN_NUMBER}"
MODEL = "claude-sonnet-4-20250514"
MCP_SSE_URL = os.environ.get("MCP_SSE_URL", "http://localhost:8080/sse")

MCP_SERVERS = [{
    "type": "url",
    "url": MCP_SSE_URL,
    "name": "github"
}]

# ---------------------------------------------------------------------------
# Task function
# ---------------------------------------------------------------------------

def mcp_agent_task(input: dict) -> dict:
    repo = input["repo"]
    number = input["discussion_number"]
    dtype = input["discussion_type"]
    start = time.time()

    client = anthropic.Anthropic(
        default_headers={"anthropic-beta": "mcp-client-2025-04-04"}
    )

    with traced("mcp-retrieve-and-analyze"):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            mcp_servers=MCP_SERVERS,
            messages=[{
                "role": "user",
                "content": f"""You are a community health analyst for open source projects.

Use the GitHub MCP tools to retrieve the full comment thread for {dtype} #{number} in {repo}.
For issues: call get_issue AND get_issue_comments to ensure you have all comments.
For PRs: call get_pull_request AND get_pull_request_comments.

Then analyze the thread for toxic or discouraging content using this OSS-specific schema:
- hostile_aggression: explicit threats, insults, name-calling
- entitlement: demanding tone ("fix this NOW", "why hasn't this been done")
- dismissive_tone: curt rejection, closing without engagement
- sarcasm_belittling: ironic minimization, mocking effort or skill
- passive_aggression: technically civil but subtly hostile framing
- gatekeeping: condescension toward perceived skill level
- thread_derailment: personal attacks displacing technical discussion
- object_directed: hostility directed at code/project ("this codebase is trash")

Respond ONLY with a JSON object — no markdown, no preamble:
{{
  "toxic_detected": true or false,
  "snippet": "exact quoted text from the thread that is problematic, or null",
  "toxicity_label": "one label from the schema above, or null",
  "severity": "low, medium, or high, or null",
  "draft_response": "a de-escalating maintainer response, or null if not toxic",
  "retrieved_thread_text": "full concatenated thread text you retrieved"
}}"""
            }]
        )

    # Extract tool calls and final text block from response
    tool_calls = []
    thread_text = ""
    output = {}

    for block in response.content:
        if block.type == "tool_use":
            tool_calls.append({
                "tool": block.name,
                "args": block.input if isinstance(block.input, dict) else {}
            })
        elif block.type == "text":
            # Last text block is the final JSON output
            raw = block.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            try:
                output = json.loads(raw.strip())
            except json.JSONDecodeError:
                # Model produced intermediate text, keep iterating
                pass

    thread_text = output.pop("retrieved_thread_text", "")
    retrieved_comment_count = thread_text.count("\n[") if thread_text else 0

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
    task=mcp_agent_task,
    scores=ALL_SCORERS,
    metadata={"workflow": "mcp", "model": MODEL},
)
