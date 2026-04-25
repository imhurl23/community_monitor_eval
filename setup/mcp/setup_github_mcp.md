# setup_github_mcp.md

Setup guide for running the GitHub MCP server locally via Docker and exposing it over SSE using `mcp-proxy`, for use with the Anthropic API in automated eval scripts.

---

## Prerequisites

- Docker Desktop installed and running (`docker ps` works)
- Python 3.10+ with `pip`
- A GitHub Personal Access Token (PAT) with `repo:read` scope
- Your PAT exported as `GITHUB_AGENT_TOKEN` in your shell profile

---

## Step 1: Store your GitHub token

Add to your `~/.zshrc` or `~/.bashrc`:

```bash
export GITHUB_AGENT_TOKEN="github_pat_yourtoken..."
```

Then reload:

```bash
source ~/.zshrc
```

Verify:

```bash
echo $GITHUB_AGENT_TOKEN | cut -c1-10
```

---

## Step 2: Install mcp-proxy

```bash
pip install mcp-proxy
```

---

## Step 3: Pull the GitHub MCP server image

```bash
docker pull ghcr.io/github/github-mcp-server
```

Verify the image works standalone:

```bash
docker run -i --rm \
  -e GITHUB_PERSONAL_ACCESS_TOKEN="$GITHUB_AGENT_TOKEN" \
  ghcr.io/github/github-mcp-server
```

You should see:

```
level=INFO msg="starting server" version=v1.0.3
GitHub MCP Server running on stdio
level=INFO msg="server run start"
level=INFO msg="server session connected"
```

Press `Ctrl+C` to stop.

---

## Step 4: Start the SSE bridge

Run this single command to start the bridge. The variable expands in your current shell before `mcp-proxy` spawns Docker, avoiding the `GITHUB_PERSONAL_ACCESS_TOKEN not set` error.

```bash
mcp-proxy --port 8080 -- docker run -i --rm \
  -e GITHUB_PERSONAL_ACCESS_TOKEN="$GITHUB_AGENT_TOKEN" \
  ghcr.io/github/github-mcp-server
```

You should see:

```
MCP proxy listening on http://localhost:8080/sse
```

Leave this running in a dedicated terminal for the duration of your eval runs.

---

## Step 5: Verify the bridge is working

```python
# test_mcp_connection.py
import anthropic

client = anthropic.Anthropic(
    default_headers={"anthropic-beta": "mcp-client-2025-04-04"}
)

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=500,
    mcp_servers=[{
        "type": "url",
        "url": "http://localhost:8080/sse",
        "name": "github"
    }],
    messages=[{
        "role": "user",
        "content": "Use the GitHub MCP tools to get issue #17823 from eslint/eslint. Return the issue title only."
    }]
)

print(response.content)
```

```bash
python test_mcp_connection.py
```

Expected: a text block containing the issue title. If you get a tool-use block followed by a text block, the bridge is working correctly.

---

## Step 6: Use in eval scripts

When constructing your Anthropic client in any eval script, always include the beta header and pass the local SSE URL:

```python
import anthropic

client = anthropic.Anthropic(
    default_headers={"anthropic-beta": "mcp-client-2025-04-04"}
)

mcp_servers = [{
    "type": "url",
    "url": "http://localhost:8080/sse",
    "name": "github"
}]
```

Pass `mcp_servers=mcp_servers` into every `client.messages.create()` call that needs GitHub tool access.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `GITHUB_PERSONAL_ACCESS_TOKEN not set` | Variable not expanding inside script | Use inline `-- docker run ... -e VAR="$GITHUB_AGENT_TOKEN"` form, not a wrapper script |
| `Connection closed` in mcp-proxy traceback | Docker container exited immediately | Run the Docker command directly first to confirm the token works |
| `mcp_servers` param silently ignored | Missing beta header | Add `"anthropic-beta": "mcp-client-2025-04-04"` to client headers |
| Proxy starts but test call hangs | Docker not running | Run `docker ps` to confirm Docker Desktop is active |
| Empty tool results on PR threads | Wrong tool selected | Confirm Claude called `get_issue_comments` not `get_pull_request_comments` for issue threads |

---

## Notes

- The `--read-only` flag can be added to the Docker command to enforce read-only mode at the server level, providing an additional safety layer on top of the `scope_containment` scorer.
- The `--toolsets` flag can scope which GitHub tools are exposed. For this eval, `issues,pull_requests,repos` is sufficient — no need to expose `actions`, `gists`, or `notifications`.
- The proxy must be restarted if Docker Desktop restarts or the container crashes.
