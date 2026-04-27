"""
run_evals.py
Coordinator script for the Community Health First Responder eval suite.

Runs CLI and MCP evals sequentially under a shared RUN_NUMBER, with
pre-flight checks for required env vars and the MCP proxy.

Usage:
    python run_evals.py                  # RUN_NUMBER auto-increments
    python run_evals.py --run 3          # explicit run number
    python run_evals.py --only cli       # single workflow
    python run_evals.py --only mcp
    python run_evals.py --skip-mcp-check # skip proxy health check
"""

import argparse
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EVAL_DIR = os.path.join(os.path.dirname(__file__), "eval")
RUNFILE = os.path.join(os.path.dirname(__file__), ".last_run_number")
MCP_SSE_URL = os.environ.get("MCP_SSE_URL", "http://localhost:8080/sse")

REQUIRED_ENV_VARS = {
    "BRAINTRUST_API_KEY": "Braintrust API key (https://www.braintrust.dev/app/settings)",
    "GITHUB_AGENT_TOKEN": "GitHub PAT with repo:read scope",
    "EVAL_OUTPUT_BOARD": "Sandbox repo for Community Health Reports, as owner/repo (e.g. imhurl23/eval-output-board)",
}


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def check_env_vars():
    missing = [k for k in REQUIRED_ENV_VARS if not os.environ.get(k)]
    if missing:
        print("ERROR: Missing required environment variables:")
        for k in missing:
            print(f"  {k}  —  {REQUIRED_ENV_VARS[k]}")
        sys.exit(1)
    print("✓ Environment variables present")


def check_mcp_proxy():
    """Probe the SSE endpoint. Returns True if reachable."""
    try:
        req = urllib.request.Request(MCP_SSE_URL, method="GET")
        # SSE endpoints keep the connection open; we just need a 200 header.
        with urllib.request.urlopen(req, timeout=4) as resp:
            if resp.status == 200:
                print(f"✓ MCP proxy reachable at {MCP_SSE_URL}")
                return True
    except (urllib.error.URLError, OSError):
        pass
    print(f"ERROR: MCP proxy not reachable at {MCP_SSE_URL}")
    print("  Start it with:")
    print('  mcp-proxy --port 8080 -- docker run -i --rm \\')
    print('    -e GITHUB_PERSONAL_ACCESS_TOKEN="$GITHUB_AGENT_TOKEN" \\')
    print('    ghcr.io/github/github-mcp-server')
    return False


# ---------------------------------------------------------------------------
# Run number management
# ---------------------------------------------------------------------------

def resolve_run_number(explicit: str | None) -> str:
    if explicit:
        _write_runfile(explicit)
        return explicit

    last = 0
    if os.path.exists(RUNFILE):
        try:
            last = int(open(RUNFILE).read().strip())
        except ValueError:
            pass

    next_run = str(last + 1)
    _write_runfile(next_run)
    return next_run


def _write_runfile(value: str):
    with open(RUNFILE, "w") as f:
        f.write(value)


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

def run_eval(script: str, run_number: str, label: str) -> bool:
    """
    Runs `bt eval <script>` in a subprocess.
    Returns True on success, False on failure.
    """
    env = {**os.environ, "RUN_NUMBER": run_number}
    cmd = ["bt", "eval", script, "--project", "community-health-eval"]

    print(f"\n{'='*60}")
    print(f"  Running {label} eval  (run {run_number})")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=EVAL_DIR, env=env)
    elapsed = time.time() - start

    status = "PASSED" if result.returncode == 0 else "FAILED"
    print(f"\n[{label}] {status} in {elapsed:.1f}s")
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run community health evals")
    parser.add_argument("--run", metavar="N", help="Explicit starting run number")
    parser.add_argument("--runs", metavar="N", type=int, default=5, help="Number of runs (default: 5)")
    parser.add_argument(
        "--only",
        choices=["cli", "mcp"],
        help="Run only one workflow",
    )
    parser.add_argument(
        "--skip-mcp-check",
        action="store_true",
        help="Skip the MCP proxy health check",
    )
    args = parser.parse_args()

    # Pre-flight
    check_env_vars()

    run_mcp = args.only in (None, "mcp")
    run_cli = args.only in (None, "cli")

    if run_mcp and not args.skip_mcp_check:
        if not check_mcp_proxy():
            sys.exit(1)

    start_run = resolve_run_number(args.run)
    run_numbers = [str(int(start_run) + i) for i in range(args.runs)]
    # Persist the last run number so the next invocation continues from there
    _write_runfile(run_numbers[-1])

    print(f"\nRuns: {', '.join(run_numbers)}")

    all_results = {}  # keyed by run_number

    for run_number in run_numbers:
        run_results = {}
        if run_cli:
            run_results["cli"] = run_eval("community_health_cli.py", run_number, "CLI")
        if run_mcp:
            run_results["mcp"] = run_eval("community_health_mcp.py", run_number, "MCP")
        all_results[run_number] = run_results

    # Summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    all_passed = True
    for run_number, run_results in all_results.items():
        for workflow, passed in run_results.items():
            icon = "✓" if passed else "✗"
            print(f"  {icon}  run {run_number}  {workflow}")
            if not passed:
                all_passed = False

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
