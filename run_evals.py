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
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EVAL_DIR = os.path.join(os.path.dirname(__file__), "eval")
RUNFILE = os.path.join(os.path.dirname(__file__), ".last_run_number")
MCP_SSE_URL = os.environ.get("MCP_SSE_URL", "http://localhost:8080/sse")

REQUIRED_ENV_VARS = {
    "BRAINTRUST_API_KEY": "Braintrust API key (https://www.braintrust.dev/app/settings)",
    "GITHUB_AGENT_TOKEN": "GitHub PAT with read access to source repos and write access to the eval output board",
    "EVAL_OUTPUT_BOARD": "Sandbox repo for Community Health Reports, as owner/repo (e.g. imhurl23/eval-output-board)",
}
ANTHROPIC_PREFLIGHT_MODEL = os.environ.get("ANTHROPIC_PREFLIGHT_MODEL", "claude-haiku-4-5-20251001")

FATAL_ANTHROPIC_ERRORS = (
    "Your credit balance is too low to access the Anthropic API",
)


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


def check_anthropic_access():
    """Make a minimal Anthropic call so we fail fast on billing/auth issues."""
    try:
        client = anthropic.Anthropic(max_retries=0)
        client.messages.create(
            model=ANTHROPIC_PREFLIGHT_MODEL,
            max_tokens=1,
            messages=[{"role": "user", "content": "Reply with OK."}],
        )
        print(f"✓ Anthropic API reachable with {ANTHROPIC_PREFLIGHT_MODEL}")
        return True
    except anthropic.BadRequestError as exc:
        error_message = str(exc)
        fatal_reason = _detect_fatal_failure(error_message)
        if fatal_reason == "anthropic credits exhausted":
            print("ERROR: Anthropic API credits are exhausted.")
            print("  Update billing or switch to a funded API key before running evals.")
            return False
        print(f"ERROR: Anthropic rejected the preflight request: {error_message}")
        return False
    except anthropic.APIError as exc:
        print(f"ERROR: Anthropic preflight failed: {exc}")
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

def _detect_fatal_failure(output: str) -> str | None:
    for marker in FATAL_ANTHROPIC_ERRORS:
        if marker in output:
            return "anthropic credits exhausted"
    return None


def run_eval(script: str, run_number: str, label: str) -> tuple[bool, str | None]:
    """
    Runs `bt eval <script>` in a subprocess.
    Returns (passed, fatal_reason).
    """
    env = {**os.environ, "RUN_NUMBER": run_number}
    cmd = ["bt", "eval", script, "--project", "community-health-eval"]

    print(f"\n{'='*60}")
    print(f"  Running {label} eval  (run {run_number})")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=EVAL_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start

    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)

    status = "PASSED" if result.returncode == 0 else "FAILED"
    print(f"\n[{label}] {status} in {elapsed:.1f}s")
    combined_output = f"{result.stdout}\n{result.stderr}"
    return result.returncode == 0, _detect_fatal_failure(combined_output)


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
    parser.add_argument(
        "--max-workers",
        metavar="N",
        type=int,
        default=None,
        help="Max parallel eval subprocesses (default: conservative cap when MCP is enabled)",
    )
    args = parser.parse_args()

    # Pre-flight
    check_env_vars()
    if not check_anthropic_access():
        sys.exit(1)

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

    # Build flat list of (run_number, script, label) tasks so all can run concurrently
    tasks = []
    for run_number in run_numbers:
        if run_cli:
            tasks.append((run_number, "community_health_cli.py", "CLI"))
        if run_mcp:
            tasks.append((run_number, "community_health_mcp.py", "MCP"))

    if args.max_workers is not None:
        max_workers = args.max_workers
    elif run_mcp and run_cli:
        max_workers = 2
    elif run_mcp:
        max_workers = 1
    else:
        max_workers = len(tasks)
    all_results = {rn: {} for rn in run_numbers}
    pending_tasks = deque(tasks)
    fatal_reason = None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_task = {}

        while pending_tasks and len(future_to_task) < max_workers:
            rn, script, label = pending_tasks.popleft()
            future_to_task[pool.submit(run_eval, script, rn, label)] = (rn, label)

        while future_to_task:
            future = next(as_completed(future_to_task))
            rn, label = future_to_task.pop(future)
            passed, task_fatal_reason = future.result()
            all_results[rn][label.lower()] = passed

            if task_fatal_reason and fatal_reason is None:
                fatal_reason = task_fatal_reason
                print(f"\nStopping remaining evals: detected fatal failure ({fatal_reason}).")
                pending_tasks.clear()

            if fatal_reason is None and pending_tasks:
                next_rn, script, next_label = pending_tasks.popleft()
                future_to_task[pool.submit(run_eval, script, next_rn, next_label)] = (next_rn, next_label)

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

    if fatal_reason:
        print(f"\nAborted early: {fatal_reason}.")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
