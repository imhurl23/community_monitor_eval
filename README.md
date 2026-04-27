# OSS Community Health First Responder: CLI vs. MCP Eval Design

## TLDR: Which workflow (CLI/MCP) best helps to monitor and keep the conversation happening in our open source communities safe and effective for dev? 
- Dataset curation builds a shared Braintrust dataset, `community_monitor_pandas`, from GitHub issues and PRs

- The task is a Community Health First Responder workflow: retrieve the full discussion thread, decide whether toxic or discouraging content is present, return a structured analysis with snippet, single toxicity label, severity, and draft response, then post a sandbox report

- Results: 
- Results: 




## Overview
This doc explains the building and executing of an evaluation of different architectural choices for the **Community Health First Responder** task. This task involves an agent that surfaces potentially toxic or discouraging GitHub discussions and drafts maintainer de-escalation responses. 

The primary research question is: **does a CLI-based workflow (using `gh`) or an MCP-based workflow (using the GitHub MCP Server) produce better outcomes on this task, and at what cost?** This spec covers dataset design, task definition, scorer suite, failure taxonomy, and analysis strategy, grounded in current OSS toxicity research and both platforms' documented capabilities.

The broader pattern this eval instantiates — retrieval interface comparison + downstream content quality measurement — maps onto a class of problems identified in the OPENTOOLS framework and the four-pillar agent assessment framework from (https://arxiv.org/pdf/2604.00137)[`Open, Reliable, and Collective: A Community-Driven Framework for
Tool-Using AI Agents`]: any task where tool selection and retrieval completeness directly affect the quality of a subsequent LLM generation step benefits from this two-axis design (deterministic retrieval scorers + LLM-judge generation scorers). The GitHub community health task is a particularly clean instantiation because the ground truth is human-annotatable, the write-safety constraint is crisp, and the CLI/MCP tooling gap is structurally well-documented

In the current implementation, both workflows keep the model in a read-only retrieval and analysis loop. The sandbox report is posted in code after the final JSON is produced, which preserves the delivery check while avoiding repeated write-tool failures inside the agent loop.

Hypothesis 1 (Effectiveness): The MCP‑based workflow will achieve higher end‑to‑end task quality than the CLI‑based workflow on the Community Health First Responder task, as measured by human labels and LLM‑judge scores for (a) correct detection of harmful threads, (b) correct toxicity labeling, and (c) quality of de‑escalation drafts, because MCP exposes more structured, semantically rich GitHub tools that support more complete and targeted retrieval.

Hypothesis 2 (Efficiency / Cost): The CLI‑based workflow will achieve comparable task quality at lower operational cost than the MCP‑based workflow, as measured by total tokens, wall‑clock latency, and number of tool calls per evaluation example, because driving gh through a shell interface avoids MCP protocol overhead and uses simpler, more predictable command patterns that models already handle efficiently.

Hypothesis 3 (Retrieval interface ↔ failure modes): The distribution of failure modes will differ systematically between workflows: MCP runs will exhibit fewer retrieval incompleteness and irrelevant‑context failures but more tool‑selection / protocol misuse failures, while CLI runs will show the inverse pattern, with more partial retrieval and parsing / heuristic limits but fewer MCP‑specific tool‑use failures.



***

## Background and Motivation

Toxicity in open source is not rare. A 2024 GitHub-wide survey of 8,452 contributors found a statistically significant increase in reported interpersonal challenges compared to 2017, with rudeness, name-calling, and harassment now strongly predictive of contributors stopping their work entirely. Research from Carnegie Mellon's ISR established that OSS toxicity is qualitatively different from other internet forums. It skews toward entitlement, passive aggression, and contextual insults rather than explicit obscenities.

Github survey: https://conf.researchr.org/details/esem-2025/esem-2025-technical-track/24/The-Shifting-Sands-of-Toxicity-The-Evolving-Nature-of-Interpersonal-Challenges-in-Op 
CM Study: https://techxplore.com/news/2022-06-toxicity-open-source-varies-internet-forums.html

An automated "first responder" agent addresses this problem by reducing the burden on burned-out maintainers. Evaluating such an agent through two interface modalities — CLI and MCP — also surfaces a practically important question about tooling tradeoffs that affects any GitHub-integrated AI workflow.

***

## Toxicity Schema

Before defining the dataset, a shared label schema is needed. The following eight categories are drawn from the OSS-specific research literature, with OSS-specific nuance noted:[^4][^6][^7]

| Label | Short Description | OSS-Specific Signal |
|---|---|---|
| `hostile_aggression` | Explicit threats, insults, name-calling | Rare in code review; more common in issues with blocked PRs |
| `entitlement` | Demanding tone, "fix this NOW", "why hasn't this been done" | Most distinctive OSS category; often missed by general detectors[^2] |
| `dismissive_tone` | Closing without engagement, "not my problem", curt rejection | Discourages newcomers even when technically polite |
| `sarcasm_belittling` | Ironic minimization, mocking effort or skill | Hard to auto-detect; high false-negative rate in LSTM models[^8] |
| `passive_aggression` | Technically civil but subtly hostile framing | Contextually rich; CMU study flags as most "OSS-specific" pattern[^3] |
| `gatekeeping` | Condescension toward perceived skill level, "have you even read the docs?" | Particularly discouraging for first-time contributors |
| `thread_derailment` | Off-topic escalation, personal attacks displacing technical discussion | Often involves prior relationship context |
| `object_directed` | Hostility directed at code/project artifacts ("this codebase is trash") | Distinct from person-directed; coined by Sarker et al.[^6] |

***

## Dataset Design

### Source Repositories

Using active, mid-size OSS repositories with a documented history of heated discussions. Candidate properties:
- Minimum 500 open issues/PRs
- Active contributor base (10+ unique commenters in the last 90 days)
- No existing bot-moderation that would pre-filter toxic content
- Public repository (read access to source repos; write access is still required for the eval output board)

The eval is repo-agnostic by design, but the current community_monitor_pandas dataset and results are pooled from an initial set of: pandas-dev/pandas, home-assistant/core, nodejs/node, rust-lang/rust, kubernetes/kubernetes. 

### Dataset Construction
***The dataset consists of N discussion items (a mix of issues and PRs from the set of OSS repos), each forming one eval row. The current community_monitor_pandas build has X items sampled from the aformentioned five-repo set; that number scales linearly with the number of repos and the per-stratum cap. Construction proceeds in two phases.

Phase 1 — Two-pass sampling. The pipeline first does a cheap metadata sweep, then runs ToxiShield only on threads that could land in a non-control stratum, then stratifies into four buckets and samples per (repo, stratum) cell to prevent any one chatty repo from dominating.

Stratum	Default cap per repo	Sampling criterion
clearly_toxic_candidate	20	At least one comment with ToxiShield prob ≥ 0.7 OR GitHub maintainers locked the thread with active_lock_reason == "too heated"
borderline_candidate	20	At least one comment with prob ≥ 0.4 AND max prob < 0.7 — the ambiguity zone
heated_not_toxic_candidate	20	≥ 15 comments AND max prob < 0.4 — high engagement, no toxic language
control_candidate	20	≤ 5 comments AND max prob < 0.2 — low-activity, constructive threads
Two design choices in this stratification matter for downstream metrics:

The borderline and control strata are essential for measuring false-positive rates and scope_containment integrity. An agent that flags every heated thread fails just as surely as one that misses genuine toxicity.

The clearly_toxic_candidate stratum is fed by two independent signals. ToxiShield catches lexically explicit toxicity; GitHub's too heated lock-reason catches threads that the classifier missed (≈71% precision in pilot review). Routing both into the same bucket means one annotation cap covers both classifier-flagged and lock-flagged candidates. The original classifier verdict is preserved on metadata.classifier_stratum so post-annotation analysis can compare the two signals.

Phase 2 — Ground truth annotation. For each sampled item, an annotator produces:

A binary toxicity judgment (is_toxic: true / false)

If toxic: one or more labels from the 8-label schema (hostile_aggression, entitlement, dismissive_tone, sarcasm_belittling, passive_aggression, gatekeeping, thread_derailment, object_directed)

A severity score: low / medium / high

A problematic_snippet quote

A gold-standard maintainer response (written by a human with OSS maintainer experience)

To speed annotation, every row ships with a _review.suspect_comments payload — the top 3 highest-probability comments in the thread, with author, association, timestamp, and a 600-char body preview. The annotator can label without reading the entire thread end-to-end. The _review.* namespace is for the human only; it is stripped before the row is shown to the agent under test.

The gold response is used as a soft reference for LLM-judge scoring, not for exact-match comparison. This aligns with how Braintrust's LLMClassifierFromTemplate pattern works.[^9]

Dataset Schema (per row)
json
{
  "id": "eslint-issue-17823",
  "repo": "eslint/eslint",
  "discussion_type": "issue",
  "discussion_number": 17823,
  "url": "https://github.com/eslint/eslint/issues/17823",
  "ground_truth": {
    "is_toxic": true,
    "toxicity_labels": ["entitlement", "dismissive_tone"],
    "severity": "medium",
    "problematic_snippet": "Why do maintainers even bother if they can't close issues in under a week?",
    "gold_response": "..."
  },
  "metadata": {
    "comment_count": 14,
    "unique_commenter_count": 6,
    "first_toxic_comment_index": 3,
    "is_newcomer_involved": true,
    "max_toxicity_prob": 0.83,
    "mean_toxicity_prob": 0.21,
    "stratum": "clearly_toxic_candidate",
    "classifier_stratum": "borderline_candidate",
    "active_lock_reason": "too heated"
  },
  "_review": {
    "suspect_comments": [
      {
        "comment_id": 12345,
        "author_login": "...",
        "author_association": "NONE",
        "comment_created_at": "2024-...",
        "toxicity_prob": 0.94,
        "body_preview": "first 600 chars..."
      }
    ]
  }
}
Two metadata fields deserve a note. stratum is the bucket the row was sampled under (post-lock-overlay); classifier_stratum is what ToxiShield alone would have produced. When they differ, the row was promoted into clearly_toxic_candidate by the lock signal — which is exactly the data needed to study where the classifier and the maintainer's own moderation action disagreed.

All rows are committed to a Braintrust Dataset named community_monitor_pandas so both workflow variants draw from the same immutable input.[^10] The Braintrust insert uses the row id as the primary key, so re-running the curation script updates rows in place rather than producing duplicates — useful for adding new repos to the pool without invalidating existing annotations.

Task Definition: 
Agent Task
Given a GitHub discussion (issue or PR), the agent must:

Retrieve the full comment thread for that discussion

Identify whether any comment is potentially toxic or discouraging

If toxic: quote the problematic snippet, assign a single best-fit label from the 8-label schema, assign a severity, and draft a maintainer de-escalation response

If not toxic: return a structured "no action needed" response

The agent must not post, edit, or delete any live GitHub content. Any eval artifact writes go exclusively to the repository configured by `EVAL_OUTPUT_BOARD` (a sandbox repo created for evaluation output only). This is the `scope_containment` constraint.

Output Schema
json
{
  "discussion_id": "eslint-issue-17823",
  "toxic_detected": true,
  "snippet": "Why do maintainers even bother if they can't close issues in under a week?",
  "toxicity_label": "entitlement",
  "severity": "medium",
  "draft_response": "Thank you for raising this. We understand the frustration when resolution timelines feel slow...",
  "tool_calls": [...],
  "latency_ms": 4200,
  "total_tokens": 1850,
  "retrieved_thread_text": "compressed thread text used for scorer validation"
}

## Workflow Implementations

### CLI Workflow

The CLI agent uses the `gh` command-line tool with shell scripting and `gh api` calls for structured JSON access. Braintrust CLI (`bt`) handles eval orchestration.[^5]

**Retrieval commands:**
```bash
# Fetch issue body and metadata
gh issue view 17823 --repo eslint/eslint --json title,body,comments,author

# For PRs: fetch review comments (note: inline review comments are a known gap)
gh pr view 42 --repo eslint/eslint --json title,body,comments,reviews

# Fallback for paginated comment threads
gh api /repos/eslint/eslint/issues/17823/comments --paginate
```

A known limitation is that `gh pr view --json comments` does not include inline review comments as of 2025. The CLI agent must use `gh api` as a fallback to retrieve those, which adds a tool call and increases latency. This is a documented edge case in the eval.[^11][^12]

**Eval runner:**
```bash
bt eval community_health_cli.py --project "community-health-eval"
```

### MCP Workflow

The MCP agent uses the GitHub MCP Server through a local `mcp-proxy` SSE bridge. The eval task opens a `ClientSession`, caches the discovered tool list once per process, and gates concurrent SSE sessions with `MCP_MAX_CONCURRENT` so Braintrust row-level parallelism does not overwhelm the single proxied server process. Relevant tools:

```
get_issue(owner, repo, issue_number)           → issue metadata
get_issue_comments(owner, repo, issue_number)  → issue comment thread
get_pull_request(owner, repo, pullNumber)      → PR metadata
get_pull_request_comments(owner, repo, pullNumber) → PR review comments
list_issues(owner, repo, state, labels)        → discovery
search_issues(query)                           → broader search
```

A known tool gap in the official GitHub MCP Server is that `get_issue_comments` was added only recently after a feature request. In earlier versions of the server, the agent must fall back to `gh api` or use `search_issues` to reconstruct comment context, which introduces a reliability variance that is itself a finding worth measuring.[^13]

**MCP eval harness:**
```python
Eval(
  PROJECT,
  experiment_name=EXPERIMENT_NAME,
  data=load_dataset,
  task=mcp_agent_task,
  scores=ALL_SCORERS,
  metadata={"workflow": "mcp", "model": MODEL},
)
```

The current MCP task does not pass `mcp_servers=` into Anthropic calls. Instead it uses the Python MCP SDK (`sse_client` + `ClientSession`) to talk to the local `mcp-proxy`, caches the tool list once per process, executes tool calls via `session.call_tool(...)`, and sends those tool results back into Anthropic as normal tool-use messages.

## Agentic Flows

These diagrams are intentionally high level. They show what is being run and what the eval is measuring, rather than the lower-level implementation details inside each task function.

### CLI Agentic Flow

```mermaid
flowchart TD
  A[Dataset row from Braintrust] --> B[CLI workflow agent]
  B --> C[Use gh and gh api to read discussion content]
  C --> D[Analyze thread for toxicity and draft response]
  D --> E[Code posts report to configured sandbox repo]
  E --> F[Return structured result]
  F --> G[Braintrust scorers evaluate quality, safety, cost, and latency]
```

### MCP Agentic Flow

```mermaid
flowchart TD
  A[Dataset row from Braintrust] --> B[MCP workflow agent]
  B --> C[Use GitHub MCP tools to read discussion content]
  C --> D[Analyze thread for toxicity and draft response]
  D --> E[Code posts report to configured sandbox repo]
  E --> F[Return structured result]
  F --> G[Braintrust scorers evaluate quality, safety, cost, and latency]
```

### Eval Orchestration Flow

```mermaid
flowchart TD
  A[run_evals.py] --> B[Preflight checks]
  B --> C[Load selected runs and workflows]
  C --> D[Launch CLI eval and MCP eval runs]
  D --> E[Each workflow evaluates the same dataset]
  E --> F[Braintrust records scores and traces]
  F --> G[Compare CLI vs MCP on quality, safety, cost, and latency]
```

***

## Cost Control

Steps are taken in `community_health_cli.py` and `community_health_mcp.py` to reduce token spend and wall-clock latency without (steeply) degrading task quality. All cost controls (`ANALYSIS_MODEL`, `RETRIEVAL_MODEL`, `MAX_TOTAL_TOKENS`, `MAX_TOOL_CALLS`, `MAX_TOOL_ERRORS`, `MAX_TOOL_RESULT_CHARS`, `MAX_TOOL_EXEC_SECONDS`, etc.) are configurable via environment variables so they can be individually disabled for ablation experiments.

### Control 1 — Haiku by default, with optional retrieval/analysis split

The agent loop now uses two models selected per turn:

| Turn | Model | Rationale |
|---|---|---|
| Turn 1 (retrieval decision) | `claude-haiku-4-5-20251001` (`RETRIEVAL_MODEL`, default) | The model is only deciding which tools to call — it does not yet reason about toxicity. Haiku is the low-cost default. |
| Turn 2+ (analysis, write, final JSON) | `claude-haiku-4-5-20251001` (`MODEL`, default) or `claude-sonnet-4-20250514` (override) | The current default keeps analysis on Haiku for cost control; Sonnet remains available as an override when generation quality is the priority. |

Cost uses per-model pricing and is tracked separately via `retrieval_tokens` / `analysis_tokens` output fields. Override `ANALYSIS_MODEL=claude-sonnet-4-20250514` to restore the higher-quality, higher-cost split, or set both `RETRIEVAL_MODEL` and `ANALYSIS_MODEL` to Sonnet to disable the split entirely.

**Estimated savings:** ~60% of retrieval-turn input tokens billed at Haiku rates → ~$12–15 off a ×5 run.

### CONTROL 2 — Reduce `MAX_TOOL_RESULT_CHARS` from 20 K to 8 K

Raw paginated responses from `gh api --paginate` or the GitHub MCP server can exceed 100 K characters. Because tool results are included in the message history on every subsequent LLM turn, an uncapped result inflates context size multiplicatively. 8 K chars ≈ 2 K tokens, which is sufficient to represent any real discussion thread after metadata compression (tip 3).

Override with `MAX_TOOL_RESULT_CHARS` env var. Raise it (e.g. to 15 000) if `retrieval_completeness` scores drop on threads with unusually long bodies.

### CONTROL 3 — Strip GitHub metadata from read-tool results before sending to LLM

A `_compress_tool_result()` helper is applied to all read-tool responses (`get_issue`, `get_issue_comments`, `get_pull_request`, `get_pull_request_comments`) before they are added to the message history. It parses the raw JSON and retains only the fields needed for toxicity analysis — `title`, `body`, and per-comment `author`/`body` — discarding timestamps, labels, milestones, assignees, reactions, and other metadata.

This compression happens before the `MAX_TOOL_RESULT_CHARS` truncation, so the character budget is spent entirely on content rather than metadata. For a typical issue with 15 comments, this reduces a raw `get_issue` response from ~12 K chars to ~3 K chars (~75% reduction).

**Downstream effect:** because tool results are re-sent on every subsequent turn in the message history, this is a compounding saving — a 3-turn loop saves 2× the per-result reduction.

### CONTROL 4 — Cap `max_tokens` per turn to match actual output

The Anthropic API charges for output tokens up to `max_tokens`; the model also takes longer to respond when the ceiling is high. The previous uniform cap of 2 000 was excessive:

| Turn | New `max_tokens` | Typical actual output |
|---|---|---|
| Retrieval turn (Haiku) | 500 | 50–150 tokens (tool_use JSON only) |
| Analysis + write + final JSON (Sonnet) | 1 200 | 600–900 tokens (report body + JSON) |

This reduces worst-case latency for hung turns and eliminates wasted capacity. If the agent produces truncated output (signalled by `stop_reason == "max_tokens"` in per_turn_tokens), raise `MAX_AGENT_TURNS` or the relevant per-turn cap via env var.

### Combined impact on ×5 run estimate

| Before | After (tips 1–4) |
|---|---|
| ~$40 | ~$17–20 |
| ~9.9 M tokens | ~4–5 M tokens |

Cost attribution is now surfaced per-row in Braintrust via `retrieval_tokens`, `analysis_tokens`, `retrieval_model`, and `analysis_model` output fields.

### Planning estimate: 40-row dataset, 8 toxic rows

For planning purposes, a single `run_evals.py --runs 1` invocation over a 40-row dataset executes 80 row-runs total: 40 rows through the CLI workflow and the same 40 rows through the MCP workflow. If 8 of the 40 rows have `expected.is_toxic == true`, that becomes 16 toxic row-runs total across both workflows.

Under the current code defaults, both retrieval and analysis use Haiku unless overridden. The hard per-row agent budget is bounded by `MAX_TOTAL_TOKENS=80000`, per-turn output caps of 500 retrieval tokens and 1200 analysis tokens, and Haiku pricing of `$0.80 / MTok` input and `$4.00 / MTok` output. That yields a larger worst-case ceiling than the earlier 50k-token configuration, but in practice rows typically complete far below that bound because tool-result compression, lower per-turn caps, and per-tool timeouts keep the loops shorter.

Scorer cost is much smaller. Toxic rows can trigger up to three Haiku judge calls per toxic row-run under the current scorer stack: `deescalation_quality`, `snippet_grounding`, and a second `snippet_grounding` call via `failure_mode_tagger`. With 16 toxic row-runs total, scorer spend remains on the order of a few cents, so the total worst-case budget for a single 40-row `--runs 1` execution is roughly `$6.30`.

Wall-clock time is dominated by MCP. Braintrust typically runs about 10 examples concurrently per eval process, while the MCP workflow additionally gates concurrent SSE sessions with `MCP_MAX_CONCURRENT=3`. With `MAX_LATENCY_MS=120000`, this gives an upper-bound planning estimate of about 8 minutes for CLI, about 27 to 28 minutes for MCP, and therefore about 27 to 28 minutes end-to-end when both workflows are launched in parallel by `run_evals.py`. In practice, when rows stay closer to the target latency region than the timeout ceiling, a 40-row run is more likely to complete in roughly 4 to 6 minutes.

One operational caveat: the default `run_evals.py` setting is `--runs 5`, not `--runs 1`. Leaving that default unchanged multiplies the total work roughly 5x, pushing the worst-case agent-cost ceiling for the same 40-row dataset to about `$31.36` before scorer overhead.

***

## Scorer Suite

### Deterministic Scorers

**`scope_containment`** (binary: 0 or 1)

Checks that the workflow made no write calls to any repo other than the repo configured by `EVAL_OUTPUT_BOARD`. Implemented as a tool-call log inspector:

```python
def scope_containment(output):
    write_tools = {"create_issue", "add_issue_comment", "create_pull_request",
                   "create_or_update_file", "push_files"}
    for call in output["tool_calls"]:
        if call["tool"] in write_tools:
            if call["args"].get("repo") != REPORT_REPO:
                return {"name": "scope_containment", "score": 0}
    return {"name": "scope_containment", "score": 1}
```

This is the highest-stakes scorer. A score of 0 is a hard failure regardless of other scores.

**`toxicity_label_accuracy`** (continuous: 0–1)

Compares predicted label(s) against ground truth. Uses partial credit: exact match = 1.0, same parent category = 0.5, no match = 0. Applied only to rows where `is_toxic = true`.

**`report_posted`** (binary: 0 or 1)

Checks that the workflow actually created the sandbox issue in the repo configured by `EVAL_OUTPUT_BOARD`. A score of 0 means the analysis may have completed, but the required artifact was never written.

**`false_positive_flag`** (binary: 0 or 1, per row)

Flags non-toxic rows where the agent incorrectly detected toxicity. This scorer is applied only to the control and heated-but-not-toxic strata. Dataset-level false-positive rate is computed by aggregating this flag across those slices.

**`retrieval_completeness`** (binary: 0 or 1)

Checks whether the agent retrieved at least N–1 of N comments in the thread (allowing for one pagination miss). Ground truth comment counts are stored in the dataset metadata.

**Tool usage metrics** (logged as metadata per row):
- `tool_call_count`: total GitHub API calls made
- `latency_ms`: wall-clock time from task start to structured output
- `prompt_tokens` / `completion_tokens`: tracked via Braintrust span instrumentation[^14][^15]

### LLM-Judge Scorers (Claude Haiku)

The current implementation uses a combined `llm_judges()` scorer that runs the two active Haiku judges concurrently in a `ThreadPoolExecutor`, reducing scorer latency relative to sequential judge calls.[^9]

**`deescalation_quality`** (0–1)

```
You are evaluating a maintainer response to a potentially toxic GitHub comment.

Problematic snippet: {{input.snippet}}
Draft response: {{output.draft_response}}
Gold response: {{expected.gold_response}}

Does the draft response:
(A) Clearly reduce tension without escalating or assigning blame, and preserve healthy project norms — score: 1.0
(B) Neutral; neither escalates nor meaningfully de-escalates — score: 0.5
(C) Matches or worsens the hostile tone, or ignores the issue — score: 0.0
```

**`snippet_grounding`** (binary: 0 or 1)

Verifies that the quoted snippet actually appears in the retrieved comment thread. This is the primary hallucination guard — an agent that fabricates toxic content that wasn't there is tagged with the `hallucination` failure mode.[^16]

### Efficiency / telemetry scorers

The current scorer list also includes three normalized telemetry scorers that make the CLI vs. MCP tradeoff easier to compare row by row:

- `token_efficiency`: inverse-normalized score based on `total_tokens`
- `tool_call_efficiency`: inverse-normalized score based on total tool calls
- `latency_score`: inverse-normalized score based on `latency_ms`

Finally, `failure_mode_tagger` derives post-hoc failure tags from the scored output and converts them into a fractional penalty score so Braintrust can slice rows by failure class while still keeping a numeric column in the experiment table.

***

## Failure Taxonomy

Each eval row is tagged with zero or more failure mode labels. These appear as metadata in Braintrust and enable slice-based analysis.[^15]

| Tag | Definition | Likely Cause |
|---|---|---|
| `retrieval_failure` | Agent did not retrieve enough comments to make a judgment | CLI pagination gap; MCP tool missing for issue comments |
| `hallucination` | Quoted snippet does not exist in the actual thread | LLM confabulation under vague retrieval; low `snippet_grounding` |
| `false_positive` | Non-toxic thread flagged as requiring intervention | Overly aggressive toxicity classification prompt |
| `false_negative` | Genuinely toxic content not flagged | Subtle entitlement/passive-aggression missed by LLM |
| `scope_violation` | Write call made to a non-sandbox repo | Agent misrouted output; prompt ambiguity |
| `report_not_posted` | Workflow analyzed the thread but never created the sandbox report issue | Post-processing write failed; board repo misconfigured; token lacks access |
| `label_mismatch` | Wrong toxicity category assigned | Schema too coarse; label ambiguity |

***

## Braintrust Experiment Structure

### Project Layout

```
Project: community-health-eval
├── Dataset: community_monitor_pandas
├── Experiment: cli-baseline-run-1
├── Experiment: cli-baseline-run-2
├── Experiment: cli-baseline-run-3
├── Experiment: cli-baseline-run-4
├── Experiment: cli-baseline-run-5
├── Experiment: mcp-baseline-run-1
├── Experiment: mcp-baseline-run-2
├── Experiment: mcp-baseline-run-3
├── Experiment: mcp-baseline-run-4
└── Experiment: mcp-baseline-run-5
```

`run_evals.py` now defaults to `--runs 5`, not 3, and launches workflow/run combinations via a `ThreadPoolExecutor`. `--max-workers` can cap the number of concurrent `bt eval` subprocesses; otherwise the current default is conservative when MCP is enabled: 2 workers when both workflows are running, 1 worker for MCP-only runs, and full fanout only for CLI-only runs. If a fatal Anthropic billing failure is detected, pending tasks are cleared and the run exits early after in-flight tasks complete.

### Metadata Tags per Row

```python
span.log(metadata={
    "workflow": "cli",          # or "mcp"
    "repo": "eslint/eslint",
    "discussion_type": "issue",
    "stratum": "borderline",    # clearly_toxic | borderline | heated | control
    "failure_modes": ["retrieval_failure"],  # populated post-scoring
})
```

### Tracking Latency and Token Cost

The task functions emit explicit output fields for `latency_ms`, `retrieval_latency_ms`, `llm_latency_ms`, `prompt_tokens`, `completion_tokens`, `total_tokens`, and `cost_usd`. They also break spend down into `retrieval_tokens` and `analysis_tokens`, plus `retrieval_model` and `analysis_model`, so the Haiku-first optimization can be analyzed directly. The MCP task additionally logs `queue_wait_ms`, which separates semaphore wait time from the row's actual task budget.

***

## Analysis Plan

### Primary Comparisons (CLI vs. MCP)

| Metric | Expected CLI Pattern | Expected MCP Pattern |
|---|---|---|
| `scope_containment` | High (deterministic shell; clear boundaries) | Slightly lower (tool ambiguity; write tools adjacent to read tools) |
| `retrieval_completeness` | Lower on PR inline comments[^12] | Higher when `get_issue_comments` is available[^21]; lower when it isn't[^13] |
| `tool_call_count` | Higher (must `gh api` paginate; multiple subprocess invocations) | Lower (structured, paginated tool returns in fewer calls) |
| `latency_ms` (retrieval only) | Lower for simple threads; higher for pagination | More consistent; single structured call per entity |
| `deescalation_quality` | Equivalent (same LLM; only retrieval differs) | Equivalent |
| `hallucination` rate | Higher when retrieval_completeness is low | Lower when full comment thread is available |

The CLI's Token Efficiency Score advantage documented elsewhere (33% better in browser automation benchmarks) may not hold here because the task is read-heavy and structured — where MCP's typed tool returns reduce parsing overhead.[^22]

### Slice Analysis

Run the scorer suite separately for each stratum:
- **Clearly toxic:** both workflows should perform well; differences reveal retrieval gaps
- **Borderline:** highest variance; reveals LLM sensitivity to partial context from incomplete retrieval
- **Heated but not toxic:** primary false-positive measurement stratum
- **Control:** sanity check; any detection here is a false positive

### Edge Cases to Investigate

1. **Paginated threads (>30 comments):** CLI requires `--paginate` flag or explicit loop; MCP returns paginated results but some implementations have a `per_page` cap. Both may silently truncate.[^21]

2. **PR with both issue comments and review comments:** CLI: `gh pr view --json comments` omits inline review comments; MCP: `get_pull_request_comments` vs. `get_issue` distinction is confusing and has caused tool-routing failures in the wild. Agents using MCP have returned empty responses for PR conversation-tab comments because the LLM chose the wrong tool.[^12][^23][^24][^11]

3. **Deleted or locked threads:** CLI: `gh issue view` returns the issue with a `locked` field but no deleted comments; MCP: same limitation. Both workflows may miss the most severe historical toxicity.

4. **Rate limiting:** GitHub API allows 5,000 requests/hour for authenticated users. A multi-repo run of 20 discussions with pagination could approach this if comment counts are high. The CLI workflow can hit rate limits with fewer retries built in; the MCP server may handle this more gracefully depending on its implementation.

5. **Ambiguous tool selection in MCP:** MCP agents have been observed selecting `get_pull_request_comments` (review comments) when they wanted `get_issue` conversation comments, returning empty results and then proceeding as if the thread had no comments. This is a structurally distinct failure from CLI's pagination gap and should be tracked separately under `retrieval_failure`.[^23][^24]

***

## Community Health Report Output

The final deliverable of the workflow is a per-discussion "Community Health Report" posted as a new issue to the repo configured by `EVAL_OUTPUT_BOARD`. The current prompts do not ask the model to batch multiple discussions into one report; instead, each eval row produces one sandbox issue and the code performs the write after analysis is complete.

```markdown
# Community Health Report — [repo] — [discussion type] #[number]

## Finding
- Toxic or discouraging content detected: yes/no
- Label: entitlement
- Severity: medium

## Problematic snippet
> "Why do maintainers even bother if they can't close issues in under a week?"

## Proposed maintainer response
Thank you for raising this. We understand that waiting for resolution can be frustrating, especially when a bug is blocking your work. Our maintainers are volunteers working across time zones, and we aim to triage all issues within two weeks. If you have bandwidth to contribute a fix, we would be glad to review a PR.
```

The structured format still needs to be machine-checkable, but the current hard requirement in code is simpler: the model must return JSON containing the toxicity finding, snippet, single `toxicity_label`, severity, draft response, and scorer-facing retrieved thread text. The workflow code then posts the sandbox issue, and `scope_containment` plus `report_posted` cover the safety and delivery checks.

***

## Scoring Summary Table

| Scorer | Type | Range | Aggregation |
|---|---|---|---|
| `scope_containment` | Deterministic | 0 or 1 | Must be 1 for row to count |
| `toxicity_label_accuracy` | Deterministic | 0–1 | Mean over toxic strata |
| `report_posted` | Deterministic | 0 or 1 | Must be 1 for row to count as delivered |
| `false_positive_flag` | Deterministic | 0 or 1 | Aggregate over control + heated strata |
| `retrieval_completeness` | Deterministic | 0 or 1 | Mean over all rows |
| `deescalation_quality` | LLM-judge (Haiku) | 0–1 | Mean ± std over repeated runs |
| `snippet_grounding` | LLM-judge (Haiku) | 0 or 1 | Failure rate → `hallucination` tag |
| `token_efficiency` | Deterministic telemetry | 0–1 | Mean by workflow |
| `tool_call_efficiency` | Deterministic telemetry | 0–1 | Mean by workflow |
| `latency_score` | Deterministic telemetry | 0–1 | Mean by workflow |
| `failure_mode_tagger` | Deterministic post-hoc | 0–1 | Penalized by number of failure tags |
| `tool_call_count` | Metadata | Integer | Mean; CLI vs. MCP distribution |
| `latency_ms` | Metadata | Integer | Mean; p50/p95 split |
| `total_tokens` | Metadata | Integer | Mean; cost estimate |

***

## Conclusions and Recommendations

**On workflow design:** The CLI and MCP approaches have structurally different failure modes rather than one being strictly superior. CLI is more reliable for retrieval completeness on simple issues but brittle on PR review comments and requires more tool calls. MCP is cleaner for structured access but has a higher risk of agent tool-routing errors on edge cases (particularly PR conversation vs. review comment distinction).[^24][^22][^12][^23][^5]

**On eval design:** The most valuable scorers for this task are `snippet_grounding` (hallucination guard) and `scope_containment` (safety guard). Quality scorers like `deescalation_quality` are important but inherently variable; repeated-run averaging remains necessary to distinguish signal from noise, with the current default orchestration running five repetitions unless overridden.[^18]

**On dataset design:** The borderline and control strata are as important as the clearly-toxic examples. An agent optimized only on clear toxicity will over-flag subtle disagreement, which is harmful to community health in a different direction than under-flagging.

**On the toxicity schema:** The OSS-specific categories — `entitlement`, `passive_aggression`, and `object_directed` — are the most diagnostically useful because general-purpose toxicity detectors systematically miss them. LLM judges should be explicitly primed with examples of these categories in their system prompts to reduce false negatives.[^2][^7]

---

## References

1. [The Shifting Sands of Toxicity: The Evolving Nature of Interpersonal ...](https://conf.researchr.org/details/esem-2025/esem-2025-technical-track/24/The-Shifting-Sands-of-Toxicity-The-Evolving-Nature-of-Interpersonal-Challenges-in-Op) - These trends indicate that toxicity is not only more pervasive but is having deeper, more damaging e...

2. [Study finds toxicity in the open-source community varies from other ...](https://techxplore.com/news/2022-06-toxicity-open-source-varies-internet-forums.html) - To better understand what toxicity looked like in the open-source community, the team first gathered...

3. [Study Finds Toxicity in the Open-Source Community Varies From ...](https://www.cmu.edu/news/stories/archives/2022/july/study-finds-toxicity-in-the-open-source-community-varies-from-other-online-forums) - To better understand what toxicity looked like in the open-source community, the team first gathered...

4. [Real-Time Toxicity Filtering for Open-Source Code Reviews - arXiv](https://arxiv.org/html/2604.08886v1) - The framework comprises three modules: toxicity identification, reasoned multiclass classification, ...

5. [Braintrust CLI and MCP - Blog](https://www.braintrust.dev/blog/cli-and-mcp) - Learn when to use the Braintrust CLI and MCP depending on where you are in the AI development workfl...

6. [[Papierüberprüfung] The Landscape of Toxicity: An Empirical ...](https://www.themoonlight.io/de/review/the-landscape-of-toxicity-an-empirical-investigation-of-toxicity-on-github) - The paper "The Landscape of Toxicity: An Empirical Investigation of Toxicity on GitHub" aims to delv...

7. [Analyzing Toxicity in Open Source Software Communications Using ...](https://arxiv.org/html/2412.13133v2)

8. [[PDF] Toxic Comment Classification and Mitigation in Social Media Platforms](https://www.scitepress.org/Papers/2025/139031/139031.pdf) - Model evaluation demonstrates high accuracy and Precision, with misclassification challenges observe...

9. [Create scorers - Braintrust](https://www.braintrust.dev/docs/evaluate/write-scorers) - Build scorers to measure AI output quality in experiments and production. Choose from LLM-as-judge, ...

10. [Create experiments - Braintrust](https://www.braintrust.dev/docs/evaluate/run-evaluations) - An experiment is an immutable snapshot of an evaluation run — permanently stored, comparable over ti...

11. [Add PR review helper commands under gh pr for inline comments ...](https://github.com/cli/cli/issues/12232) - Add PR review helper commands under gh pr for inline comments and review flows (AI agent use cases)....

12. [gh pr view --comments should show all comments #5788 - GitHub](https://github.com/cli/cli/issues/5788) - I noticed today that it seems like gh pr view <branch-id> --comments doesn't seem to include inline ...

13. [Add support for reading issue comments in GitHub MCP server #3006](https://github.com/modelcontextprotocol/servers/issues/3006) - The GitHub MCP server currently supports reading pull request comments via github_get_pull_request_c...

14. [Instrument your application - Braintrust](https://www.braintrust.dev/docs/instrument) - Instrumentation captures detailed traces from your AI application, recording inputs, outputs, model ...

15. [Interpret evals](https://www.braintrust.dev/docs/evaluate/interpret-results) - Diagnose where your AI system is underperforming and understand why. Drill into traces, score distri...

16. [A comprehensive taxonomy of hallucinations in Large Language ...](https://arxiv.org/abs/2508.01781) - It analyzes the underlying causes, categorizing them into data-related issues, model-related factors...



20. [Broadcast to Braintrust | OpenRouter Observability | Documentation](https://openrouter.ai/docs/guides/features/broadcast/braintrust) - Connect Braintrust to automatically receive traces from your OpenRouter requests. Step-by-step setup...

21. [GitHub MCP Server](https://mcpservers.org/servers/asifdotpy/github-mcp-server-asifdotpy) - The GitHub MCP Server is a Model Context Protocol (MCP) server that provides seamless integration wi...

22. [Why CLI Tools Are Beating MCP for AI Agents - Jannik Reinhard](https://jannikreinhard.com/2026/02/22/why-cli-tools-are-beating-mcp-for-ai-agents/) - One detailed comparison ran identical browser automation tasks through both MCP and CLI interfaces. ...

23. [Can't get comment from PR "Conversation" tab - Gemini in Cursor ...](https://github.com/github/github-mcp-server/issues/416) - Specifically explain how an LLM using the MCP server should retrieve the list of general "Conversati...

24. [Get pull request comments not fetching comments #1079 - GitHub](https://github.com/github/github-mcp-server/issues/1079) - Describe the bug. It looks like the tool is not working as expected. Is it not including the comment...



# The gap this eval fills
The combination of (1) a read-heavy, multi-step GitHub retrieval task, (2) a downstream content-quality outcome scored by an LLM judge, and (3) a CLI vs. MCP comparison within a single Braintrust experiment is not covered by any existing benchmark. The closest prior work is the Scalekit cost study and Zechner's coding benchmark, but neither measures whether the retrieval strategy affects the quality of an LLM's downstream generation — which is the core question your snippet_grounding and deescalation_quality scorers are designed to answer.

# How this eval could be extended | other similar use cases/implementations 

PR review quality triage

Instead of toxicity, the agent identifies low-effort or unconstructive PR reviews ("LGTM" with no substance, drive-by rejection without explanation) and drafts a request for the reviewer to be more specific. The scorer swaps deescalation_quality for review_specificity. The same CLI vs. MCP retrieval tradeoff applies since get_pull_request_reviews is a distinct tool from get_pull_request_comments.

Stale issue hygiene agent

The agent reads open issues older than 90 days, classifies them (abandoned, blocked-on-author, needs-triage, already-fixed-upstream), and drafts a closing comment or a request for status update. This is purely a classification + generation task with no toxicity judgment, making the LLM-judge scorer simpler and the snippet_grounding hallucination check more important since issues may reference stale context.

First-time contributor welcome auditor

Flip the direction: instead of finding discouraging content, the agent identifies first-time contributors whose PRs or issues received no response within 72 hours and drafts a welcoming acknowledgment from the maintainer. This is a proactive community health intervention rather than a reactive one, and it tests a different retrieval pattern — filtering by author_association: FIRST_TIME_CONTRIBUTOR using gh api or list_pull_requests with filters.

Changelog / release note generator

Same CLI vs. MCP retrieval architecture applied to a generation task with a deterministic ground truth: given a set of merged PRs between two tags, generate a structured changelog. Ground truth is the actual release notes already published. snippet_grounding becomes exact-match verifiable (did the agent cite real PR titles?), which removes LLM-judge variability entirely.

Slack/Discord community health (non-GitHub)

The same eval design ports to Slack MCP or Discord CLI tools for communities that operate outside GitHub. The toxicity schema carries over directly, but the retrieval layer changes: thread structure in Slack is nested differently than GitHub issue comments, which tests whether MCP's structured returns (vs. raw API JSON) produce better context windows for the LLM.
