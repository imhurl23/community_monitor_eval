# OSS Community Health First Responder: CLI vs. MCP Eval Design

## TLDR: Which workflow (CLI/MCP) best helps to monitor and keep the conversation happening in our open source communities safe and effective for dev? 

## Overview

This document covers the design and execution of an evaluation comparing different architectural approaches for the **Community Health First Responder** task — an agent that surfaces potentially toxic or discouraging GitHub discussions and drafts maintainer de-escalation responses.

The primary research question is: **does a CLI-based workflow (using `gh`) or an MCP-based workflow (using the GitHub MCP Server) produce better outcomes on this task, and at what cost?** This spec covers dataset design, task definition, scorer suite, failure taxonomy, and analysis strategy, grounded in current OSS toxicity research and both platforms' documented capabilities.

The broader pattern this eval instantiates is retrieval interface comparison plus downstream content quality measurement. The pattern maps onto a class of problems identified in the OPENTOOLS framework and the four-pillar agent assessment framework from ["Open, Reliable, and Collective: A Community-Driven Framework for Tool-Using AI Agents"](https://arxiv.org/pdf/2604.00137): any task where tool selection and retrieval completeness directly affect the quality of a subsequent LLM generation step benefits from this two-axis design (deterministic retrieval scorers + LLM-judge generation scorers).

---

## Hypotheses
Hypothesis 1 (Effectiveness): The MCP‑based workflow will achieve higher end‑to‑end task quality than the CLI‑based workflow on the Community Health First Responder task, as measured by: (a) correct detection of harmful threads — false‑positive rate via `false_positive_flag` (direct scorer, control/heated strata only) and false‑negative rate derived post‑hoc by filtering rows tagged `false_negative` in `tag_failure_modes`; (b) correct toxicity labeling via `toxicity_label_accuracy`; and (c) quality of de‑escalation drafts via the `deescalation_quality` LLM judge. Note: there is no standalone per‑row FN scorer; FN rate is a post‑hoc aggregate over the `false_negative` tag and should be reported alongside FP rate rather than as a symmetric scorer pair.

**Hypothesis 1 — Effectiveness:** The MCP-based workflow will achieve higher end-to-end task quality than the CLI-based workflow on the Community Health First Responder task, as measured by human labels and LLM-judge scores for (a) correct detection of harmful threads, (b) correct toxicity severity and type labeling, and (c) quality of de-escalation drafts, because MCP exposes more structured, semantically rich GitHub tools that support more complete and targeted retrieval.

Hypothesis 3 (Retrieval interface ↔ failure modes):
The distribution of failure modes will differ between CLI and MCP workflows. CLI runs will show higher rates of retrieval_failure (retrieval_completeness) and hallucination (snippet_grounding), reflecting pagination limits and looser context grounding. MCP runs will show lower tool_call_efficiency scores — more tool calls for the same task — reflecting the overhead of navigating a richer tool surface, along with higher rates of scope_violation and report_not_posted failures from incorrect tool routing.

**Hypothesis 3 — Retrieval interface ↔ failure modes:** The distribution of failure modes will differ systematically between workflows. MCP runs will exhibit fewer retrieval incompleteness and irrelevant-context failures but more tool-selection and protocol misuse failures, while CLI runs will show the inverse pattern, with more partial retrieval and parsing/heuristic limits but fewer overhead and tool-use failures.

---

## Background and Motivation

Toxicity in open source is not rare. A 2024 GitHub-wide survey of 8,452 contributors found a statistically significant increase in reported interpersonal challenges compared to 2017, with rudeness, name-calling, and harassment now strongly predictive of contributors stopping their work entirely.[^1] Research from Carnegie Mellon's ISR established that OSS toxicity is qualitatively different from other internet forums. The toxicity on these forums skews toward entitlement, passive aggression, and contextual insults rather than explicit obscenities.[^2]

An automated "first responder" agent addresses this problem by reducing the burden on burned-out maintainers. Evaluating such an agent through two interface modalities — CLI and MCP  also surfaces a practically important question about tooling tradeoffs that affects any GitHub-integrated AI workflow.

---

## Toxicity Schema

The following eight categories are drawn from the OSS-specific research literature, with OSS-specific nuance noted.[^3][^4][^5]

| Label | Short description | OSS-specific signal |
|---|---|---|
| `hostile_aggression` | Explicit threats, insults, name-calling | Rare in code review; more common in issues with blocked PRs |
| `entitlement` | Demanding tone, "fix this NOW", "why hasn't this been done" | Most distinctive OSS category; often missed by general detectors[^2] |
| `dismissive_tone` | Closing without engagement, "not my problem", curt rejection | Discourages newcomers even when technically polite |
| `sarcasm_belittling` | Ironic minimization, mocking effort or skill | Hard to auto-detect; high false-negative rate in LSTM models[^8] |
| `passive_aggression` | Technically civil but subtly hostile framing | Contextually rich; CMU study flags as most "OSS-specific" pattern[^2] |
| `gatekeeping` | Condescension toward perceived skill level, "have you even read the docs?" | Particularly discouraging for first-time contributors |
| `thread_derailment` | Off-topic escalation, personal attacks displacing technical discussion | Often involves prior relationship context |
| `object_directed` | Hostility directed at code/project artifacts ("this codebase is trash") | Distinct from person-directed; coined by Sarker et al.[^4] |

---

## Dataset Design

### Construction

The dataset consists of N discussion items (a mix of issues and PRs from the source OSS repos), each forming one eval row. The current `community_monitor_pandas` build has X items sampled from the five-repo set described below; that number scales linearly with the number of repos and the per-stratum cap. Construction proceeds in two phases.

**Phase 1 — Two-pass sampling.** The pipeline first runs a cheap metadata sweep, then applies ToxiShield only on threads that could land in a non-control stratum, then stratifies into four buckets and samples per (repo, stratum) cell to prevent any one high-volume repo from dominating.

| Stratum | Default cap per repo | Sampling criterion |
|---|---|---|
| `clearly_toxic_candidate` | 20 | At least one comment with ToxiShield prob ≥ 0.7, OR GitHub maintainers locked the thread with `active_lock_reason == "too heated"` |
| `borderline_candidate` | 20 | At least one comment with prob ≥ 0.4 AND max prob < 0.7 — the ambiguity zone |
| `heated_not_toxic_candidate` | 20 | ≥ 15 comments AND max prob < 0.4 — high engagement, no toxic language |
| `control_candidate` | 20 | ≤ 5 comments AND max prob < 0.2 — low-activity, constructive threads |

Two design choices in this stratification matter for downstream metrics. First, the `borderline` and `control` strata are essential for measuring false-positive rates and `scope_containment` integrity, an agent that flags every heated thread fails just as surely as one that misses genuine toxicity. Second, `clearly_toxic_candidate` is fed by two independent signals: ToxiShield catches lexically explicit toxicity, while GitHub's `too heated` lock reason catches threads the classifier missed (≈71% precision in pilot review). The original classifier verdict is preserved on `metadata.classifier_stratum` so post-annotation analysis can compare the two signals.

**Phase 2 — Ground truth annotation.** For each sampled item, an annotator produces:

- A binary toxicity judgment (`is_toxic: true / false`)
- If toxic: one or more labels from the 8-label schema
- A severity score: `low` / `medium` / `high`
- A `problematic_snippet` quote
- A gold-standard maintainer response (written by a human with OSS maintainer experience)

To speed annotation, every row ships with a `_review.suspect_comments` payload — the top 3 highest-probability comments in the thread as idnetified by ToxicShield classifier, with author, association, timestamp, and a 600-char body preview. The annotator can label without reading the entire thread end-to-end. The `_review.*` namespace is for the human only; it is stripped before the row is shown to the agent under test.

The gold response is used as a soft reference for LLM-judge scoring, not for exact-match comparison.

### Dataset Schema (per row)

```json
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
```

When `stratum` and `classifier_stratum` differ, the row was promoted into `clearly_toxic_candidate` by the lock signal — which is exactly the data needed to study where the classifier and the maintainer's own moderation action disagreed.

### Browser Labeler

The repo includes a browser-based annotation tool at `dataset_curation/braintrust-toxicity-labeler.html` for reviewing sampled rows locally before re-importing them into Braintrust. It is designed around the same row shape produced by the curation pipeline: each row carries the discussion metadata, the current `ground_truth` / `expected` label object, and the `_review.suspect_comments` helper payload.

This tool is used instead of Braintrust's built-in review flow because it surfaces model-ranked suspect comments, makes it easy to copy candidate snippets, and supports direct assignment of toxicity, severity, and label annotations in one place. The labeler supports the full manual review loop on one page:

- Import JSONL or JSON exports built from the curation pipeline
- Filter rows by search text and labeling status
- Review suspect comments, copy candidate snippets, and assign toxicity / severity / labels
- Save edits locally in-browser, then export updated JSONL or patch JSON for upload

It is especially useful for the toxic and borderline strata because the UI surfaces the top-ranked suspect comments, lets the annotator paste a problematic snippet directly from those comments, and keeps the gold response alongside the labeling controls.


<img width="1418" height="913" alt="labeler" src="https://github.com/user-attachments/assets/8c5553d1-ad86-4ccb-a68e-1bbdab078e44" />


### Source Repositories

The eval uses active, mid-size OSS repositories with a documented history of heated discussions. Candidate properties:

- Minimum 500 open issues/PRs
- Active contributor base (10+ unique commenters in the last 90 days)
- No existing bot-moderation that would pre-filter toxic content
- Public repository (read access to source repos; write access required for the eval output board only)

The eval is repo-agnostic by design, but the current `community_monitor_pandas` dataset and results are pooled from an initial pull of just `pandas-dev/pandas`. 

---

## Task Definition

### Agent Task

Given a GitHub discussion (issue or PR), the agent must:

1. Retrieve the full comment thread for that discussion
2. Identify whether any comment is potentially toxic or discouraging
3. If toxic: quote the problematic snippet, assign a single best-fit label from the 8-label schema, assign a severity, and draft a maintainer de-escalation response
4. If not toxic: return a structured "no action needed" response

The agent must not post, edit, or delete any live GitHub content. Any eval artifact writes go exclusively to the repository configured by `EVAL_OUTPUT_BOARD` (a sandbox repo created for evaluation output only). This is the `scope_containment` constraint.

### Output Schema

```json
{
  "discussion_id": "eslint-issue-17823",
  "analysis_status": "completed",
  "toxic_detected": true,
  "snippet": "Why do maintainers even bother if they can't close issues in under a week?",
  "toxicity_label": "entitlement",
  "severity": "medium",
  "draft_response": "Thank you for raising this. We understand the frustration when resolution timelines feel slow...",
  "task_error": null,
  "tool_timeout_count": 0,
  "tool_calls": ["..."],
  "latency_ms": 4200,
  "total_tokens": 1850,
  "retrieved_thread_text": "compressed thread text used for scorer validation"
}
```

---

## Workflow Implementations

### CLI Workflow

The CLI agent uses the `gh` command-line tool with shell scripting and `gh api` calls for structured JSON access. Braintrust CLI (`bt`) handles eval orchestration.

**Retrieval commands:**

```bash
# Fetch issue body and metadata
gh issue view 17823 --repo eslint/eslint --json title,body,comments,author

# For PRs: fetch review comments (note: inline review comments are a known gap)
gh pr view 42 --repo eslint/eslint --json title,body,comments,reviews

# Fallback for paginated comment threads
gh api /repos/eslint/eslint/issues/17823/comments --paginate
```

A known limitation is that `gh pr view --json comments` does not include inline review comments as of 2025. The CLI agent must use `gh api` as a fallback to retrieve those, which adds a tool call and increases latency. This is a documented edge case in the eval.[^6][^7]

**Eval runner:**

```bash
bt eval community_health_cli.py --project "community-health-eval"
```

### MCP Workflow

For this experiment, MCP agent uses the GitHub MCP Server through a local `mcp-proxy` SSE bridge. The eval task opens a `ClientSession`, caches the discovered tool list once per process, and gates concurrent SSE sessions with `MCP_MAX_CONCURRENT` so Braintrust row-level parallelism does not overwhelm the single proxied server process. The code normalizes generic MCP tools into a canonical read surface:[^8]

```
issue_read(method=get, ...)                        → canonical get_issue
issue_read(method=get_comments, ...)               → canonical get_issue_comments
pull_request_read(method=get, ...)                 → canonical get_pull_request
pull_request_read(method=get_comments, ...)        → canonical get_pull_request_comments
pull_request_read(method=get_review_comments, ...) → canonical get_pull_request_comments
```

This normalization is necessary because the server does not expose the same stable top-level tool names as the CLI workflow.

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

---

## Scorer Suite

### Deterministic Scorers

**`scope_containment`** (binary: 0 or 1)

Checks that the workflow made no write calls to any repo other than the repo configured by `EVAL_OUTPUT_BOARD`. This is the highest-stakes scorer — a score of 0 is a hard failure regardless of other scores.

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

**`toxicity_label_accuracy`** (continuous: 0–1)

Compares predicted label(s) against ground truth. Uses partial credit: exact match = 1.0, same parent category = 0.5, no match = 0. Applied only to rows where `is_toxic = true`.

**`report_posted`** (binary: 0 or 1)

Checks that the workflow actually created the sandbox issue in the repo configured by `EVAL_OUTPUT_BOARD`. A score of 0 means the analysis may have completed, but the required artifact was never written.

**`false_positive_flag`** (binary: 0 or 1, per row)

Flags non-toxic rows where the agent incorrectly detected toxicity. Applied only to the control and heated-but-not-toxic strata. Dataset-level false-positive rate is computed by aggregating this flag across those slices.

**`retrieval_completeness`** (binary: 0 or 1)

Checks whether the agent retrieved at least N–1 of N comments in the thread (allowing for one pagination miss). Ground truth comment counts are stored in the dataset metadata.

**Tool usage metrics** (logged as metadata per row):
- `tool_call_count`: total GitHub API calls made
- `latency_ms`: wall-clock time from task start to structured output
- `prompt_tokens` / `completion_tokens`: tracked via Braintrust span instrumentation

### LLM-Judge Scorers (Claude Haiku)

The current implementation uses a combined `llm_judges()` scorer that runs the two active Haiku judges concurrently in a `ThreadPoolExecutor`, reducing scorer latency relative to sequential judge calls.

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

Verifies that the quoted snippet actually appears in the retrieved comment thread. This is the primary hallucination guard — an agent that fabricates toxic content that wasn't there is tagged with the `hallucination` failure mode.

### Efficiency / Telemetry Scorers

Three normalized telemetry scorers make the CLI vs. MCP tradeoff easier to compare row by row:

- `token_efficiency`: inverse-normalized score based on `total_tokens`, anchored at 10K tokens and decaying to 0 at 50K
- `tool_call_efficiency`: inverse-normalized score based on total tool calls
- `latency_score`: inverse-normalized score based on `latency_ms`

`failure_mode_tagger` derives the highest-priority failure mode from the scored output and returns a numeric ordinal for Braintrust compatibility. Current mapping: `0 = none`, `1 = scope_violation`, `2 = hallucination`, `3 = false_negative`, `4 = false_positive`, `5 = retrieval_failure`, `6 = report_not_posted`, `7 = label_mismatch`, `99 = scorer error`.

### Scoring Summary

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
| `failure_mode_tagger` | Deterministic post-hoc | Numeric ordinal | Primary failure mode rank (`0 = none`, `99 = scorer error`) |
| `tool_call_count` | Metadata | Integer | Mean; CLI vs. MCP distribution |
| `latency_ms` | Metadata | Integer | Mean; p50/p95 split |
| `total_tokens` | Metadata | Integer | Mean; cost estimate |

---

## Failure Taxonomy #TODO figure out what is going on here

The scorer code defines the following failure taxonomy and collapses each row to a single primary failure mode via `failure_mode_tagger`. The numeric primary-mode score is what is stored in Braintrust.

| Tag | Definition | Likely cause |
|---|---|---|
| `retrieval_failure` | Agent did not retrieve enough comments to make a judgment | CLI pagination gap; MCP tool missing for issue comments |
| `hallucination` | Quoted snippet does not exist in the actual thread | LLM confabulation under vague retrieval; low `snippet_grounding` |
| `false_positive` | Non-toxic thread flagged as requiring intervention | Overly aggressive toxicity classification prompt |
| `false_negative` | Genuinely toxic content not flagged | Subtle entitlement/passive-aggression missed by LLM |
| `scope_violation` | Write call made to a non-sandbox repo | Agent misrouted output; prompt ambiguity |
| `report_not_posted` | Workflow analyzed the thread but never created the sandbox report issue | Post-processing write failed; board repo misconfigured; token lacks access |
| `label_mismatch` | Wrong toxicity category assigned | Schema too coarse; label ambiguity |

---

## Braintrust Experiment Structure

### Project Layout

```
Project: community-health-eval
├── Dataset: community_monitor_pandas
├── Experiment: cli-improved-run-1
├── Experiment: cli-improved-run-2
├── Experiment: ...
├── Experiment: mcp-improved-run-1
├── Experiment: mcp-improved-run-2
└── Experiment: ...
```

`run_evals.py` defaults to `--runs 5` and launches workflow/run combinations via a `ThreadPoolExecutor`. `--max-workers` caps the number of concurrent `bt eval` subprocesses. The current default is conservative when MCP is enabled: 2 workers when both workflows are running, 1 worker for MCP-only runs, and full fanout for CLI-only runs. Experiment names are configurable via `--cli-experiment-prefix` / `--mcp-experiment-prefix` (defaulting to `cli-improved` / `mcp-improved`). If a fatal Anthropic billing failure is detected, pending tasks are cleared and the run exits early after in-flight tasks complete.

### Metadata Tags per Row

```python
span.log(metadata={
    "workflow": "cli",                          # or "mcp"
    "repo": "eslint/eslint",
    "discussion_type": "issue",
    "stratum": "borderline",                    # clearly_toxic | borderline | heated | control
    "discussion_id": "eslint-issue-17823",
    "retrieval_model": "claude-haiku-4-5-20251001",
    "analysis_model": "claude-haiku-4-5-20251001",
})
```

### Tracking Latency and Token Cost

The task functions emit explicit output fields for `latency_ms`, `retrieval_latency_ms`, `llm_latency_ms`, `prompt_tokens`, `completion_tokens`, `total_tokens`, and `cost_usd`. They also break spend down into `retrieval_tokens` and `analysis_tokens`, plus `retrieval_model` and `analysis_model`, so the Haiku-first optimization can be analyzed directly. Both tasks emit `analysis_status`, `task_error`, `tool_error_messages`, and `tool_timeout_count` so incomplete rows can be analyzed rather than treated as silent failures. The MCP task additionally logs `queue_wait_ms`, which separates semaphore wait time from the row's actual task budget.

---

## Community Health Report Output

The final deliverable of the workflow is a per-discussion "Community Health Report" posted as a new issue to the eval repo. Each eval row produces one sandbox issue; the code performs the write after analysis is complete.

```markdown
# Community Health Report — [repo] — [discussion type] #[number]

## Finding
- Toxic or discouraging content detected: yes/no
- Label: entitlement
- Severity: medium

## Problematic snippet
> "Why do maintainers even bother if they can't close issues in under a week?"

## Proposed maintainer response
Thank you for raising this. We understand that waiting for resolution can be frustrating,
especially when a bug is blocking your work. Our maintainers are volunteers working across
time zones, and we aim to triage all issues within two weeks. If you have bandwidth to
contribute a fix, we would be glad to review a PR.
```

The current hard requirement in code is: the model must return JSON containing the toxicity finding, snippet, single `toxicity_label`, severity, and draft response. The workflow code then normalizes missing fields into a stable schema, attaches scorer-facing retrieval fields, and posts the sandbox issue. If retrieval or tool execution aborts, the row is still returned with `analysis_status="incomplete"` plus `task_error` / timeout metadata so the failure is analyzable inside Braintrust rather than crashing the whole eval.

---

## The Gap This Eval Fills

The combination of (1) a read-heavy, multi-step GitHub retrieval task, (2) a downstream content-quality outcome scored by an LLM judge, and (3) a CLI vs. MCP comparison within a single Braintrust experiment is not covered by any existing benchmark. The closest prior work is the Scalekit cost study and Zechner's coding benchmark, but neither measures whether the retrieval strategy affects the quality of an LLM's downstream generation — which is the core question the `snippet_grounding` and `deescalation_quality` scorers are designed to answer.

---

## How This Eval Could Be Extended

**PR review quality triage.** Instead of toxicity, the agent identifies low-effort or unconstructive PR reviews ("LGTM" with no substance, drive-by rejection without explanation) and drafts a request for the reviewer to be more specific. The scorer swaps `deescalation_quality` for `review_specificity`. The same CLI vs. MCP retrieval tradeoff applies since PR review retrieval is a distinct surface from PR conversation comments in both workflows.

**Stale issue hygiene agent.** The agent reads open issues older than 90 days, classifies them (`abandoned`, `blocked-on-author`, `needs-triage`, `already-fixed-upstream`), and drafts a closing comment or a request for status update. This is a classification + generation task with no toxicity judgment, making the LLM-judge scorer simpler and `snippet_grounding` more important since issues may reference stale context.

**First-time contributor welcome auditor.** Instead of finding discouraging content, the agent identifies first-time contributors whose PRs or issues received no response within 72 hours and drafts a welcoming acknowledgment. This is a proactive community health intervention rather than a reactive one, and it tests a different retrieval pattern — filtering by `author_association: FIRST_TIME_CONTRIBUTOR` using `gh api` or `list_pull_requests` with filters.

**Changelog / release note generator.** The same CLI vs. MCP retrieval architecture applied to a generation task with a deterministic ground truth: given a set of merged PRs between two tags, generate a structured changelog. Ground truth is the actual release notes already published. `snippet_grounding` becomes exact-match verifiable (did the agent cite real PR titles?), which removes LLM-judge variability entirely.

**Slack / Discord community health (non-GitHub).** The same eval design ports to Slack MCP or Discord CLI tools for communities that operate outside GitHub. The toxicity schema carries over directly, but the retrieval layer changes — thread structure in Slack is nested differently than GitHub issue comments, which tests whether MCP's structured returns (vs. raw API JSON) produce better context windows for the LLM.

---

## References

[^1]: ["The Shifting Sands of Toxicity: The Evolving Nature of Interpersonal Challenges in Open Source"](https://conf.researchr.org/details/esem-2025/esem-2025-technical-track/24/The-Shifting-Sands-of-Toxicity-The-Evolving-Nature-of-Interpersonal-Challenges-in-Op) — ESEM 2025. 2024 GitHub-wide survey of 8,452 contributors on interpersonal challenges and contributor attrition.

[^2]: ["Study Finds Toxicity in the Open-Source Community Varies From Other Online Forums"](https://www.cmu.edu/news/stories/archives/2022/july/study-finds-toxicity-in-the-open-source-community-varies-from-other-online-forums) — Carnegie Mellon University News, July 2022. Primary source for CMU ISR findings on passive aggression as the most OSS-specific toxicity pattern.

[^3]: ["Real-Time Toxicity Filtering for Open-Source Code Reviews"](https://arxiv.org/html/2604.08886v1) — arXiv:2604.08886. Framework comprising toxicity identification, reasoned multiclass classification, and mitigation modules relevant to the 8-label schema.

[^4]: ["The Landscape of Toxicity: An Empirical Investigation of Toxicity on GitHub"](https://www.themoonlight.io/de/review/the-landscape-of-toxicity-an-empirical-investigation-of-toxicity-on-github) — Sarker et al. Source for the `object_directed` label category (hostility directed at artifacts rather than persons).

[^5]: ["Analyzing Toxicity in Open Source Software Communications"](https://arxiv.org/html/2412.13133v2) — arXiv:2412.13133. OSS-specific toxicity communication analysis; informs label schema design.

[^6]: ["Add PR review helper commands under `gh pr` for inline comments"](https://github.com/cli/cli/issues/12232) — GitHub CLI issue #12232. Documents the absence of inline review comment support in `gh pr` and the AI agent use case implications.

[^7]: ["`gh pr view --comments` should show all comments"](https://github.com/cli/cli/issues/5788) — GitHub CLI issue #5788. Documents that `gh pr view <branch-id> --comments` does not include inline review comments, requiring `gh api` as a fallback.

[^8]: ["Add support for reading issue comments in GitHub MCP server"](https://github.com/modelcontextprotocol/servers/issues/3006) — MCP Servers issue #3006. Documents that the GitHub MCP server's tool naming for issue vs. PR comments differs from stable top-level names; basis for the normalization layer.

[^9]: ["A Comprehensive Taxonomy of Hallucinations in Large Language Models"](https://arxiv.org/abs/2508.01781) — arXiv:2508.01781. Taxonomy of hallucination types and underlying causes; basis for treating fabricated snippets as a distinct, high-priority failure mode.

