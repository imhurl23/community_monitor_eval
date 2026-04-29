# OSS Community Health First Responder: CLI vs. MCP Performance on Pandas Discussion Threads

Comparative evaluation of CLI and MCP moderation agents on `pandas-dev/pandas` discussion samples. This report compares two implementations of the same GitHub community-health triage workflow: a CLI path built on `gh` and an MCP path built on the GitHub MCP Server, both running Anthropic Claude Haiku 4.5.

**Main finding:** MCP is stronger on retrieval completeness. CLI is stronger on downstream moderation quality, with roughly one-fifth the token footprint and substantially lower variance across runs.

| | |
|---|---|
| **Project** | `community-health-eval` |
| **Run cohort** | CLI_final + MCP_final |
| **Runs compared** | 16 fixed final runs (8 CLI, 8 MCP) |
| **Rows observed** | 8,933 (558 average per run) |

---

## TL;DR

| Dimension | CLI | MCP |
|---|---|---|
| Retrieval completeness | 0.829 ± 0.052 | **0.868 ± 0.020** |
| Toxicity label accuracy | **0.191 ± 0.009** | 0.146 ± 0.008 |
| De-escalation quality | **0.833 ± 0.000** | 0.563 ± 0.165 |
| Scope containment | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Tokens (mean) | **3,442 ± 12** | 18,293 ± 131 |
| Latency (mean) | **5,097ms ± 307ms** | 14,421ms ± 1,246ms |
| Cost per item | **$0.0045** | $0.0203 |
| Report posted | 0.888 ± 0.287 | 0.810 ± 0.350 |

> `report_posted` is partially contaminated by a GitHub secondary rate-limit incident — see [Caveats](#caveats).

---

## H1 — Effectiveness, safety, and moderation quality

> Tests whether MCP improves end-to-end moderation quality on toxicity detection, labeling, and de-escalation.

**Summary:** MCP is better at retrieval completeness and consistency. CLI is better on toxicity labeling and de-escalation quality. Safety containment is effectively tied at ceiling.

### Claim 1 — Retrieval completeness

MCP retrieves more completely on average (0.868 vs. 0.829) with tighter variance (±0.020 vs. ±0.052), suggesting the tool-mediated path is more consistent at pulling available evidence into context. This is the primary advantage of MCP for this workflow.

| | Mean | Std |
|---|---|---|
| CLI | 0.829 | 0.052 |
| **MCP** | **0.868** | **0.020** |

<img width="680" alt="Retrieval completeness — mean ± std across fixed CLI and MCP runs" src="https://github.com/user-attachments/assets/00c65d51-aaf6-408d-bbda-aa16cf90927a" />

### Claim 2 — Scope containment

Both workflows saturate at 1.0 across all runs. Neither path shows evidence of inappropriate write attempts or scope-escape behavior in the observed rows — both implementations correctly restrict themselves to read operations and sandbox report creation.

| | Mean | Std |
|---|---|---|
| CLI | 1.000 | 0.000 |
| MCP | 1.000 | 0.000 |

### Claim 3 — Toxicity label accuracy

CLI performs materially better on label accuracy (0.191 vs. 0.146), but both pipelines still struggle overall. MCP's weaker recall (its ability to correctly identify toxic content) is the more concerning safety outcome.

MCP achieves near-perfect precision (rarely generating false positives) but misses most toxic threads, flagging only the most egregious, lexically explicit cases while letting the vast majority of harmful content through undetected. In a content moderation context, this is arguably the worse failure mode: contributors are still exposed to most of the harm, and the system creates a false sense of safety by appearing to work on the cases it does catch. CLI's broader but noisier detection makes it a more complete safety net, even accounting for its higher false-positive rate.

| | Mean | Std |
|---|---|---|
| **CLI** | **0.191** | **0.009** |
| MCP | 0.146 | 0.008 |

<img width="680" alt="Toxicity label accuracy — mean ± std across fixed CLI and MCP runs" src="https://github.com/user-attachments/assets/3236d774-3e25-4161-956b-bdc4c3fd3715" />
<img width="680" alt="Toxicity confusion matrix — mean ± std across fixed CLI and MCP runs" src="https://github.com/user-attachments/assets/1840e42f-19a8-449f-a30d-cc1afa79d76c" />

### Claim 4 — De-escalation quality

CLI delivers substantially stronger de-escalation quality (0.833 vs. 0.563) despite worse retrieval completeness. More complete retrieval does not automatically improve downstream response quality — this is a direct counter to the intuition that a better context window produces better outputs.

| | Mean | Std |
|---|---|---|
| **CLI** | **0.833** | **0.000** |
| MCP | 0.563 | 0.165 |

CLI's stronger snippet-grounding behavior is the likely explanation: CLI tends to anchor its draft response to a specific quoted passage from the thread, producing de-escalation messages that address the actual toxic content directly. MCP's broader, less-grounded context appears to generate more generic responses that acknowledge a problem without clearly connecting to it.

<img width="553" height="269" alt="Screenshot 2026-04-29 at 8 45 29 AM" src="https://github.com/user-attachments/assets/51b2a278-a343-4a3c-9c63-0166f377f276" />

This finding is a direct signal that retrieval quality and response quality are not interchangeable objectives in this eval design.

<img width="680" alt="De-escalation quality — mean ± std across fixed CLI and MCP runs" src="https://github.com/user-attachments/assets/f05df5dc-66d8-4a5c-9c0e-173655d7a01b" />



---

## H2 — Efficiency, token footprint, and operational variance

> Tests whether CLI reaches comparable quality at lower token cost, lower latency, and lower operational variance.

**Summary:** CLI is the more efficient and more predictable workflow on every measured dimension.

### Claim 1 — Cost per moderation event

At the current pandas intake rate (GitHub Search API snapshot 2026-04-21 to 2026-04-28: 15 issues + 76 PRs = 91 events per week):

| | Cost per item | Std | Weekly cost |
|---|---|---|---|
| **CLI** | **$0.0045** | $0.0001 | **$0.41** |
| MCP | $0.0203 | $0.0002 | $1.85 |

CLI is **$1.43 cheaper per week** at current pandas intake — a 4.5× cost difference driven almost entirely by token volume.

<img width="680" alt="Token footprint — mean ± std across fixed CLI and MCP runs" src="https://github.com/user-attachments/assets/073f8be2-922a-4356-b457-3c15991ff6ba" />

### Claim 2 — Latency and variance

CLI is 2.8× faster on average (5,097ms vs. 14,421ms) and substantially tighter in both token and latency variance. The lower variance matters operationally: CLI runs are easier to budget and far less likely to produce surprise tail-latency spikes.

| | Mean latency | Std | Mean tokens | Std |
|---|---|---|---|---|
| **CLI** | **5,097ms** | **307ms** | **3,442** | **12** |
| MCP | 14,421ms | 1,246ms | 18,293 | 131 |

<img width="680" alt="Latency — mean ± std across fixed CLI and MCP runs" src="https://github.com/user-attachments/assets/25bae825-92ea-4e85-96eb-a27fbeb21709" />

### Claim 3 — Per-run token distributions

Token distributions show MCP carrying a consistently higher and wider usage band across every paired run. The `mcp-final-run-44` outlier (max 45,546 tokens vs. the typical MCP ceiling of ~27,000) is the single highest-cost row in the dataset and warrants separate investigation.

| Run | Token range | Std |
|---|---|---|
| `mcp-final-run-51` | 739 – 27,278 | 5,940 |
| `cli-final-run-51` | 713 – 8,463 | 1,615 |
| `mcp-final-run-50` | 739 – 27,279 | 5,984 |
| `cli-final-run-50` | 713 – 8,485 | 1,603 |
| `mcp-final-run-49` | 739 – 26,172 | 5,938 |
| `cli-final-run-49` | 713 – 8,474 | 1,601 |
| `mcp-final-run-48` | 739 – 27,272 | 6,017 |
| `cli-final-run-48` | 713 – 8,485 | 1,621 |
| `mcp-final-run-47` | 739 – 26,153 | 5,913 |
| `cli-final-run-47` | 713 – 8,473 | 1,612 |
| `mcp-final-run-46` | 739 – 27,186 | 5,934 |
| `cli-final-run-45` | 713 – 8,473 | 1,622 |
| `mcp-final-run-44` | 739 – **45,546** | 6,935 |
| `cli-final-run-44` | 652 – 8,463 | 1,624 |
| `mcp-final-run-43` | 739 – 26,158 | 5,944 |
| `cli-final-run-43` | 652 – 8,468 | 1,602 |

---

## H3 — Retrieval interface and failure modes

> Tests whether CLI and MCP induce different retrieval-linked failure patterns and tool-surface overhead.

**Summary:** MCP improves retrieval completeness and uses more tool calls, but completed rows still distribute across scorer-defined failure modes. Explicit runtime errors are a separate execution-risk surface concentrated in MCP. The interface changes the *mix* of failures, not just the retrieval rate.

### Claim 1 — The interface changes the failure mix, not just the retrieval rate

MCP is stronger on retrieval completeness but CLI is stronger on snippet grounding: better retrieval coverage does not automatically translate into better-grounded moderation outputs. The more comprehensive MCP retrievals combined with tool overhead appear to dilute the model's attention; flooding the context with additional comments makes it harder, not easier, to identify and anchor on the specific toxic passage. CLI's constrained context forces tighter focus.

**Failure priority order:** `scope_violation → hallucination → false_negative → false_positive → retrieval_failure → report_not_posted → label_mismatch`

MCP's zero hallucination and near-zero false-positive counts reflect its 100% precision / ~15% recall tradeoff: it only flags when very confident, missing most toxic content. CLI may not retrieve the full conversation context, which occasionally leads it to flag a comment that reads as toxic in isolation but is neutralized by surrounding replies.

### Claim 2 — MCP carries higher tool-surface overhead

MCP uses more tool calls per row, as expected from the richer tool surface. More importantly, MCP is the only workflow that produces explicit runtime errors, and at a non-trivial rate. However, it is likely the cli fails more silently. 

All 117 MCP execution errors trace to SSE transport failures (`sse_client`, `httpx`, `httpcore`) rather than logic errors. MCP's server-sent event connection drops under sustained concurrent load. CLI has no equivalent failure surface because it shells out to `gh` commands synchronously. Full error counts appear below.

<img width="680" alt="Tool calls — mean ± std across fixed CLI and MCP runs" src="https://github.com/user-attachments/assets/62014872-5c47-46e4-b1ea-b37b3a92d42f" />

### Claim 3 — Scorer-defined failure distribution

The `report_posted` gap cannot be cleanly attributed to workflow behavior due to the rate-limit incident (see Caveats). Excluding the final two runs, the gap narrows considerably and is not statistically distinguishable from noise.

**Failure taxonomy by workflow** (scorer-tagged failures only; MCP execution errors counted separately):

| Failure type | CLI count | CLI % | MCP count | MCP % |
|---|---|---|---|---|
| Scope violation | 0 | 0% | 0 | 0% |
| Hallucination | 8 | 7% | 0 | 0% |
| **False negative** | **39** | **35%** | **47** | **47%** |
| False positive | 7 | 6% | 0 | 0% |
| Retrieval failure | 31 | 28% | 18 | 18% |
| Report not posted | 25 | 23% | 35 | 35% |
| Label mismatch | 0 | 0% | 0 | 0% |
| **Total** | **110** | | **100** | |
| Completed rows | 4,712 | | 4,104 | |

False negative is the largest scorer-defined failure mode for both workflows — 35% of CLI failures and 47% of MCP failures. CLI distributes its remaining failures across retrieval (28%) and report-posting (23%). MCP's remaining failures split between report-not-posted (35%) and retrieval failure (18%), with zero hallucinations, zero false positives, and zero scope violations or label mismatches.

**Execution error summary** (rows that failed during execution, not scored as moderation outcomes):

| Workflow | Rows | Explicit errors | Error rate | Frequent signatures |
|---|---|---|---|---|
| CLI | 4,712 across 8 runs | 0 | 0% | — |
| MCP | 4,221 across 8 runs | 117 | 3% | `sse_client` 117 · `httpx` 57 · `httpcore` 57 |

All 117 MCP errors are transport failures, not moderation logic failures. They are excluded from the scorer-tagged failure totals above.
 
## Failure examples — run 43
 
These cases are drawn from run 43 of each implementation but reflect patterns observed across other runs as well.
 
### CLI — retrieval truncation cascades into wrong diagnosis (PR #63831)
 
CLI's 19,049 total tokens made it the costliest row in the CLI run, and it produced a cascade of failures: token efficiency = 0, retrieval completeness = 0, snippet grounding = 0, wrong severity.
 
The thread had 36 comments. CLI's pagination dropped the final two, including the human-annotated toxic comment at position 32: "I don't have the bandwidth to walk you through this. I need you to figure this out for yourself." (`gatekeeping`, severity high). The model anchored instead on an earlier, syntactically obvious comment: "Closing as stale. can reopen when youre ready to address comments" at position 6, labeling it `dismissive_tone` at medium severity.
 
The failure chain has a single root cause. Retrieval drops the tail → model anchors on first salient signal → all downstream scores degrade. It is not three independent errors.
 
### CLI — false positive on policy enforcement (PR #63898)
 
"I suspect this pull request used AI in an irresponsible manner." is a legitimate code-review concern under the pandas `AGENTS.md` policy. CLI labeled it `dismissive_tone` and drafted a de-escalation response; MCP correctly passed this row without flagging. If posted, CLI's draft would have implicitly invalidated a reasonable reviewer concern. Policy-enforcement language framed as suspicion is a category neither model handles well, but CLI's broader detection threshold makes it the one that misfires here.
 
### MCP — high cost on a correctly-classified non-toxic thread (PR #64366)
 
MCP's highest-token row in run 43: 52,316 tokens, $0.0228, on a 35-comment thread where 32 were retrieved and the correct output was "not toxic." The detection was correct, but MCP spent 5× the CLI per-item mean to reach the same conclusion on a thread that required no action. This is a representative example of MCP's efficiency problem: it issues multiple tool calls regardless of whether the content warrants them, and large non-toxic threads are where that overhead is most visible.
 
### Shared misses — arc-level toxicity evades both workflows (#63444, #63991, #64588)
 
Both CLI and MCP retrieved the relevant comments and still failed to flag these threads. They are not retrieval failures. PR #63444 (`clearly_toxic`, prob = 0.998) involves `gatekeeping` through an implication about contributor bias; PR #63991 (`clearly_toxic`, prob = 0.999) involves `thread_derailment` embedded in a technical dispute; PR #64588 involves `dismissive_tone` expressed as a casual deferral ("Just wait 3 more days I will review, I am little busy!").
 
All three require reading the conversational arc rather than identifying a single flagged passage. The one toxic thread MCP did catch PR #63446's explicit "why are you so dumb 😊" is the only case in the sample with unambiguous lexical toxicity. Retrieval improvements will not fix these misses; they require either prompting that explicitly asks for arc-level analysis or a dedicated pre-pass for these subtler categories.
 
---
## Caveats

**This eval is most useful as a measure of tradeoffs along the dimensions measured, not a single-score leaderboard. It compares retrieval interface tradeoffs for a downstream content-quality task, not a universal verdict on MCP vs. CLI.**

**Tool call efficiency carries no signal for current CLI setup.** Every row in this run uses exactly three tool calls (`get_pull_request → get_pull_request_comments → create_issue`), producing a locked `tool_call_efficiency` score of 0.833 for all 39 rows unless there is an error/retry. 
### Rate-limit incident

`report_posted` was affected by a GitHub secondary rate limit rather than a clean workflow difference. The failure affected both workflows inconsistently while evaluations were running concurrently. The final two runs should be treated as outliers for direct `report_posted` comparisons.

<img width="680" alt="Report posted — mean ± std across fixed CLI and MCP runs" src="https://github.com/user-attachments/assets/5b680753-b0b9-4a9a-8f2e-23c7e30630ab" />

```
GitHub issue create failed: HTTP 403
"You have exceeded a secondary rate limit and have been temporarily
blocked from content creation."
Request ID: F3EA:8DB75:230E067:8A7B957:69F133E5
Timestamp: 2026-04-28 22:25:42 UTC
```

### Tradeoff frame

MCP is stronger on retrieval completeness and consistency. CLI is stronger on downstream toxicity labeling, de-escalation quality, token footprint, and variance. The most important finding is the mismatch between retrieval strength and downstream moderation quality: **more complete retrieval does not automatically produce better-grounded responses.** The mechanism may be attentional: a broader context window in a high-noise thread makes it harder, not easier, to isolate the relevant toxic passage and respond to it specifically.

A promising direction for future work is combining MCP's retrieval completeness with CLI-style snippet-first prompting: retrieve comprehensively, then force the model to identify and quote the specific passage before drafting a response. Whether this closes the de-escalation quality gap without inheriting MCP's token and latency costs is an open question.
