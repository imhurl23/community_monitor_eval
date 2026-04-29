# OSS Community Health First Responder: CLI vs. MCP performance on Pandas discussion threads 

Comparative evaluation of CLI and MCP moderation agents on `pandas-dev/pandas` discussion samples. This report compares two implementations of the same GitHub community-health triage workflow: a CLI path built on `gh` and an MCP path built on the GitHub MCP Server running with Anthropic Haiku 4-5    .

**Main Finding:** While is stronger on retrieval completeness, CLI is stronger on downstream moderation quality with __ the token and much less variance.

| | |
|---|---|
| **Project** | `community-health-eval` |
| **Run cohort** | CLI_final + MCP_final , more runs are shown in the Braintrust project but do not reflect a stable implementation of the evaluation pipeline |
| **Runs compared** | 16 fixed final CLI and MCP runs |
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
| Cost/item | **$0.0045** | $0.0203 |
| Report posted | 0.888 ± 0.287 | 0.810 ± 0.350 |

> `report_posted` is partially contaminated by a GitHub secondary rate-limit incident — see [Caveats](#caveats).

---

## H1 — Effectiveness, safety, and moderation quality

> Tests whether MCP improves end-to-end moderation quality on toxicity detection, labeling, and de-escalation.

**Summary:** MCP is better at retrieval completeness and consistency. CLI is better on toxicity labeling and de-escalation quality. Safety containment is effectively tied at ceiling.

### Claim 1 — Retrieval completeness

MCP retrieves more completely on average with less variance, suggesting the tool-mediated path is more consistent at pulling available evidence into context.

<img width="478" height="238" alt="Screenshot 2026-04-29 at 6 37 09 AM" src="https://github.com/user-attachments/assets/00c65d51-aaf6-408d-bbda-aa16cf90927a" />


### Claim 2 — Scope containment

Both workflows saturate at 1.0. Neither path shows evidence of innapropriate write attempts or scope-escape behavior in the observed rows.

| | Mean | Std |
|---|---|---|
| CLI | 1.000 | 0.000 |
| MCP | 1.000 | 0.000 |

### Claim 3 — Toxicity label accuracy

CLI performs materially better on label accuracy, but both pipelines still struggle. MCP's weaker recall (ability to correctly identify more of the tocix content) is the more concerning safety outcome.

<img width="473" height="229" alt="Screenshot 2026-04-29 at 6 39 33 AM" src="https://github.com/user-attachments/assets/3236d774-3e25-4161-956b-bdc4c3fd3715" />
<img width="478" height="231" alt="Screenshot 2026-04-29 at 6 39 02 AM" src="https://github.com/user-attachments/assets/1840e42f-19a8-449f-a30d-cc1afa79d76c" />

MCP achieves perfect precision (never adding a false positive) but misses most of toxic threads; it only flags the most egregious, unambiguous cases while letting the vast majority of toxic content through undetected. In a content moderation context, this is arguably the worse failure mode: users are still exposed to most of the harm, and the system creates a false sense of safety by appearing to "work" on the cases it does catch. It is possible with further tuning or specific toxicity tooling this could improve but the implementation evaluated here is insufficient. 

### Claim 4 — De-escalation quality

CLI delivers stronger de-escalation quality despite worse retrieval completeness. More complete retrieval does not automatically improve downstream response quality. CLI's stronger snippet-grounding behavior is the likely explanation.
<img width="478" height="255" alt="Screenshot 2026-04-29 at 6 40 01 AM" src="https://github.com/user-attachments/assets/f05df5dc-66d8-4a5c-9c0e-173655d7a01b" />


This is a clear sign that retrieval quality and response quality are not interchangeable objectives in this eval design.

---

## H2 — Efficiency, token footprint, and operational variance

> Tests whether CLI reaches comparable quality at lower token cost, lower latency, and lower operational variance.

**Summary:** CLI is the more efficient and more predictable workflow on every measured dimension (as expected).

### Claim 1 — Cost per moderation event

**Weekly pandas workload scenario** (GitHub Search API snapshot 2026-04-21 to 2026-04-28: 15 issues + 76 PRs = 91 events/week):

| | Cost/item | Std | Weekly cost |
|---|---|---|---|
| CLI | **$0.0045** | $0.0001 | **$0.41** |
| MCP | $0.0203 | $0.0002 | $1.85 |

CLI is **$1.43 cheaper per week** at the current pandas intake.

<img width="1040" height="439" alt="Screenshot 2026-04-29 at 7 17 07 AM" src="https://github.com/user-attachments/assets/073f8be2-922a-4356-b457-3c15991ff6ba" />

### Claim 2 — Latency and variance

Lower variance shows up in both token usage and latency, making CLI easier to budget and less likely to produce surprise-tail runs.
<img width="1045" height="433" alt="Screenshot 2026-04-29 at 7 16 50 AM" src="https://github.com/user-attachments/assets/25bae825-92ea-4e85-96eb-a27fbeb21709" />

### Claim 3 — Per-run token distributions

Token distributions show MCP carrying a consistently higher and wider usage band. The `mcp-final-run-44` outlier (max 45,546 tokens) is worth investigating separately.

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
| `mcp-final-run-44` | 739 – 45,546 | 6,935 |
| `cli-final-run-44` | 652 – 8,463 | 1,624 |
| `mcp-final-run-43` | 739 – 26,158 | 5,944 |
| `cli-final-run-43` | 652 – 8,468 | 1,602 |

---

## H3 — Retrieval interface and failure modes

> Tests whether CLI and MCP induce different retrieval-linked failure patterns and tool-surface overhead.

**Summary:** MCP improves retrieval completeness and uses more tool calls, but completed rows still distribute across scorer-defined failure modes. Explicit runtime errors are a separate execution-risk surface concentrated in MCP.

### Claim 1 — The interface changes the failure mix, not just the retrieval rate

MCP is stronger on retrieval completeness but CLI is stronger on snippet grounding, so better retrieval coverage does not automatically translate into better-grounded moderation outputs. It seems the more comprehensive retrievals + tool overhead may actually limit the ___. 

### Claim 2 — MCP carries higher tool-surface overhead... which was expected 

<img width="478" height="219" alt="Screenshot 2026-04-29 at 7 12 27 AM" src="https://github.com/user-attachments/assets/62014872-5c47-46e4-b1ea-b37b3a92d42f" />


**Execution errors by workflow:**

| Workflow | Rows | Explicit errors | Error rate | Frequent error tokens |
|---|---|---|---|---|
| CLI | 4,712 across 8 runs | 0 | 0% | — |
| MCP | 4,221 across 8 runs | 117 | 3% | `sse_client` 117 · `httpx` 57 · `httpcore` 57 |

### Claim 3 — Scorer-defined failure distribution

The `report_posted` gap cannot be cleanly attributed to workflow behavior due to the rate-limit incident.

# NEED PLOT 

Failure priority order: `scope_violation → hallucination → false_negative → false_positive → retrieval_failure → report_not_posted → label_mismatch`

MCP's zero hallucination and zero false positive counts reflect its 100% precision / 15% recall tradeoff : it only flags when very confident, missing most toxic content. Whereas, CLI may not pul the entire context of a conversation and resultingly tag someting as toxic that in the greater conversation does not carry that tone. 
## Caveats

**This eval is most useful as a measure of tradeoffs along the dimensions measured, not a single-score leaderboard. It is a comparison of retrieval interface tradeoffs for a downstream content-quality task, not a universal verdict on MCP vs CLI.**

### Rate-limit incident

`report_posted` was affected by a GitHub secondary rate limit rather than a clean workflow difference. The failure affected both workflows inconsistently while evaluations were running concurrently. The final two runs should be treated as outliers for direct `report_posted` comparisons.
<img width="478" height="219" alt="Screenshot 2026-04-29 at 7 13 17 AM" src="https://github.com/user-attachments/assets/5b680753-b0b9-4a9a-8f2e-23c7e30630ab" />


```
GitHub issue create failed: HTTP 403
"You have exceeded a secondary rate limit and have been temporarily
blocked from content creation."
Request ID: F3EA:8DB75:230E067:8A7B957:69F133E5
Timestamp: 2026-04-28 22:25:42 UTC
```

### Tradeoff frame

MCP is stronger on retrieval completeness and consistency. CLI is stronger on downstream toxicity labeling, de-escalation quality, token footprint, and variance. The most important finding is the mismatch between retrieval strength and downstream moderation quality — more complete retrieval does not automatically produce better-grounded responses.

---


