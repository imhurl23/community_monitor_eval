# OSS Community Health First Responder  Eval Card
 
**Eval Card · EvalEval Framework**
CLI vs. MCP workflow comparison · pandas-dev/pandas
 
| | |
|---|---|
| **Author** | Izzy Hurley |
| **Date** | April 2026 |
| **Platform** | Braintrust  project: community-health-eval |
| **Tags** | Agent eval · Claude Haiku 4.5 · N=39 · 8 runs/workflow · 8,933 rows observed |
 
---
 
## Context & Scope
 
| | |
|---|---|
| **Research question** | Does a CLI-based workflow (`gh`) or MCP-based workflow (GitHub MCP Server) produce better outcomes on community health triage, and at what cost? |
| **Task** | Given a GitHub issue or PR: retrieve the comment thread → detect toxic/discouraging content → classify it → draft a maintainer de-escalation response. No writes to live repos; sandbox output only. |
| **Model** | Claude Haiku 4.5 (retrieval + analysis + judge); Claude Sonnet for toxicity label classification |
| **Deployment state** | Pre-deployment · offline evaluation |
| **Experiment cohorts** | CLI_final + MCP_final |
 
---
 
## Dataset  `community_monitor_pandas`
 
| | |
|---|---|
| **Source** | pandas-dev/pandas (public, mid-size, documented heated discussions) |
| **Size** | N = 39 items (issues + PRs) |
| **Sampling** | Two-pass: metadata sweep → ToxiShield classifier (prob thresholds) → stratified sample per stratum |
| **Ground truth** | Human annotated: binary toxicity, 8-label schema, severity (low/medium/high), problematic snippet, gold maintainer response |
 
### Strata
 
| Stratum | Sampling criterion |
|---|---|
| `clearly_toxic` | ToxiShield prob ≥ 0.7, or GitHub "too heated" lock reason |
| `borderline` | Prob ≥ 0.4 and < 0.7  the ambiguity zone |
| `heated_not_toxic` | ≥ 15 comments and max prob < 0.4 |
| `control` | ≤ 5 comments and max prob < 0.2 |

**Why N = 39, not the full 4 × 20 pool.** The sampling cap is 20 per stratum, which would yield up to 80 rows from a single repo. Three strata came in under cap on `pandas-dev/pandas`: `clearly_toxic` supplied only 5 rows (the classifier found few high-confidence examples because pandas maintainers moderate aggressively), `heated_not_toxic` supplied 14, and `borderline` supplied 0. The `control` stratum filled its full 20 — these are mostly PRs with 0–1 comments that produce identical "not toxic / nothing to do" outputs, and their repetitiveness is a known limitation of this dataset rather than a signal-bearing contribution. Final composition: `control` (20) + `heated_not_toxic` (14) + `clearly_toxic` (5) + `borderline` (0) = 39.
 
### 8-Label Toxicity Schema
 
`hostile_aggression` · `entitlement` · `dismissive_tone` · `sarcasm_belittling` · `passive_aggression` · `gatekeeping` · `thread_derailment` · `object_directed`
 
---
 
## Scorer Suite
 
| Scorer | Type | Range |
|---|---|---|
| `scope_containment` | Deterministic · safety | Binary 0/1 · hard failure if 0 |
| `toxicity_label_accuracy` | Deterministic · accuracy | 0–1 · partial credit for parent-category match |
| `retrieval_completeness` | Deterministic · retrieval | Binary 0/1 · allows one-comment miss |
| `false_positive_flag` | Deterministic · detection | Binary 0/1 · control + heated strata only |
| `report_posted` | Deterministic · delivery | Binary 0/1 · contaminated by rate-limit incident |
| `deescalation_quality` | LLM-judge (Haiku) · generation | A→1.0 / B→0.5 / C→0.0 |
| `snippet_grounding` | LLM-judge (Haiku) · hallucination guard | 0–1 · checks quote exists in retrieved text |
| `token / latency / tool efficiency` | Telemetry · cost | 0–1 normalized · inverse-scaled per row |
 
---
 
## Key Findings  CLI vs. MCP
 
| Dimension | CLI | MCP | Winner |
|---|---|---|---|
| Retrieval completeness | 0.829 ±0.052 | **0.868 ±0.020** | MCP |
| Toxicity label accuracy | **0.191 ±0.009** | 0.146 ±0.008 | CLI |
| De-escalation quality | **0.833 ±0.000** | 0.563 ±0.165 | CLI |
| Scope containment | 1.000 | 1.000 | Tie |
| Mean tokens | **3,442 ±12** | 18,293 ±131 | CLI (5.3× fewer) |
| Mean latency | **5,097ms ±307ms** | 14,421ms ±1,246ms | CLI (2.8× faster) |
| Cost per item | **$0.0045** | $0.0203 | CLI (4.5× cheaper) |
| Execution errors | **0 / 0%** | 117 / 3% | CLI (all SSE transport) |
 
---
 
## Failure Mode Distribution (scorer-tagged)
 
| Failure type | CLI count / % | MCP count / % |
|---|---|---|
| `scope_violation` | 0 / 0% | 0 / 0% |
| `hallucination` | 8 / 7% | 0 / 0% |
| `false_negative` ★ largest mode | 39 / 35% | 47 / 47% |
| `false_positive` | 7 / 6% | 0 / 0% |
| `retrieval_failure` | 31 / 28% | 18 / 18% |
| `report_not_posted` | 25 / 23% | 35 / 35% |
| `label_mismatch` | 0 / 0% | 0 / 0% |
 
> MCP execution errors (SSE transport failures) are counted separately and not included in scorer-tagged totals above.
 
---
 
## Hypothesis Verdicts
 
### H1  MCP produces better end-to-end moderation quality · `Partial`
 
MCP is stronger on retrieval completeness and run-to-run consistency. CLI is stronger on label accuracy, de-escalation quality, and snippet grounding. The key finding: better retrieval does not automatically produce better moderation outputs. CLI's constrained context appears to force tighter snippet focus, which improves grounded response drafting even when the full thread isn't retrieved.
 
### H2  CLI reaches comparable quality at lower cost · `Supported`
 
CLI is 4.5× cheaper per item, 2.8× faster, and substantially lower variance on both tokens and latency. Weekly cost at current pandas intake: CLI $0.41 vs. MCP $1.85.
 
### H3  Interface drives different failure mode distributions · `Confirmed`
 
CLI distributes failures across retrieval (28%) and report-not-posted (23%) beyond false negatives. MCP shows near-zero false positives and hallucinations  but 47% false negatives and 3% transport errors. MCP's 100% precision / ~15% recall tradeoff is the more dangerous safety profile: it creates a false sense of coverage while missing most toxic content.
 
---
 
## Known Limitations
 
**LLM-judge circularity.** Haiku is both the analysis model and the LLM-judge scorer. This introduces self-evaluation bias. A different model not used elsewhere in the pipeline should serve as judge in the next iteration.
 
**Snippet grounding edge case.** The scorer checks whether the quoted snippet appears in *retrieved* text, not the ground-truth thread. Truncation at 8,000 chars means a fabricated snippet that matches the truncated window will pass  even if the actual toxic content appeared later in the thread.
 
**Rate-limit incident.** GitHub secondary rate limiting affected both workflows during the final two runs, contaminating `report_posted` scores. Those runs should be treated as outliers for delivery-metric comparisons.
 
**`clearly_toxic` stratum conflation.** Lock-promoted rows may have no single flaggable comment  the thread was locked for cumulative tone or off-platform context. An agent that correctly returns `is_toxic: false` on such a row is penalized as a false negative, even if the label reflects a locking decision rather than schema-visible toxic content.
 
**Arc-level toxicity misses.** Both workflows retrieved relevant comments and still failed to flag threads where toxicity is embedded in the conversational arc (entitlement, passive-aggression, thread-derailment). These require prompting that explicitly asks for arc-level analysis  retrieval improvements alone will not fix them.
 
**Inline PR review comments.** Structurally invisible to CLI without an extra tool call (counts against `MAX_TOOL_CALLS = 6`). An agent can pass `retrieval_completeness` while missing toxic inline review comments entirely, producing a false negative with a passing retrieval score.
 
---
 
## Next Iteration Priorities
 
**Independent judge model.** Swap Haiku out as the LLM-judge scorer and use a model not present anywhere else in the pipeline to eliminate circularity bias.

**Alter Strata.** Strata lead to undersampling on borderline, and heated strata. The numerical ToxiShield values should be tuned based on labeler experience. 
 
**Snippet-first prompting on MCP.** Retrieve comprehensively (MCP), then force the model to identify and quote the specific passage before drafting. Tests whether CLI's grounding advantage is prompt-recoverable.
 
**Arc-level scorer.** Add a scorer that evaluates whether the draft response addresses the conversational arc, not just a single flagged passage  targeting the shared false-negative failure mode.
 
**Multi-repo expansion.** Extend beyond pandas-dev/pandas to reduce repo-specific sampling bias and test whether failure mode distributions generalize.
 
**Inline comment coverage.** Add an explicit inline review comment retrieval step to both workflows and score it separately from top-level comment retrieval.
 
**MCP transport stability.** Investigate SSE connection drops under sustained concurrent load  likely addressable with connection pooling or retry logic before the next run.
 
---
 
*OSS Community Health First Responder · Eval Card v1.0 · EvalEval Coalition framework · evalevalai.com/projects/eval-cards*
