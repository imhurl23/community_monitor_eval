# Community Health Eval — Dataset Curation

Unified pipeline that builds the labeling pool for the **Community Health
First-Responder** eval. It pulls discussions from one or more OSS repos,
scores every comment with the ToxiShield toxicity classifier, stratifies
threads, samples a balanced labeling pool, and (optionally) uploads to
Braintrust.
---

## Pipeline at a glance

```
              ┌────────────────────────────────────────────────┐
              │ for each repo:                                 │
              │                                                │
   GitHub ───▶│  PASS 1  (cheap metadata)                      │
              │  ──────                                        │
              │  1. list issues + PRs since SINCE              │
              │  2. discover locked-too-heated threads         │
              │  3. split: candidates (will score) vs          │
              │     controls (low-volume tail, skip scoring)   │
              │                                                │
              │  PASS 2  (expensive — only for candidates)     │
              │  ──────                                        │
              │  4. fetch issue_comments + reviews +           │
              │     review_comments for candidates             │
              │  5. score with ToxiShield                      │
              │  6. aggregate to thread level                  │
              │  7. assign stratum                             │
              │  8. lock-signal overlay → promote too-heated   │
              │     threads into `clearly_toxic_candidate`     │
              │     (preserve original verdict on              │
              │      `classifier_stratum`)                     │
              │  9. merge with un-scored controls              │
              └─────────────────┬──────────────────────────────┘
                                │
                ┌───────────────▼────────────────┐
                │ 10. per-(repo, stratum) sample │
                │ 11. attach top-K suspect       │
                │     snippets per thread        │
                │ 12. emit JSON + Braintrust     │
                └────────────────────────────────┘
```

---

## Stages

### 1. Listing

`GitHub.list_issues_and_prs(repo, since=…, max_items=…)` paginates
`/repos/{repo}/issues` and returns issues+PRs in a single list (GitHub's
`issues` endpoint is unified). Configured via `Config.since` and
`Config.max_items_per_repo`.

### 2. Lock-reason discovery (recall booster)

We call `/search/issues?q=repo:{repo}+is:locked&sort=comments` (paginated
up to `lock_search_max_pages × lock_search_per_page`, default 300
candidates per repo), then GET each one to read `active_lock_reason`.
Threads with value `"too heated"` are kept. Locked threads created
before `cfg.since` are filtered out so they don't pollute the eval
window.

The lock signal is **a recall booster, not a stratum**. Threads that
GitHub maintainers locked as too heated are routed into
`clearly_toxic_candidate` in step 8 below. The classifier's original
stratum verdict is preserved on `metadata.classifier_stratum` for
diagnostics.

### 3. Two-pass candidate selection

For every listed item we have a `comments` count from the issues API for
free (no extra request). We split:

| bucket | rule | downstream treatment |
|---|---|---|
| **candidates** | `comment_count ≥ candidate_min_comment_count` (default 3) **OR** locked-too-heated | full comment fetch + scoring |
| **controls** | everything else | go straight into `control_candidate` without scoring |

Locked threads outside the listing window are appended to candidates
unconditionally — we never want to silently drop the lock signal.

This is the change that buys most of the speedup. ~70–90% of items in a
typical repo have ≤2 comments and could only ever land in
`control_candidate` anyway; scoring them is wasted work.

### 4. Comment fetch (candidates only)

For each candidate we collect three comment kinds, mirroring the
notebook:

- `issue_comment` — top-level discussion comments (issues and PRs).
- `review` — PR review bodies (skipped when empty).
- `review_comment` — inline diff comments on PRs.

All three are normalized to a flat row with `repo`, `discussion_type`,
`discussion_number`, `comment_source`, `comment_id`, timestamps, author
fields, and `body`.

### 5. ToxiShield scoring

Every candidate's comments run through
[`toxishield/toxic-classifier-38k`](https://huggingface.co/toxishield/toxic-classifier-38k)
with the `bert-base-uncased` tokenizer (kept identical to the notebook
to preserve label compatibility with the existing dataset). Batch size
32, max length 256. The probability of the toxic class is stored on
`df_comments["toxicity_prob"]`.

The model is loaded lazily so `--help` and `--dry-run` don't need GPUs
or the transformers stack.

### 6. Thread aggregation

`aggregate_repo` collapses each `(repo, discussion_type, discussion_number)`
into:

| field | meaning |
|---|---|
| `comment_count` | total comments collected |
| `unique_commenter_count` | distinct authors |
| `max_toxicity_prob` | highest comment-level prob |
| `mean_toxicity_prob` | mean comment-level prob |
| `comments_ge_0_4` | count of comments with prob ≥ 0.4 |
| `comments_ge_0_7` | count of comments with prob ≥ 0.7 |
| `first_toxic_comment_index` | 1-based index of the first ≥0.7 comment in chronological order |
| `is_newcomer_involved` | did any `FIRST_TIME_CONTRIBUTOR` comment? |

Implementation note: the notebook's `groupby().apply()` form trips a
`pandas 3.0` deprecation about `include_groups`. We use vectorized
`groupby().agg()` here for both forward-compat and a measurable speedup
on large frames.

### 7. Stratification

`_stratum_vectorized` assigns one of (rules unchanged from the
notebook):

| stratum | rule |
|---|---|
| `control_candidate` | `comment_count ≤ 5` AND `max_prob < 0.2` |
| `clearly_toxic_candidate` | `comments_ge_0_7 ≥ 1` |
| `borderline_candidate` | `comments_ge_0_4 ≥ 1` AND `max_prob < 0.7` |
| `heated_not_toxic_candidate` | `comment_count ≥ 15` AND `max_prob < 0.4` |
| `other` | none of the above |

Thresholds are fields on `Config` so a tuning sweep can override them
without code edits.

### 8. Lock-signal overlay

For every thread in the locked-too-heated set:
- `metadata.active_lock_reason` ← `"too heated"`
- `metadata.classifier_stratum` ← whatever the classifier said
- `metadata.stratum` ← `"clearly_toxic_candidate"` (overwriting the
  classifier's verdict if it differed)

Threads the classifier already flagged as toxic stay where they are
(no-op). Threads the classifier missed get promoted, which is the
whole reason we look at the lock signal. Sampling in step 10 then sees
both kinds in the same `clearly_toxic_candidate` bucket under one cap,
which is what you wanted.

### 9. Per-repo stratified sampling

For each `(repo, stratum)` cell we draw up to `samples_per_stratum`
threads with `random_state=42`. Per-repo sampling prevents a single
chatty repo (looking at you, `kubernetes/kubernetes`) from dominating
any stratum.

### 10. Suspect-snippet annotation aid

Per sampled thread, the top `suspect_top_k` (default 3) highest-
probability comments are attached under `_review.suspect_comments`:

```json
"_review": {
  "suspect_comments": [
    {
      "comment_id": 12345,
      "author_login": "...",
      "author_association": "MEMBER",
      "comment_created_at": "2024-…",
      "toxicity_prob": 0.94,
      "body_preview": "first 600 chars…"
    }
  ]
}
```

The `_review.*` namespace is a convention: anything under it is for the
human annotator and is **not** part of the eval input. Local dataset rows
keep it for labeling, but Braintrust upload strips the key before insert.

### 11. Output

- `community_monitor_pandas.json` — list of dataset rows (see schema below).
- Optional Braintrust upload (`--upload`) inserts each row with
  `input=model_input`, `expected=expected`, and `id=row.id`, where
  `model_input` is the row with `ground_truth`, `expected`, and `_review`
  removed. `expected` is taken from `row.expected` when present, otherwise
  it falls back to `row.ground_truth`. The `id` makes the upload
  idempotent — re-running won't create duplicates, it'll update existing
  rows in place.

---

## Output schema

Each local row matches the labeling format used by the existing
`braintrust-labeled-N.jsonl` workflow so new pulls can be merged with
prior labeled data. Before Braintrust upload, the script strips
`ground_truth` and `_review` from `input` and sends the label object as
`expected`.

```json
{
  "id": "pandas-issue-12345",
  "repo": "pandas-dev/pandas",
  "discussion_type": "issue",
  "discussion_number": 12345,
  "url": "https://github.com/pandas-dev/pandas/issues/12345",
  "ground_truth": {
    "is_toxic": null,
    "toxicity_labels": [],
    "severity": null,
    "problematic_snippet": "",
    "gold_response": ""
  },
  "metadata": {
    "comment_count": 12,
    "unique_commenter_count": 5,
    "first_toxic_comment_index": 4,
    "is_newcomer_involved": false,
    "max_toxicity_prob": 0.83,
    "mean_toxicity_prob": 0.21,
    "stratum": "clearly_toxic_candidate",
    "classifier_stratum": "borderline_candidate",
    "active_lock_reason": "too heated"
  },
  "_review": {
    "suspect_comments": [ … ]
  }
}
```

`stratum` is what the row was sampled under (post-lock-overlay).
`classifier_stratum` is what ToxiShield alone would have produced.
When they differ, the row was promoted by the lock signal; when they
match, the classifier and the lock signal agree (or there is no lock
signal). This pair is what you'd plot to study lock-vs-classifier
disagreement after annotation.

The 8-label `toxicity_labels` schema (filled in by the annotator) is:
`hostile_aggression`, `entitlement`, `dismissive_tone`,
`sarcasm_belittling`, `passive_aggression`, `gatekeeping`,
`thread_derailment`, `object_directed`.

---

## Configuration

Defaults live in the `Config` dataclass at the top of
`curate_dataset.py`. Override per-run with CLI flags:

| flag | what it does |
|---|---|
| `--repos OWNER/REPO …` | repo list (space-separated) |
| `--since 2023-01-01T00:00:00Z` | issue cutoff |
| `--max-items 2000` | issues+PRs per repo |
| `--samples-per-stratum 20` | cap per (repo, stratum) cell |
| `--candidate-min-comments 3` | two-pass scoring threshold |
| `--no-lock-signal` | disable the lock-reason recall booster |
| `--output path.json` | output file |
| `--no-cache` | re-pull from GitHub even if `cache/` has data |
| `--upload` | push to Braintrust after writing the JSON |
| `--dry-run` | skip scoring + upload (plumbing test) |

Recommended repo set for a high-severity-rich pull (validated in pass 2):

```
pandas-dev/pandas
home-assistant/core
nodejs/node
rust-lang/rust
kubernetes/kubernetes
eslint/eslint
```

---

## Running it

```bash
export GITHUB_TOKEN=ghp_...           # required
export BRAINTRUST_API_KEY=sk_...      # required only with --upload

# Pandas-only, same as the original notebook
python curate_dataset.py

# Multi-repo, write JSON only
python curate_dataset.py \
  --repos pandas-dev/pandas home-assistant/core nodejs/node rust-lang/rust \
          kubernetes/kubernetes eslint/eslint \
  --output community-health-v2.json

# Same, plus push to Braintrust
python curate_dataset.py --repos … --upload
```

Dependencies: `pandas`, `requests`, `transformers`, `torch`, `braintrust`.
A GPU is recommended once you go past one repo — comment counts run a
few thousand per repo and ToxiShield is ~110M params.

---

## Caching

Each repo's raw `items`, locked-thread list, and normalized
`df_comments` are cached under `cache/<repo>__<kind>__<since>.pkl`.
Re-runs reuse the cache unless `--no-cache` is set or `cfg.since`
changes. ToxiShield scores are stored on `df_comments` and survive the
comments cache, so adding a new repo doesn't force a re-score of
existing repos.

---

## Why this pipeline 
Want to be repo agnostic and prepared to do multirepo to work with different repo design preferences. With the text text classifier as the sole signal we were getting to many false positives and emitted the dataset without any annotation aids... leading to slow and cumbersome human review. Aftertwo enrichment passes major lessons learned: 

1. **Pandas alone undersupplies high-severity toxicity.** Pandas
   maintainers moderate aggressively, so blatant hostility is rare. To
   surface high-severity examples I had to broaden the repo set. 
2. **GitHub's `active_lock_reason == "too heated"` is a strong prior the
   text classifier cannot see.** It is metadata, not text. In pass 2,
   5/7 hand-reviewed too-heated threads were judged toxic — useful as a
   stratum prior, **not** as a ground-truth label.
3. **Annotators (...me) are slow without snippets.** Thread-level rows leave the
   annotator to find the offending comment themselves. Emitting the
   top-K highest-probability comments per thread cuts review time
   substantially.

---

The script also adds a **two-pass fetch** so we don't waste GPU time
scoring threads that can never land in the dataset (e.g. one-comment
PRs that aren't candidates for any stratum). Empirically this cuts
ToxiShield wall time ~5–10× on multi-repo runs.

## Resilience

`GitHub._get` retries with exponential backoff on:
- 429 / 403 with `rate limit` in the body (primary + secondary limits)
- 500 / 502 / 503 / 504 (transient server errors)
- network errors (DNS, connection reset, timeouts)

Up to 4 attempts per request. Beyond that the run continues for other
repos via the `try/except` wrapping each `run_repo` call — one repo's
failure won't kill a multi-repo run.

---

## Known limitations

- **Lock-reason recall is bounded.** GitHub's search returns at most
  the top 1000 results. We page up to 300 by default; bump
  `lock_search_max_pages` if you need more on a huge repo.
- **Search results lack `active_lock_reason`.** We deliberately follow
  up with a per-issue GET because the search payload omits it.
- **Maintainer comments dominate volume.** ToxiShield scores aren't
  conditioned on author role; consider weighting by
  `author_association` if false positives from staff become a problem.
- **Cross-repo label drift.** The 8-label schema was tuned on pandas
  threads; some labels (e.g. `gatekeeping`) trigger more often in
  systems-language repos like rust. Spot-check the first batch of
  labels per new repo before scaling.
- **Two-pass floor is heuristic.** With
  `candidate_min_comment_count=3`, threads with 1–2 comments can never
  land in `clearly_toxic_candidate` even if that one comment was
  hostile. If high-recall on solo-comment hostility matters, lower the
  floor (slower) or pre-filter listings by some other signal.
