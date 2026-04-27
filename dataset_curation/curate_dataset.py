"""
Community Health Eval — Dataset Curation Pipeline (unified)

Combines:
  • (issue/PR/review/review-comment fetch → ToxiShield scoring →
     thread aggregation → 4-stratum stratification → Braintrust upload)
  • Multi-repo support (configurable list of OWNER/REPO targets,
    per-repo sampling then merge).
  • A `too heated` lock-reason recall booster: GitHub maintainers' own
    `active_lock_reason == "too heated"` flag is invisible to the text
    classifier (it's metadata, not text). We surface those threads,
    fetch their comments, and route them into `clearly_toxic_candidate`
    so the same stratum cap covers classifier-flagged AND lock-flagged
    threads. The signal is preserved on `metadata.active_lock_reason`.
  • Per-thread `_review.suspect_comments` snippets — top-K highest-
    probability comments emitted next to each row to speed annotation.
  • Two-pass fetch: cheap metadata first, score+collect comments only
    for threads that survive a candidate filter (high comment count OR
    locked-too-heated). Cuts ToxiShield wall time ~5–10× on multi-repo
    runs.

Run end-to-end:
    GITHUB_TOKEN=ghp_xxx BRAINTRUST_API_KEY=sk_xxx \\
      python curate_dataset.py --upload

See README.md for full docs.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

# ─── Logging ─────────────────────────────────────────────────────────────────
log = logging.getLogger("curate")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


# ─── Config ──────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # Repos to pull. Each gets the full pipeline applied independently, then
    # samples are merged. Pandas-only? leave default. Multi-repo? add more.
    repos: list[str] = field(default_factory=lambda: [
        # "pandas-dev/pandas",
        "home-assistant/core",
        # "nodejs/node",
        # "rust-lang/rust",
        "kubernetes/kubernetes",
        "eslint/eslint",
    ])
    since: str = "2023-01-01T00:00:00Z"
    max_items_per_repo: int = 100

    # ToxiShield model — same as user's notebook
    model_name: str = "toxishield/toxic-classifier-38k"
    base_tokenizer: str = "bert-base-uncased"
    batch_size: int = 32
    max_length: int = 256

    # Stratification thresholds
    ctrl_max_comments: int = 5
    ctrl_max_prob: float = 0.2
    toxic_ge_07_threshold: int = 1
    borderline_ge_04_threshold: int = 1
    borderline_max_prob: float = 0.7
    heated_min_comments: int = 15
    heated_max_prob: float = 0.4

    # Two-pass candidate filter — only threads passing this cheap test get
    # their comments fetched + scored. Tuned to keep all candidates that
    # COULD land in a non-control stratum:
    #   • clearly_toxic_candidate / borderline_candidate / heated_not_toxic_candidate
    #     all require comment_count ≥ 2 in practice (one comment can't be
    #     both ≥0.4 and trigger anything else interesting). We use a soft
    #     floor here — set lower to widen the funnel.
    #   • control_candidate is sampled from the leftover low-volume tail
    #     without scoring (we only need to know the comment_count is low).
    candidate_min_comment_count: int = 3

    # Lock-reason discovery — recall booster for clearly_toxic_candidate
    lock_search_per_page: int = 100        # GitHub max
    lock_search_max_pages: int = 3         # 300 candidates per repo, plenty
    enable_lock_signal: bool = True

    # Sampling
    samples_per_stratum: int = 20
    random_state: int = 42

    # Suspect-snippet emission
    suspect_top_k: int = 3
    suspect_snippet_chars: int = 600

    # I/O
    cache_dir: Path = Path("cache")
    output_path: Path = Path("community-health-v3.json")

    # Braintrust
    braintrust_project: str = "community-health-eval"
    braintrust_dataset: str = "community-health-v3"
    braintrust_description: str = (
        "Multi-repo OSS discussions for community-health first-responder eval. "
        "Stratified across control / borderline / heated-not-toxic / clearly-toxic. "
        "Lock-reason `too heated` is used as a recall booster for the toxic stratum."
    )


# ─── GitHub client ───────────────────────────────────────────────────────────
class GitHub:
    """Tiny GitHub REST client. Retries 429/secondary-rate-limit AND 5xx."""

    TRANSIENT_5XX = {500, 502, 503, 504}

    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        })

    def _get(self, url, params=None, retries: int = 4):
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                r = self.session.get(url, params=params, timeout=60)
            except requests.RequestException as e:
                last_exc = e
                wait = 2 ** attempt * 2
                log.warning("Network error (%s); retrying in %ds", e, wait)
                time.sleep(wait)
                continue
            # Secondary / primary rate limits.
            if r.status_code in (403, 429) and "rate limit" in r.text.lower():
                wait = 2 ** attempt * 5
                log.warning("Rate limited on %s, sleeping %ds", url, wait)
                time.sleep(wait)
                continue
            # Transient server errors.
            if r.status_code in self.TRANSIENT_5XX:
                wait = 2 ** attempt * 3
                log.warning("GitHub %d on %s; retrying in %ds",
                            r.status_code, url, wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        if last_exc:
            raise last_exc
        r.raise_for_status()
        return r

    def paginate(self, url, params=None, sleep: float = 0.1) -> list:
        items: list = []
        while url:
            r = self._get(url, params=params)
            data = r.json()
            if isinstance(data, list):
                items.extend(data)
            else:
                raise ValueError(f"Expected list response for {url}")
            url = r.links.get("next", {}).get("url")
            params = None
            time.sleep(sleep)
        return items

    # — list-issues style endpoints
    def list_issues_and_prs(self, repo, state="all", since=None,
                            per_page=100, max_items=None):
        params = {"state": state, "per_page": per_page}
        if since:
            params["since"] = since
        items = self.paginate(f"https://api.github.com/repos/{repo}/issues",
                              params=params)
        if max_items:
            items = items[:max_items]
        return items

    def list_issue_comments(self, repo, number):
        return self.paginate(
            f"https://api.github.com/repos/{repo}/issues/{number}/comments",
            params={"per_page": 100},
        )

    def list_pr_reviews(self, repo, number):
        return self.paginate(
            f"https://api.github.com/repos/{repo}/pulls/{number}/reviews",
            params={"per_page": 100},
        )

    def list_pr_review_comments(self, repo, number):
        return self.paginate(
            f"https://api.github.com/repos/{repo}/pulls/{number}/comments",
            params={"per_page": 100},
        )

    # — locked thread discovery
    def search_locked_issues(self, repo, per_page=100, max_pages=3) -> list:
        """Locked issues+PRs in a repo, sorted by comment count desc.
        Paginates manually; the search API caps at 1000 results.
        Search payload does NOT include `active_lock_reason`, so we
        follow up with a per-issue GET in `discover_locked_too_heated`."""
        items: list = []
        for page in range(1, max_pages + 1):
            url = (
                "https://api.github.com/search/issues"
                f"?q=repo:{repo}+is:locked"
                f"&per_page={per_page}&page={page}&sort=comments&order=desc"
            )
            r = self._get(url)
            payload = r.json()
            chunk = payload.get("items") or []
            items.extend(chunk)
            if len(chunk) < per_page:
                break
            time.sleep(0.1)
        return items

    def get_issue(self, repo, number) -> dict:
        r = self._get(f"https://api.github.com/repos/{repo}/issues/{number}")
        return r.json()


# ─── Normalization ───────────────────────────────────────────────────────────
def normalize_issue_comment(c, repo, discussion_type, number):
    return {
        "repo": repo,
        "discussion_type": discussion_type,
        "discussion_number": number,
        "comment_source": "issue_comment",
        "comment_id": c["id"],
        "comment_created_at": c["created_at"],
        "author_login": c["user"]["login"] if c.get("user") else None,
        "author_type": c["user"]["type"] if c.get("user") else None,
        "author_association": c.get("author_association"),
        "body": c.get("body") or "",
    }


def normalize_pr_review(rev, repo, number):
    return {
        "repo": repo,
        "discussion_type": "pull_request",
        "discussion_number": number,
        "comment_source": "review",
        "comment_id": rev["id"],
        "comment_created_at": rev.get("submitted_at") or rev.get("created_at"),
        "author_login": rev["user"]["login"] if rev.get("user") else None,
        "author_type": rev["user"]["type"] if rev.get("user") else None,
        "author_association": rev.get("author_association"),
        "body": rev.get("body") or "",
    }


def normalize_pr_review_comment(c, repo, number):
    return {
        "repo": repo,
        "discussion_type": "pull_request",
        "discussion_number": number,
        "comment_source": "review_comment",
        "comment_id": c["id"],
        "comment_created_at": c["created_at"],
        "author_login": c["user"]["login"] if c.get("user") else None,
        "author_type": c["user"]["type"] if c.get("user") else None,
        "author_association": c.get("author_association"),
        "body": c.get("body") or "",
    }


# ─── Comment collection ──────────────────────────────────────────────────────
def collect_thread_comments(gh: GitHub, repo: str, dtype: str,
                            number: int) -> list[dict]:
    """Fetch all comment kinds for a single thread."""
    rows: list[dict] = []
    for c in gh.list_issue_comments(repo, number):
        rows.append(normalize_issue_comment(c, repo, dtype, number))
    if dtype == "pull_request":
        for rev in gh.list_pr_reviews(repo, number):
            if (rev.get("body") or "").strip():
                rows.append(normalize_pr_review(rev, repo, number))
        for rc in gh.list_pr_review_comments(repo, number):
            rows.append(normalize_pr_review_comment(rc, repo, number))
    return rows


def collect_comments_for_threads(gh: GitHub, repo: str,
                                 thread_keys: list[tuple[str, int]]
                                 ) -> pd.DataFrame:
    """Fetch + normalize comments for the given threads only (two-pass)."""
    rows: list[dict] = []
    for i, (dtype, number) in enumerate(thread_keys):
        try:
            rows.extend(collect_thread_comments(gh, repo, dtype, number))
        except requests.HTTPError as e:
            log.warning("Skipping %s#%s — %s", repo, number, e)
            continue
        if (i + 1) % 50 == 0:
            log.info("  [%s] %d/%d threads pulled",
                     repo, i + 1, len(thread_keys))
    if not rows:
        # Preserve schema even when empty so downstream code doesn't blow up.
        return pd.DataFrame(columns=[
            "repo", "discussion_type", "discussion_number", "comment_source",
            "comment_id", "comment_created_at", "author_login", "author_type",
            "author_association", "body",
        ])
    return pd.DataFrame(rows)


# ─── ToxiShield scoring ──────────────────────────────────────────────────────
def load_model(cfg: Config):
    """Imported lazily so `--help` and dry runs don't load torch."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)
    model.to(device)
    model.eval()
    log.info("Loaded %s on %s", cfg.model_name, device)
    return tokenizer, model, device


def score_texts(texts: Iterable[str], tokenizer, model, device,
                batch_size: int = 32, max_length: int = 256) -> list[float]:
    import torch
    texts = list(texts)
    probs: list[float] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            if logits.ndim == 2 and logits.shape[1] == 1:
                p = torch.sigmoid(logits[:, 0])
            elif logits.ndim == 2 and logits.shape[1] == 2:
                p = torch.softmax(logits, dim=1)[:, 1]
            else:
                raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
        probs.extend(p.detach().cpu().tolist())
    return probs


# ─── Thread-level aggregation + stratification ───────────────────────────────
def aggregate_repo(df_comments: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Groupby+agg implementation. Faster and avoids the pandas 3.0
    `include_groups` deprecation that the original .apply() form trips."""
    if df_comments.empty:
        return pd.DataFrame(columns=[
            "repo", "discussion_type", "discussion_number",
            "comment_count", "unique_commenter_count",
            "max_toxicity_prob", "mean_toxicity_prob",
            "comments_ge_0_4", "comments_ge_0_7",
            "first_toxic_comment_index", "is_newcomer_involved", "stratum",
        ])

    df = df_comments.copy()
    df["_ge_04"] = df["toxicity_prob"] >= 0.4
    df["_ge_07"] = df["toxicity_prob"] >= 0.7
    df["_is_newcomer"] = df["author_association"] == "FIRST_TIME_CONTRIBUTOR"

    grouped = df.groupby(["repo", "discussion_type", "discussion_number"],
                         sort=False)
    df_threads = grouped.agg(
        comment_count=("comment_id", "count"),
        unique_commenter_count=("author_login", "nunique"),
        max_toxicity_prob=("toxicity_prob", "max"),
        mean_toxicity_prob=("toxicity_prob", "mean"),
        comments_ge_0_4=("_ge_04", "sum"),
        comments_ge_0_7=("_ge_07", "sum"),
        is_newcomer_involved=("_is_newcomer", "any"),
    ).reset_index()
    df_threads["comments_ge_0_4"] = df_threads["comments_ge_0_4"].astype(int)
    df_threads["comments_ge_0_7"] = df_threads["comments_ge_0_7"].astype(int)

    # first_toxic_comment_index needs ordering — compute separately.
    df_threads["first_toxic_comment_index"] = _first_toxic_index(df)
    df_threads["stratum"] = _stratum_vectorized(df_threads, cfg)
    return df_threads


def _first_toxic_index(df_comments: pd.DataFrame) -> pd.Series:
    """1-based chronological index of the first ≥0.7 comment per thread."""
    ordered = df_comments.sort_values("comment_created_at")
    ordered = ordered.assign(
        _idx=ordered.groupby(
            ["repo", "discussion_type", "discussion_number"]
        ).cumcount() + 1
    )
    flagged = ordered[ordered["toxicity_prob"] >= 0.7]
    first = (
        flagged.groupby(["repo", "discussion_type", "discussion_number"])["_idx"]
        .min()
    )
    # Reindex to match thread order; missing → NaN.
    keys = (
        df_comments[["repo", "discussion_type", "discussion_number"]]
        .drop_duplicates()
        .set_index(["repo", "discussion_type", "discussion_number"])
        .index
    )
    return first.reindex(keys).reset_index(drop=True)


def _stratum_vectorized(df_threads: pd.DataFrame, cfg: Config) -> pd.Series:
    """Vectorized version of the original suggest_stratum() rules. Order
    matters and matches the notebook: control → toxic → borderline →
    heated → other."""
    mp = df_threads["max_toxicity_prob"].fillna(0)
    cc = df_threads["comment_count"]
    ge4 = df_threads["comments_ge_0_4"]
    ge7 = df_threads["comments_ge_0_7"]

    out = pd.Series("other", index=df_threads.index, dtype=object)
    is_ctrl = (cc <= cfg.ctrl_max_comments) & (mp < cfg.ctrl_max_prob)
    is_toxic = ge7 >= cfg.toxic_ge_07_threshold
    is_border = (ge4 >= cfg.borderline_ge_04_threshold) & (mp < cfg.borderline_max_prob)
    is_heated = (cc >= cfg.heated_min_comments) & (mp < cfg.heated_max_prob)

    out[is_ctrl] = "control_candidate"
    out[~is_ctrl & is_toxic] = "clearly_toxic_candidate"
    out[~is_ctrl & ~is_toxic & is_border] = "borderline_candidate"
    out[~is_ctrl & ~is_toxic & ~is_border & is_heated] = "heated_not_toxic_candidate"
    return out


# ─── Lock-reason discovery ───────────────────────────────────────────────────
def discover_locked_too_heated(gh: GitHub, repo: str,
                               cfg: Config) -> list[dict]:
    """Returns issues+PRs in `repo` locked with reason 'too heated'.

    Why: GitHub's own moderation signal is invisible to a text classifier
    (it's metadata, not text). On a 7-thread sample in pass 2, ≈71% of
    `too heated` threads were judged toxic on manual review — strong
    enough to use as a recall booster for `clearly_toxic_candidate`,
    but NOT a ground-truth label. The annotator still decides
    `is_toxic` and `severity`.
    """
    candidates = gh.search_locked_issues(
        repo,
        per_page=cfg.lock_search_per_page,
        max_pages=cfg.lock_search_max_pages,
    )
    confirmed: list[dict] = []
    for c in candidates:
        number = c["number"]
        try:
            full = gh.get_issue(repo, number)
        except requests.HTTPError as e:
            log.warning("Skipping %s#%s — %s", repo, number, e)
            continue
        if (full.get("active_lock_reason") or "").lower() == "too heated":
            confirmed.append({
                "number": number,
                "discussion_type": (
                    "pull_request" if full.get("pull_request") else "issue"
                ),
                "html_url": full.get("html_url"),
                "active_lock_reason": full["active_lock_reason"],
                "title": full.get("title"),
                "created_at": full.get("created_at"),
            })
    log.info("[%s] %d locked-too-heated threads found", repo, len(confirmed))
    return confirmed


def filter_locked_by_since(locked: list[dict], since: str) -> list[dict]:
    """Drop locked threads created before `since`."""
    if not since:
        return locked
    cutoff = since
    return [l for l in locked if (l.get("created_at") or "") >= cutoff]


def merge_locked_signal(df_threads: pd.DataFrame,
                        locked: list[dict]) -> pd.DataFrame:
    """Mark `active_lock_reason` and route too-heated threads into
    `clearly_toxic_candidate` so they share the toxic stratum cap.

    Threads that already have classifier-flagged toxicity keep their
    stratum (it'd already be `clearly_toxic_candidate`). Threads where
    the classifier missed toxicity — the entire reason we look at the
    lock signal — get promoted from `borderline`/`heated`/`other` into
    the toxic stratum. The original classifier-derived stratum is
    preserved on `metadata.classifier_stratum` for diagnostics.
    """
    df_threads = df_threads.copy()
    df_threads["active_lock_reason"] = None
    df_threads["classifier_stratum"] = df_threads["stratum"]

    locked_keys = {(l["discussion_type"], l["number"]) for l in locked}
    if not locked_keys:
        return df_threads

    mask = df_threads.apply(
        lambda r: (r["discussion_type"], r["discussion_number"]) in locked_keys,
        axis=1,
    )
    df_threads.loc[mask, "active_lock_reason"] = "too heated"
    df_threads.loc[mask, "stratum"] = "clearly_toxic_candidate"
    n_promoted = int(mask.sum() & (df_threads.loc[mask, "classifier_stratum"]
                                   != "clearly_toxic_candidate").sum())
    log.info("Lock signal: %d threads tagged, of which %d promoted into "
             "clearly_toxic_candidate", int(mask.sum()), n_promoted)
    return df_threads


# ─── Sampling ────────────────────────────────────────────────────────────────
TARGET_STRATA = [
    "clearly_toxic_candidate",
    "borderline_candidate",
    "heated_not_toxic_candidate",
    "control_candidate",
]


def stratified_sample_per_repo(df_threads: pd.DataFrame,
                               cfg: Config) -> pd.DataFrame:
    """Sample within each (repo, stratum) cell, then concat. Per-repo
    sampling prevents a single chatty repo from dominating any stratum."""
    parts: list[pd.DataFrame] = []
    for repo, df_repo in df_threads.groupby("repo"):
        for stratum in TARGET_STRATA:
            grp = df_repo[df_repo["stratum"] == stratum]
            if grp.empty:
                continue
            n = min(cfg.samples_per_stratum, len(grp))
            if n < cfg.samples_per_stratum:
                log.info("  [%s/%s] only %d of %d available",
                         repo, stratum, n, cfg.samples_per_stratum)
            parts.append(grp.sample(n=n, random_state=cfg.random_state))
    if not parts:
        return df_threads.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


# ─── Suspect-snippet builder ─────────────────────────────────────────────────
def build_suspect_snippet_index(df_comments: pd.DataFrame,
                                top_k: int, max_chars: int) -> dict:
    """Pre-build a {(repo, dtype, number) -> [snippet, ...]} lookup so we
    don't re-filter the full comments dataframe per sampled row."""
    if df_comments.empty:
        return {}

    df = df_comments.sort_values("toxicity_prob", ascending=False)
    out: dict[tuple, list[dict]] = {}
    grouped = df.groupby(["repo", "discussion_type", "discussion_number"],
                         sort=False)
    for key, group in grouped:
        snippets = []
        for _, c in group.head(top_k).iterrows():
            body = (c.get("body") or "").strip().replace("\r\n", "\n")
            if len(body) > max_chars:
                body = body[:max_chars] + "…"
            snippets.append({
                "comment_id": int(c["comment_id"]),
                "author_login": c.get("author_login"),
                "author_association": c.get("author_association"),
                "comment_created_at": c.get("comment_created_at"),
                "toxicity_prob": float(c["toxicity_prob"]),
                "body_preview": body,
            })
        out[key] = snippets
    return out


# ─── Row assembly ────────────────────────────────────────────────────────────
def assemble_dataset_rows(sampled: pd.DataFrame,
                          df_comments_all: pd.DataFrame,
                          cfg: Config) -> list[dict]:
    snippet_index = build_suspect_snippet_index(
        df_comments_all, cfg.suspect_top_k, cfg.suspect_snippet_chars
    )
    out: list[dict] = []
    for _, row in sampled.iterrows():
        repo = row["repo"]
        dtype = row["discussion_type"]
        number = int(row["discussion_number"])
        dtype_short = "pr" if dtype == "pull_request" else "issue"
        url_segment = "pull" if dtype == "pull_request" else "issues"
        repo_slug = repo.split("/")[-1]
        row_id = f"{repo_slug}-{dtype_short}-{number}"

        suspects = snippet_index.get((repo, dtype, number), [])

        out.append({
            "id": row_id,
            "repo": repo,
            "discussion_type": dtype,
            "discussion_number": number,
            "url": f"https://github.com/{repo}/{url_segment}/{number}",
            "ground_truth": {
                "is_toxic": None,           # annotator fills in
                "toxicity_labels": [],      # see schema in eval design doc
                "severity": None,           # low / medium / high
                "problematic_snippet": "",
                "gold_response": "",
            },
            "metadata": {
                "comment_count": int(row["comment_count"]),
                "unique_commenter_count": int(row["unique_commenter_count"]),
                "first_toxic_comment_index": (
                    int(row["first_toxic_comment_index"])
                    if pd.notna(row["first_toxic_comment_index"]) else None
                ),
                "is_newcomer_involved": bool(row["is_newcomer_involved"]),
                "max_toxicity_prob": (
                    float(row["max_toxicity_prob"])
                    if pd.notna(row["max_toxicity_prob"]) else None
                ),
                "mean_toxicity_prob": (
                    float(row["mean_toxicity_prob"])
                    if pd.notna(row["mean_toxicity_prob"]) else None
                ),
                "stratum": row["stratum"],
                "classifier_stratum": (
                    row["classifier_stratum"]
                    if "classifier_stratum" in row.index
                    and pd.notna(row.get("classifier_stratum"))
                    else row["stratum"]
                ),
                "active_lock_reason": (
                    row["active_lock_reason"]
                    if "active_lock_reason" in row.index
                    and pd.notna(row.get("active_lock_reason"))
                    else None
                ),
            },
            "_review": {
                # Emitted to make manual annotation faster — top-K highest-
                # probability comments per thread. NOT for model consumption.
                "suspect_comments": suspects,
            },
        })
    return out


# ─── Per-repo cache helpers ──────────────────────────────────────────────────
def _safe(name: str) -> str:
    return name.replace("/", "_")


def cache_path(cfg: Config, repo: str, kind: str) -> Path:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    return cfg.cache_dir / f"{_safe(repo)}__{kind}__{cfg.since[:10]}.pkl"


# ─── Two-pass pipeline ───────────────────────────────────────────────────────
def select_candidate_threads(items: list[dict],
                             locked: list[dict],
                             cfg: Config
                             ) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Pass 1 → Pass 2 boundary.

    Splits the listing into:
      • candidates: threads whose comments WILL be fetched + scored. Any
        thread with comment_count ≥ candidate_min_comment_count, plus
        every locked-too-heated thread regardless of its position in the
        listing (so we never silently lose the lock signal).
      • controls: low-volume tail used to populate `control_candidate`
        without scoring. We only need comment_count to decide.

    Returns (candidate_keys, control_keys) where each key is
    (discussion_type, number).
    """
    locked_keys = {(l["discussion_type"], l["number"]) for l in locked}

    candidate_keys: list[tuple[str, int]] = []
    control_keys: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()

    for item in items:
        number = item["number"]
        is_pr = "pull_request" in item
        dtype = "pull_request" if is_pr else "issue"
        key = (dtype, number)
        if key in seen:
            continue
        seen.add(key)
        cc = int(item.get("comments") or 0)
        if (cc >= cfg.candidate_min_comment_count) or (key in locked_keys):
            candidate_keys.append(key)
        else:
            control_keys.append(key)

    # Locked threads outside the listing window — fetch them anyway so the
    # signal survives.
    for k in locked_keys:
        if k not in seen:
            candidate_keys.append(k)
            seen.add(k)

    return candidate_keys, control_keys


def synthesize_control_threads(items: list[dict],
                               control_keys: set[tuple[str, int]],
                               repo: str) -> pd.DataFrame:
    """Build a thread-level frame for low-volume threads we deliberately
    didn't score. They go into `control_candidate` if comment_count is
    low enough, else dropped (they can't land in any other stratum
    without scoring)."""
    rows = []
    for item in items:
        is_pr = "pull_request" in item
        dtype = "pull_request" if is_pr else "issue"
        number = item["number"]
        if (dtype, number) not in control_keys:
            continue
        rows.append({
            "repo": repo,
            "discussion_type": dtype,
            "discussion_number": number,
            "comment_count": int(item.get("comments") or 0),
            "unique_commenter_count": 0,  # unknown without fetching
            "max_toxicity_prob": 0.0,
            "mean_toxicity_prob": 0.0,
            "comments_ge_0_4": 0,
            "comments_ge_0_7": 0,
            "first_toxic_comment_index": None,
            "is_newcomer_involved": False,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # All eligible by construction (low comment count, low max_prob=0).
    df["stratum"] = "control_candidate"
    return df


def run_repo(gh: GitHub, repo: str, cfg: Config,
             tokenizer, model, device,
             use_cache: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (df_threads, df_comments) for one repo using the two-pass
    fetch."""
    import pickle
    items_cache = cache_path(cfg, repo, "items")
    comments_cache = cache_path(cfg, repo, "comments")
    locked_cache = cache_path(cfg, repo, "locked")

    # ── Pass 1: cheap metadata ────────────────────────────────────────────
    if use_cache and items_cache.exists():
        log.info("[%s] loading cached items", repo)
        items = pickle.loads(items_cache.read_bytes())
    else:
        log.info("[%s] fetching issues+PRs since %s", repo, cfg.since)
        items = gh.list_issues_and_prs(
            repo, since=cfg.since, max_items=cfg.max_items_per_repo
        )
        items_cache.write_bytes(pickle.dumps(items))
    log.info("[%s] %d items", repo, len(items))

    # Lock-reason discovery (also cheap, search + per-issue GETs).
    if cfg.enable_lock_signal:
        if use_cache and locked_cache.exists():
            locked = pickle.loads(locked_cache.read_bytes())
        else:
            locked = discover_locked_too_heated(gh, repo, cfg)
            locked_cache.write_bytes(pickle.dumps(locked))
        locked = filter_locked_by_since(locked, cfg.since)
    else:
        locked = []

    # ── Pass 2: score only candidates ─────────────────────────────────────
    candidate_keys, control_keys = select_candidate_threads(items, locked, cfg)
    log.info("[%s] candidates=%d controls=%d",
             repo, len(candidate_keys), len(control_keys))

    if use_cache and comments_cache.exists():
        log.info("[%s] loading cached comments", repo)
        df_comments = pickle.loads(comments_cache.read_bytes())
    else:
        df_comments = collect_comments_for_threads(gh, repo, candidate_keys)
        comments_cache.write_bytes(pickle.dumps(df_comments))
    log.info("[%s] %d candidate comments", repo, len(df_comments))

    # Score
    if df_comments.empty:
        df_comments["toxicity_prob"] = []
    elif "toxicity_prob" not in df_comments.columns:
        log.info("[%s] scoring %d comments", repo, len(df_comments))
        df_comments["toxicity_prob"] = score_texts(
            df_comments["body"].fillna("").tolist(),
            tokenizer, model, device,
            batch_size=cfg.batch_size, max_length=cfg.max_length,
        )

    # Aggregate scored threads + tack on the un-scored controls.
    df_scored = aggregate_repo(df_comments, cfg)
    df_controls = synthesize_control_threads(items, set(control_keys), repo)
    if not df_controls.empty:
        df_threads = pd.concat([df_scored, df_controls], ignore_index=True)
    else:
        df_threads = df_scored

    # Lock-reason overlay (now reroutes into clearly_toxic_candidate).
    df_threads = merge_locked_signal(df_threads, locked)

    log.info("[%s] strata: %s", repo,
             df_threads["stratum"].value_counts().to_dict())
    return df_threads, df_comments


def upload_to_braintrust(rows: list[dict], cfg: Config) -> None:
    import braintrust
    ds = braintrust.init_dataset(
        project=cfg.braintrust_project,
        name=cfg.braintrust_dataset,
        description=cfg.braintrust_description,
    )
    for row in rows:
        model_input = {
            key: value
            for key, value in row.items()
            if key not in {"ground_truth", "expected", "_review"}
        }
        expected = row.get("expected") or row.get("ground_truth")
        ds.insert(input=model_input, expected=expected, id=row["id"])
    log.info("Uploaded %d rows to Braintrust %s/%s",
             len(rows), cfg.braintrust_project, cfg.braintrust_dataset)


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repos", nargs="*",
                   help="Override repo list, e.g. pandas-dev/pandas nodejs/node")
    p.add_argument("--since", help="ISO date filter, e.g. 2023-01-01T00:00:00Z")
    p.add_argument("--max-items", type=int,
                   help="Max issues+PRs per repo (default 2000)")
    p.add_argument("--samples-per-stratum", type=int,
                   help="Samples per (repo, stratum) cell (default 20)")
    p.add_argument("--candidate-min-comments", type=int,
                   help="Two-pass threshold: only score threads with at least "
                        "this many comments (default 3)")
    p.add_argument("--no-lock-signal", action="store_true",
                   help="Disable the locked-too-heated recall booster")
    p.add_argument("--output", help="Output JSON path (default community_monitor_pandas.json)")
    p.add_argument("--no-cache", action="store_true",
                   help="Ignore cached fetches and re-pull from GitHub")
    p.add_argument("--upload", action="store_true",
                   help="Upload to Braintrust after assembling rows")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip ToxiShield scoring and Braintrust upload (for plumbing tests)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = Config()
    if args.repos:
        cfg.repos = args.repos
    if args.since:
        cfg.since = args.since
    if args.max_items:
        cfg.max_items_per_repo = args.max_items
    if args.samples_per_stratum:
        cfg.samples_per_stratum = args.samples_per_stratum
    if args.candidate_min_comments is not None:
        cfg.candidate_min_comment_count = args.candidate_min_comments
    if args.no_lock_signal:
        cfg.enable_lock_signal = False
    if args.output:
        cfg.output_path = Path(args.output)

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        log.error("GITHUB_TOKEN env var is required")
        return 2
    gh = GitHub(token)

    if args.dry_run:
        tokenizer = model = device = None
    else:
        tokenizer, model, device = load_model(cfg)

    all_threads: list[pd.DataFrame] = []
    all_comments: list[pd.DataFrame] = []
    for repo in cfg.repos:
        try:
            df_threads, df_comments = run_repo(
                gh, repo, cfg, tokenizer, model, device,
                use_cache=not args.no_cache,
            )
        except Exception as e:
            log.exception("Repo %s failed: %s", repo, e)
            continue
        all_threads.append(df_threads)
        all_comments.append(df_comments)

    if not all_threads:
        log.error("No repos produced data")
        return 1

    df_threads_all = pd.concat(all_threads, ignore_index=True)
    df_comments_all = pd.concat(all_comments, ignore_index=True)

    sampled = stratified_sample_per_repo(df_threads_all, cfg)
    log.info("Sampled %d threads across %d repos",
             len(sampled), sampled["repo"].nunique())
    log.info("Per-stratum totals: %s",
             sampled["stratum"].value_counts().to_dict())

    rows = assemble_dataset_rows(sampled, df_comments_all, cfg)
    cfg.output_path.write_text(json.dumps(rows, indent=2))
    log.info("Wrote %d rows to %s", len(rows), cfg.output_path)

    if args.upload:
        if not os.environ.get("BRAINTRUST_API_KEY"):
            log.error("BRAINTRUST_API_KEY not set; cannot upload")
            return 2
        upload_to_braintrust(rows, cfg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
