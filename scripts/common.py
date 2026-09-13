#!/usr/bin/env python3
"""common.py — the SINGLE canonical implementation of the method's math & metrics.

Every eval/generation script must import from here instead of re-implementing. This module is
the executable form of the frozen method spec; every constant in the paper lives here.

Canonical protocol:
  folds        : K=5, interleaved by stream order (np.arange(n) % 5) for reasoning; random-by-task
                 (group_folds) for agent domains.
  PCA          : 30 dims per embedding block, fit on the TRAINING fold only, applied to test
                 (never fit on train+test pooled).
  key          : diagonal correctness-supervised reweight |w|/max|w| from a GD logistic
                 (l2=1e-2, 1500 iters, lr=0.15) fit on the training fold only.
  retrieval    : cosine over the reweighted, L2-normalised space; top-k=50.
  proxy        : confidence-conditioned hit-rate over the top-50, Gaussian kernel sigma=0.08
                 (sigma=None/inf -> plain neighbour hit-rate; used for the pre-hoc goal-only proxy).
  blend        : proxy+metapost = 0.5 * (proxy + metapost). Missing metapost fallback: reasoning
                 line falls back to VERBALIZED conf; agent line
                 falls back to the TRAIN-fold outcome mean (eval_agent_posthoc.fill_metap_by_fold).
  ECE          : primary = 10-bin equal-width; secondary robustness = 15-bin equal-mass.
  CI           : nonparametric bootstrap over rows, B=1000, seed 0, 95%.
  truncation   : no truncation of model-visible/embedded text. The only allowed cut is
                 fit_context(), which fires ONLY on genuine context-budget overflow, keeps head+tail,
                 marks the elision, and logs.
"""
from __future__ import annotations

import sys

import numpy as np

K_NEIGHBOURS = 50
SIGMA_CONF = 0.08
PCA_DIM = 30
N_FOLDS = 5
LOGIT_L2, LOGIT_IT, LOGIT_LR = 1e-2, 1500, 0.15
ECE_BINS_EW, ECE_BINS_EM = 10, 15
BOOT_B, BOOT_SEED = 1000, 0
# Gemini 2.5 context = 1M tokens ~ 4M chars; leave ample headroom for system/instructions/output.
PROMPT_BUDGET_CHARS = 3_000_000


# ---------------------------------------------------------------- metrics
def auroc(y, s):
    """Tie-aware Mann-Whitney AUROC of score s predicting binary y."""
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    if len(y) == 0 or y.min() == y.max():
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(s)
    p = y.sum()
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * (len(y) - p)))


def ece_ew(y, p, bins=ECE_BINS_EW):
    """PRIMARY ECE: equal-width bins over [0,1]."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = ~np.isnan(p)
    y, p = y[m], p[m]
    if len(y) == 0:
        return float("nan")
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = (p >= lo) & (p < hi) if b < bins - 1 else (p >= lo) & (p <= hi)
        if sel.any():
            e += abs(p[sel].mean() - y[sel].mean()) * sel.mean()
    return float(e)


def ece_em(y, p, bins=ECE_BINS_EM):
    """SECONDARY ECE (robustness): equal-mass (adaptive) bins."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = ~np.isnan(p)
    y, p = y[m], p[m]
    if len(y) == 0:
        return float("nan")
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    ids = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    e = 0.0
    for b in np.unique(ids):
        sel = ids == b
        e += abs(p[sel].mean() - y[sel].mean()) * sel.mean()
    return float(e)


def brier(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = ~np.isnan(p)
    return float(np.mean((p[m] - y[m]) ** 2)) if m.any() else float("nan")


def risk_coverage(y, conf):
    """Selective prediction: sort by conf desc, sweep coverage.
    Returns (coverages, risks, aurc). Lower AURC is better."""
    y = np.asarray(y, float)
    conf = np.asarray(conf, float)
    m = ~np.isnan(conf)
    y, conf = y[m], conf[m]
    if len(y) == 0:
        return np.array([]), np.array([]), float("nan")
    order = np.argsort(-conf)
    err = 1.0 - y[order]
    n = len(y)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)
    return cov, risk, float(risk.mean())


def acc_at_coverage(y, conf, coverage):
    """Accuracy among the most-confident `coverage` fraction."""
    cov, risk, _ = risk_coverage(y, conf)
    if len(cov) == 0:
        return float("nan")
    idx = np.searchsorted(cov, coverage, side="left")
    idx = min(idx, len(cov) - 1)
    return float(1.0 - risk[idx])


def bootstrap_ci(metric_fn, y, s, B=BOOT_B, seed=BOOT_SEED, alpha=0.05):
    """Nonparametric row bootstrap 95% CI for metric_fn(y, s). Returns (lo, hi)."""
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    rng = np.random.RandomState(seed)
    n = len(y)
    vals = []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        v = metric_fn(y[idx], s[idx])
        if v == v:
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


# ---------------------------------------------------------------- folds
def fold_ids(n, K=N_FOLDS):
    """Reasoning line: interleaved modulo folds over stream order."""
    return np.arange(n) % K


def group_folds(groups, K=N_FOLDS, seed=0):
    """Agent line: random K-fold over unique group ids (task_id)."""
    uniq = sorted(set(groups))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uniq))
    gf = {g: int(perm[i] % K) for i, g in enumerate(uniq)}
    return np.array([gf[g] for g in groups])


# ---------------------------------------------------------------- PCA (per-fold, train-only fit)
def pca_fit(X, m=PCA_DIM):
    X = np.asarray(X, np.float64)
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:m]


def pca_apply(X, mu, Vt):
    return (np.asarray(X, np.float64) - mu) @ Vt.T


# ---------------------------------------------------------------- capability key + retrieval
def fit_logit(X, y, l2=LOGIT_L2, it=LOGIT_IT, lr=LOGIT_LR):
    """GD logistic F->correct on the TRAINING fold. Returns (w, mu, sd)."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    mu = X.mean(0)
    sd = X.std(0) + 1e-9
    Xs = (X - mu) / sd
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(it):
        p = 1 / (1 + np.exp(-(Xs @ w + b)))
        g = p - y
        w -= lr * (Xs.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return w, mu, sd


def diag_space(F, w, mu, sd):
    """Reweight by |w|/max|w| and L2-normalise rows -> cosine-ready retrieval space."""
    Wt = np.abs(w) / (np.abs(w).max() + 1e-9)
    Fw = ((np.asarray(F, np.float64) - mu) / sd) * Wt
    return Fw / (np.linalg.norm(Fw, axis=1, keepdims=True) + 1e-9)


def proxy_scores(A_tr, y_tr, v_tr, A_te, v_te, k=K_NEIGHBOURS, sigma=SIGMA_CONF):
    """Conf-conditioned hit-rate proxy. sigma=None -> plain top-k hit-rate (pre-hoc variant)."""
    y_tr = np.asarray(y_tr, float)
    v_tr = np.asarray(v_tr, float)
    out = np.empty(len(A_te))
    for j in range(len(A_te)):
        s = A_tr @ A_te[j]
        top = np.argsort(-s)[:k]
        if sigma is None or not np.isfinite(sigma):
            out[j] = y_tr[top].mean()
        else:
            wgt = np.exp(-((v_tr[top] - v_te[j]) ** 2) / (2 * sigma**2))
            out[j] = float((y_tr[top] * wgt).sum() / (wgt.sum() + 1e-9))
    return out


# ---------------------------------------------------------------- context guard
def fit_context(text, budget_chars=PROMPT_BUDGET_CHARS, label=""):
    """Return text unchanged unless it would genuinely overflow the context budget.
    On overflow: keep head+tail, insert an explicit elision marker, and LOG the event."""
    if len(text) <= budget_chars:
        return text
    keep = budget_chars // 2 - 50
    elided = len(text) - 2 * keep
    print(f"[fit_context] {label}: prompt {len(text)} chars > budget {budget_chars}; "
          f"elided {elided} chars from the middle (head+tail kept)", file=sys.stderr, flush=True)
    return text[:keep] + f"\n[... {elided} characters elided to fit the context window ...]\n" + text[-keep:]
