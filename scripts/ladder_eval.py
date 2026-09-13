#!/usr/bin/env python3
"""Retrieval ladder R0 -> L3.

Holds the folds, the bank, k and the metric fixed and changes ONLY how neighbours
are chosen:

  R0  uniform random k from the training fold          -- is retrieval needed at all?
  R1  random k from the SAME `subject` stratum         -- does cheap metadata suffice?
  L0  raw question-embedding cosine top-k              -- is topical similarity enough?
  L1  correctness-supervised diagonal key, top-k       -- the core increment
  L2  L1 + confidence-conditioned hit rate  (= Recall) -- the calibration backbone
  L3  0.5 * (L2 + Reflect)                  (= headline)

R0 is the lower anchor: with neighbours drawn at random the estimate keeps only the
global base rate. R1 is stratified by SUBJECT, not by stated confidence.

Pure CPU, no API calls. Seeds are derived from a stable crc32 of the dataset name,
matching the shuffle ablation, so reruns reproduce exactly.

Usage:
    python3 scripts/ladder_eval.py [--cols COL ...] [--datasets DS ...]
    -> table on stdout + data/baselines/ladder.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from zlib import crc32

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (auroc, ece_ew as ece, bootstrap_ci, fit_logit, pca_fit, pca_apply,
                    N_FOLDS as NF, K_NEIGHBOURS as K, SIGMA_CONF as SIGMA)

PRED = "data/predictions"
COLS = ["gemini-2.5-flash", "gemini-3.5-flash",
        "Qwen_Qwen3.5-397B-A17B-FP8", "claude-sonnet-4-6"]
RUNGS = ["R0", "R1", "L0", "L1", "L2", "L3"]


def lj(md, f):
    p = os.path.join(md, f)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def lemb(md, f):
    return {r["id"]: np.asarray(r["embedding"], np.float32) for r in lj(md, f)}


def conf_of(rec):
    c = rec.get("confidence")
    return float(c) if c is not None and c == c else 0.5


def load(md, ds):
    """Frozen loader: same ids, same order, same fields as the headline path."""
    recs = {r["id"]: r for r in lj(md, f"{ds}.jsonl")}
    refl = {r["id"]: r for r in lj(md, f"{ds}.episodes_reflect.jsonl")}
    qe = lemb(md, f"{ds}.emb.jsonl")
    re_ = lemb(md, f"{ds}.reflkey.emb.jsonl")
    rsn = lemb(md, f"{ds}.reasonkey.emb.jsonl")
    meta = {r["id"]: r.get("conf") for r in lj(md, f"{ds}.incontext_metapost.jsonl")}

    ids = [i for i in refl if i in qe and i in re_ and i in recs and (not rsn or i in rsn)]
    if len(ids) < 200:
        return None
    ids.sort(key=lambda i: recs[i].get("order_index", 0))

    y = np.array([float(refl[i]["correct"]) for i in ids])
    if y.min() == y.max():
        return None
    verb = np.array([conf_of(refl[i]) for i in ids])
    rl = np.array([np.log1p(refl[i].get("reason_len", 0)) for i in ids])
    fl = np.array([np.log1p(refl[i].get("reflect_len", 0)) for i in ids])

    blocks = [np.stack([qe[i] for i in ids])]
    if rsn:
        blocks.append(np.stack([rsn[i] for i in ids]))
    blocks.append(np.stack([re_[i] for i in ids]))

    # Stratum for R1. Missing subject collapses to one bucket, which makes R1
    # degenerate to R0 -- reported rather than silently hidden.
    subj = np.array([str(recs[i].get("subject", "_none")) for i in ids])

    # Reflect scores may be absent for a column/dataset; L3 is then skipped.
    reflect = np.array([float(meta[i]) if meta.get(i) is not None else np.nan for i in ids])

    return dict(ids=ids, y=y, verb=verb, rl=rl, fl=fl, Q=blocks[0],
                blocks=blocks, subj=subj, reflect=reflect)


def hit_rate(y_tr, idx, v_tr=None, v_j=None, sigma=None):
    """Neighbour hit rate; Gaussian confidence weighting when sigma is given."""
    if sigma is None:
        return float(y_tr[idx].mean())
    w = np.exp(-((v_tr[idx] - v_j) ** 2) / (2 * sigma ** 2))
    return float((y_tr[idx] * w).sum() / (w.sum() + 1e-9))


def topk(S_row, k):
    return np.argsort(-S_row)[:k]


def run(d, ds):
    y, verb, subj = d["y"], d["verb"], d["subj"]
    n = len(y)
    fold = np.arange(n) % NF
    out = {r: np.full(n, np.nan) for r in RUNGS}
    fallback = 0  # R1 draws that had too small a stratum and fell back to uniform

    Qn = d["Q"] / (np.linalg.norm(d["Q"], axis=1, keepdims=True) + 1e-9)

    for f in range(NF):
        tr, te = fold != f, fold == f
        y_tr, v_tr = y[tr], verb[tr]
        tr_idx = np.where(tr)[0]
        rng = np.random.RandomState((crc32(f"{ds}|{f}".encode()) & 0xFFFFFFFF))

        # ---- correctness-supervised diagonal key, refit per fold (causal)
        Z = np.concatenate(
            [pca_apply(B, *pca_fit(B[tr])) for B in d["blocks"]]
            + [verb[:, None], d["rl"][:, None], d["fl"][:, None]], axis=1)
        w, mu, sd = fit_logit(Z[tr], y_tr)
        Wt = np.abs(w) / (np.abs(w).max() + 1e-9)
        A = ((Z - mu) / sd) * Wt
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        S_key = A[te] @ A[tr].T
        S_raw = Qn[te] @ Qn[tr].T

        subj_tr = subj[tr]
        for row, j in enumerate(np.where(te)[0]):
            # R0 uniform random
            pick = rng.choice(len(tr_idx), min(K, len(tr_idx)), replace=False)
            out["R0"][j] = hit_rate(y_tr, pick)

            # R1 random within the same subject stratum
            pool = np.where(subj_tr == subj[j])[0]
            if len(pool) < K:
                fallback += 1
                pool = np.arange(len(tr_idx))
            pick = rng.choice(len(pool), min(K, len(pool)), replace=False)
            out["R1"][j] = hit_rate(y_tr, pool[pick])

            out["L0"][j] = hit_rate(y_tr, topk(S_raw[row], K))
            nb = topk(S_key[row], K)
            out["L1"][j] = hit_rate(y_tr, nb)
            out["L2"][j] = hit_rate(y_tr, nb, v_tr, verb[j], SIGMA)

    ref = d["reflect"]
    if np.isfinite(ref).mean() > 0.5:
        out["L3"] = 0.5 * (out["L2"] + np.where(np.isfinite(ref), ref, out["L2"]))
    else:
        out["L3"] = np.full(n, np.nan)
    return out, fallback / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", nargs="*", default=COLS)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--out", default="data/baselines/ladder.json")
    args = ap.parse_args()

    res = {}
    hdr = f"{'col/dataset':38s} {'n':>6} " + " ".join(f"{r:>13}" for r in RUNGS) + "  R1-fb"
    print(hdr)
    print("-" * len(hdr))
    for col in args.cols:
        md = os.path.join(PRED, col)
        if not os.path.isdir(md):
            continue
        dss = args.datasets or sorted({f.split(".")[0] for f in os.listdir(md)
                                       if f.endswith(".episodes_reflect.jsonl")})
        for ds in dss:
            d = load(md, ds)
            if d is None:
                continue
            scores, fb = run(d, ds)
            y = d["y"]
            cell, line = {}, f"{col[:20]}/{ds}"[:38]
            row = f"{line:38s} {len(y):>6} "
            for r in RUNGS:
                s = scores[r]
                if not np.isfinite(s).all():
                    cell[r] = None
                    row += f"{'--':>13} "
                    continue
                cell[r] = {"auroc": auroc(y, s), "ece": ece(y, s)}
                row += f"{cell[r]['auroc']:.3f}/{cell[r]['ece']:.3f} "
            lo, hi = bootstrap_ci(auroc, y, scores["R0"])
            cell["R0_auroc_ci"] = [lo, hi]
            print(row + f" {fb:.2f}")
            res[f"{col}/{ds}"] = {"n": int(len(y)), "r1_fallback_rate": fb, **cell}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}  ({len(res)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
