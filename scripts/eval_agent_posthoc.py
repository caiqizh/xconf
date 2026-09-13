#!/usr/bin/env python3
"""Agent calibration under the POST-HOC setting, CONSISTENT with the reasoning line.

The reasoning line was always post-hoc: proxy retrieves on the capability key
  F = [task_goal_emb(30), reflection_emb(30), verbalized, log reason_len, log reflect_len]
i.e. it uses the model's produced self-reflection + stated confidence. STEP 1 here applies the SAME
recipe to the agent domains so agents are measured like QA.

STEP 2 (opt-in per domain): add a few TASK-SPECIFIC trace features. Excluded: (a) strictly
tautological features (ALFWorld: fail == ran out of steps, so n_steps IS the label) and (b)
task-length proxies.

Protocol (common.py): proxy = diagonal capability-key retrieval +
conf-conditioned hit-rate (top-50, sigma=0.08; pre-hoc ablation uses sigma=None = plain hit-rate).
PCA fit PER FOLD on training folds only. proxy+metapost = 0.5*(proxy + metapost_conf); missing
metapost falls back to the TRAIN-fold base rate (never the pooled mean). 5-fold OOS by task.
Bootstrap 95% CI + AURC on the headline rows. Usage: eval_agent_posthoc.py [dom ...]
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (K_NEIGHBOURS, SIGMA_CONF, PCA_DIM, N_FOLDS, auroc, ece_ew as ece,
                           risk_coverage, bootstrap_ci, group_folds, pca_fit, pca_apply, fit_logit,
                           diag_space, proxy_scores)

BOOT_B = int(os.environ.get("XCONF_BOOT", "1000"))

DOMS = {
    "sciworld":  ("data/predictions/sciworld-25f", "sci"),
    "alfworld":  ("data/predictions/alfworld-25f", "alf"),
    "swebench":  ("data/predictions/swebench-25f", "swe"),
    "appworld":  ("data/predictions/appworld-25f", "aw"),
    # claude-sonnet-4-6 agent column (same recipes, s46 dirs)
    "appworld_s46": ("data/predictions/appworld-s46", "aw"),
    "sciworld_s46": ("data/predictions/sciworld-s46", "sci"),
    "swebench_s46": ("data/predictions/swebench-s46", "swe"),
    "alfworld_s46": ("data/predictions/alfworld-s46", "alf"),
    "appworld_q35": ("data/predictions/appworld-q35", "aw"),
    "swebench_q35": ("data/predictions/swebench-q35", "swe"),
    "alfworld_q35": ("data/predictions/alfworld-q35", "alf"),
    "sciworld_q35": ("data/predictions/sciworld-q35", "sci"),
    "appworld_g35f": ("data/predictions/appworld-g35f", "aw"),
    "swebench_g35f": ("data/predictions/swebench-g35f", "swe"),
    "alfworld_g35f": ("data/predictions/alfworld-g35f", "alf"),
    "sciworld_g35f": ("data/predictions/sciworld-g35f", "sci"),
}
CAVEAT = {
    "alfworld": "fail==ran out of steps -> n_steps tautological, dropped",
    "swebench": "Agentless single-call rollouts -> trace features are constant (STEP2 == STEP1 by "
                "construction)",
}

# STEP-2 per-domain task-specific features (drawn from {ds}.feats.jsonl).
TASK_FEATS = {
    "swebench": ["n_steps", "think_chars", "last_obs_len"],
    "sciworld": ["n_steps", "n_distinct_actions", "max_action_repeat", "distinct_ratio", "last_obs_len"],
    "alfworld": ["n_distinct_actions", "distinct_ratio", "nothing_ratio"],  # exclude n_steps/hit_max_steps
    "appworld": ["n_steps", "n_distinct_actions", "repeat_ratio", "last_obs_len"],
    "appworld_s46": ["n_steps", "n_distinct_actions", "repeat_ratio", "last_obs_len"],
    "sciworld_s46": ["n_steps", "n_distinct_actions", "max_action_repeat", "distinct_ratio", "last_obs_len"],
    "swebench_s46": ["n_steps", "think_chars", "last_obs_len"],
    "alfworld_s46": ["n_distinct_actions", "distinct_ratio", "nothing_ratio"],
    "appworld_q35": ["n_steps", "n_distinct_actions", "repeat_ratio", "last_obs_len"],
    "swebench_q35": ["n_steps", "think_chars", "last_obs_len"],
    "alfworld_q35": ["n_distinct_actions", "distinct_ratio", "nothing_ratio"],
    "sciworld_q35": ["n_steps", "n_distinct_actions", "max_action_repeat", "distinct_ratio", "last_obs_len"],
    "appworld_g35f": ["n_steps", "n_distinct_actions", "repeat_ratio", "last_obs_len"],
    "swebench_g35f": ["n_steps", "think_chars", "last_obs_len"],
    "alfworld_g35f": ["n_distinct_actions", "distinct_ratio", "nothing_ratio"],
    "sciworld_g35f": ["n_steps", "n_distinct_actions", "max_action_repeat", "distinct_ratio", "last_obs_len"],
}


def lj(md, f):
    p = os.path.join(md, f)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def load(md, ds):
    recs = {r["id"]: r for r in lj(md, f"{ds}.jsonl")}
    feats = {r["id"]: r.get("feats", r) for r in lj(md, f"{ds}.feats.jsonl")}
    emb = {r["id"]: np.array(r["embedding"], np.float32) for r in lj(md, f"{ds}.emb.jsonl")}
    remb = {r["id"]: np.array(r["embedding"], np.float32) for r in lj(md, f"{ds}.reflkey.emb.jsonl")}
    refl = {r["id"]: r for r in lj(md, f"{ds}.episodes_reflect.jsonl")}
    meta = {r["id"]: r for r in lj(md, f"{ds}.metapost.jsonl")}
    ids = [i for i in recs if i in emb and i in remb and i in refl and (not feats or i in feats)]
    ids.sort(key=lambda i: recs[i].get("order_index", 0))
    def label(i):
        if ds == "ws":
            return float(recs[i]["extra"].get("final_score", 0.0) >= 0.5)
        return float(recs[i]["correct"])
    y = np.array([label(i) for i in ids], float)
    gfield = os.environ.get("XCONF_GROUP_FIELD", "group_id")  # agent folds group by task FAMILY
    grp = np.array([recs[i]["extra"].get(gfield, recs[i]["extra"]["task_id"]) for i in ids])
    def conf(d, i):
        v = d.get(i, {}).get("confidence")
        return v if (v is not None and v == v) else float("nan")
    verb_raw = np.array([conf(refl, i) for i in ids])
    verb = np.where(np.isnan(verb_raw), 0.5, verb_raw)  # constant fallback: no label information
    metap = np.array([conf(meta, i) for i in ids])      # NaN kept; filled per fold with TRAIN mean
    rlen = np.array([np.log1p(refl[i].get("reason_len", 0)) for i in ids])
    flen = np.array([np.log1p(refl[i].get("reflect_len", 0)) for i in ids])
    Qraw = np.stack([emb[i] for i in ids]); Rraw = np.stack([remb[i] for i in ids])
    return ids, y, grp, Qraw, Rraw, verb, metap, rlen, flen, feats


def fill_metap_by_fold(metap, y, fold):
    """Missing metapost -> the TRAIN-fold base rate for that test fold (never the pooled mean)."""
    out = metap.copy()
    for kf in range(fold.max() + 1):
        te = fold == kf; tr = fold != kf
        fill = y[tr].mean() if tr.sum() else 0.5
        out[te & np.isnan(metap)] = fill
    return out


def oos_proxy(blocks, scalars, y, verb, fold, k=K_NEIGHBOURS, sigma=SIGMA_CONF):
    """Per-fold: PCA(train) each embedding block -> F -> diagonal key(train) -> retrieve test from train.
    sigma=None -> plain neighbour hit-rate (pre-hoc variant)."""
    n = len(y); cc = np.zeros(n)
    for kf in range(fold.max() + 1):
        tr = fold != kf; te = fold == kf
        if tr.sum() < k or te.sum() == 0:
            cc[te] = y[tr].mean() if tr.sum() else 0.5
            continue
        cols = []
        for B in blocks:
            bmu, bVt = pca_fit(B[tr], PCA_DIM)
            cols.append(pca_apply(B, bmu, bVt))
        F = np.concatenate(cols + ([scalars] if scalars is not None and scalars.shape[1] else []), 1)
        w, mu, sd = fit_logit(F[tr], y[tr])
        A = diag_space(F, w, mu, sd)
        cc[te] = proxy_scores(A[tr], y[tr], verb[tr], A[te], verb[te], k, sigma)
    return cc


def rep(name, y, p, ci=False):
    a = auroc(y, p)
    _, _, aurc = risk_coverage(y, p)
    cis = ""
    if ci:
        lo, hi = bootstrap_ci(auroc, y, p, B=BOOT_B)
        cis = f"  CI[{lo:.3f},{hi:.3f}]"
    print(f"  {name:44s} AUROC={a:.3f}{cis}  ECE={ece(y,p):.3f}  AURC={aurc:.3f}")
    return a


def run(dom):
    md, ds = DOMS[dom]
    ids, y, grp, Qraw, Rraw, verb, metap, rlen, flen, feats = load(md, ds)
    fold = group_folds(grp, N_FOLDS)
    metap_f = fill_metap_by_fold(metap, y, fold)
    print(f"\n===== {dom.upper()}  (n={len(ids)}, base={y.mean():.3f}, 5-fold OOS by task) =====")
    if dom in CAVEAT:
        print(f"  [caveat] {CAVEAT[dom]}")

    # pre-hoc ablation: task-goal only, plain hit-rate (sigma=None), no scalars
    pre = oos_proxy([Qraw], np.zeros((len(y), 0)), y, verb, fold, sigma=None)
    a_pre = rep("pre-hoc proxy+metapost [ablation]", y, 0.5 * (pre + metap_f), ci=True)
    rep("verbalized (post-hoc, 1 call)", y, verb, ci=True)

    # STEP 1: same recipe as reasoning line
    scalars = np.stack([verb, rlen, flen], 1)
    p1 = oos_proxy([Qraw, Rraw], scalars, y, verb, fold)
    rep("STEP1 post-hoc proxy (same recipe)", y, p1)
    a1 = rep("STEP1 post-hoc proxy+metapost", y, 0.5 * (p1 + metap_f))

    # STEP 2: + task-specific features
    tf = TASK_FEATS.get(dom, []) if feats else []
    Xf = np.array([[feats[i].get(k, 0.0) for k in tf] for i in ids], float) if tf else np.zeros((len(ids), 0))
    if Xf.shape[1]:
        Xf = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-9)
        p2 = oos_proxy([Qraw, Rraw], np.concatenate([scalars, Xf], 1), y, verb, fold)
        rep(f"STEP2 + task-feats {tf}", y, p2)
        a2 = rep("STEP2 post-hoc proxy+metapost (task feats)", y, 0.5 * (p2 + metap_f), ci=True)
    else:
        a2 = a1
    print(f"  -> pre-hoc {a_pre:.3f}  |  step1 {a1:.3f}  |  step2 {a2:.3f}")
    return dom, y.mean(), a_pre, a1, a2


def main():
    doms = sys.argv[1:] or list(DOMS)
    rows = []
    for d in doms:
        try:
            rows.append(run(d))
        except Exception as ex:
            import traceback; traceback.print_exc(); print(f"{d}: FAILED {ex}")
    print("\n\n===== POST-HOC (consistent recipe) — AUROC =====")
    print(f"{'domain':11s} {'base':>5} {'pre-hoc':>8} {'step1':>7} {'step2(+taskfeat)':>17}")
    for d, base, a_pre, a1, a2 in rows:
        print(f"{d:11s} {base:5.2f} {a_pre:8.3f} {a1:7.3f} {a2:17.3f}")


if __name__ == "__main__":
    main()
