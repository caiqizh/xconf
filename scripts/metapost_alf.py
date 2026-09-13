#!/usr/bin/env python3
"""PRE-HOC metapost for agent domains: recalibrate expected success from the retrieved track record.

Shown ONLY the current task goal (pre-hoc: no trace of the current attempt exists yet) + k similar PAST
tasks (task-OOS retrieval), the model names its recurring pattern and estimates the probability it will
complete THIS task. 1 call/task. Output: <md>/<ds>.metapost.jsonl {id, confidence, self_reflection}

Neighbour cards (all fields are HISTORICAL artifacts — no leakage):
  goal + COMPLETED/FAILED + the FULL post-rollout self-reflection of that episode + the confidence it
  stated then + (if available) the FULL outcome-aware hindsight (<ds>.hindsight.jsonl, hindsight_alf.py)
  + compact trace stats (steps, invalid-action ratio). No truncation — the only cut is the
  fit_context() overflow guard.

Causal discipline: neighbour retrieval is 5-fold task-OOS; the PCA basis AND the diagonal key are fit
per fold on the training folds only. System prompt is parameterised by domain.
"""
from __future__ import annotations
import os, re, sys, json, zlib, argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xconf.config import Config
from xconf.llm import build_client
from xconf.prompts import _CONF_RE
from common import PCA_DIM, N_FOLDS, group_folds, pca_fit, pca_apply, fit_logit, diag_space, fit_context

MD = os.environ.get("XCONF_MD", "data/predictions/alfworld-25f")
DS = os.environ.get("XCONF_DS", "alf")
_REFLECT = re.compile(r"self[- ]?reflection\s*:\s*(.*?)(?:\n\s*confidence\s*:|\Z)", re.IGNORECASE | re.DOTALL)

DOMAIN_DESC = {
    "alf": "an ALFWorld household task (navigate a room, pick/place/clean/heat/cool objects via text commands)",
    "sci": "a ScienceWorld science-experiment task (conduct a multi-step experiment via text commands)",
    "swe": "a SWE-bench software bug-fix task (edit the repository so the failing unit tests pass)",
}

def system_prompt():
    dom = DOMAIN_DESC.get(DS, "an interactive agent task")
    return (
        f"You are about to attempt {dom}. Estimate the probability you will fully COMPLETE it. Study "
        "your TRACK RECORD on similar past tasks (each marked COMPLETED or FAILED, with the reflection "
        "you wrote after attempting it). Find your recurring pattern -- which KINDS of tasks you "
        "reliably finish vs. get stuck on, and WHY you failed when you failed -- and calibrate. Be "
        "honest and specific; if tasks like this one tripped you up before, your confidence must drop. "
        "Never be reflexively confident."
    )


def _load_jsonl_map(path, key="id"):
    d = {}
    if os.path.exists(path):
        for l in open(path):
            if l.strip():
                o = json.loads(l); d[o[key]] = o
    return d


def load():
    recs = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(MD, f"{DS}.jsonl")))}
    emb = {}
    for l in open(os.path.join(MD, f"{DS}.emb.jsonl")):
        o = json.loads(l); emb[o["id"]] = np.array(o["embedding"], np.float32)
    epi = _load_jsonl_map(os.path.join(MD, f"{DS}.episodes_reflect.jsonl"))
    hind = _load_jsonl_map(os.path.join(MD, f"{DS}.hindsight.jsonl"))
    feats = _load_jsonl_map(os.path.join(MD, f"{DS}.feats.jsonl"))
    ids = [i for i in recs if i in emb]
    ids.sort(key=lambda i: recs[i].get("order_index", 0))
    return recs, emb, epi, hind, feats, ids


def neighbours(recs, emb, ids, k):
    """5-fold task-OOS retrieval; PCA + diagonal key fit on the TRAIN folds only."""
    y = np.array([recs[i]["correct"] for i in ids], float)
    grp = np.array([recs[i]["extra"].get("group_id", recs[i]["extra"]["task_id"]) for i in ids])
    E = np.stack([emb[i] for i in ids])
    fold = group_folds(grp, N_FOLDS)
    nbr = [None] * len(ids)
    for kf in range(N_FOLDS):
        tr = np.where(fold != kf)[0]; te = np.where(fold == kf)[0]
        mu, Vt = pca_fit(E[tr], PCA_DIM)
        Q = pca_apply(E, mu, Vt)
        w, kmu, ksd = fit_logit(Q[tr], y[tr])
        A = diag_space(Q, w, kmu, ksd)
        S = A[te] @ A[tr].T
        for r, j in enumerate(te):
            top = np.argpartition(-S[r], k - 1)[:k]; top = top[np.argsort(-S[r][top])]
            nbr[j] = tr[top].tolist()
    return nbr


def card(recs, epi, hind, feats, ids, n, fake_outcome=None):
    i = ids[n]
    corr = fake_outcome if fake_outcome is not None else recs[i]["correct"]
    outcome = "COMPLETED" if corr else "FAILED"
    seg = [f"  - Task: \"{recs[i]['question']}\"  -> {outcome}"]
    e = epi.get(i)
    if e:
        if isinstance(e.get("confidence"), (int, float)) and e["confidence"] == e["confidence"]:
            seg.append(f"    Confidence you stated after attempting it: {int(round(e['confidence'] * 100))}%")
        if e.get("self_reflection"):
            seg.append(f"    Your reflection after attempting it: \"{e['self_reflection']}\"")
    h = hind.get(i)
    if h and h.get("hindsight"):
        seg.append(f"    In hindsight (knowing the outcome): \"{h['hindsight']}\"")
    f = (feats.get(i) or {}).get("feats") or {}
    if f:
        stats = [f"steps={int(f.get('n_steps', 0))}"]
        if "nothing_ratio" in f:
            stats.append(f"invalid-action ratio={f['nothing_ratio']:.2f}")
        seg.append(f"    Trace stats: {', '.join(stats)}")
    return "\n".join(seg)


def build_prompt(recs, epi, hind, feats, ids, j, nbr_idx, k, variant="metapost", proxy=None):
    cur = recs[ids[j]]["question"]
    if variant == "scalar":  # NO cards — only the scalar track-record hit-rate
        return fit_context(
            f"CURRENT task you are about to attempt:\n  \"{cur}\"\n\n"
            f"Your measured TRACK RECORD on the {k} most similar past tasks was "
            f"{int(round(proxy*100))}% COMPLETED.\n\n"
            "Trust that measured rate over your gut. Output a single probability (integer 0-100) that "
            "you will fully COMPLETE the current task.\nConfidence:",
            label=f"scalar:{DS}:{ids[j]}")
    if variant == "shuffle":
        # shuffle: same retrieval, cards WITHOUT hindsight (it would leak the real outcome),
        # verdicts replaced by random train-fold outcomes. Paired control = --no-hindsight metapost.
        rng = np.random.RandomState(zlib.crc32(str(ids[j]).encode()) % (2**31))
        fakes = proxy  # here `proxy` carries the pool of train outcomes (array), sampled below
        picks = rng.choice(len(fakes), size=len(nbr_idx[:k]), replace=True)
        track = "\n".join(card(recs, epi, {}, feats, ids, n, fake_outcome=int(fakes[p]))
                          for n, p in zip(nbr_idx[:k], picks))
        return fit_context(
            "YOUR TRACK RECORD on similar past tasks (COMPLETED = you finished it; FAILED = you did not):\n"
            f"{track}\n\n"
            f"CURRENT task you are about to attempt:\n  \"{cur}\"\n\n"
            "Based on your track record on tasks like this, name your recurring pattern and estimate the "
            "probability you will fully COMPLETE the current task.\n\n"
            "Respond EXACTLY in this format:\n"
            "Self-reflection: <2-4 sentences: your pattern on tasks like this and how it applies>\n"
            "Confidence: <integer 0-100 that you will complete the current task>",
            label=f"shuffle:{DS}:{ids[j]}")
    track = "\n".join(card(recs, epi, hind, feats, ids, n) for n in nbr_idx[:k])
    if variant == "plain":  # full cards, plain recalibration, NO pattern-naming
        return fit_context(
            "YOUR TRACK RECORD on similar past tasks (COMPLETED = you finished it; FAILED = you did not):\n"
            f"{track}\n\n"
            f"CURRENT task you are about to attempt:\n  \"{cur}\"\n\n"
            "Given this track record, output a single probability (integer 0-100) that you will fully "
            "COMPLETE the current task.\nConfidence:",
            label=f"plain:{DS}:{ids[j]}")
    return fit_context(
        "YOUR TRACK RECORD on similar past tasks (COMPLETED = you finished it; FAILED = you did not):\n"
        f"{track}\n\n"
        f"CURRENT task you are about to attempt:\n  \"{cur}\"\n\n"
        "Based on your track record on tasks like this, name your recurring pattern and estimate the "
        "probability you will fully COMPLETE the current task.\n\n"
        "Respond EXACTLY in this format:\n"
        "Self-reflection: <2-4 sentences: your pattern on tasks like this and how it applies>\n"
        "Confidence: <integer 0-100 that you will complete the current task>",
        label=f"metapost:{DS}:{ids[j]}",
    )


def parse(text):
    refl = ""; m = _REFLECT.search(text or "")
    if m: refl = " ".join(m.group(1).split())
    cm = _CONF_RE.search(text or "")
    if cm: conf = max(0.0, min(1.0, float(cm.group(1)) / 100.0))
    else:
        pcts = re.findall(r"(\d{1,3}(?:\.\d+)?)\s*%", text or "")
        conf = max(0.0, min(1.0, float(pcts[-1]) / 100.0)) if pcts else float("nan")
    if conf != conf:
        # bare-number continuation: the prompt ends with a "Confidence:" cue, and some model
        # revisions complete it with just the integer instead of echoing the label
        m2 = re.fullmatch(r"\s*([0-9]{1,3})(?:\.[0-9]+)?\s*", text or "")
        if m2: conf = max(0.0, min(1.0, float(m2.group(1)) / 100.0))
    return refl, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--location", default=os.environ.get("XCONF_LOCATION", "us-central1"))
    ap.add_argument("--no-hindsight", action="store_true",
                    help="ablation variant: identical retrieval + prompt, cards WITHOUT the "
                         "hindsight line -> <ds>.metapost_nohind.jsonl (paired hindsight ablation)")
    ap.add_argument("--variant", choices=["metapost", "plain", "scalar", "shuffle"], default="metapost",
                    help="ablation: metapost=name pattern; plain=full cards, no pattern "
                         "-> <ds>.incontext_posthoc.jsonl; scalar=only the hit-rate number "
                         "-> <ds>.incontext_scalar.jsonl")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    recs, emb, epi, hind, feats, ids = load()
    if args.no_hindsight:
        hind = {}
    if not epi:
        print("[metapost] WARNING: no episodes_reflect file — neighbour cards will lack reflections "
              "(run elicit_alf.py first)", flush=True)
    if not hind:
        print("[metapost] note: no hindsight in cards "
              + ("(--no-hindsight ablation variant)" if args.no_hindsight else "(optional; hindsight_alf.py)"), flush=True)
    nbr = neighbours(recs, emb, ids, args.k)
    VARIANT_OUT = {"metapost": f"{DS}.metapost{'_nohind' if args.no_hindsight else ''}.jsonl",
                   "plain": f"{DS}.incontext_posthoc.jsonl", "scalar": f"{DS}.incontext_scalar.jsonl",
                   "shuffle": f"{DS}.incontext_shuffle.jsonl"}
    out_path = os.path.join(MD, VARIANT_OUT[args.variant])
    yv = np.array([recs[i]["correct"] for i in ids], float)  # for the scalar proxy (plain hit-rate, pre-hoc)
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            if l.strip(): done.add(json.loads(l)["id"])
    todo = [j for j in range(len(ids)) if ids[j] not in done]
    print(f"[metapost] {len(ids)} rows, {len(done)} cached, {len(todo)} to do (k={args.k})", flush=True)

    cfg = Config(); cfg.model.name = args.model; cfg.vertex.location = args.location
    cfg.model.max_output_tokens = 2048; cfg.model.max_workers = args.workers
    client = build_client(cfg)

    def work(j):
        proxy = float(np.mean([yv[n] for n in nbr[j][:args.k]])) if args.variant == "scalar" else None
        if args.variant == "shuffle":
            proxy = yv  # outcome pool for randomised verdicts (marginal base-rate preserved)
        prompt = build_prompt(recs, epi, hind, feats, ids, j, nbr[j], args.k,
                              variant=args.variant, proxy=proxy)
        txt = client.generate(prompt, system=system_prompt())
        r, c = parse(txt)
        if c != c:
            # some replies bury the number in prose that all three parse layers miss;
            # one follow-up with a changed prompt (temp-0-safe) fixes it
            txt2 = client.generate(prompt + "\n\nReply with ONLY the line 'Confidence: <integer 0-100>'.",
                                   system=system_prompt())
            r2, c2 = parse(txt2)
            if c2 == c2:
                r, c = (r2 or r), c2
        if args.variant in ("plain", "scalar"):
            return {"id": ids[j], "conf": c}  # align with reasoning incontext schema
        return {"id": ids[j], "confidence": c, "self_reflection": r}

    fout = open(out_path, "a", encoding="utf-8")
    for c0 in range(0, len(todo), args.chunk):
        batch = todo[c0:c0 + args.chunk]; rows = {}
        with ThreadPoolExecutor(max_workers=cfg.model.max_workers) as pool:
            futs = {pool.submit(work, j): j for j in batch}
            for fut in as_completed(futs):
                j = futs[fut]
                try: rows[ids[j]] = fut.result()
                except Exception as e: print(f"[metapost] FAIL {ids[j]}: {e}", file=sys.stderr)
        for j in batch:
            if ids[j] in rows: fout.write(json.dumps(rows[ids[j]], ensure_ascii=False) + "\n")
        fout.flush(); print(f"[metapost] {min(c0+args.chunk,len(todo))}/{len(todo)}", flush=True)
    fout.close()
    rows_all = [json.loads(l) for l in open(out_path)]
    ck = "conf" if args.variant in ("plain", "scalar") else "confidence"
    print(f"[metapost:{args.variant}] wrote {len(rows_all)} (conf_parsed={sum(1 for x in rows_all if x[ck]==x[ck])/max(1,len(rows_all)):.2f})", flush=True)


if __name__ == "__main__":
    main()
