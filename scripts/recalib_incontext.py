#!/usr/bin/env python3
"""In-context experiential recalibration (1 call): the MODEL reads its own
retrieved track record and states a calibrated confidence. Tests whether the POST-HOC (hindsight)
reflection helps, by an internally-controlled A/B on the SAME retrieved neighbors:

  --mode prehoc : show each neighbor's [approach + stated conf + CORRECT/WRONG]            (no hindsight)
  --mode posthoc: show each neighbor's [approach + stated conf + CORRECT/WRONG + HINDSIGHT] (with the 2nd reflection)

The delta (posthoc - prehoc) isolates the value of the second reflection. Format-general: neighbors are
described by their REASONING/REFLECTION + outcome, never by the answer string. History-safe: the test
point's own outcome is never shown; retrieval excludes self. Output: <ds>.incontext_<mode>.jsonl {id,conf}.

CAUSAL DISCIPLINE: the retrieval key (PCA + logistic reweight) is fit PER FOLD on the
training folds only, and neighbours are retrieved ONLY from the training folds — mirroring the 5-fold OOS
protocol of the eval scripts. Neighbour cards show FULL reflections/hindsights (no truncation).
"""
from __future__ import annotations
import os, re, sys, json, zlib, argparse
# The recalibration call outputs ONE fixed-format confidence value — thinking must never be on for
# it. Auxiliary calls run think-off, same as grading + recovery rungs.
os.environ["XCONF_THINKING"] = "off"
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tqdm import tqdm
from xconf.config import Config
from xconf.llm import build_client
from xconf.prompts import _CONF_RE
from common import PCA_DIM, N_FOLDS, SIGMA_CONF, fold_ids, pca_fit, pca_apply, fit_logit, diag_space, fit_context
md = os.environ.get("XCONF_MD", "data/predictions/gemini-2.5-flash")


def loademb(f):
    p = md + "/" + f; d = {}
    if os.path.exists(p):
        for l in open(p):
            o = json.loads(l); d[o["id"]] = np.array(o["embedding"], dtype=np.float32)
    return d
def C(e): c = e.get("confidence"); return c if isinstance(c, (int, float)) and c == c else 0.5

SYSTEM = (
    "You are recalibrating your confidence in an answer you already gave, using your OWN track record on "
    "the most similar problems you have faced before. Trust the track record over your gut: if on similar "
    "problems you were systematically over- or under-confident, correct for it. Follow the reply format "
    "at the end of the prompt EXACTLY, and always end with the Confidence line."
)
# hide-track-record ablation: cards carry NO outcomes and NO hindsight — only the similar
# past problems, the reflections written back then, and the stated confidences. Isolates "reading
# retrieved experience" from "reading retrieved OUTCOMES" (same supervised retrieval, same cards
# otherwise). Separate output file (.incontext_nooutcome.jsonl).
SYSTEM_NOOUT = (
    "You are recalibrating your confidence in an answer you already gave, by comparing the problem "
    "against the most similar problems you have faced before. You do NOT know whether you got those "
    "past problems right; you only see them, the reflections you wrote at the time, and how confident "
    "you said you were. Look for shared structure and for recurring doubts in your own notes. Follow "
    "the reply format at the end of the prompt EXACTLY, and always end with the Confidence line."
)
# no-memory control: same second call, no neighbour cards, no track record, no proxy.
# Output: .incontext_nomem.jsonl
SYSTEM_NOMEM = (
    "You are re-examining an answer you already gave and restating your confidence that it is "
    "correct. No external track record is available; judge from the problem, your answer, and your "
    "own reflection. Follow the reply format at the end of the prompt EXACTLY, and always end with "
    "the Confidence line."
)


def build_prompt(cur, neighbors, mode):
    opts = ""
    if cur.get("choices"):
        opts = "\nOptions:\n" + "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(cur["choices"]))
    if mode == "nomem":
        # no cards, no proxy: the pure ask-again control.
        return (
            f"Your CURRENT problem:\nQuestion: {cur['question']}{opts}\n"
            f"Your answer: {cur.get('answer','')}\n"
            f"Your self-reflection: {cur.get('self_reflection','')}\n"
            f"Your initial confidence: {int(round(cur['conf'] * 100))}%\n\n"
            "Take a second, careful look at your answer and your self-reflection, and re-judge how "
            "likely your answer is to be correct. Output a single recalibrated confidence (integer "
            "0-100) that your CURRENT answer is correct.\nConfidence:"
        )
    if mode == "scalar":
        # NO neighbour cards — the model sees ONLY the scalar proxy (conf-conditioned hit-rate).
        # Isolates "does the model need the raw retrieved episodes, or just the number?"
        return (
            f"Your CURRENT problem:\nQuestion: {cur['question']}{opts}\n"
            f"Your answer: {cur.get('answer','')}\n"
            f"Your self-reflection: {cur.get('self_reflection','')}\n"
            f"Your initial confidence: {int(round(cur['conf'] * 100))}%\n\n"
            f"Your measured TRACK RECORD on the most similar past problems you have faced (weighted "
            f"toward the ones where you felt similarly confident) is {int(round(cur['proxy']*100))}% correct.\n\n"
            "Trust that measured rate over your gut. Output a single recalibrated confidence (integer "
            "0-100) that your CURRENT answer is correct.\nConfidence:"
        )
    lines = []
    for k, nb in enumerate(neighbors, 1):
        # shuffle mode: verdict comes from nb["shuffled_correct"] (a random train-fold outcome) —
        # cards carry NO hindsight there (hindsight text would leak the real outcome).
        corr = nb["shuffled_correct"] if mode == "shuffle" else nb["correct"]
        verdict = "CORRECT" if corr else "WRONG"
        appr = nb.get("self_reflection", "") or "(no note)"
        # full question stem + full reflection + full hindsight — no truncation of neighbour
        # cards. (Neighbour distractor OPTIONS are the one permitted omission — never the stem.)
        seg = (f"  {k}. [similar past problem] \"{nb.get('question', '')}\"\n"
               f"     your approach/uncertainty then: \"{appr}\"\n")
        if mode == "nooutcome":
            seg += f"     you stated {int(round(nb['conf'] * 100))}% confidence."
        else:
            seg += f"     you stated {int(round(nb['conf'] * 100))}% confidence; you were actually {verdict}."
        if mode in ("posthoc", "metapost") and nb.get("posthoc"):
            seg += f"\n     your hindsight afterward: \"{nb['posthoc']}\""
        lines.append(seg)
    track = "\n".join(lines)
    navg = float(np.mean([nb["conf"] for nb in neighbors]))
    nacc = float(np.mean([(nb["shuffled_correct"] if mode == "shuffle" else nb["correct"])
                          for nb in neighbors]))
    if mode == "nooutcome":
        head = (
            f"Your CURRENT problem:\nQuestion: {cur['question']}{opts}\n"
            f"Your answer: {cur.get('answer','')}\n"
            f"Your self-reflection: {cur.get('self_reflection','')}\n"
            f"Your initial confidence: {int(round(cur['conf'] * 100))}%\n\n"
            f"The {len(neighbors)} most similar problems from your past (outcomes unknown; you averaged "
            f"{int(round(navg*100))}% confidence on them):\n{track}\n\n"
        )
        return head + (
            "First, in one sentence, name what these past problems share with your CURRENT one — the "
            "procedure, the structure, or the kind of step that decides success. Then, judging only from "
            "your past reflections and stated confidences (no outcomes are available), decide whether your "
            "current approach deserves more or less confidence than you initially felt. Reply as:\n"
            "Pattern: <one sentence>\nConfidence: <integer 0-100>"
        )
    head = (
        f"Your CURRENT problem:\nQuestion: {cur['question']}{opts}\n"
        f"Your answer: {cur.get('answer','')}\n"
        f"Your self-reflection: {cur.get('self_reflection','')}\n"
        f"Your initial confidence: {int(round(cur['conf'] * 100))}%\n\n"
        f"Your TRACK RECORD on the {len(neighbors)} most similar past problems "
        f"(you averaged {int(round(navg*100))}% confidence and were right {int(round(nacc*100))}% of the time):\n"
        f"{track}\n\n"
    )
    if mode == "metapost":
        return head + (
            "First, in one sentence, name the RECURRING failure pattern these hindsights reveal about you on "
            "this kind of problem (e.g. 'I confidently misremember specific facts', 'my logic is sound but I "
            "miss edge cases'). Then judge whether your CURRENT problem is exposed to that same pattern, and "
            "calibrate accordingly. Reply as:\nPattern: <one sentence>\nConfidence: <integer 0-100>"
        )
    return head + (
        "Given this track record, output a single recalibrated confidence (integer 0-100) that your "
        "CURRENT answer is correct.\nConfidence:"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["prehoc", "posthoc", "metapost", "nooutcome", "scalar", "shuffle", "nomem"],
                    required=True)  # shuffle = prehoc cards but neighbour VERDICTS randomised
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--chunk", type=int, default=120)
    ap.add_argument("--bank-limit", type=int, default=0, help="restrict retrieval pool to first N episodes (0=all) -> metapost scaling curve")
    ap.add_argument("--tag", default="", help="suffix on the output filename (to keep multiple bank sizes side by side)")
    args = ap.parse_args()

    cfg = Config.load(args.config); cfg.model.name = args.model
    # 1024: the output is Pattern+Confidence; the cap is an overflow guard, not a target.
    cfg.model.max_output_tokens = 1024; cfg.model.max_workers = int(os.environ.get("XCONF_WORKERS", "64"))
    epi = {json.loads(l)["id"]: json.loads(l) for l in open(f"{md}/{args.dataset}.episodes_reflect.jsonl") if l.strip()}
    rec = {json.loads(l)["id"]: json.loads(l) for l in open(f"{md}/{args.dataset}.jsonl")}
    posth = {json.loads(l)["id"]: json.loads(l).get("posthoc_reflection", "")
             for l in open(f"{md}/{args.dataset}.posthoc.jsonl") if l.strip()} if os.path.exists(f"{md}/{args.dataset}.posthoc.jsonl") else {}
    qe = loademb(f"{args.dataset}.emb.jsonl"); re_ = loademb(f"{args.dataset}.reflkey.emb.jsonl")
    rsn = loademb(f"{args.dataset}.reasonkey.emb.jsonl"); use_rsn = len(rsn) > 0
    # Fold-alignment guarantee: the id list MUST equal the eval scripts' id list
    # (epi ∩ rec ∩ emb ∩ reflkey), or the interleaved fold split silently diverges from eval's and
    # neighbour cards can carry eval-test-fold outcomes. posthoc/metapost therefore REQUIRE complete
    # posthoc coverage instead of silently dropping uncovered ids.
    ids = [i for i in epi if i in rec and i in qe and i in re_ and (not use_rsn or i in rsn)]
    if args.mode in ("posthoc", "metapost"):
        miss = [i for i in ids if i not in posth]
        if miss:
            sys.exit(f"[incontext-{args.mode}] REFUSING to run: {len(miss)} ids lack posthoc "
                     f"reflections (e.g. {miss[:3]}); complete {args.dataset}.posthoc.jsonl first "
                     f"so the fold split stays identical to eval's.")
    ids.sort(key=lambda i: rec[i]["order_index"])
    if args.dataset == "supergpqa":
        ids = ids[:8000]  # paper protocol: 8k prefix
    y = np.array([epi[i]["correct"] for i in ids], float); verb = np.array([C(epi[i]) for i in ids])
    rl = np.array([np.log1p(epi[i].get("reason_len", 0)) for i in ids])
    fl = np.array([np.log1p(epi[i].get("reflect_len", 0)) for i in ids])
    blocks = [np.stack([qe[i] for i in ids]), np.stack([re_[i] for i in ids])]
    if use_rsn: blocks.insert(1, np.stack([rsn[i] for i in ids]))
    scalars = np.stack([verb, rl, fl], 1)
    n = len(ids)
    # CAUSAL: per-fold PCA + diagonal co-correctness key, fit on TRAIN folds only; retrieval pool = train
    # folds only (mirrors the eval scripts' 5-fold OOS). Both A/B modes share the same retrieval.
    folds = fold_ids(n, N_FOLDS)
    bank_limit = args.bank_limit or n  # ids sorted by order_index -> first bank_limit = the bank
    A_by_fold, tr_by_fold = {}, {}
    for f in range(N_FOLDS):
        tr = np.where((folds != f) & (np.arange(n) < bank_limit))[0]
        cols = []
        for B in blocks:
            bmu, bVt = pca_fit(B[tr], PCA_DIM)
            cols.append(pca_apply(B, bmu, bVt))
        F = np.concatenate(cols + [scalars], 1)
        w, kmu, ksd = fit_logit(F[tr], y[tr])
        A_by_fold[f] = diag_space(F, w, kmu, ksd)
        tr_by_fold[f] = tr
    pos = {i: k for k, i in enumerate(ids)}
    sub = ids[: args.limit]

    out = f"{md}/{args.dataset}.incontext_{args.mode}{args.tag}.jsonl"
    done = set()
    if os.path.exists(out):
        for l in open(out):
            if l.strip(): done.add(json.loads(l)["id"])
    todo = [i for i in sub if i not in done]
    print(f"[incontext-{args.mode}] {args.dataset}: {len(sub)} pts, {len(done)} cached, {len(todo)} to do (k={args.k})")
    client = build_client(cfg)

    def work(i):
        j = pos[i]; f = int(folds[j])
        A = A_by_fold[f]; tr = tr_by_fold[f]  # retrieval pool = training folds only (self never included)
        s = A[tr] @ A[j]
        top = tr[np.argsort(-s)[: args.k]]
        neighbors = [{"conf": verb[t], "correct": int(y[t]),
                      "question": rec[ids[t]].get("question", ""),
                      "self_reflection": epi[ids[t]].get("self_reflection", ""),
                      "posthoc": posth.get(ids[t], "")} for t in top]
        if args.mode == "shuffle":
            # random TRAIN-fold outcomes replace the true verdicts (marginal base-rate preserved,
            # per-card correspondence destroyed). Stable per-id seed -> reproducible/resumable.
            rng = np.random.RandomState(zlib.crc32(str(i).encode()) % (2**31))
            fake = y[rng.choice(tr, size=len(neighbors), replace=True)]
            for nb, fc in zip(neighbors, fake):
                nb["shuffled_correct"] = int(fc)
        # conf-conditioned hit-rate over the top-k (= the Recall proxy) — fed to the model in scalar mode
        wv = np.exp(-((verb[top] - verb[j]) ** 2) / (2 * SIGMA_CONF ** 2))
        proxy = float((y[top] * wv).sum() / (wv.sum() + 1e-9))
        cur = {"question": rec[i]["question"], "choices": rec[i].get("choices"),
               "answer": epi[i].get("answer", ""), "self_reflection": epi[i].get("self_reflection", ""),
               "conf": verb[j], "proxy": proxy}
        raw = client.generate(fit_context(build_prompt(cur, neighbors, args.mode), label=f"recalib:{args.dataset}:{i}"),
                              system=SYSTEM_NOMEM if args.mode == "nomem" else
                                     SYSTEM_NOOUT if args.mode == "nooutcome" else SYSTEM)
        m = _CONF_RE.search(raw or "")
        if m: conf = float(m.group(1)) / 100.0
        else:
            # fallback ladder: explicit percentage anywhere, else the reply being a bare number.
            # NEVER "last number in prose" (it grabs years/counts).
            p = re.findall(r"(\d{1,3}(?:\.\d+)?)\s*%", raw or "")
            if p: conf = float(p[-1]) / 100.0
            else:
                b = re.fullmatch(r"\s*([0-9]{1,3})(?:\.[0-9]+)?\s*", raw or "")
                conf = float(b.group(1)) / 100.0 if b else float("nan")
        return {"id": i, "conf": max(0.0, min(1.0, conf)) if conf == conf else None}

    fout = open(out, "a", encoding="utf-8")
    try:
        for c0 in tqdm(range(0, len(todo), args.chunk), desc=f"ic-{args.mode}"):
            batch = todo[c0:c0 + args.chunk]; rows = {}
            with ThreadPoolExecutor(max_workers=cfg.model.max_workers) as pool:
                futs = {pool.submit(work, i): i for i in batch}
                for f in as_completed(futs):
                    try: rows[futs[f]] = f.result()
                    except Exception as e: print("FAIL", futs[f], e, file=sys.stderr)
            for i in batch:
                if i in rows: fout.write(json.dumps(rows[i]) + "\n")
            fout.flush()
    finally:
        fout.close()
    print(f"[incontext-{args.mode}] wrote -> {out}")


if __name__ == "__main__":
    main()
