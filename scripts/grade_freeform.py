#!/usr/bin/env python3
"""Grade free-form answers (SimpleQA / HLE) against the KNOWN gold answer.

This is a ground-truth verifier implemented with an LLM (the grader SEES the gold
answer and only checks semantic equivalence) -- NOT a self-judge of correctness
without truth. Mirrors the SimpleQA/HLE official grading. Updates `correct` in the
episodes_reflect rows in place (and writes a sidecar <dataset>.graded.jsonl).

Reads:  <dataset>.episodes_reflect.jsonl (answer), <dataset>.jsonl (gold/question)
Writes: overwrites correct in episodes_reflect.jsonl; sidecar grade map.
"""
from __future__ import annotations
import os, sys, json, re, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from xconf.config import Config
from xconf.llm import build_client
from xconf.records import read_records

SYSTEM = (
    "You are a strict grader. You are given a question, the GOLD correct answer, and a "
    "candidate answer. Decide if the candidate answer is correct -- i.e. semantically "
    "equivalent to the gold answer (same entity/number/fact; ignore phrasing, case, extra "
    "words). If the candidate is missing, refuses, or says it doesn't know, it is INCORRECT. "
    "Respond EXACTLY: Grade: <CORRECT|INCORRECT>"
)
_G = re.compile(r"grade\s*:\s*(correct|incorrect)", re.IGNORECASE)


def build_prompt(q, gold, cand):
    return (f"Question: {q}\n\nGold correct answer: {gold}\n\nCandidate answer: {cand}\n\n"
            "Is the candidate answer correct? Respond EXACTLY: Grade: <CORRECT|INCORRECT>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--grader-model", default="gemini-2.5-flash")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pred-dir", default="data/predictions")
    ap.add_argument("--config", default=None)
    ap.add_argument("--chunk", type=int, default=100)
    args = ap.parse_args()

    cfg = Config.load(args.config); cfg.model.name = args.grader_model
    cfg.model.max_workers = int(os.environ.get("XCONF_WORKERS", "64")); cfg.model.max_output_tokens = 2048
    md = os.path.join(args.pred_dir, args.model.replace("/", "_"))
    rec = {r.id: r for r in read_records(os.path.join(md, f"{args.dataset}.jsonl"))}
    ep_path = os.path.join(md, f"{args.dataset}.episodes_reflect.jsonl")
    eps = [json.loads(l) for l in open(ep_path) if l.strip()]
    sidecar = os.path.join(md, f"{args.dataset}.graded.jsonl")
    done = {}
    if os.path.exists(sidecar):
        for l in open(sidecar):
            if l.strip(): o = json.loads(l); done[o["id"]] = o["correct"]
    todo = [e for e in eps if e["id"] not in done and e["id"] in rec]
    print(f"[grade] {args.dataset}: {len(eps)} eps, {len(done)} cached, {len(todo)} to grade")

    client = build_client(cfg)

    def work(e):
        r = rec[e["id"]]; cand = (e.get("answer") or "").strip() or "<no answer>"
        raw = client.generate(build_prompt(r.question, r.gold, cand), system=SYSTEM)
        m = _G.search(raw or "")
        return e["id"], int(bool(m) and m.group(1).lower() == "correct")

    fs = open(sidecar, "a", encoding="utf-8")
    try:
        for c0 in tqdm(range(0, len(todo), args.chunk), desc="grade"):
            batch = todo[c0:c0 + args.chunk]; rows = {}
            with ThreadPoolExecutor(max_workers=cfg.model.max_workers) as pool:
                futs = {pool.submit(work, e): e for e in batch}
                for f in as_completed(futs):
                    try:
                        i, c = f.result(); rows[i] = c
                    except Exception as ex:
                        print("FAIL", futs[f]["id"], ex, file=sys.stderr)
            for e in batch:
                if e["id"] in rows:
                    done[e["id"]] = rows[e["id"]]
                    fs.write(json.dumps({"id": e["id"], "correct": rows[e["id"]]}) + "\n")
            fs.flush()
    finally:
        fs.close()

    # rewrite episodes_reflect with graded correctness
    for e in eps:
        if e["id"] in done:
            e["correct"] = done[e["id"]]
    with open(ep_path, "w", encoding="utf-8") as f:
        for e in eps:
            f.write(json.dumps(e) + "\n")
    acc = sum(done.get(e["id"], 0) for e in eps) / max(1, len(eps))
    print(f"[grade] {args.dataset}: graded acc = {acc:.3f}  (updated {ep_path})")


if __name__ == "__main__":
    main()
