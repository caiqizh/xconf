#!/usr/bin/env python3
"""Outcome-aware HINDSIGHT reflection for agent bank episodes (agent-line counterpart of
reflect_posthoc.py on the reasoning line).

For each episode the model re-reads the task goal + its OWN FULL trace, is TOLD the true outcome
(COMPLETED/FAILED), and writes a hindsight: the decisive mistake (if failed) or what made it work
(if succeeded). Hindsights live ONLY in the historical bank and are injected into metapost neighbour
cards (metapost_alf.py) — the leakage discipline matches reflect_posthoc.py: a test point's own
hindsight is never used for that point.

Full trace, full hindsight — the only cut is the fit_context() overflow guard.
Output: <md>/<ds>.hindsight.jsonl {id, hindsight, correct}. Resumable, parallel.
"""
from __future__ import annotations
import os, re, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concurrent.futures import ThreadPoolExecutor, as_completed
from xconf.config import Config
from xconf.llm import build_client
from elicit_alf import digest
from common import fit_context

SYSTEM = (
    "You are reviewing a task you attempted in a text-based interactive environment. You now KNOW the "
    "true outcome. Re-read your own action trace and explain, concretely and honestly, WHY it ended the "
    "way it did: if you FAILED, name the decisive mistake (the step where it went wrong and what you "
    "should have done instead); if you COMPLETED it, name what made it work and any step where it nearly "
    "failed. Cite specific steps. Do not rationalise."
)
_HIND = re.compile(r"hindsight\s*:\s*(.*)\Z", re.IGNORECASE | re.DOTALL)


def build_prompt(ep):
    outcome = "COMPLETED" if ep["correct"] else "FAILED"
    return fit_context(
        f"{digest(ep)}\n\n"
        f"TRUE OUTCOME: you {outcome} this task.\n\n"
        "Respond EXACTLY in this format:\n"
        "Hindsight: <3-5 sentences: why it ended this way, citing specific steps>",
        label=f"hindsight:{ep.get('id','?')}",
    )


def work(client, ep):
    txt = client.generate(build_prompt(ep), system=SYSTEM, temperature=0.0)
    m = _HIND.search(txt or "")
    hind = " ".join((m.group(1) if m else (txt or "")).split())
    return {"id": ep["id"], "hindsight": hind, "correct": int(ep["correct"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=os.environ.get("XCONF_MD", "data/predictions/alfworld-25f"))
    ap.add_argument("--ds", default=os.environ.get("XCONF_DS", "alf"))
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--location", default=os.environ.get("XCONF_LOCATION", "us-central1"))
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(os.path.join(args.md, f"{args.ds}.jsonl")) if l.strip()]
    out = os.path.join(args.md, f"{args.ds}.hindsight.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            if l.strip():
                done.add(json.loads(l)["id"])
    todo = [r for r in recs if r["id"] not in done]
    print(f"[hindsight] {len(recs)} recs, {len(done)} cached, {len(todo)} to do (model={args.model})", flush=True)

    cfg = Config(); cfg.model.name = args.model; cfg.vertex.location = args.location
    cfg.model.max_output_tokens = 2048
    client = build_client(cfg)
    fout = open(out, "a", encoding="utf-8")
    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(work, client, ep): ep for ep in todo}
        for fut in as_completed(futs):
            try:
                row = fut.result()
            except Exception as e:
                print(f"[hindsight] FAIL {futs[fut]['id']}: {e}", flush=True); continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n"); fout.flush()
            n += 1
            if n % 300 == 0:
                print(f"[hindsight] {n}/{len(todo)}", flush=True)
    fout.close()
    print(f"[hindsight] wrote -> {out}", flush=True)


if __name__ == "__main__":
    main()
