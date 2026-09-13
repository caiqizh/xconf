#!/usr/bin/env python3
"""Elicit a post-hoc reliability reflection + verbalized confidence for ALFWorld trajectories.

The actor (gemini-2.5-flash) sees the task + its OWN full action trace (from extra.transcript, NO
explicit success signal shown), names the shakiest part, and states a calibrated confidence that the
task was accomplished.
Output: <md>/<ds>.episodes_reflect.jsonl  Resumable, parallel.
"""
from __future__ import annotations
import os, sys, re, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor, as_completed
from xconf.config import Config
from xconf.llm import build_client
from xconf.prompts import _CONF_RE
from common import fit_context

SYSTEM = (
    "You are a meticulous reviewer judging whether an AI agent ACCOMPLISHED a given task in a "
    "text-based interactive environment. You see the task goal and the agent's FULL action trace (each "
    "action and the environment's response). You are NOT told whether it succeeded. Judge only from the "
    "trace: did it actually reach the goal state the task requires? Watch for signs of failure: going in "
    "circles, repeated identical actions, many invalid / no-op responses, or running long without "
    "completing the goal. Be skeptical; if the trace looks stuck or incomplete, your confidence must "
    "drop. Never be reflexively confident."
)
_REFLECT = re.compile(r"self[- ]?reflection\s*:\s*(.*?)(?:\n\s*confidence\s*:|\Z)", re.IGNORECASE | re.DOTALL)


def digest(ep):
    """Render the FULL trace — every step, full observations. The only cut allowed is the fit_context()
    overflow guard applied to the assembled prompt."""
    tr = ep["extra"].get("transcript", [])
    lines = []
    for h in tr:
        o = (h.get("obs", "") or "").replace("\n", " ")
        ol = o.lower()
        flag = "  [INVALID]" if ("nothing happens" in ol or "no known action" in ol) else ""
        lines.append(f"{h['step']}. {h['action']} -> {o}{flag}")
    return f"GOAL: {ep['question']}\n\nACTION TRACE ({len(tr)} steps):\n" + "\n".join(lines)


def build_prompt(ep):
    return fit_context(
        f"{digest(ep)}\n\n"
        "Assess how likely the agent FULLY accomplished the task (reached the exact goal state).\n"
        "Respond EXACTLY in this format:\n"
        "Self-reflection: <3-5 sentences: the shakiest part / most likely failure, citing specific steps "
        "of the trace>\n"
        "Confidence: <integer 0-100 that the task was accomplished>",
        label=f"elicit:{ep.get('id','?')}",
    )


def parse(text):
    text = text or ""
    m = _REFLECT.search(text)
    refl = " ".join(m.group(1).split()) if m else ""
    cm = _CONF_RE.search(text)
    if cm:
        conf = max(0.0, min(1.0, float(cm.group(1)) / 100.0))
    else:
        pcts = re.findall(r"(\d{1,3})\s*%", text)
        conf = max(0.0, min(1.0, float(pcts[-1]) / 100.0)) if pcts else float("nan")
    return refl, conf


def work(client, ep):
    txt = client.generate(build_prompt(ep), system=SYSTEM, temperature=0.0)
    refl, conf = parse(txt)
    return {"id": ep["id"], "correct": int(ep["correct"]), "self_reflection": refl,
            "reflect_len": len(refl), "reason_len": int(ep["extra"].get("reason_len", 0)),
            "confidence": conf, "final_score": float(ep["extra"].get("final_score", ep["correct"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="data/predictions/alfworld-25f")
    ap.add_argument("--ds", default="alf")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--location", default=os.environ.get("XCONF_LOCATION", "us-central1"))
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(os.path.join(args.md, f"{args.ds}.jsonl")) if l.strip()]
    out = os.path.join(args.md, f"{args.ds}.episodes_reflect.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            if l.strip():
                done.add(json.loads(l)["id"])
    todo = [r for r in recs if r["id"] not in done]
    print(f"[elicit] {len(recs)} recs, {len(done)} cached, {len(todo)} to do (model={args.model})", flush=True)

    cfg = Config(); cfg.model.name = args.model; cfg.vertex.location = args.location
    cfg.model.max_output_tokens = 2048  # overflow guard, not a target
    client = build_client(cfg)
    fout = open(out, "a", encoding="utf-8")
    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(work, client, ep): ep for ep in todo}
        for fut in as_completed(futs):
            try:
                row = fut.result()
            except Exception as e:
                print(f"[elicit] FAIL {futs[fut]['id']}: {e}", flush=True); continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n"); fout.flush()
            n += 1
            if n % 300 == 0:
                print(f"[elicit] {n}/{len(todo)}", flush=True)
    fout.close()
    rows = [json.loads(l) for l in open(out)]
    cpar = sum(1 for x in rows if x["confidence"] == x["confidence"]) / max(1, len(rows))
    print(f"[elicit] wrote {len(rows)} -> {out}  (conf_parsed={cpar:.2f})", flush=True)


if __name__ == "__main__":
    main()
