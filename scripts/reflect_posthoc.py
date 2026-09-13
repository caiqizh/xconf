#!/usr/bin/env python3
"""Pass-2 HINDSIGHT (post-hoc) reflection: the second of two reflections.

After pass-1 (run_reflect: Q -> CoT -> answer -> self-reflection -> verbalized conf), we show the
model EVERYTHING plus the GROUND TRUTH and ask it to reflect with hindsight: which step actually
held up or broke, whether its confidence was justified, and what this reveals about its reliability
on THIS KIND of problem.

LEAKAGE DISCIPLINE: this reflection is answer-aware, so it may ONLY live in the
memory bank (historical, already-labeled episodes). It is NEVER built for a test point and NEVER a
test-time feature -- exactly like the binary `correct` label. Downstream use:
in-context lessons: retrieve by pass-1 key, feed these hindsight lessons to the model.

Output: <dataset>.posthoc.jsonl  ({id, posthoc_reflection}), resumable.
"""
from __future__ import annotations
import os, re, sys, json, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from xconf.config import Config
from xconf.llm import build_client

SYSTEM = (
    "You are reviewing one of your own past attempts, now that the correct answer is revealed. "
    "Be brutally honest -- the point is to learn your real failure modes, not to save face. "
    "Identify where your reasoning actually held up or broke and whether your stated confidence "
    "was justified in hindsight."
)
_HIND = re.compile(r"hindsight\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)


def gold_text(rec):
    g = str(rec.get("gold", ""))
    ch = rec.get("choices")
    if ch and len(g) == 1 and g.isalpha():
        gi = ord(g.upper()) - 65
        if 0 <= gi < len(ch):
            return f"{g}. {ch[gi]}"
    return g


def build_prompt(rec, epi):
    opts = ""
    if rec.get("choices"):
        opts = "\nOptions:\n" + "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(rec["choices"]))
    verdict = "CORRECT" if epi.get("correct") else "INCORRECT"
    conf = epi.get("confidence")
    conf_s = f"{int(round(conf * 100))}%" if isinstance(conf, (int, float)) and conf == conf else "unstated"
    return (
        f"Question: {rec['question']}{opts}\n\n"
        f"Your earlier reasoning: {epi.get('reasoning', '') or '(not recorded)'}\n"
        f"Your answer: {epi.get('answer', '')}\n"
        f"Your self-reflection (made BEFORE knowing the truth): {epi.get('self_reflection', '')}\n"
        f"Your stated confidence: {conf_s}\n\n"
        f"The CORRECT answer is: {gold_text(rec)}\n"
        f"So your answer was {verdict}.\n\n"
        "Now reflect with hindsight, honestly and concisely (<=5 sentences): which step of your "
        "reasoning actually held up or broke, whether your confidence was justified, and what this "
        "reveals about how reliable you are on THIS KIND of problem. Start your reply with 'Hindsight:'."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pred-dir", default="data/predictions")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=100)
    args = ap.parse_args()

    cfg = Config.load(args.config); cfg.model.name = args.model
    cfg.model.max_output_tokens = 4096; cfg.model.max_workers = int(os.environ.get("XCONF_WORKERS", "64"))
    md = os.path.join(args.pred_dir, args.model.replace("/", "_"))
    rec = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(md, f"{args.dataset}.jsonl"))}
    epi = {json.loads(l)["id"]: json.loads(l)
           for l in open(os.path.join(md, f"{args.dataset}.episodes_reflect.jsonl")) if l.strip()}
    ids = [i for i in epi if i in rec]
    ids.sort(key=lambda i: rec[i]["order_index"])
    ids = ids[: args.limit]
    out = os.path.join(md, f"{args.dataset}.posthoc.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            if l.strip(): done.add(json.loads(l)["id"])
    todo = [i for i in ids if i not in done]
    print(f"[posthoc] {args.dataset}: {len(ids)} episodes, {len(done)} cached, {len(todo)} to do")
    client = build_client(cfg)

    def work(i):
        raw = client.generate(build_prompt(rec[i], epi[i]), system=SYSTEM)
        m = _HIND.search(raw or "")
        txt = " ".join((m.group(1) if m else (raw or "")).split())  # full hindsight (no truncation)
        return {"id": i, "posthoc_reflection": txt, "correct": epi[i].get("correct", 0)}

    fout = open(out, "a", encoding="utf-8")
    try:
        for c0 in tqdm(range(0, len(todo), args.chunk), desc="posthoc"):
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
    print(f"[posthoc] wrote -> {out}")


if __name__ == "__main__":
    main()
