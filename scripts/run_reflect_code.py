#!/usr/bin/env python3
"""Code-dataset elicitation (lcb): run_reflect's machinery with a fenced-code Answer.

REUSE, NOT A FORK: SYSTEM, build_prompt, the recovery ladder (elicit_with_ladder) and the
label-line regexes all come from run_reflect. This wrapper only adds, for code tasks:

  1. a prompt ADDITION -- the Answer must be EXACTLY ONE fenced ```python block (the
     elicitation semantics are unchanged: Reasoning -> Answer(code) -> Self-reflection ->
     Confidence, same system prompt, same ladder, temp 0, thinking off);
  2. a fence-aware parser -- run_reflect's _ANSWER regex stops at the first line matching
     "Self-reflection:"/"Confidence:", so a multi-line code answer CONTAINING such a line
     (e.g. in a string literal) would be silently cut. Here the answer is the fenced block
     itself, and Self-reflection/Confidence are parsed from the text AFTER the closing
     fence, which removes that failure mode entirely. `--selftest` proves both the hazard
     and the fix on synthetic responses (no network).

The overridden build_prompt/parse are installed into run_reflect's module globals, which is
exactly what elicit_with_ladder resolves at call time -- rungs 2/3 (salvage continuation,
concise retry) therefore inherit the code format automatically. Do not import this module
in a process that also runs the plain run_reflect elicitation.

RUNG 4 (code-only recovery): for items where ALL three rungs fail -- the salvage replay
re-triggers the loop and the 15-sentence concise retry cannot converge on a hard problem it
cannot solve. For such items the methodologically correct episode is a COMMITTED best-effort
attempt (even partial/brute-force) + a reflection saying what is unsolved + low confidence;
the unit tests supply the (failing) label. work_code therefore adds up to 2 extra temp-1.0
attempts demanding exactly that commitment (logged retry_commit=1; QC reports the rate).
Recovery mechanism, not elicitation design -- same status as rungs 2/3.

`correct` stays a 0 placeholder: lcb_verify.py fills it by executing the
problem's unit tests locally (ground truth -- never self-judged).
"""
from __future__ import annotations

import os
import re
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_reflect as rr  # noqa: E402  (the single source of the elicitation design)
from xconf.prompts import _CONF_RE  # noqa: E402

CODE_FORMAT = (
    "\n\nCODE ANSWER FORMAT (this is a coding task): on the Answer line, give your complete, "
    "final, self-contained solution as EXACTLY ONE fenced Python code block:\n"
    "Answer:\n```python\n<your complete solution code>\n```\n"
    "Do not place any other fenced code block after the Answer block. The Self-reflection and "
    "Confidence lines come AFTER the closing ``` of the Answer block."
)

_FENCE = re.compile(r"```[ \t]*(?:python|py)?[ \t]*\r?\n(.*?)```", re.IGNORECASE | re.DOTALL)
_ANS_LABEL = re.compile(r"^[ \t]*answer\s*:", re.IGNORECASE | re.MULTILINE)

_orig_build_prompt = rr.build_prompt
_orig_parse = rr.parse  # captured BEFORE the patch below (the fallback must not recurse)


def build_prompt_code(rec):
    return _orig_build_prompt(rec) + CODE_FORMAT


def parse_code(text, rec):
    """Fence-first parse: answer = the fenced block after the last 'Answer:' label;
    reflection/confidence from the text AFTER the closing fence. Falls back to
    run_reflect.parse when no fence exists (rung-2/3 outputs occasionally drop it)."""
    text = text or ""
    labels = list(_ANS_LABEL.finditer(text))
    fence = _FENCE.search(text, labels[-1].end()) if labels else None
    if fence is None:  # no fence after the Answer label -> any last fence in the response
        fences = list(_FENCE.finditer(text))
        fence = fences[-1] if fences else None
    if fence is None:
        f = _orig_parse(text, rec)  # legacy line parser (lcb registered in verify.TEST_VERIFIED)
        f["fenced"] = 0
        return f
    code = fence.group(1).rstrip("\n")
    head, tail = text[: fence.start()], text[fence.end():]
    refl = ""
    m = rr._REFLECT.search(tail)
    if m:
        refl = " ".join(m.group(1).split())
    reasoning, reason_len = "", 0
    m = rr._REASON.search(head)
    if m:
        reason_len = len(m.group(1))
        reasoning = " ".join(m.group(1).split())  # stored FULL (no truncation)
    cm = _CONF_RE.search(tail) or _CONF_RE.search(head)  # tail first: never a stray match inside code
    if cm:
        conf = max(0.0, min(1.0, float(cm.group(1)) / 100.0))
    else:  # same lenient fallback as run_reflect: last percentage mentioned
        pcts = re.findall(r"(\d{1,3}(?:\.\d+)?)\s*%", tail) or re.findall(r"(\d{1,3}(?:\.\d+)?)\s*%", text)
        conf = max(0.0, min(1.0, float(pcts[-1]) / 100.0)) if pcts else float("nan")
    return {
        "answer": code,  # the FULL code block, verbatim
        "correct": 0,  # placeholder -- filled by lcb_verify.py (unit tests)
        "self_reflection": refl,
        "reflect_len": len(refl),
        "reason_len": reason_len,
        "reasoning": reasoning,
        "confidence": conf,
        "fenced": 1,
    }


# install into run_reflect's globals: elicit_with_ladder (all three rungs) resolves these there
rr.build_prompt = build_prompt_code
rr.parse = parse_code


def work_code(client, rec):
    """elicit_with_ladder + code-only rung 4: bounded commit-or-bruteforce retries (see docstring)."""
    f = rr.elicit_with_ladder(client, rec)
    for _ in range(2):
        if rr._ok(f):
            break
        commit = (rr.build_prompt(rec)
                  + "\n\n(FINAL ATTEMPT -- your previous attempts failed to converge. COMMIT NOW to "
                  "your best attempt: even a partial or brute-force solution is acceptable. Keep "
                  "Reasoning under 10 sentences. The Answer MUST be exactly one fenced ```python "
                  "block; then Self-reflection -- state plainly what is unsolved or shaky -- then "
                  "Confidence, honestly low if the solution is partial.)")
        try:
            f4 = rr.parse(client.generate(commit, system=rr.SYSTEM, temperature=1.0), rec)
        except Exception as e:  # noqa: BLE001
            print(f"[reflect_code] rung-4 exception on {rec.id}: {e}", flush=True)
            continue
        if rr._ok(f4):
            f4["retry_commit"] = 1
            f4["id"] = rec.id
            f = f4
            break
    return f


def selftest():
    from xconf.datasets import Example

    rec = Example(id="lcb/x", dataset="lcb", question="q", gold="", choices=None)
    # the genuine hazard: a code line that BEGINS (mod whitespace) with a label the legacy
    # _ANSWER regex stops at -- e.g. a report template in a multi-line string
    tricky_code = (
        'import sys\n\nTEMPLATE = """\n'
        'Self-reflection: {refl}\n'
        'Confidence: 99\n'
        '"""\n\ndef main():\n    s = sys.stdin.read()\n    print(TEMPLATE.format(refl=s))\n\nmain()'
    )
    resp = (
        "Reasoning: Step 1: read input. Step 2: print the tricky strings.\n"
        "Answer:\n```python\n" + tricky_code + "\n```\n"
        "Self-reflection: The shakiest step is the echo of label-like strings; the code itself is trivial.\n"
        "Confidence: 72"
    )
    f = parse_code(resp, rec)
    assert f["fenced"] == 1
    assert f["answer"] == tricky_code, "fenced answer must be the FULL multi-line code, verbatim"
    assert "def main():" in f["answer"] and "Self-reflection: {refl}" in f["answer"]
    assert f["self_reflection"].startswith("The shakiest step"), f["self_reflection"]
    assert f["confidence"] == 0.72, f["confidence"]
    assert f["reasoning"].startswith("Step 1"), f["reasoning"]
    # prove the hazard exists in the legacy line parser (why this wrapper exists):
    legacy_ans = rr._ANSWER.search(resp).group(1)
    assert "def main():" not in legacy_ans, "expected the legacy regex to cut the code at the embedded label"
    # DOTALL sanity: legacy parser DOES handle benign multi-line answers (no embedded labels)
    benign = "Reasoning: r\nAnswer: line1\nline2\nline3\nSelf-reflection: ok fine\nConfidence: 60"
    lf = _orig_parse(benign, rec)
    assert lf["answer"] == "line1\nline2\nline3", lf["answer"]
    assert lf["confidence"] == 0.60
    # fence variant: casing/space after backticks
    resp2 = "Answer:\n``` PY \na = 1\nb = 2\n```\nSelf-reflection: fine.\nConfidence: 55"
    f2 = parse_code(resp2, rec)
    assert f2["answer"] == "a = 1\nb = 2" and f2["confidence"] == 0.55 and f2["fenced"] == 1
    # no fence at all -> legacy fallback still yields a usable episode
    f3 = parse_code("Reasoning: r\nAnswer: print(1)\nSelf-reflection: meh.\nConfidence: 40", rec)
    assert f3["fenced"] == 0 and f3["answer"] == "print(1)" and f3["confidence"] == 0.40
    # 'confidence: 99' INSIDE the code must not shadow the real Confidence line (tail-first search)
    assert 'Confidence: 99' in f["answer"] and f["confidence"] == 0.72
    print("[selftest] all parse_code assertions passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=False, default="gemini-2.5-flash")
    ap.add_argument("--dataset", default="lcb", choices=["lcb"])
    ap.add_argument("--pred-dir", default="data/predictions")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--ids", default=None, help="comma-separated record ids (smoke tests)")
    ap.add_argument("--selftest", action="store_true", help="run parser unit tests (no network)")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    from xconf.config import Config
    from xconf.llm import build_client
    from xconf.records import read_records

    cfg = Config.load(args.config)
    cfg.model.name = args.model
    cfg.model.max_output_tokens = int(os.environ.get("XCONF_MAXTOK", "65536"))
    cfg.model.max_workers = int(os.environ.get("XCONF_WORKERS", "64"))
    md = os.path.join(args.pred_dir, args.model.replace("/", "_"))
    records = read_records(os.path.join(md, f"{args.dataset}.jsonl"))
    records.sort(key=lambda r: r.order_index)
    if args.ids:
        want = set(args.ids.split(","))
        records = [r for r in records if r.id in want]
        missing = want - {r.id for r in records}
        if missing:
            raise SystemExit(f"[reflect_code] ids not found: {sorted(missing)}")
    records = records[: args.limit]
    out_path = os.path.join(md, f"{args.dataset}.episodes_reflect.jsonl")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [r for r in records if r.id not in done]
    print(f"[reflect_code] {len(records)} records, {len(done)} cached, {len(todo)} to do")

    client = build_client(cfg)
    # STREAMING writer (same design as run_reflect): one pool over ALL items, write each
    # episode on completion.
    fout = open(out_path, "a", encoding="utf-8")
    done_n = 0
    try:
        from tqdm import tqdm

        with ThreadPoolExecutor(max_workers=cfg.model.max_workers) as pool:
            futs = {pool.submit(work_code, client, r): r for r in todo}
            for fut in tqdm(as_completed(futs), total=len(todo), desc="reflect_code"):
                r = futs[fut]
                try:
                    row = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"[reflect_code] FAILED {r.id}: {e}", file=sys.stderr)
                    continue
                fout.write(json.dumps(row) + "\n")
                done_n += 1
                if done_n % 50 == 0:
                    fout.flush()
        fout.flush()
    finally:
        fout.close()
    rows_all = [json.loads(l) for l in open(out_path) if l.strip()]
    n = max(1, len(rows_all))
    fenced = sum(x.get("fenced", 0) for x in rows_all) / n
    cpar = sum(1 for x in rows_all if x["confidence"] == x["confidence"]) / n
    ans = sum(1 for x in rows_all if str(x.get("answer", "")).strip()) / n
    refl = sum(1 for x in rows_all if str(x.get("self_reflection", "")).strip()) / n
    print(f"[reflect_code] wrote {len(rows_all)} -> {out_path}")
    print(f"[reflect_code] fenced={fenced:.2f} answer_nonempty={ans:.2f} refl_nonempty={refl:.2f} "
          f"conf_parsed={cpar:.2f}  (correct=placeholder until lcb_verify.py)")


if __name__ == "__main__":
    main()
