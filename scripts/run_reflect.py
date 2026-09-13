#!/usr/bin/env python3
"""The logically-correct elicitation: reason -> answer -> self-reflect -> confidence.

Fixed order (non-negotiable): the model sees the question, reasons step by step,
commits to an answer, THEN critically reflects on how reliable that reasoning was,
and only then states a confidence. The confidence is derived AFTER reasoning and
AFTER a genuine self-critique -- not blurted, not a pre-commitment guess.

The self_reflection text is the rich reliability-episode content (for retrieval /
recalibration). Uses the model's CoT answer (re-verified) as label.
Output: <dataset>.episodes_reflect.jsonl
"""

from __future__ import annotations

import os
import re
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from xconf.config import Config
from xconf.llm import build_client
from xconf.records import read_records
from xconf.verify import verify
from xconf.prompts import _CONF_RE

SYSTEM = (
    "You solve a question by reasoning step by step, commit to an answer, then "
    "CRITICALLY reflect on how reliable your reasoning actually was -- name the "
    "shakiest step, any assumption that could be wrong, and whether a different "
    "answer is defensible -- and only then give a calibrated confidence. Do not be "
    "reflexively confident: if your reasoning has a weak link, your confidence must drop. "
    "Use as many reasoning steps as you need, keeping each step concise. If you catch yourself "
    "re-reading the question or circling without NEW progress, COMMIT to your current best answer "
    "and move on -- remaining doubts belong in the Self-reflection, not in more reasoning. ALWAYS "
    "finish with the Answer, Self-reflection, and Confidence lines in the exact format (never stop "
    "before the Confidence line)."
)

_ANSWER = re.compile(r"answer\s*:\s*(.*?)(?:\n\s*self[- ]?reflection\s*:|\n\s*confidence\s*:|\Z)", re.IGNORECASE | re.DOTALL)
_REFLECT = re.compile(r"self[- ]?reflection\s*:\s*(.*?)(?:\n\s*confidence\s*:|\Z)", re.IGNORECASE | re.DOTALL)
_REASON = re.compile(r"reasoning\s*:\s*(.*?)(?:\n\s*answer\s*:|\Z)", re.IGNORECASE | re.DOTALL)


def build_prompt(rec):
    opts = ""
    if rec.choices:
        opts = "\nOptions:\n" + "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(rec.choices))
    return (
        f"Question: {rec.question}{opts}\n\n"
        "Respond EXACTLY in this format (use as many steps as you need; keep each step concise):\n"
        "Reasoning: <your step-by-step reasoning, each step concise>\n"
        "Answer: <your single best answer>\n"
        "Self-reflection: <critically assess your reasoning: the shakiest step, assumptions that "
        "could be wrong, whether another answer is defensible>\n"
        "Confidence: <integer 0-100 that your Answer is correct, AFTER that reflection>"
    )


def parse(text, rec):
    n_choices = len(rec.choices) if rec.choices else 0
    ans = ""
    m = _ANSWER.search(text or "")
    if m:
        ans = m.group(1).strip()
    correct, extracted = verify(rec.dataset, ans, rec.gold, n_choices)
    refl = ""
    m = _REFLECT.search(text or "")
    if m:
        refl = " ".join(m.group(1).split())
    reason_len = 0
    reasoning = ""
    m = _REASON.search(text or "")
    if m:
        reason_len = len(m.group(1))
        reasoning = " ".join(m.group(1).split())  # store FULL text (no truncation in storage)
    cm = _CONF_RE.search(text or "")
    conf_src = "label"
    if cm:
        conf = max(0.0, min(1.0, float(cm.group(1)) / 100.0))
    else:  # lenient fallback: last percentage mentioned (handles minor format drift)
        pcts = re.findall(r"(\d{1,3}(?:\.\d+)?)\s*%", text or "")
        conf = max(0.0, min(1.0, float(pcts[-1]) / 100.0)) if pcts else float("nan")
        # the fallback can harvest a % from the REASONING ("a 90% chance the reaction..."),
        # which is indistinguishable from a clean parse without this marker
        conf_src = "pct_fallback" if pcts else "none"
    return {
        "answer": extracted or ans,
        "correct": int(correct),
        "self_reflection": refl,
        "reflect_len": len(refl),
        "reason_len": reason_len,
        "reasoning": reasoning,  # full approach text -> reasoning key, captured in this single pass
        "confidence": conf,
        "conf_src": conf_src,
    }


def _ok(f):
    # reflection is a CORE method signal — an episode without it is unusable, so it gates recovery too
    return (bool(str(f.get("answer", "")).strip()) and f["confidence"] == f["confidence"]
            and bool(str(f.get("self_reflection", "")).strip()))


def elicit_with_ladder(client, rec, images=None):
    # Recovery ladder (uniform, every rung logged; QC reports the rung rates). SHARED by the text
    # (run_reflect) and multimodal (run_reflect_mmmu) elicitation — one implementation, no forks:
    #  rung 1: normal single pass.
    #  rung 2 (salvaged=1): response ran past the output budget before the Answer/Confidence lines ->
    #          ONE continuation call forces the final three lines WITHOUT redoing or discarding the
    #          produced reasoning (no information loss).
    #  rung 3 (retry_concise=1): last resort for items that failed BOTH passes (a second long response
    #          or an empty/errored first response) -> one fresh attempt at temp 1.0 with a hard
    #          convergence demand. A bounded retry beats an unusable episode; the concision hint here
    #          is a recovery mechanism, not part of the elicitation design.
    from common import fit_context
    try:
        raw = client.generate(build_prompt(rec), system=SYSTEM, images=images)
    except Exception as e:
        # e.g. persistent 499 CANCELLED: a long response that outruns the SERVER's own
        # deadline on every identical greedy attempt. Fall through to rung 3 (temp-1.0 escape)
        # instead of losing the episode.
        print(f"[reflect] rung-1 exception on {rec.id}: {e} -> falling to rung 3", flush=True)
        raw = ""
    f = parse(raw, rec)
    if not _ok(f) and (raw or "").strip():
        must = " (one of the option letters)" if rec.choices else ""
        # Bounded replay (head+tail, marked elision): feeding a 200k-char response back verbatim
        # re-triggers the loop. This is the recovery layer, not a method signal — the FULL
        # reasoning is still stored in the episode untouched.
        cont = (build_prompt(rec)
                + "\n\nYour reasoning so far (it ran long and was cut off):\n"
                + fit_context(raw, budget_chars=60_000, label=f"salvage:{rec.id}")
                + "\n\nDo NOT reason further. Based on the reasoning above, output ONLY the three "
                "final lines now, in the exact format. You MUST output an Answer line with your "
                f"single best answer{must}, even if unsure (state doubts in the Self-reflection).\n"
                "Answer: <your single best answer>\n"
                "Self-reflection: <critically assess the reasoning above: the shakiest step, whether "
                "another answer is defensible>\n"
                "Confidence: <integer 0-100 that your Answer is correct>")
        f2 = parse(client.generate(cont, system=SYSTEM, images=images, force_no_think=True), rec)
        if _ok(f2):
            # the continuation may commit to a DIFFERENT answer than rung 1's partial attempt;
            # when it does, the stored (rung-1) reasoning argues for an answer other than the
            # graded one — mark it so the rate is measurable post-hoc
            if f.get("answer") and f2.get("answer") and str(f["answer"]).strip() != str(f2["answer"]).strip():
                f2["salvage_answer_flipped"] = 1
            if f.get("reasoning"):
                f2["reasoning"] = f["reasoning"]
                f2["reason_len"] = f["reason_len"]
            f = f2
            f["salvaged"] = 1
    n_last_resort = 2 if os.environ.get("XCONF_THINKING", "off") == "on" else 1
    # think-ablation arm: thinking-on models intermittently drop format labels in the
    # visible channel (private deliberation -> prose summary). One extra bounded rung-3 attempt for
    # that arm only; recovery layer, logged via retry_concise count.
    for _attempt in range(n_last_resort):
        if _ok(f):
            break
        must = " (one of the option letters)" if rec.choices else ""
        retry = (build_prompt(rec)
                 + "\n\n(IMPORTANT: your previous attempts failed to converge. Keep Reasoning under "
                 "15 sentences this time. You MUST end with the Answer"
                 f"{must}, Self-reflection, and Confidence lines.)")
        # temperature 1.0: at temp 0 the same question falls into the SAME deterministic loop
        # every time — sampling breaks it. Flagged.
        f3 = parse(client.generate(retry, system=SYSTEM, temperature=1.0, images=images, force_no_think=True), rec)
        if _ok(f3):
            f = f3
            f["retry_concise"] = 1 + _attempt
    f["id"] = rec.id
    return f


def work(client, rec):
    return elicit_with_ladder(client, rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="mmlu_pro")
    ap.add_argument("--pred-dir", default="data/predictions")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cfg.model.name = args.model
    cfg.model.max_output_tokens = int(os.environ.get("XCONF_MAXTOK", "65536"))  # flash max; uncapped reasoning needs headroom
    cfg.model.max_workers = int(os.environ.get("XCONF_WORKERS", "64"))
    md = os.path.join(args.pred_dir, args.model.replace("/", "_"))
    records = read_records(os.path.join(md, f"{args.dataset}.jsonl"))
    records.sort(key=lambda r: r.order_index)
    records = records[: args.limit]
    out_path = os.path.join(md, f"{args.dataset}.episodes_reflect.jsonl")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [r for r in records if r.id not in done]
    print(f"[reflect] {len(records)} records, {len(done)} cached, {len(todo)} to do")

    client = build_client(cfg)
    # STREAMING writer: one pool over ALL items, write each episode as it completes.
    fout = open(out_path, "a", encoding="utf-8")
    done_n = 0
    try:
        with ThreadPoolExecutor(max_workers=cfg.model.max_workers) as pool:
            futs = {pool.submit(work, client, r): r for r in todo}
            for fut in tqdm(as_completed(futs), total=len(todo), desc="reflect"):
                r = futs[fut]
                try:
                    row = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"[reflect] FAILED {r.id}: {e}", file=sys.stderr)
                    continue
                fout.write(json.dumps(row) + "\n")
                done_n += 1
                if done_n % 50 == 0:
                    fout.flush()
        fout.flush()
    finally:
        fout.close()
    rows_all = [json.loads(l) for l in open(out_path)]
    acc = sum(x["correct"] for x in rows_all) / max(1, len(rows_all))
    cpar = sum(1 for x in rows_all if x["confidence"] == x["confidence"]) / max(1, len(rows_all))
    print(f"[reflect] wrote {len(rows_all)} -> {out_path}  (CoT acc={acc:.3f}, conf_parsed={cpar:.2f})")


if __name__ == "__main__":
    main()
