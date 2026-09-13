#!/usr/bin/env python3
"""Ingest LiveCodeBench (code_generation_lite, release_v6) as a REASONING-LINE dataset.

The HF repo `livecodebench/code_generation_lite` is a script-based dataset (its loader
script is unsupported by datasets>=4), so we download the underlying jsonl shards
directly and reproduce the release mapping from the upstream loader:
release_v6 = test.jsonl + test2..test6.jsonl (~1055 problems, contests 2023-05..2025-04).

Question record (PredictionRecord, data/predictions/<model>/lcb.jsonl):
  question = FULL problem statement (+ starter code for LeetCode-style functional
             problems / a stdin-stdout note for contest problems) -- no truncation.
  choices  = None; gold = "" (correctness comes from local unit-test execution by
             lcb_verify.py, never from string match or an LLM judge).
  subject  = "<platform>/<difficulty>" (metadata only, never a retrieval key).

Tests sidecar (lcb.tests.jsonl, keyed by id; NEVER model-visible):
  public_test_cases  = decoded list [{input, output, testtype}]
  private_test_cases = the upstream string VERBATIM (plain JSON for early shards;
        base64+zlib+pickle for the later, larger shards). Kept encoded at rest so the
        sidecar stays ~4.5GB instead of ~3x that; lcb_verify.py decodes with the exact
        upstream recipe. Lossless compression of never-model-visible data -- NOT
        truncation.
  func_name (from metadata) for functional tests; platform/difficulty/contest_date.
"""
from __future__ import annotations
import os, sys, json, random, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xconf.records import PredictionRecord, write_records

RELEASE_FILES = {  # mirrors ALLOWED_FILES in the upstream code_generation_lite.py loader
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
}


def build_question(row: dict) -> str:
    q = row["question_content"].rstrip()
    starter = (row.get("starter_code") or "").strip("\n")
    if starter.strip():
        q += (
            "\n\nYour solution must complete the following starter code, keeping this exact "
            "class/method signature (the grader calls it directly):\n"
            "```python\n" + starter + "\n```"
        )
    else:
        q += (
            "\n\nYour solution must be a standalone Python program that reads the input from "
            "stdin and writes the answer to stdout."
        )
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--pred-dir", default="data/predictions")
    ap.add_argument("--release", default="release_v6", choices=sorted(RELEASE_FILES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    rows = []
    for fn in RELEASE_FILES[args.release]:
        p = hf_hub_download("livecodebench/code_generation_lite", fn, repo_type="dataset")
        n0 = len(rows)
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        print(f"[ingest_lcb] {fn}: {len(rows) - n0} problems")
    ids = [r["question_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate question_id across shards"

    # same streaming-order convention as every other loader: shuffle indices by seed
    idx = list(range(len(rows)))
    random.Random(args.seed).shuffle(idx)
    if args.limit is not None:
        idx = idx[: args.limit]

    md = os.path.join(args.pred_dir, args.model.replace("/", "_"))
    os.makedirs(md, exist_ok=True)
    recs, sidecar = [], []
    for oi, i in enumerate(idx):
        row = rows[i]
        rid = f"lcb/{row['question_id']}"
        meta = json.loads(row.get("metadata") or "{}")
        recs.append(PredictionRecord(
            id=rid, dataset="lcb", order_index=oi, question=build_question(row),
            gold="", answer="", correct=0, verbalized_confidence=None,
            reasoning="", raw_response="", model=args.model, choices=None,
            subject=f"{row['platform']}/{row['difficulty']}", parse_ok=False))
        sidecar.append({
            "id": rid,
            "question_id": row["question_id"],
            "platform": row["platform"],
            "difficulty": row["difficulty"],
            "contest_date": row["contest_date"],
            "func_name": meta.get("func_name"),
            "starter_code": row.get("starter_code") or "",
            "public_test_cases": json.loads(row["public_test_cases"]),
            # verbatim upstream string (JSON or base64+zlib+pickle); decoded by lcb_verify
            "private_test_cases": row["private_test_cases"],
        })

    out = os.path.join(md, "lcb.jsonl")
    n = write_records(out, recs)
    tpath = os.path.join(md, "lcb.tests.jsonl")
    with open(tpath, "w", encoding="utf-8") as f:
        for s in sidecar:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    n_func = sum(1 for s in sidecar if s["func_name"])
    print(f"[ingest_lcb] {args.release}: wrote {n} records -> {out}")
    print(f"[ingest_lcb] tests sidecar -> {tpath}  (functional={n_func}, stdin={n - n_func})")


if __name__ == "__main__":
    main()
