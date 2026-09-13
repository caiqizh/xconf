#!/usr/bin/env python3
"""Ingest a dataset into the Stage-A records JSONL (the streaming order is fixed here).

Loads examples via the registered loader, shuffles them into a single stream (seed),
assigns order_index = stream position, and writes placeholder PredictionRecords
(answer/correct filled later by run_reflect + grading). Output:
data/predictions/<model>/<dataset>.jsonl
"""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xconf.datasets import load_examples
from xconf.records import PredictionRecord, write_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--pred-dir", default="data/predictions")
    ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    split = args.split or ("train" if args.dataset in ("gpqa", "math", "medmcqa", "supergpqa", "olympiadbench") else "test")
    exs = load_examples(args.dataset, split=split, n=args.limit, seed=args.seed)
    recs = []
    for oi, ex in enumerate(exs):
        recs.append(PredictionRecord(
            id=ex.id, dataset=ex.dataset, order_index=oi, question=ex.question,
            gold=ex.gold, answer="", correct=0, verbalized_confidence=None,
            reasoning="", raw_response="", model=args.model, choices=ex.choices,
            subject=ex.subject, parse_ok=False))
    md = os.path.join(args.pred_dir, args.model.replace("/", "_"))
    os.makedirs(md, exist_ok=True)
    out = os.path.join(md, f"{args.dataset}.jsonl")
    n = write_records(out, recs)
    nmc = sum(1 for r in recs if r.choices)
    print(f"[ingest] {args.dataset}: wrote {n} records -> {out}  (MC={nmc}, free-form={n-nmc})")


if __name__ == "__main__":
    main()
