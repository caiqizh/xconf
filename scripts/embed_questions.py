#!/usr/bin/env python3
"""Embed QUESTION-ONLY text as the one-pass retrieval key (resumable, batched).

One-pass retrieval keys on the question (the reflection doesn't exist yet), so the
key embedding must NOT include any model output. Writes <dataset>.emb.jsonl.
"""
from __future__ import annotations
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xconf.config import Config
from xconf.embed import Embedder
from xconf.records import read_records

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemini-2.5-flash")
ap.add_argument("--dataset", required=True)
ap.add_argument("--pred-dir", default="data/predictions")
ap.add_argument("--config", default=None)
ap.add_argument("--batch", type=int, default=64)
args = ap.parse_args()

cfg = Config.load(args.config); cfg.model.name = args.model
md = os.path.join(args.pred_dir, args.model.replace("/", "_"))
recs = read_records(os.path.join(md, f"{args.dataset}.jsonl"))
out = os.path.join(md, f"{args.dataset}.emb.jsonl")
done = set()
if os.path.exists(out):
    for l in open(out):
        if l.strip(): done.add(json.loads(l)["id"])
todo = [r for r in recs if r.id not in done]
print(f"[emb] {args.dataset}: {len(recs)} records, {len(done)} cached, {len(todo)} to do", flush=True)
emb = Embedder(cfg)
fout = open(out, "a", encoding="utf-8")
for i in range(0, len(todo), args.batch):
    chunk = todo[i:i + args.batch]
    texts = [r.retrieval_text(include_output=False) for r in chunk]  # QUESTION ONLY
    vecs = emb.embed(texts)
    for r, v in zip(chunk, vecs):
        fout.write(json.dumps({"id": r.id, "embedding": v.astype(float).tolist()}) + "\n")
    fout.flush()
    print(f"[emb] {min(i + args.batch, len(todo))}/{len(todo)}", flush=True)
fout.close()
print(f"[emb] wrote {len(todo)} -> {out}", flush=True)
