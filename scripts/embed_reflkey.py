#!/usr/bin/env python3
"""Embed self-reflections as the reasoning/cue retrieval key (incremental, resumable)."""
from __future__ import annotations
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xconf.config import Config
from xconf.embed import Embedder

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemini-2.5-flash")
ap.add_argument("--dataset", default="mmlu_pro")
ap.add_argument("--pred-dir", default="data/predictions")
ap.add_argument("--config", default=None)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--tag", default="", help="suffix: reads episodes_reflect<tag>.jsonl -> reflkey<tag>.emb.jsonl")
args = ap.parse_args()

cfg = Config.load(args.config); cfg.model.name = args.model
md = os.path.join(args.pred_dir, args.model.replace("/", "_"))
rf = [json.loads(l) for l in open(os.path.join(md, f"{args.dataset}.episodes_reflect{args.tag}.jsonl")) if l.strip()]
out = os.path.join(md, f"{args.dataset}.reflkey{args.tag}.emb.jsonl")
done = set()
if os.path.exists(out):
    for l in open(out):
        if l.strip(): done.add(json.loads(l)["id"])
todo = [x for x in rf if x["id"] not in done]
print(f"[reflkey] {len(rf)} reflections, {len(done)} cached, {len(todo)} to do", flush=True)
emb = Embedder(cfg)
fout = open(out, "a", encoding="utf-8")
for i in range(0, len(todo), args.batch):
    chunk = todo[i:i+args.batch]
    texts = [f"Reasoning and uncertainty: {x.get('self_reflection','')}" for x in chunk]
    vecs = emb.embed(texts)
    for x, v in zip(chunk, vecs):
        fout.write(json.dumps({"id": x["id"], "embedding": v.astype(float).tolist()}) + "\n")
    fout.flush()
    print(f"[reflkey] {min(i+args.batch,len(todo))}/{len(todo)}", flush=True)
fout.close()
print(f"[reflkey] wrote {len(todo)} -> {out}", flush=True)
