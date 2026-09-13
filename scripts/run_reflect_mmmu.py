#!/usr/bin/env python3
"""Multimodal elicitation for MMMU-Pro (standard, 10 options) — the only modality-specific step.

Passes the question's images to gemini-2.5-flash alongside the text prompt; everything downstream
(embeddings, post-hoc, in-context, eval) is identical to the text datasets because the retrieval keys
are the model's REASONING/REFLECTION text, which is modality-agnostic. Writes both the text records
(mmmu_pro.jsonl) and the episodes (mmmu_pro.episodes_reflect.jsonl), so the existing pipeline scripts
run unchanged with --dataset mmmu_pro. MC -> verify_mc (no LLM grading).
"""
from __future__ import annotations
import os, sys, json, argparse, ast, random
from io import BytesIO
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from xconf.config import Config
from xconf.llm import build_client
from xconf.records import PredictionRecord, write_records
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_reflect import elicit_with_ladder


def to_png_bytes(img):
    buf = BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--config", default=None)
    ap.add_argument("--pred-dir", default="data/predictions")
    ap.add_argument("--limit", type=int, default=1730)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config.load(args.config); cfg.model.name = args.model
    cfg.model.max_output_tokens = int(os.environ.get("XCONF_MAXTOK", "65536"))  # match run_reflect (uncapped reasoning needs headroom)
    from datasets import load_dataset
    ds = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")
    idx = list(range(len(ds))); random.Random(args.seed).shuffle(idx); idx = idx[: args.limit]

    md = os.path.join(args.pred_dir, args.model.replace("/", "_")); os.makedirs(md, exist_ok=True)
    rec_path = os.path.join(md, "mmmu_pro.jsonl")
    # build + persist text records (so downstream scripts work unchanged)
    recs = []
    rowmap = {}
    for oi, i in enumerate(idx):
        row = ds[i]
        opts = row["options"]
        if isinstance(opts, str):
            try: opts = ast.literal_eval(opts)
            except Exception: opts = [opts]
        rid = f"mmmu_pro/{i:06d}"
        recs.append(PredictionRecord(
            id=rid, dataset="mmmu_pro", order_index=oi, question=row["question"],
            gold=str(row["answer"]).strip().upper(), answer="", correct=0, verbalized_confidence=None,
            reasoning="", raw_response="", model=args.model, choices=list(opts),
            subject=row.get("subject"), parse_ok=False))
        rowmap[rid] = (row, list(opts))
    write_records(rec_path, recs)
    print(f"[mmmu] wrote {len(recs)} records -> {rec_path}")

    out_path = os.path.join(md, "mmmu_pro.episodes_reflect.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            if l.strip(): done.add(json.loads(l)["id"])
    todo = [r for r in recs if r.id not in done]
    print(f"[mmmu] {len(recs)} recs, {len(done)} cached, {len(todo)} to do")
    client = build_client(cfg)

    def work(r):
        row, opts = rowmap[r.id]
        imgs = []
        img_fail = 0
        for k in range(1, 8):
            im = row.get(f"image_{k}")
            if im is not None:
                try: imgs.append((to_png_bytes(im), "image/png"))
                except Exception:
                    # a silently dropped image turns this into a text-only
                    # elicitation banked as genuine — count and flag instead of hiding it
                    img_fail += 1
        if img_fail:
            print(f"[mmmu] {r.id}: {img_fail} image(s) failed to decode", flush=True)
        rec = SimpleNamespace(id=r.id, dataset="mmmu_pro", question=r.question, choices=opts, gold=r.gold)
        # shared 3-rung recovery ladder from run_reflect (salvage + temp-1.0 escape), images included
        f = elicit_with_ladder(client, rec, images=imgs or None)
        if img_fail:
            f["img_decode_failures"] = img_fail
        f["id"] = r.id; return f

    fout = open(out_path, "a", encoding="utf-8")
    try:
        for c0 in tqdm(range(0, len(todo), args.chunk), desc="mmmu"):
            batch = todo[c0:c0 + args.chunk]; rows = {}
            with ThreadPoolExecutor(max_workers=cfg.model.max_workers) as pool:
                futs = {pool.submit(work, r): r for r in batch}
                for fut in as_completed(futs):
                    rr = futs[fut]
                    try: rows[rr.id] = fut.result()
                    except Exception as e: print("FAIL", rr.id, e, file=sys.stderr)
            for r in batch:
                if r.id in rows: fout.write(json.dumps(rows[r.id]) + "\n")
            fout.flush()
    finally:
        fout.close()
    allr = [json.loads(l) for l in open(out_path)]
    acc = sum(x["correct"] for x in allr) / max(1, len(allr))
    print(f"[mmmu] wrote {len(allr)} episodes (CoT acc={acc:.3f})")


if __name__ == "__main__":
    main()
