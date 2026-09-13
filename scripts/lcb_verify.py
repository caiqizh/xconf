#!/usr/bin/env python3
"""Verify LiveCodeBench answers by LOCAL unit-test execution (ground truth, no LLM).

Reads:  <md>/lcb.episodes_reflect.jsonl  (answer = the model's full code block)
        <md>/lcb.tests.jsonl             (sidecar from ingest_lcb.py, streamed -- the
                                          private tests total ~GBs, never all in RAM)
Writes: <md>/lcb.graded.jsonl sidecar (per-id verdict + FULL first-failure detail:
        input, expected, got, stderr -- stored untruncated), then rewrites `correct`
        into the episodes file in place, exactly like grade_freeform.py.

Execution sandbox (per test, no docker): a fresh subprocess in its own session +
temp cwd, with resource rlimits (address space, CPU seconds, file size, stack) set
via preexec_fn; wall-clock timeout enforced by killing the whole process group.
This is crash/loop/memory-safe for benchmark code, but it is NOT a security boundary
against genuinely malicious code (fine here: the code under test is the model's own
answer to public contest problems).

Test semantics (per LCB's format):
  stdin      -- run the answer as a standalone script; feed `input`; compare stdout to
                `output` by stripped-line equality, falling back to token-wise float
                comparison (tol 1e-6).
  functional -- LeetCode-style: a harness execs the answer (with LCB's conventional
                import prelude + typing.*), resolves Solution().<func_name> (or a bare
                function), parses each input LINE as one JSON argument, calls the
                function, and prints the JSON result; compared with float tolerance.
All tests must pass -> correct=1 (public first, then private; early-exit on failure).
"""
from __future__ import annotations

import os
import re
import sys
import json
import base64
import pickle
import shutil
import signal
import zlib
import argparse
import tempfile
import resource
import subprocess
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait

MARKER = "___LCB_OUT___"

# LCB's conventional import prelude: LeetCode starter code assumes List/Optional etc.
# exist, and reference solutions lean on the usual competitive-programming stdlib.
HARNESS = r'''
import sys, json
import time, itertools, collections, math, fractions, heapq, bisect, string, re, random, functools, operator, copy
from itertools import accumulate, product, permutations, combinations
from collections import Counter, OrderedDict, deque, defaultdict, ChainMap
from functools import lru_cache, cache, reduce
from math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2, inf
from heapq import heappush, heappop, heapify, heappushpop, nlargest, nsmallest
from bisect import bisect_left, bisect_right, insort
from typing import *

sys.setrecursionlimit(30000)

def _norm(o):
    if isinstance(o, tuple) or isinstance(o, list):
        return [_norm(x) for x in o]
    if isinstance(o, set):
        return sorted(_norm(x) for x in o)
    if isinstance(o, dict):
        return {k: _norm(v) for k, v in o.items()}
    return o

def _main():
    sol_path, func_name, spec_path = sys.argv[1], sys.argv[2], sys.argv[3]
    ns = dict(globals())
    ns["__name__"] = "__lcb_solution__"
    with open(sol_path) as f:
        code = f.read()
    exec(compile(code, "solution.py", "exec"), ns)
    fn = None
    if "Solution" in ns:
        try:
            fn = getattr(ns["Solution"](), func_name, None)
        except Exception:
            fn = None
    if fn is None:
        fn = ns.get(func_name)
    if fn is None or not callable(fn):
        print("function %r not found in answer" % func_name, file=sys.stderr)
        sys.exit(3)
    with open(spec_path) as f:
        spec = json.load(f)
    args = [json.loads(l) for l in spec["input"].split("\n") if l.strip() != ""]
    out = fn(*args)
    sys.stdout.write("\n" + "___LCB_OUT___" + json.dumps(_norm(out)))

_main()
'''

_ENV = dict(os.environ, MPLBACKEND="Agg", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")


def run_limited(cmd, input_text, timeout, mem_gb, cwd):
    """Subprocess with rlimits + own session; wall timeout kills the process group.
    Returns (returncode, stdout, stderr, timed_out)."""
    mem = int(mem_gb * (1 << 30))
    cpu = int(timeout) + 3

    def _pre():
        for rl, lim in ((resource.RLIMIT_AS, (mem, mem)),
                        (resource.RLIMIT_CPU, (cpu, cpu)),
                        (resource.RLIMIT_FSIZE, (1 << 28, 1 << 28)),
                        (resource.RLIMIT_STACK, (64 << 20, resource.RLIM_INFINITY))):
            try:
                resource.setrlimit(rl, lim)
            except (ValueError, OSError):
                pass

    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, cwd=cwd, env=_ENV,
                         preexec_fn=_pre, start_new_session=True)
    try:
        out, err = p.communicate(input=input_text, timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            p.kill()
        out, err = p.communicate()
        timed_out = True
    except BrokenPipeError:  # child died before reading all stdin
        out, err = "", "broken pipe (child exited early)"
        timed_out = False
        p.wait()
    return p.returncode, out or "", err or "", timed_out


def decode_private(s: str):
    """Upstream encoding: plain JSON for early shards; base64+zlib+pickle(json) later."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(s.encode("utf-8")))))


def _cmp_stdout(got: str, exp: str) -> bool:
    gl = [l.rstrip() for l in got.strip().split("\n")] if got.strip() else []
    el = [l.rstrip() for l in exp.strip().split("\n")] if exp.strip() else []
    if gl == el:
        return True
    gt, et = got.split(), exp.split()
    if len(gt) != len(et):
        return False
    for a, b in zip(gt, et):
        if a == b:
            continue
        try:
            fa, fb = float(a), float(b)
        except ValueError:
            return False
        if not (fa == fb or abs(fa - fb) <= 1e-6 * max(1.0, abs(fb))):
            return False
    return True


def _feq(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b or abs(a - b) <= 1e-6 * max(1.0, abs(b))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_feq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_feq(a[k], b[k]) for k in a)
    return a == b


def grade_one(ep: dict, trow: dict, timeout: float, mem_gb: float) -> dict:
    rid = ep["id"]
    code = (ep.get("answer") or "")
    if not code.strip():
        return {"id": rid, "correct": 0, "n_tests": 0, "n_passed": 0, "error": "empty answer"}
    tests = list(trow["public_test_cases"]) + list(decode_private(trow["private_test_cases"]))
    if not tests:
        return {"id": rid, "correct": 0, "n_tests": 0, "n_passed": 0, "error": "no tests in sidecar"}
    td = tempfile.mkdtemp(prefix="lcb_")
    try:
        sol = os.path.join(td, "sol.py")
        with open(sol, "w", encoding="utf-8") as f:
            f.write(code)
        harness = os.path.join(td, "harness.py")
        spec = os.path.join(td, "spec.json")
        if any(t.get("testtype") == "functional" for t in tests):
            with open(harness, "w", encoding="utf-8") as f:
                f.write(HARNESS)
        n_passed, fail = 0, None
        for ti, t in enumerate(tests):
            if t.get("testtype") == "functional":
                fname = trow.get("func_name")
                if not fname:
                    return {"id": rid, "correct": 0, "n_tests": len(tests), "n_passed": n_passed,
                            "error": "functional test but no func_name in sidecar"}
                with open(spec, "w", encoding="utf-8") as f:
                    json.dump({"input": t["input"]}, f)
                rc, out, errtxt, to = run_limited(
                    [sys.executable, harness, sol, fname, spec], "", timeout, mem_gb, td)
                ok, got = False, out
                if rc == 0 and MARKER in out:
                    got = out[out.rfind(MARKER) + len(MARKER):].strip()
                    try:
                        exp_obj = json.loads(t["output"])
                        ok = _feq(json.loads(got), exp_obj)
                    except (json.JSONDecodeError, ValueError):
                        ok = got.strip() == t["output"].strip()
            else:  # stdin
                rc, out, errtxt, to = run_limited(
                    [sys.executable, sol], t["input"], timeout, mem_gb, td)
                got = out
                ok = (rc == 0) and _cmp_stdout(out, t["output"])
            if ok:
                n_passed += 1
                continue
            # FULL failure detail stored (no truncation of test outputs)
            fail = {"test_index": ti, "testtype": t.get("testtype"), "timeout": to,
                    "returncode": rc, "input": t["input"], "expected": t["output"],
                    "got": got, "stderr": errtxt}
            break
        return {"id": rid, "correct": int(fail is None), "n_tests": len(tests),
                "n_passed": n_passed, "fail": fail}
    finally:
        shutil.rmtree(td, ignore_errors=True)


_ID_HEAD = re.compile(r'^\{"id":\s*"([^"]+)"')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--dataset", default="lcb")
    ap.add_argument("--pred-dir", default="data/predictions")
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds per test")
    ap.add_argument("--mem-gb", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ids", default=None, help="comma-separated ids (smoke tests)")
    args = ap.parse_args()

    md = os.path.join(args.pred_dir, args.model.replace("/", "_"))
    ep_path = os.path.join(md, f"{args.dataset}.episodes_reflect.jsonl")
    eps = [json.loads(l) for l in open(ep_path) if l.strip()]
    sidecar = os.path.join(md, f"{args.dataset}.graded.jsonl")
    done = {}
    if os.path.exists(sidecar):
        for l in open(sidecar):
            if l.strip():
                o = json.loads(l)
                done[o["id"]] = o
    want = set(args.ids.split(",")) if args.ids else None
    todo = {e["id"]: e for e in eps
            if e["id"] not in done and (want is None or e["id"] in want)}
    print(f"[lcb_verify] {len(eps)} episodes, {len(done)} cached, {len(todo)} to grade")

    tests_path = os.path.join(md, f"{args.dataset}.tests.jsonl")
    fs = open(sidecar, "a", encoding="utf-8")
    n_done = 0

    def _drain(pending, block):
        nonlocal n_done
        if not pending:
            return pending
        finished, rest = wait(pending, return_when=FIRST_COMPLETED) if block else wait(pending, timeout=0)
        for fut in finished:
            try:
                row = fut.result()
            except Exception as ex:  # noqa: BLE001
                print(f"[lcb_verify] grader crashed: {ex}", file=sys.stderr)
                continue
            done[row["id"]] = row
            fs.write(json.dumps(row, ensure_ascii=False) + "\n")
            fs.flush()
            n_done += 1
            tag = "PASS" if row["correct"] else "fail"
            print(f"[lcb_verify] {row['id']}: {tag} ({row.get('n_passed', 0)}/{row.get('n_tests', 0)} tests)"
                  + (f" err={row['error']}" if row.get("error") else ""), flush=True)
        return rest

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = set()
            seen = set()
            # stream the multi-GB sidecar; only fully parse lines we need
            with open(tests_path, "r", encoding="utf-8") as tf:
                for line in tf:
                    m = _ID_HEAD.match(line)
                    if not m or m.group(1) not in todo:
                        continue
                    trow = json.loads(line)
                    seen.add(trow["id"])
                    pending.add(pool.submit(grade_one, todo[trow["id"]], trow,
                                            args.timeout, args.mem_gb))
                    while len(pending) >= args.workers * 2:
                        pending = _drain(pending, block=True)
            while pending:
                pending = _drain(pending, block=True)
        missing = set(todo) - seen
        if missing:
            print(f"[lcb_verify] WARNING: {len(missing)} episode ids missing from tests sidecar "
                  f"(e.g. {sorted(missing)[:3]})", file=sys.stderr)
    finally:
        fs.close()

    # rewrite episodes with test-verified correctness (same pattern as grade_freeform.py)
    for e in eps:
        if e["id"] in done:
            e["correct"] = int(done[e["id"]]["correct"])
    with open(ep_path, "w", encoding="utf-8") as f:
        for e in eps:
            f.write(json.dumps(e) + "\n")
    graded = [e for e in eps if e["id"] in done]
    acc = sum(e["correct"] for e in graded) / max(1, len(graded))
    n_to = sum(1 for r in done.values() if (r.get("fail") or {}).get("timeout"))
    print(f"[lcb_verify] graded {len(graded)}/{len(eps)} (this run: {n_done}); "
          f"acc={acc:.3f}; first-fail-was-timeout on {n_to}  (updated {ep_path})")


if __name__ == "__main__":
    main()
