"""Dataset loaders -> a unified Example stream.

Subjects are carried as metadata only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from .verify import gsm8k_gold


@dataclass
class Example:
    id: str
    dataset: str
    question: str
    gold: str  # canonical gold (letter for MC, number for math)
    choices: Optional[list[str]] = None
    subject: Optional[str] = None


def load_mmlu(split: str = "test", n: Optional[int] = None, seed: int = 0) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    if n is not None:
        idx = idx[:n]
    out: list[Example] = []
    for i in idx:
        row = ds[i]
        choices = list(row["choices"])
        gold_letter = chr(65 + int(row["answer"]))
        out.append(
            Example(
                id=f"mmlu/{split}/{i:06d}",
                dataset="mmlu",
                question=row["question"],
                gold=gold_letter,
                choices=choices,
                subject=row.get("subject"),
            )
        )
    return out


def load_mmlu_pro(split: str = "test", n: Optional[int] = None, seed: int = 0) -> list[Example]:
    """MMLU-Pro: harder MMLU with up to 10 options, still clean exact-match MC.

    A good fit when the model is near-ceiling on vanilla MMLU (more errors -> more
    calibration signal) while staying in the zero-ambiguity verification regime.
    """
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    if n is not None:
        idx = idx[:n]
    out: list[Example] = []
    for i in idx:
        row = ds[i]
        choices = list(row["options"])
        gold_letter = row["answer"].strip().upper()  # already a letter A..J
        out.append(
            Example(
                id=f"mmlu_pro/{split}/{i:06d}",
                dataset="mmlu_pro",
                question=row["question"],
                gold=gold_letter,
                choices=choices,
                subject=row.get("category"),
            )
        )
    return out


def load_gsm8k(split: str = "test", n: Optional[int] = None, seed: int = 0) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    if n is not None:
        idx = idx[:n]
    out: list[Example] = []
    for i in idx:
        row = ds[i]
        out.append(
            Example(
                id=f"gsm8k/{split}/{i:06d}",
                dataset="gsm8k",
                question=row["question"],
                gold=gsm8k_gold(row["answer"]),
                choices=None,
                subject=None,
            )
        )
    return out


def load_gpqa(split: str = "train", n: Optional[int] = None, seed: int = 0,
              config: str = "gpqa_extended") -> list[Example]:
    """GPQA: graduate-level science MC (4 options). Hard enough that gemini-2.5-flash
    drops well below ceiling, creating REAL uncertainty to calibrate. Options are
    shuffled per-question (deterministic by seed+index) so the gold letter is not
    positionally biased. gpqa_extended ~546 (>=500); main=448, diamond=198.
    """
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", config, split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    if n is not None:
        idx = idx[:n]
    out: list[Example] = []
    for i in idx:
        row = ds[i]
        correct = str(row["Correct Answer"]).strip()
        opts = [correct] + [str(row[f"Incorrect Answer {j}"]).strip() for j in (1, 2, 3)]
        random.Random(seed * 100003 + i).shuffle(opts)  # per-question option order
        gold_letter = chr(65 + opts.index(correct))
        out.append(
            Example(
                id=f"gpqa/{config}/{i:06d}",
                dataset="gpqa",
                question=row["Question"],
                gold=gold_letter,
                choices=opts,
                subject=row.get("High-level domain"),
            )
        )
    return out


def load_simpleqa(split: str = "test", n: Optional[int] = None, seed: int = 0) -> list[Example]:
    """SimpleQA: short-answer factual QA, the calibration benchmark ('does the model
    know what it knows'). Free-form -> graded by an LLM against the known gold answer
    (ground truth, not a self-judge). No options.
    """
    from datasets import load_dataset

    ds = load_dataset("basicv8vc/SimpleQA", split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    if n is not None:
        idx = idx[:n]
    out: list[Example] = []
    for i in idx:
        row = ds[i]
        meta = row.get("metadata") or {}
        topic = meta.get("topic") if isinstance(meta, dict) else None
        out.append(
            Example(
                id=f"simpleqa/{i:06d}",
                dataset="simpleqa",
                question=row["problem"],
                gold=str(row["answer"]).strip(),
                choices=None,
                subject=topic,
            )
        )
    return out


def load_hle(split: str = "test", n: Optional[int] = None, seed: int = 0) -> list[Example]:
    """HLE (Humanity's Last Exam): extremely hard, frontier models <20%. We keep only
    TEXT-ONLY questions (drop image/multimodal) since the model runs text-only. Free-form
    exactMatch / multipleChoice -> graded by an LLM against the gold answer.
    """
    from datasets import load_dataset

    ds = load_dataset("cais/hle", split=split)
    text_idx = [i for i in range(len(ds)) if not (ds[i].get("image") or "").strip()]
    random.Random(seed).shuffle(text_idx)
    if n is not None:
        text_idx = text_idx[:n]
    out: list[Example] = []
    for i in text_idx:
        row = ds[i]
        out.append(
            Example(
                id=f"hle/{row['id']}",
                dataset="hle",
                question=row["question"],
                gold=str(row["answer"]).strip(),
                choices=None,  # MC options are inline in the question text for HLE
                subject=row.get("category"),
            )
        )
    return out


def load_bbh(split: str = "test", n: Optional[int] = None, seed: int = 0) -> list[Example]:
    """BIG-Bench Hard: 27 diverse reasoning tasks. Short answers (letter / bool / word / number)
    but the questions need CoT -> reasoning-heavy. Graded free-form (target format varies by task).
    """
    from datasets import load_dataset, get_dataset_config_names

    cfgs = get_dataset_config_names("lukaemon/bbh")
    items = []
    for c in cfgs:
        ds = load_dataset("lukaemon/bbh", c, split="test")
        for j in range(len(ds)):
            row = ds[j]
            items.append((c, j, row["input"], str(row["target"]).strip()))
    random.Random(seed).shuffle(items)
    if n is not None:
        items = items[:n]
    out: list[Example] = []
    for c, j, q, tgt in items:
        out.append(Example(id=f"bbh/{c}/{j:04d}", dataset="bbh", question=q, gold=tgt,
                           choices=None, subject=c))
    return out


def load_math(split: str = "train", n: Optional[int] = None, seed: int = 0) -> list[Example]:
    """Hendrycks MATH: competition math, structured by subject + difficulty level. 12000 train
    examples. Symbolic answers (fractions, tuples, ...) -> graded free-form against the gold
    `answer` field.
    """
    from datasets import load_dataset

    ds = load_dataset("nlile/hendrycks-MATH-benchmark", split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    if n is not None:
        idx = idx[:n]
    out: list[Example] = []
    for i in idx:
        row = ds[i]
        out.append(Example(id=f"math/{i:06d}", dataset="math", question=row["problem"],
                           gold=str(row["answer"]).strip(), choices=None, subject=row.get("subject")))
    return out


def load_medmcqa(split: str = "train", n: Optional[int] = None, seed: int = 0) -> list[Example]:
    """MedMCQA: ~182k medical entrance-exam MC (4 options), structured by 21 subjects.
    Clean MC verification.
    """
    from datasets import load_dataset

    ds = load_dataset("openlifescienceai/medmcqa", split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    if n is not None:
        idx = idx[:n]
    out: list[Example] = []
    for i in idx:
        row = ds[i]
        ch = [row["opa"], row["opb"], row["opc"], row["opd"]]
        out.append(Example(id=f"medmcqa/{i:06d}", dataset="medmcqa", question=row["question"],
                           gold=chr(65 + int(row["cop"])), choices=ch, subject=row.get("subject_name")))
    return out


def load_agieval(split: str = "test", n: Optional[int] = None, seed: int = 0) -> list[Example]:
    """AGIEval: human admission/qualification exams (LSAT/SAT/GRE/Gaokao/civil-service). MC, graded
    free-form (formats vary by task). subject = the exam section.
    """
    from datasets import load_dataset

    tasks = ["aqua-rat", "logiqa-en", "lsat-ar", "lsat-lr", "lsat-rc",
             "sat-en", "sat-en-without-passage", "sat-math"]  # English MC sections
    items = []
    for t in tasks:
        try:
            ds = load_dataset(f"hails/agieval-{t}", split="test")
        except Exception:
            continue
        for j in range(len(ds)):
            r = ds[j]
            q = r.get("query"); ch = r.get("choices"); gold = r.get("gold")
            if not q or not ch or not gold:
                continue
            gi = gold[0] if isinstance(gold, list) else gold
            try:
                gi = int(gi)
            except Exception:
                continue
            if 0 <= gi < len(ch):
                items.append((f"agieval/{t}/{j:04d}", q, str(ch[gi]).strip(), t))
    random.Random(seed).shuffle(items)
    if n is not None:
        items = items[:n]
    return [Example(id=i, dataset="agieval", question=q, gold=g, choices=None, subject=s) for i, q, g, s in items]


def load_supergpqa(split: str = "train", n: Optional[int] = None, seed: int = 0) -> list[Example]:
    """SuperGPQA: 26.5k graduate-level MC (4-10 options) across 285 subfields. Structured by
    discipline/field/subfield. Clean letter-match MC verification. subject = discipline.
    """
    from datasets import load_dataset

    ds = load_dataset("m-a-p/SuperGPQA", split="train")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    if n is not None:
        idx = idx[:n]
    out: list[Example] = []
    for i in idx:
        row = ds[i]
        out.append(Example(
            id=f"supergpqa/{i:06d}", dataset="supergpqa", question=row["question"],
            gold=str(row["answer_letter"]).strip().upper(), choices=list(row["options"]),
            subject=row.get("discipline")))
    return out


def load_bbeh(split: str = "test", n: Optional[int] = None, seed: int = 0) -> list[Example]:
    """BIG-Bench Extra Hard: 23 tasks (~4520 examples) that replace each BBH task with a much harder
    variant (Gemini-2.0-Flash drops 85% -> 24%). Hosted on GitHub (not HF); fetched per task.json.
    Mixed short-answer targets -> graded free-form. subject = task.
    """
    import json as _json
    import urllib.request

    tasks = ["boardgame_qa", "boolean_expressions", "buggy_tables", "causal_understanding",
             "disambiguation_qa", "dyck_languages", "geometric_shapes", "hyperbaton", "linguini",
             "movie_recommendation", "multistep_arithmetic", "nycc", "object_counting",
             "object_properties", "sarc_triples", "shuffled_objects", "spatial_reasoning", "sportqa",
             "temporal_sequence", "time_arithmetic", "web_of_lies", "word_sorting", "zebra_puzzles"]
    base = "https://raw.githubusercontent.com/google-deepmind/bbeh/main/bbeh/benchmark_tasks"
    items = []
    for t in tasks:
        url = f"{base}/bbeh_{t}/task.json"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = _json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"[bbeh] skip {t}: {e}")
            continue
        for j, ex in enumerate(data.get("examples", [])):
            items.append((t, j, ex["input"], str(ex["target"]).strip()))
    random.Random(seed).shuffle(items)
    if n is not None:
        items = items[:n]
    return [Example(id=f"bbeh/{t}/{j:04d}", dataset="bbeh", question=q, gold=g, choices=None, subject=t)
            for t, j, q, g in items]


def load_olympiadbench(split: str = "train", n: Optional[int] = None, seed: int = 0) -> list[Example]:
    """OlympiadBench: olympiad math/physics. We keep ENGLISH, TEXT-ONLY, OPEN-ENDED (has final_answer,
    not proof, no figures). Graded free-form against final_answer. subject = Math/Physics.
    """
    from datasets import load_dataset

    configs = ["OE_TO_maths_en_COMP", "OE_TO_physics_en_COMP"]  # only EN text-only open-ended configs that exist
    items = []
    for c in configs:
        try:
            ds = load_dataset("Hothan/OlympiadBench", c, split="train")
        except Exception as e:  # noqa: BLE001
            print(f"[olympiadbench] skip {c}: {e}")
            continue
        for j in range(len(ds)):
            row = ds[j]
            if any(row.get(f"image_{k}") for k in range(1, 10)):
                continue  # text-only
            fa = row.get("final_answer")
            if not fa:
                continue
            gold = fa[0] if isinstance(fa, list) and fa else (fa if isinstance(fa, str) else None)
            if not gold:
                continue
            items.append((f"olympiadbench/{c}/{j:04d}", row["question"], str(gold).strip(), row.get("subject")))
    random.Random(seed).shuffle(items)
    if n is not None:
        items = items[:n]
    return [Example(id=i, dataset="olympiadbench", question=q, gold=g, choices=None, subject=s)
            for i, q, g, s in items]


LOADERS = {"mmlu": load_mmlu, "mmlu_pro": load_mmlu_pro, "gsm8k": load_gsm8k,
           "gpqa": load_gpqa, "simpleqa": load_simpleqa, "hle": load_hle, "bbh": load_bbh,
           "math": load_math, "medmcqa": load_medmcqa, "agieval": load_agieval,
           "supergpqa": load_supergpqa, "bbeh": load_bbeh, "olympiadbench": load_olympiadbench}


def load_examples(dataset: str, split: str = "test", n: Optional[int] = None, seed: int = 0) -> list[Example]:
    if dataset not in LOADERS:
        raise ValueError(f"unknown dataset '{dataset}'; have {list(LOADERS)}")
    return LOADERS[dataset](split=split, n=n, seed=seed)
