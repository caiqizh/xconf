"""Core data records and JSONL I/O.

A PredictionRecord is the single unit produced by Stage A and consumed by
Stage B. It holds everything needed to (a) reconstruct the retrieval key,
(b) know the ground-truth correctness, and (c) report verbalized confidence.

Embeddings are stored in a *separate* parallel JSONL keyed by ``id`` so that
the heavy float vectors do not bloat the human-readable prediction file and so
we can recompute embeddings (e.g. a different embed model) without re-querying
the LLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Iterator, Optional

import numpy as np


@dataclass
class PredictionRecord:
    """One (question, model answer, verbalized confidence, ground truth) tuple."""

    id: str  # globally unique, stable across runs (e.g. "mmlu/test/01234")
    dataset: str  # "mmlu", "gsm8k", ...
    order_index: int  # position in the streaming order (assigned at collection)

    question: str
    gold: str  # canonical gold answer (letter for MC; normalized number for math)
    answer: str  # extracted model answer in the same canonical space as ``gold``
    correct: int  # 1 if answer matches gold under the dataset verifier, else 0

    verbalized_confidence: Optional[float]  # model self-report in [0, 1]; None if absent
    reasoning: str  # extracted reasoning trace (may be empty)
    raw_response: str  # full untouched model output

    model: str  # model id that produced this record
    choices: Optional[list[str]] = None  # MC options, if any
    subject: Optional[str] = None  # metadata only (e.g. MMLU subject); NOT a retrieval key
    parse_ok: bool = True  # False if answer/confidence could not be parsed
    extra: dict = field(default_factory=dict)

    def retrieval_text(self, include_output: bool = True) -> str:
        """Text whose embedding is the task key.

        The pipeline embeds the question only (include_output=False);
        include_output=True appends the model's own output.
        """
        parts = [f"Question: {self.question}"]
        if self.choices:
            opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(self.choices))
            parts.append(f"Options:\n{opts}")
        if include_output:
            if self.reasoning:
                parts.append(f"Model reasoning: {self.reasoning}")
            parts.append(f"Model answer: {self.answer}")
            if self.verbalized_confidence is not None:
                parts.append(f"Model confidence: {round(self.verbalized_confidence * 100)}%")
        return "\n".join(parts)

    # --- serialization -----------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "PredictionRecord":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def write_records(path: str, records: Iterator[PredictionRecord]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_json() + "\n")
            n += 1
    return n


def read_records(path: str) -> list[PredictionRecord]:
    out: list[PredictionRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(PredictionRecord.from_dict(json.loads(line)))
    return out


# --- embedding sidecar (id -> vector) --------------------------------------
def append_embedding(path: str, rec_id: str, vec: np.ndarray) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": rec_id, "embedding": np.asarray(vec, dtype=np.float32).tolist()}) + "\n")


def read_embeddings(path: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out[obj["id"]] = np.asarray(obj["embedding"], dtype=np.float32)
    return out
