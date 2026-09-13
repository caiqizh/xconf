"""Ground-truth verifiers (NOT an LLM judge).

Correctness comes from deterministic comparison against dataset ground truth.
Everything here is string/number processing.
"""

from __future__ import annotations

import re
from typing import Optional

from .prompts import extract_mc_letter


def verify_mc(answer_field: str, gold_letter: str, n_choices: int) -> tuple[int, str]:
    """Multiple choice: exact letter match. Returns (correct, extracted_letter)."""
    letter = extract_mc_letter(answer_field, n_choices)
    if letter is None:
        return 0, ""
    return int(letter == gold_letter.strip().upper()), letter


_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _normalize_number(s: str) -> Optional[str]:
    """Extract the last number from a string and canonicalize it."""
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").replace("%", "")
    matches = _NUM_RE.findall(s.replace(",", ""))
    if not matches:
        return None
    raw = matches[-1]
    try:
        val = float(raw)
    except ValueError:
        return None
    # canonical form: drop trailing .0 so "18" == "18.0"
    if val == int(val):
        return str(int(val))
    return repr(val)


def gsm8k_gold(answer_text: str) -> str:
    """GSM8K gold strings look like '... #### 18'. Return the canonical number."""
    after = answer_text.split("####")[-1] if "####" in answer_text else answer_text
    norm = _normalize_number(after)
    return norm if norm is not None else answer_text.strip()


def verify_numeric(answer_field: str, gold: str) -> tuple[int, str]:
    """Verifiable math: normalize both sides to a canonical number, exact match."""
    pred = _normalize_number(answer_field)
    gold_norm = _normalize_number(gold)
    if pred is None or gold_norm is None:
        return 0, pred or ""
    return int(pred == gold_norm), pred


# Registry: dataset -> verifier callable signature (answer_field, gold, meta) -> (correct, extracted)
FREEFORM = {"simpleqa", "hle", "bbh", "math", "agieval", "bbeh", "olympiadbench"}  # graded by an LLM
# Code generation: correctness comes from LOCAL unit-test execution (lcb_verify.py),
# never an LLM judge. At parse time the answer (a full code block) is kept verbatim
# and `correct` stays a 0 placeholder until the test-execution pass.
TEST_VERIFIED = {"lcb"}


def verify(dataset: str, answer_field: str, gold: str, n_choices: int = 0) -> tuple[int, str]:
    if dataset in ("mmlu", "mmlu_pro", "gpqa", "medmcqa", "supergpqa", "mmmu_pro"):
        return verify_mc(answer_field, gold, n_choices)
    if dataset == "gsm8k":
        return verify_numeric(answer_field, gold)
    if dataset in FREEFORM or dataset in TEST_VERIFIED:
        # No deterministic check here: keep the raw answer; correctness is filled by a
        # later ground-truth pass (grade_freeform.py for FREEFORM; local unit-test
        # execution for TEST_VERIFIED code datasets).
        return 0, (answer_field or "").strip()
    raise ValueError(f"no verifier registered for dataset '{dataset}'")
