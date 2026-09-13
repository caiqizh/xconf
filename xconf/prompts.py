"""Prompts that elicit an answer + verbalized confidence in one call, and the
deterministic parser for the response.

Single-call elicitation (reasoning + answer + confidence) is the standard
verbalized-confidence setup (Tian et al. 2023) and is cheap. The strict output
format makes parsing deterministic so that *answer extraction is not an LLM
judge* -- it is regex over a constrained template.
"""

from __future__ import annotations

import re
from typing import Optional

_CONF_RE = re.compile(r"confidence\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?", re.IGNORECASE)

SYSTEM = (
    "You are a careful test-taker. Answer the question, then state how confident "
    "you are that your answer is correct. Always follow the exact output format."
)

_MC_TEMPLATE = """Answer the following multiple-choice question.

Question: {question}
Options:
{options}

Think briefly, then respond in EXACTLY this format (no extra text):
Reasoning: <one or two short sentences>
Answer: <a single option letter, e.g. {letters}>
Confidence: <integer 0-100, the probability that your Answer is correct>"""

_FREEFORM_TEMPLATE = """Solve the following problem.

Problem: {question}

Think briefly, then respond in EXACTLY this format (no extra text):
Reasoning: <brief solution>
Answer: <your final answer only, as a single number or short string>
Confidence: <integer 0-100, the probability that your Answer is correct>"""


def build_prompt(question: str, choices: Optional[list[str]]) -> str:
    if choices:
        letters = ", ".join(chr(65 + i) for i in range(len(choices)))
        options = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
        return _MC_TEMPLATE.format(question=question, options=options, letters=letters)
    return _FREEFORM_TEMPLATE.format(question=question)


# a single uppercase letter not adjacent to other letters: matches "C", "(C)",
# "C.", "C)" but NOT the "A" inside "answer".
_STANDALONE_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])")


def extract_mc_letter(answer: str, n_choices: int) -> Optional[str]:
    """Pull a single valid option letter out of a possibly noisy answer field."""
    if not answer:
        return None
    valid = {chr(65 + i) for i in range(n_choices)}
    up = answer.upper()
    # 1) prefer a standalone letter token (the common, clean case)
    for m in _STANDALONE_LETTER_RE.finditer(up):
        if m.group(1) in valid:
            return m.group(1)
    # 2) last resort: first valid letter char anywhere
    for ch in up:
        if ch in valid:
            return ch
    return None
