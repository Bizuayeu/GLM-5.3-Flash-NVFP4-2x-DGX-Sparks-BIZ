"""CPU-portable HLE answer extraction, and the exact-match rule for the grader.

The format, normalization and exact-match rule follow the private lineage
experiment that sampled the question set, so GLM rows can sit beside its
Claude rows. The runner (hle.py) only extracts; grading happens off the host,
and ``normalize``/``exact`` are kept here as that grader's rule so both sides
read one definition. Anything not settled by exact match goes to a judge; this
module never guesses equivalence itself.
"""

import re
import unicodedata

# The exam instruction of the lineage experiment (itself the HLE answer format).
SYSTEM_PROMPT = (
    "You are answering an exam question. Answer from your own knowledge only.\n"
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your final answer}\n"
    "Exact Answer: {your succinct, final answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)

_ANSWER = re.compile(
    r"^\s*(?:\*\*)?Exact Answer:(?:\*\*)?\s*(.*?)(?=^\s*(?:\*\*)?Confidence:|\Z)",
    re.M | re.S,
)
_CONFIDENCE = re.compile(r"Confidence:(?:\*\*)?\s*(\d+(?:\.\d+)?)\s*%")


def normalize(value):
    return unicodedata.normalize("NFKC", value).strip().casefold().rstrip(".").strip()


def extract(content):
    """Return (answer, confidence) from the final content; reasoning is not read."""
    if not content:
        return None, None
    match = _ANSWER.search(content)
    answer = match.group(1).strip() if match else None
    found = _CONFIDENCE.search(content)
    score = float(found.group(1)) if found else None
    if score is not None and not 0 <= score <= 100:
        score = None
    return answer or None, score


def final_content(response):
    choices = response.get("choices") or []
    if len(choices) != 1:
        return None, None
    choice = choices[0]
    return (choice.get("message") or {}).get("content"), choice.get("finish_reason")


def exact(answer, reference):
    """True/False on a parsed answer; None when there is nothing to compare."""
    if answer is None:
        return None
    return normalize(answer) == normalize(reference)
