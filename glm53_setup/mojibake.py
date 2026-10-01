"""Japanese and Korean generation check for broken characters.

A checkpoint whose gate/up scales disagree was reported to garble multibyte
text (vLLM #54150). The pinned NVIDIA checkpoint is not affected, so this is a
monitor, not a fix: it asks for long answers and counts replacement characters,
lone surrogates and control characters in both the answer and the reasoning.
A short answer or one in another language proves nothing, so it never passes.
"""

import unicodedata

MIN_CHARACTERS = 400  # The length the prompts ask for.
PROMPTS = {
    "ja": (
        "日本の四季それぞれについて、気候の特徴と代表的な行事を、"
        "400文字以上の日本語の文章で説明してください。箇条書きは使わないでください。"
    ),
    "ko": (
        "한국의 사계절 각각에 대해 기후의 특징과 대표적인 행사를 "
        "400자 이상의 한국어 문장으로 설명해 주세요. 글머리 기호는 사용하지 마세요."
    ),
}
SCRIPTS = {
    "ja": ((0x3040, 0x30FF), (0x4E00, 0x9FFF)),  # Kana and CJK ideographs
    "ko": ((0xAC00, 0xD7A3), (0x1100, 0x11FF), (0x3130, 0x318F)),  # Hangul
}


def count_invalid_text(text):
    counts = {"replacement": 0, "lone_surrogate": 0, "control": 0}
    for character in text:
        code = ord(character)
        if character == "\N{REPLACEMENT CHARACTER}":
            counts["replacement"] += 1
        elif 0xD800 <= code <= 0xDFFF:
            counts["lone_surrogate"] += 1
        elif unicodedata.category(character) == "Cc" and character not in "\n\r\t":
            counts["control"] += 1
    return counts


def script_share(text, language):
    characters = [c for c in text if not c.isspace()]
    if not characters:
        return 0.0
    inside = sum(
        any(low <= ord(c) <= high for low, high in SCRIPTS[language])
        for c in characters
    )
    return inside / len(characters)


def judge(language, result):
    """One run's verdict: pass, fail, or inconclusive (never a silent pass)."""
    row = {"language": language}
    if isinstance(result, Exception):
        return {**row, "verdict": "inconclusive", "error": type(result).__name__}
    choice = result["choices"][0]
    content = choice["message"].get("content") or ""
    reasoning = choice["message"].get("reasoning") or ""
    row.update(
        finish_reason=choice["finish_reason"],
        truncated=choice["finish_reason"] == "length",
        characters=len(content),
        script_share=round(script_share(content, language), 3),
        content_invalid=count_invalid_text(content),
        reasoning_invalid=count_invalid_text(reasoning),
    )
    if any(row["content_invalid"].values()) or any(row["reasoning_invalid"].values()):
        row["verdict"] = "fail"
    elif len(content) < MIN_CHARACTERS or row["script_share"] <= 0.5:
        row["verdict"] = "inconclusive"
    else:
        row["verdict"] = "pass"
    row["content"], row["reasoning"] = content, reasoning
    return row


def run(ask, repeats=3, sampling=None):
    """Ask each prompt repeats times at temperature 0; never raise on answers.

    ``sampling`` ({temperature, top_p, seed}) counts broken characters in
    sampled answers instead: run i draws with seed + i, so repeats are distinct
    draws, and the record names the settings.
    """
    if sampling is not None and not (
        sampling.get("temperature", 0) > 0
        and 0 < sampling.get("top_p", 0) <= 1
        and isinstance(sampling.get("seed"), int)
    ):
        raise ValueError("Sampling needs temperature > 0, 0 < top_p <= 1 and a seed")
    rows = []
    for language, prompt in PROMPTS.items():
        for _ in range(repeats):
            request = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
            if sampling is not None:
                request.update(
                    temperature=sampling["temperature"],
                    top_p=sampling["top_p"],
                    seed=sampling["seed"] + len(rows),
                )
            try:
                row = judge(language, ask(request))
            except Exception as error:  # noqa: BLE001 - a failed or malformed answer is recorded, not a pass
                row = judge(language, error)
            if sampling is not None:
                row["seed"] = request["seed"]
            rows.append(row)
    verdicts = {row["verdict"] for row in rows}
    verdict = (
        "fail"
        if "fail" in verdicts
        else "pass"
        if verdicts == {"pass"}
        else "inconclusive"
    )
    result = {"runs": rows, "verdict": verdict, "passed": verdict == "pass"}
    if sampling is not None:
        result["sampling"] = {**sampling, "repeats": repeats}
    return result
