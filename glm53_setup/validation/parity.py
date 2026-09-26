"""The BF16 parity bound and verdicts the attention component checks share."""

BF16_EPS = 2**-7  # torch.finfo(torch.bfloat16).eps, without importing torch


def bf16_bound(max_abs):
    """Two BF16 ulps at the largest reference magnitude, never below magnitude one."""
    return 2 * BF16_EPS * max(1.0, max_abs)


def judge(case):
    """A compared output: finite, within its bound, and zero where no candidate was.

    An error that is not a number is above the bound.
    """
    reasons = []
    if not case["finite"]:
        reasons.append("non-finite")
    if not case["max_abs_error"] <= case["tolerance"]:
        reasons.append("error-above-bound")
    if case.get("empty_row_zero") is False:
        reasons.append("empty-row-nonzero")
    return {"passed": not reasons, "reasons": reasons}


def judge_tail(tail):
    """The trailing candidate is kept (within bound) and dropping it visibly matters."""
    reasons = []
    if not tail["native_error"] <= tail["tolerance"]:
        reasons.append("error-above-bound")
    if not tail["omission_difference"] > 1:
        reasons.append("tail-insensitive")
    return {"passed": not reasons, "reasons": reasons}
