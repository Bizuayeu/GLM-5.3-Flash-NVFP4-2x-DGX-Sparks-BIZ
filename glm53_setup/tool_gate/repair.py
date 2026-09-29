"""Stop a tool call whose arguments break its schema and let the model answer once more (assembly layer).

The violating turn is not returned. It is added to the conversation with a tool reply per call,
the one the model would have received had the call reached a real tool, and the model is asked
once more. A second violation is returned as it is: the gate does not hide what the model does.
"""

import copy

from .check import violations

REASONS = {
    "missing": "required argument '{argument}' is missing",
    "empty": "required argument '{argument}' is empty",
    "not_object": "the arguments are not a JSON object",
}
# Saying only that a call was rejected left the model reporting a failed run (TC-43 on 2026-09-29,
# records/20260929-tool-gate/REPORT.md). The reply also says what to do next.
INVALID = (
    "Error: not executed: {reasons}. The call did not run. If you do not know the value of a"
    " required argument, do not call {tool} again without it; ask the user for it instead."
)
NOT_RUN = (
    "Error: not executed, because another call in this turn had invalid arguments."
)


def applies(request):
    """Only a single-choice request that lets the model call a declared tool is checked."""
    return (
        bool(request.get("tools"))
        and request.get("tool_choice") != "none"
        and request.get("n") in (None, 1)
    )


def _message(response):
    try:
        return response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return {}


def retry_request(request, message, found):
    """The request with the rejected turn and one tool reply for each of its calls."""
    bad = {}
    for v in found:
        bad.setdefault(v["call_id"], []).append(
            REASONS[v["kind"]].format(argument=v["argument"])
        )
    turn = {
        "role": "assistant",
        "content": message.get("content"),
        "tool_calls": message["tool_calls"],
    }
    replies = [
        {
            "role": "tool",
            "tool_call_id": call.get("id"),
            "content": (
                INVALID.format(
                    reasons="; ".join(bad[call.get("id")]),
                    tool=(call.get("function") or {}).get("name"),
                )
                if call.get("id") in bad
                else NOT_RUN
            ),
        }
        for call in message["tool_calls"]
    ]
    retry = copy.deepcopy(request)
    retry["messages"] = [*retry["messages"], turn, *replies]
    return retry


def gate(request, send, check=violations):
    """Return (response, outcome, first violations); outcome is none, passed, repaired or unrepaired."""
    if not applies(request):
        return send(request), "none", []
    response = send(request)
    message = _message(response)
    found = check(request["tools"], message.get("tool_calls"))
    if not found:
        return response, "passed", []
    second = send(retry_request(request, message, found))
    again = check(request["tools"], _message(second).get("tool_calls"))
    return second, "unrepaired" if again else "repaired", found
