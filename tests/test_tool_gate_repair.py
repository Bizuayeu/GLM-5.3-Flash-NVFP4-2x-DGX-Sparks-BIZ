import copy
import json
import unittest

from glm53_setup.tool_gate.repair import applies, gate

SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}
REQUEST = {
    "model": "glm",
    "messages": [{"role": "user", "content": "Just call web_search."}],
    "tools": [SEARCH],
}


def reply(content=None, calls=None):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {
                "id": f"c{i}",
                "type": "function",
                "function": {"name": n, "arguments": json.dumps(a)},
            }
            for i, (n, a) in enumerate(calls)
        ]
    return {
        "id": "x",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if calls else "stop",
            }
        ],
    }


class Upstream:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.sent = []

    def __call__(self, body):
        self.sent.append(copy.deepcopy(body))
        return self.responses.pop(0)


class GateTests(unittest.TestCase):
    def test_requests_without_tools_or_with_tool_choice_none_pass_through(self):
        self.assertFalse(applies({"messages": []}))
        self.assertFalse(applies(dict(REQUEST, tool_choice="none")))
        self.assertFalse(applies(dict(REQUEST, n=2)))
        self.assertTrue(applies(REQUEST))
        upstream = Upstream(reply("hi"))
        response, outcome, found = gate({"messages": []}, upstream)
        self.assertEqual((outcome, found, len(upstream.sent)), ("none", [], 1))

    def test_a_valid_call_is_returned_after_one_request(self):
        upstream = Upstream(reply(calls=[("web_search", {"query": "news"})]))
        response, outcome, found = gate(REQUEST, upstream)
        self.assertEqual(outcome, "passed")
        self.assertEqual(len(upstream.sent), 1)
        self.assertEqual(response["choices"][0]["finish_reason"], "tool_calls")

    def test_an_empty_argument_is_repaired_by_one_more_request(self):
        first = reply(calls=[("web_search", {"query": ""})])
        second = reply("What should I search for?")
        upstream = Upstream(first, second)
        original = copy.deepcopy(REQUEST)
        response, outcome, found = gate(REQUEST, upstream)
        self.assertEqual(outcome, "repaired")
        self.assertEqual(response, second)
        self.assertEqual(found[0]["argument"], "query")
        self.assertEqual(REQUEST, original)
        retry = upstream.sent[1]["messages"]
        self.assertEqual(retry[:1], REQUEST["messages"])
        self.assertEqual(retry[1]["role"], "assistant")
        self.assertEqual(
            retry[1]["tool_calls"], first["choices"][0]["message"]["tool_calls"]
        )
        self.assertEqual(retry[2]["role"], "tool")
        self.assertEqual(retry[2]["tool_call_id"], "c0")
        self.assertIn("query", retry[2]["content"])
        self.assertIn("not executed", retry[2]["content"])
        # The reply says what to do next, so the model neither reports a failed run nor retries empty.
        self.assertIn("did not run", retry[2]["content"])
        self.assertIn("ask the user", retry[2]["content"])
        self.assertIn("web_search", retry[2]["content"])

    def test_the_other_calls_of_the_turn_are_answered_as_not_executed(self):
        first = reply(
            calls=[("get_weather", {"city": "Tokyo"}), ("web_search", {"query": ""})]
        )
        upstream = Upstream(first, reply("ok"))
        gate(REQUEST, upstream)
        tools = [m for m in upstream.sent[1]["messages"] if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tools], ["c0", "c1"])
        self.assertIn("another call", tools[0]["content"])

    def test_a_second_violation_is_returned_as_it_is(self):
        again = reply(calls=[("web_search", {"query": " "})])
        upstream = Upstream(reply(calls=[("web_search", {"query": ""})]), again)
        response, outcome, found = gate(REQUEST, upstream)
        self.assertEqual(outcome, "unrepaired")
        self.assertEqual(response, again)
        self.assertEqual(len(upstream.sent), 2)
