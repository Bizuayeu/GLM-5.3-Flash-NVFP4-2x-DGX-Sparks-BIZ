import json
import unittest

from glm53_setup.tool_gate.check import violations

SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        },
    },
}
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def call(name, arguments, call_id="c1"):
    text = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": text},
    }


class ViolationTests(unittest.TestCase):
    def test_an_empty_or_blank_required_string_is_a_violation(self):
        for args in ({"query": ""}, {"query": "   \n"}):
            found = violations([SEARCH], [call("web_search", args)])
            self.assertEqual(
                found,
                [
                    {
                        "call_id": "c1",
                        "tool": "web_search",
                        "kind": "empty",
                        "argument": "query",
                    }
                ],
            )

    def test_a_missing_or_null_required_argument_is_a_violation(self):
        for args in ({"limit": 3}, {"query": None}):
            found = violations([SEARCH], [call("web_search", args)])
            self.assertEqual(found[0]["kind"], "missing")
            self.assertEqual(found[0]["argument"], "query")

    def test_a_required_string_of_another_type_is_left_alone(self):
        # Type checking is out of scope; only blank strings count as empty.
        self.assertEqual(violations([SEARCH], [call("web_search", {"query": 5})]), [])

    def test_arguments_that_are_not_a_json_object_are_a_violation(self):
        for text in ("{not json", "[1, 2]", '"query"'):
            found = violations([SEARCH], [call("web_search", text)])
            self.assertEqual(
                found,
                [
                    {
                        "call_id": "c1",
                        "tool": "web_search",
                        "kind": "not_object",
                        "argument": None,
                    }
                ],
            )

    def test_valid_calls_and_unchecked_tools_pass(self):
        self.assertEqual(
            violations([SEARCH], [call("web_search", {"query": "news"})]), []
        )
        # An unknown tool, and a tool without a required list, are left to the server and the client.
        self.assertEqual(violations([SEARCH], [call("send_email", {})]), [])
        loose = {
            "type": "function",
            "function": {"name": "web_search", "parameters": {"type": "object"}},
        }
        self.assertEqual(violations([loose], [call("web_search", {})]), [])

    def test_a_required_argument_of_another_type_is_not_checked_for_emptiness(self):
        tool = {
            "type": "function",
            "function": {
                "name": "set_count",
                "parameters": {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "required": ["n"],
                },
            },
        }
        self.assertEqual(violations([tool], [call("set_count", {"n": 0})]), [])

    def test_only_the_violating_call_of_a_turn_is_reported(self):
        calls = [
            call("get_weather", {"city": "Tokyo"}, "a"),
            call("web_search", {"query": ""}, "b"),
        ]
        found = violations([SEARCH, WEATHER], calls)
        self.assertEqual([v["call_id"] for v in found], ["b"])
