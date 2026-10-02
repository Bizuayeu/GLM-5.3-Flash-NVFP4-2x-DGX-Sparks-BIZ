import http.client
import io
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from glm53_setup.tool_gate import proxy

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
TOOL_REQUEST = {
    "model": "glm",
    "messages": [{"role": "user", "content": "Just call web_search."}],
    "tools": [SEARCH],
}


def completion(content=None, args=None):
    message = {"role": "assistant", "content": content}
    if args is not None:
        message["tool_calls"] = [
            {
                "id": "c0",
                "type": "function",
                "function": {"name": "web_search", "arguments": json.dumps(args)},
            }
        ]
    return {
        "id": "x",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }


def sse(content=None, args=None):
    """An upstream stream: reasoning, then either text or one tool call split over two deltas."""
    chunks = [
        {
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "reasoning_content": "hmm"}}
            ]
        }
    ]
    if args is None:
        chunks.append({"choices": [{"index": 0, "delta": {"content": content}}]})
        finish = "stop"
    else:
        text = json.dumps(args)
        chunks.append(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c0",
                                    "type": "function",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": text[:3],
                                    },
                                }
                            ]
                        },
                    }
                ]
            }
        )
        chunks.append(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": text[3:]}}
                            ]
                        },
                    }
                ]
            }
        )
        finish = "tool_calls"
    chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    chunks.append({"choices": [], "usage": {"completion_tokens": 7}})
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


class FakeUpstream:
    """A local server that answers queued responses and records what it received."""

    def __init__(self):
        self.queue, self.received = [], []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _answer(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                outer.received.append(
                    (self.command, self.path, dict(self.headers), body)
                )
                status, content_type, payload = outer.queue.pop(0)
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = do_POST = do_DELETE = _answer

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.port = self.server.server_address[1]

    def push_json(self, body, status=200):
        self.queue.append((status, "application/json", json.dumps(body).encode()))

    def push_sse(self, text):
        self.queue.append((200, "text/event-stream", text.encode()))


class LoopbackOriginTests(unittest.TestCase):
    # The only guard that keeps the gate from relaying requests, credentials included, off the host.
    def test_only_a_plain_http_origin_on_the_loopback_is_accepted(self):
        self.assertEqual(
            proxy._loopback_origin("http://127.0.0.1:8893"), ("127.0.0.1", 8893)
        )
        self.assertEqual(proxy._loopback_origin("http://localhost/"), ("localhost", 80))
        for url in (
            "https://127.0.0.1:8893",
            "http://10.41.12.1:8893",
            "http://localhost.evil:8893",
            "http://127.0.0.1:8893/v1",
            "127.0.0.1:8893",
        ):
            with self.subTest(url=url):
                with self.assertRaisesRegex(ValueError, "loopback"):
                    proxy._loopback_origin(url)


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.upstream = FakeUpstream()
        self.log = io.StringIO()
        self.gate = proxy.make_server(
            0, f"http://127.0.0.1:{self.upstream.port}", log=self.log
        )
        threading.Thread(target=self.gate.serve_forever, daemon=True).start()
        self.port = self.gate.server_address[1]

    def tearDown(self):
        for server in (self.gate, self.upstream.server):
            server.shutdown()
            server.server_close()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        data = None if body is None else json.dumps(body).encode()
        conn.request(
            method,
            path,
            body=data,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        response = conn.getresponse()
        try:
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    def test_other_paths_and_requests_without_tools_pass_through(self):
        self.upstream.push_json({"data": [{"id": "glm"}]})
        status, headers, body = self.request(
            "GET", "/v1/models", headers={"Authorization": "Bearer k"}
        )
        self.assertEqual((status, json.loads(body)), (200, {"data": [{"id": "glm"}]}))
        self.assertEqual(self.upstream.received[0][2].get("Authorization"), "Bearer k")
        self.upstream.push_json(completion("hi"))
        status, headers, body = self.request(
            "POST", "/v1/chat/completions", {"messages": []}
        )
        self.assertEqual(json.loads(body), completion("hi"))
        self.assertEqual(len(self.upstream.received), 2)

    def test_a_violating_call_is_repaired_without_reaching_the_client(self):
        self.upstream.push_json(completion(args={"query": ""}))
        self.upstream.push_json(completion("What should I search for?"))
        status, headers, body = self.request(
            "POST", "/v1/chat/completions", TOOL_REQUEST
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("x-glm53-tool-gate"), "repaired")
        self.assertEqual(
            json.loads(body)["choices"][0]["message"]["content"],
            "What should I search for?",
        )
        retry = json.loads(self.upstream.received[1][3])
        self.assertEqual(
            [m["role"] for m in retry["messages"]], ["user", "assistant", "tool"]
        )
        record = json.loads(self.log.getvalue().splitlines()[-1])
        self.assertEqual(record["outcome"], "repaired")
        self.assertEqual(
            record["violations"],
            [{"tool": "web_search", "kind": "empty", "argument": "query"}],
        )
        self.assertNotIn("Just call", self.log.getvalue())

    def test_an_upstream_error_is_returned_as_it_is(self):
        self.upstream.push_json({"error": {"message": "bad"}}, status=400)
        status, headers, body = self.request(
            "POST", "/v1/chat/completions", TOOL_REQUEST
        )
        self.assertEqual(
            (status, json.loads(body)), (400, {"error": {"message": "bad"}})
        )

    def raw_request(self, method, path, length, body=b""):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest(method, path)
        conn.putheader("Content-Length", length)
        conn.endheaders(body)
        response = conn.getresponse()
        try:
            return response.status, response.read()
        finally:
            conn.close()

    def test_an_oversized_body_is_refused_once_and_never_relayed(self):
        # DELETE used to relay the refused request and answer a second time.
        for method in ("POST", "DELETE"):
            with self.subTest(method=method), patch.object(proxy, "MAX_BODY", 8):
                status, _ = self.raw_request(method, "/v1/x", "9", b"123456789")
                self.assertEqual(status, 413)
        time.sleep(
            0.3
        )  # the old handler relayed after answering; give it the time to show
        self.assertEqual(self.upstream.received, [])

    def test_a_malformed_content_length_is_refused(self):
        # A negative length made read(-n) wait for the client to close the connection.
        for length in ("-1", "abc"):
            with self.subTest(length=length):
                status, _ = self.raw_request("POST", "/v1/chat/completions", length)
                self.assertEqual(status, 400)
        self.assertEqual(self.upstream.received, [])

    def test_a_body_that_is_not_json_is_relayed_unchecked(self):
        self.upstream.push_json({"error": {"message": "bad json"}}, status=400)
        status, body = self.raw_request("POST", "/v1/chat/completions", "5", b"{oops")
        self.assertEqual(status, 400)
        self.assertEqual(self.upstream.received[0][3], b"{oops")
        self.assertEqual(self.log.getvalue(), "")

    def test_a_delete_is_relayed_with_its_body(self):
        self.upstream.push_json({"deleted": True})
        status, body = self.raw_request("DELETE", "/v1/files/f", "2", b"{}")
        self.assertEqual((status, json.loads(body)), (200, {"deleted": True}))
        self.assertEqual(self.upstream.received[0][:2], ("DELETE", "/v1/files/f"))

    def test_a_stream_the_upstream_refuses_is_returned_as_it_is(self):
        self.upstream.push_json({"error": {"message": "busy"}}, status=503)
        status, headers, body = self.request(
            "POST", "/v1/chat/completions", dict(TOOL_REQUEST, stream=True)
        )
        self.assertEqual(
            (status, json.loads(body)), (503, {"error": {"message": "busy"}})
        )
        self.assertNotIn("x-glm53-tool-gate", headers)
        self.assertEqual(self.log.getvalue(), "")

    def stream(self, body):
        status, headers, raw = self.request(
            "POST", "/v1/chat/completions", dict(body, stream=True)
        )
        events = [
            line[6:] for line in raw.decode().splitlines() if line.startswith("data: ")
        ]
        comments = [line for line in raw.decode().splitlines() if line.startswith(":")]
        chunks = [json.loads(e) for e in events if e != "[DONE]"]
        return status, events, chunks, comments

    def test_a_stream_with_a_valid_call_keeps_its_content_and_gathers_the_call(self):
        self.upstream.push_sse(sse(args={"query": "news"}))
        status, events, chunks, comments = self.stream(TOOL_REQUEST)
        self.assertEqual(events[-1], "[DONE]")
        calls = [
            c
            for c in chunks
            if c["choices"] and c["choices"][0]["delta"].get("tool_calls")
        ]
        self.assertEqual(len(calls), 1)
        call = calls[0]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(json.loads(call["function"]["arguments"]), {"query": "news"})
        self.assertEqual(chunks[0]["choices"][0]["delta"]["reasoning_content"], "hmm")
        self.assertEqual(chunks[-1]["usage"], {"completion_tokens": 7})
        self.assertIn(": tool-gate passed", comments)

    def test_a_stream_with_a_violating_call_continues_with_the_repair(self):
        self.upstream.push_sse(sse(args={"query": ""}))
        self.upstream.push_sse(sse(content="What should I search for?"))
        status, events, chunks, comments = self.stream(TOOL_REQUEST)
        self.assertFalse(
            any(
                c["choices"] and c["choices"][0]["delta"].get("tool_calls")
                for c in chunks
            )
        )
        text = "".join(
            c["choices"][0]["delta"].get("content") or ""
            for c in chunks
            if c["choices"]
        )
        self.assertEqual(text, "What should I search for?")
        finishes = [
            c["choices"][0]["finish_reason"]
            for c in chunks
            if c["choices"] and c["choices"][0].get("finish_reason")
        ]
        self.assertEqual(finishes, ["stop"])
        self.assertEqual(events.count("[DONE]"), 1)
        self.assertIn(": tool-gate repaired", comments)
        retry = json.loads(self.upstream.received[1][3])
        self.assertTrue(retry["stream"])
        self.assertEqual(
            json.loads(retry["messages"][1]["tool_calls"][0]["function"]["arguments"]),
            {"query": ""},
        )

    def test_a_failed_repair_is_recorded_as_an_error(self):
        self.upstream.push_json(completion(args={"query": ""}))
        self.upstream.push_json({"error": {"message": "busy"}}, status=503)
        status, headers, body = self.request(
            "POST", "/v1/chat/completions", TOOL_REQUEST
        )
        self.assertEqual(status, 503)
        first = json.loads(self.log.getvalue().splitlines()[-1])
        self.assertEqual(first["outcome"], "error")
        # Both paths name the violation the failed repair was for.
        self.assertEqual(
            first["violations"],
            [{"tool": "web_search", "kind": "empty", "argument": "query"}],
        )
        self.upstream.push_sse(sse(args={"query": ""}))
        self.upstream.push_json({"error": {"message": "busy"}}, status=503)
        status, events, chunks, comments = self.stream(TOOL_REQUEST)
        self.assertIn(": tool-gate error", comments)
        self.assertTrue(any("error" in c for c in chunks))
        last = json.loads(self.log.getvalue().splitlines()[-1])
        self.assertEqual(last["outcome"], "error")
        self.assertEqual(last["violations"], first["violations"])
