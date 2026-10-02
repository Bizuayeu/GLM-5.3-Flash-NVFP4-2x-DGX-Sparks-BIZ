"""Serve the tool-argument gate on a loopback port in front of the model API (entry and side effects).

    python -m glm53_setup tool-gate [--port 8894] [--upstream http://127.0.0.1:8893] [--log FILE]

Only POST /v1/chat/completions requests that declare tools are checked (repair.applies); every other
request is relayed unchanged, credentials included. The listener binds 127.0.0.1 only, like the
model API behind it. The log is one JSON line per checked request with the tool and argument names
of any violation; it never holds argument values, prompts or replies.
"""

import argparse
import datetime
import http.client
import json
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .check import violations
from .repair import applies, gate, retry_request

CHAT_PATH = "/v1/chat/completions"
# cc-defer: provisional ceiling (a 262,144-token prompt with a few images stays far below it);
# raise it when a real request is refused with 413.
MAX_BODY = 64 * 2**20
# cc-defer: provisional; the HLE client used 1,200 s for 16,384 tokens, and a repair doubles a
# request. Revisit when a client reports a gate timeout.
DEFAULT_TIMEOUT = 2400
HOP = {
    "connection",
    "keep-alive",
    "transfer-encoding",
    "content-length",
    "host",
    "proxy-connection",
    "te",
    "upgrade",
}


class Passthrough(Exception):
    """An upstream answer that is not a 200 JSON completion; the client gets it unchanged."""

    def __init__(self, status, headers, body):
        super().__init__(status)
        self.status, self.headers, self.body = status, headers, body


def _loopback_origin(url):
    parts = urllib.parse.urlsplit(url)
    if (
        parts.scheme != "http"
        or parts.hostname not in ("127.0.0.1", "localhost")
        or parts.path not in ("", "/")
    ):
        raise ValueError("The upstream must be an http origin on this host's loopback")
    return parts.hostname, parts.port or 80


def _events(response):
    """The data payloads of an SSE response, in order."""
    for line in response:
        line = line.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            yield line[5:].strip()


def _relay(response, emit):
    """Forward content deltas at once; gather tool-call deltas and hold the finish and usage chunks."""
    content, calls, held = [], {}, []
    for payload in _events(response):
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        choices = chunk.get("choices") or []
        if not choices:
            held.append(chunk)
            continue
        delta = choices[0].get("delta") or {}
        gathered = delta.pop("tool_calls", None)
        for part in gathered or []:
            slot = calls.setdefault(
                part.get("index", 0),
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if part.get("id"):
                slot["id"] = part["id"]
            function = part.get("function") or {}
            slot["function"]["name"] += function.get("name") or ""
            slot["function"]["arguments"] += function.get("arguments") or ""
        if delta.get("content"):
            content.append(delta["content"])
        if choices[0].get("finish_reason"):
            held.append(chunk)
        elif delta or not gathered:
            emit(chunk)
    return "".join(content) or None, [calls[i] for i in sorted(calls)], held


def _calls_chunk(held, calls):
    base = next((c for c in held if c.get("choices")), {})
    chunk = {k: v for k, v in base.items() if k != "choices"}
    chunk["choices"] = [
        {
            "index": 0,
            "delta": {
                "tool_calls": [dict(call, index=i) for i, call in enumerate(calls)]
            },
            "finish_reason": None,
        }
    ]
    return chunk


def make_server(port, upstream, log=sys.stderr, timeout=DEFAULT_TIMEOUT):
    host, upstream_port = _loopback_origin(upstream)
    lock = threading.Lock()

    def record(stream, outcome, found):
        line = {
            "time": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "stream": stream,
            "outcome": outcome,
            "violations": [
                {k: v[k] for k in ("tool", "kind", "argument")} for v in found
            ],
        }
        with lock:
            log.write(json.dumps(line) + "\n")
            log.flush()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"  # the connection closes after each answer, so a stream needs no length

        def log_message(self, *args):
            pass

        def setup(self):
            super().setup()
            self._conns = []

        def finish(self):
            for conn in self._conns:
                conn.close()
            super().finish()

        def _upstream(self, method, body):
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            if body is not None:
                headers["Content-Length"] = str(len(body))
            conn = http.client.HTTPConnection(host, upstream_port, timeout=timeout)
            self._conns.append(conn)
            conn.request(method, self.path, body=body, headers=headers)
            return conn.getresponse()

        def _start(self, status, headers, extra=()):
            self.send_response(status)
            for key, value in headers:
                if key.lower() not in HOP:
                    self.send_header(key, value)
            for key, value in extra:
                self.send_header(key, value)
            self.end_headers()

        def _relay_raw(self, response):
            self._start(response.status, response.getheaders())
            while chunk := response.read(65536):
                self.wfile.write(chunk)
                self.wfile.flush()

        def _body(self):
            """The request body, or None after answering 400 or 413 itself."""
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0:
                self.send_error(400, "Bad Content-Length")
                return None
            if length > MAX_BODY:
                self.send_error(413)
                return None
            return self.rfile.read(length) if length else b""

        def _send_json(self, request):
            response = self._upstream("POST", json.dumps(request).encode())
            raw = response.read()
            if response.status != 200:
                raise Passthrough(response.status, response.getheaders(), raw)
            return json.loads(raw)

        def _chat(self, raw):
            try:
                request = json.loads(raw)
            except ValueError:
                request = None
            if not isinstance(request, dict) or not applies(request):
                self._relay_raw(self._upstream("POST", raw))
                return
            if request.get("stream"):
                self._stream(request)
                return
            sent, checked = [], []

            def send(body):
                sent.append(body)
                return self._send_json(body)

            def check(tools, calls):
                checked.append(violations(tools, calls))
                return checked[-1]

            try:
                answer, outcome, found = gate(request, send, check)
            except Passthrough as held:
                if len(sent) > 1:
                    # The repair request failed; the violating turn is not returned.
                    record(False, "error", checked[0])
                self._start(held.status, held.headers)
                self.wfile.write(held.body)
                return
            record(False, outcome, found)
            data = json.dumps(answer).encode()
            self._start(
                200,
                [("Content-Type", "application/json")],
                [("x-glm53-tool-gate", outcome)],
            )
            self.wfile.write(data)

        def _stream(self, request):
            first = self._upstream("POST", json.dumps(request).encode())
            if first.status != 200:
                self._relay_raw(first)
                return
            self._start(200, first.getheaders(), [("x-glm53-tool-gate", "stream")])

            def emit(chunk):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

            content, calls, held = _relay(first, emit)
            found = violations(request["tools"], calls)
            outcome = "passed"
            if found:
                message = {"content": content, "tool_calls": calls}
                second = self._upstream(
                    "POST", json.dumps(retry_request(request, message, found)).encode()
                )
                if second.status != 200:
                    # The client already has the stream's first chunks; it gets the error inside it.
                    text = second.read().decode("utf-8", "replace")
                    emit(
                        {
                            "error": {
                                "message": "tool-gate repair request failed",
                                "status": second.status,
                                "upstream": text[:2000],
                            }
                        }
                    )
                    record(True, "error", found)
                    self.wfile.write(b": tool-gate error\n\ndata: [DONE]\n\n")
                    self.wfile.flush()
                    return
                content, calls, held = _relay(second, emit)
                outcome = (
                    "unrepaired" if violations(request["tools"], calls) else "repaired"
                )
            if calls:
                emit(_calls_chunk(held, calls))
            for chunk in held:
                emit(chunk)
            record(True, outcome, found)
            self.wfile.write(f": tool-gate {outcome}\n\ndata: [DONE]\n\n".encode())
            self.wfile.flush()

        def do_POST(self):
            raw = self._body()
            if raw is None:
                return
            if urllib.parse.urlsplit(self.path).path == CHAT_PATH:
                self._chat(raw)
            else:
                self._relay_raw(self._upstream("POST", raw))

        def do_GET(self):
            self._relay_raw(self._upstream("GET", None))

        def do_DELETE(self):
            raw = self._body()
            if raw is not None:
                self._relay_raw(self._upstream("DELETE", raw))

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=8894)
    parser.add_argument("--upstream", default="http://127.0.0.1:8893")
    parser.add_argument(
        "--log", help="Append the JSON-line log to this file (default: stderr)"
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args(argv)
    log = open(args.log, "a", encoding="utf-8") if args.log else sys.stderr
    server = make_server(args.port, args.upstream, log=log, timeout=args.timeout)
    print(
        f"tool-gate on 127.0.0.1:{args.port} -> {args.upstream}",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
