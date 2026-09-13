import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from glm53_setup import model_http, startup


class ModelHTTPTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        requests = self.requests

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                requests.append((self.path, self.headers.get("Authorization")))
                if self.path == "/redirect":
                    self.send_response(307)
                    self.send_header("Location", self.server.redirect)
                    self.end_headers()
                    return
                if self.path in ("/private", "/forbidden"):
                    expected = "Bearer test-model-key"
                    if (
                        self.path == "/forbidden"
                        or self.headers.get("Authorization") != expected
                    ):
                        self.send_response(403 if self.path == "/forbidden" else 401)
                        self.end_headers()
                        # A hostile API error body must not appear in a saved exception.
                        self.wfile.write(
                            (self.headers.get("Authorization") or "").encode()
                        )
                        return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(
                    b'data: {"token":"ok"}\n\ndata: [DONE]\n\n'
                    if self.path == "/sse"
                    else b'{"ok":true}'
                )

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.do_GET()

        self.servers = [
            ThreadingHTTPServer(("127.0.0.1", 0), Handler) for _ in range(2)
        ]
        self.threads = []
        for server in self.servers:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.threads.append(thread)
        self.origin = f"http://127.0.0.1:{self.servers[0].server_port}"
        self.servers[
            0
        ].redirect = f"http://127.0.0.1:{self.servers[1].server_port}/leaked"

    def tearDown(self):
        for server, thread in zip(self.servers, self.threads):
            server.shutdown()
            server.server_close()
            thread.join()

    def test_precedence_empty_and_no_key_on_actual_requests(self):
        cases = [
            ({}, None),
            ({"API_KEY": "", "VLLM_API_KEY": ""}, None),
            ({"VLLM_API_KEY": "second"}, "Bearer second"),
            ({"API_KEY": "", "VLLM_API_KEY": "second"}, "Bearer second"),
            ({"API_KEY": "first", "VLLM_API_KEY": "second"}, "Bearer first"),
        ]
        for env, expected in cases:
            with model_http.open_response(
                self.origin, "/public", environ=env
            ) as response:
                self.assertTrue(json.load(response)["ok"])
            self.assertEqual(self.requests[-1][1], expected)

    def test_wrong_and_correct_keys_and_sanitized_errors(self):
        for path, key, code in [
            ("/private", "wrong-secret", 401),
            ("/forbidden", "test-model-key", 403),
        ]:
            with self.assertRaises(model_http.ModelHTTPError) as caught:
                model_http.open_response(self.origin, path, environ={"API_KEY": key})
            self.assertEqual(caught.exception.code, code)
            self.assertIn("authentication failed", str(caught.exception))
            self.assertNotIn(key, repr(caught.exception))
            self.assertNotIn("Bearer", repr(caught.exception))
        with model_http.open_response(
            self.origin, "/private", environ={"API_KEY": "test-model-key"}
        ) as response:
            self.assertEqual(response.status, 200)

    def test_sse_and_startup_post_use_the_same_auth_contract(self):
        with model_http.open_response(
            self.origin,
            "/sse",
            body={"stream": True},
            environ={"API_KEY": "stream-key"},
        ) as response:
            self.assertEqual(response.readline(), b'data: {"token":"ok"}\n')
            self.assertIn(b"[DONE]", response.read())
        self.assertEqual(self.requests[-1][1], "Bearer stream-key")
        with patch.dict("os.environ", {"API_KEY": "test-model-key"}, clear=True):
            result = startup.post(
                {
                    "api": {"port": self.servers[0].server_port},
                    "generation": {"timeout_seconds": 5},
                },
                "/private",
                {},
            )
        self.assertTrue(result["ok"])

    def test_redirect_never_contacts_second_origin(self):
        with self.assertRaises(model_http.ModelHTTPError) as caught:
            model_http.open_response(
                self.origin, "/redirect", environ={"API_KEY": "redirect-secret"}
            )
        self.assertEqual(caught.exception.code, 307)
        self.assertEqual(self.requests, [("/redirect", "Bearer redirect-secret")])

    def test_unscoped_urls_and_header_injection_are_refused_before_network(self):
        for path in ["//evil.invalid/", "https://evil.invalid/", "relative", "/\n"]:
            with self.assertRaises(ValueError):
                model_http.open_response(self.origin, path, environ={"API_KEY": "key"})
        for base in ["file:///tmp/a", "http://key@localhost", self.origin + "/v1"]:
            with self.assertRaises(ValueError):
                model_http.open_response(base, "/private", environ={})
        with self.assertRaises(ValueError) as caught:
            model_http.open_response(
                self.origin, "/private", environ={"API_KEY": "secret\r\nInjected: key"}
            )
        self.assertNotIn("secret", str(caught.exception))
        self.assertFalse(self.requests)
