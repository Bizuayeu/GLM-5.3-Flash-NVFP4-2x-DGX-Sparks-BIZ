"""Pinned upstream middleware contract, using no model weights or GPU."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("vllm"), "Pinned vLLM image required")
class NativeAuthTests(unittest.TestCase):
    def test_guarded_and_control_paths_in_fixed_runtime(self):
        from starlette.responses import JSONResponse
        from starlette.testclient import TestClient
        from vllm.entrypoints.serve.middleware.authenticate import (
            AuthenticationMiddleware,
        )

        async def app(scope, receive, send):
            await JSONResponse({"ok": True})(scope, receive, send)

        middleware = AuthenticationMiddleware(app, ["test-model-key"])

        async def authenticated(scope, receive, send):
            # The native middleware returns an awaitable from a synchronous
            # __call__. Expose the ASGI3 signature explicitly to TestClient.
            await middleware(scope, receive, send)

        client = TestClient(authenticated)
        for path in ("/v1/chat/completions", "/v1/completions", "/v1/models"):
            self.assertEqual(client.post(path).status_code, 401)
            self.assertEqual(
                client.post(
                    path, headers={"Authorization": "Bearer wrong"}
                ).status_code,
                401,
            )
            self.assertEqual(
                client.post(
                    path, headers={"Authorization": "Bearer test-model-key"}
                ).status_code,
                200,
            )
        # The client sends auth consistently, but these paths are not protected
        # by this middleware. Keep the development/control listener loopback-only.
        for path in (
            "/health",
            "/metrics",
            "/tokenize",
            "/collective_rpc",
            "/reset_prefix_cache",
            "/start_profile",
            "/stop_profile",
        ):
            self.assertEqual(client.post(path).status_code, 200)
