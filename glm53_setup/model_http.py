"""HTTP transport scoped to an explicit model API; never follow redirects."""

import json
import os
import urllib.error
import urllib.parse
import urllib.request


class ModelHTTPError(urllib.error.URLError):
    """A reportable failure with no headers, response body or credentials."""

    def __init__(self, code=None):
        self.code = code
        kind = "authentication failed" if code in (401, 403) else "request failed"
        super().__init__(f"Model API {kind}" + (f" (HTTP {code})" if code else ""))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def authorization(environ=None):
    env = os.environ if environ is None else environ
    key = env.get("API_KEY") or env.get("VLLM_API_KEY")
    if not key:
        return {}
    if not isinstance(key, str) or any(ord(c) < 33 or ord(c) > 126 for c in key):
        raise ValueError("Model API key must contain printable ASCII without spaces")
    return {"Authorization": "Bearer " + key}


def open_response(base_url, path, *, body=None, timeout=600, environ=None):
    """Return a closing/context-managed response, including an SSE stream.

    base_url explicitly selects an origin; path is relative to that origin.
    Even same-origin redirects fail, making retries and credential scope explicit.
    Download clients must not use this authenticated model-only transport.
    """
    base = urllib.parse.urlsplit(base_url)
    endpoint = urllib.parse.urlsplit(path)
    if (
        base.scheme not in ("http", "https")
        or not base.hostname
        or base.username is not None
        or base.password is not None
        or base.path not in ("", "/")
        or base.query
        or base.fragment
        or endpoint.scheme
        or endpoint.netloc
        or not path.startswith("/")
        or path.startswith("//")
        or endpoint.fragment
        or any(ord(c) < 33 for c in base_url + path)
    ):
        raise ValueError("Use an explicit HTTP model origin and an absolute local path")
    headers = authorization(environ)
    data = None if body is None else json.dumps(body).encode("utf-8")
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(base_url.rstrip("/") + path, data, headers)
    try:
        return urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        code = error.code
        error.close()
        raise ModelHTTPError(code) from None
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        raise ModelHTTPError() from None


def post_json(base_url, path, body, *, timeout=600):
    with open_response(base_url, path, body=body, timeout=timeout) as response:
        return json.load(response)
