"""Which tool calls break the declared schema (decision layer: pure, no I/O).

Only rules the tool's JSON schema states are applied, so nothing here names a benchmark scenario:
arguments that are not a JSON object, a required argument that is absent, and a required string
argument that is empty or blank. Types, enums and ranges are not checked. A tool the request did
not declare, or one without a required list, is left to the server and the client.
"""

import json


def _schemas(tools):
    out = {}
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            parameters = function.get("parameters")
            out[function["name"]] = parameters if isinstance(parameters, dict) else {}
    return out


def violations(tools, tool_calls):
    """One record per violating call: call_id, tool, kind (not_object/missing/empty), argument."""
    schemas = _schemas(tools)
    found = []
    for tool_call in tool_calls or []:
        function = tool_call.get("function") or {}
        name = function.get("name")
        schema = schemas.get(name)
        required = schema.get("required") if schema else None
        if not required:
            continue

        def record(kind, argument=None, _id=tool_call.get("id"), _name=name):
            found.append(
                {"call_id": _id, "tool": _name, "kind": kind, "argument": argument}
            )

        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except (TypeError, ValueError):
            arguments = None
        if not isinstance(arguments, dict):
            record("not_object")
            continue
        properties = schema.get("properties") or {}
        for key in required:
            value = arguments.get(key)
            if value is None:
                record("missing", key)
            elif (properties.get(key) or {}).get("type") == "string" and (
                isinstance(value, str) and not value.strip()
            ):
                record("empty", key)
    return found
