"""The running head that the host-side probe tools (weight_digest, kernel_hashes) read."""

import contextlib
from unittest.mock import patch


@contextlib.contextmanager
def served_tool(tool, **rpc):
    """Patch ``tool``'s server and settings and yield the collective_rpc mock.

    ``rpc`` is collective_rpc's ``return_value`` or ``side_effect``. The head is
    container ``c`` on image ``sha256:img``; the profile's fingerprint is ``fp``.
    """
    head = ({"name": "c"}, {"Image": "sha256:img"})
    with (
        patch.object(tool.server, "running_head", return_value=head),
        patch.object(
            tool.server_config, "load", return_value={"runtime": {}, "api": {}}
        ),
        patch.object(tool.server_config, "fingerprint", return_value="fp"),
        patch.object(tool.server, "collective_rpc", **rpc) as mock,
    ):
        yield mock
