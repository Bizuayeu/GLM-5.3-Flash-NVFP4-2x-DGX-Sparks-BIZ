"""Source-pinned patch: each safetensors tensor is copied off the file mapping before it is loaded.

vLLM 385dce36's default (lazy) iterator yields ``safe_open(...).get_tensor(name)``, a tensor backed
by the checkpoint file's memory mapping, and the loader copies it to the GPU from there. On GB10 with
a CUDA context, that host-to-device copy from a file-backed mapping runs at about 0.16 GiB/s; the
same tensor cloned into anonymous memory first moves at about 1.55 GiB/s (edgexpert03, image
0eede6e0, 3 GiB of one shard per run, two runs each; records/20260926-release-1190). Rank 0 of the
reference pair spent 532 s in ``Loading weights`` on 1.18.0. The clone holds one tensor at a time,
unlike vLLM's ``eager`` strategy, which keeps a whole shard twice (22.81 GiB peak for an 11.15 GiB
shard, measured the same day) and ran rank 1 out of memory. Upstream: the same idea appears in
MiaAI-Lab's recipe (no code taken, AGPL-3.0); vLLM has no such change at the pin.
"""

# cc-defer: carries a local staging copy on the pinned loader; drop it (and the
# Dockerfile RUN) when a vLLM pin loads file-backed tensors without the slow path,
# or when the GB10 driver copies from file mappings at anonymous-memory speed.

from . import pinned_patch
from .pinned_patch import replace_once

TARGET = "model_executor/model_loader/weight_utils.py"
SOURCE_SHA256 = "8ef6e3e92b0d6ee3386088427fb97f96c544649c7baf0379a82a73dd541d2e8d"
MISMATCH = "weight loader source hash mismatch"
RECORD = "glm53-load-clone-patch.json"
LAZY = (
    "                    param = f.get_tensor(name)\n"
    "                    yield name, param\n"
)
CLONED = (
    "                    # GLM setup: copy off the file mapping; GB10 copies to the GPU\n"
    "                    # from a file-backed mapping about ten times slower.\n"
    "                    param = f.get_tensor(name).clone()\n"
    "                    yield name, param\n"
)
HEADER = (
    "# Modified by GLM setup: safetensors tensors are cloned off the file mapping before loading.\n"
    "# Original vLLM Apache-2.0 notices below remain applicable.\n"
)


def patch_text(text):
    if CLONED in text:
        raise ValueError("load clone patch already applied")
    patched = HEADER + replace_once(text, LAZY, CLONED)
    compile(patched, TARGET, "exec")
    return patched


def prepare(package):
    return pinned_patch.prepare(package, TARGET, SOURCE_SHA256, MISMATCH, patch_text)


def main(argv=None):
    pinned_patch.main(
        argv,
        doc=__doc__,
        target=TARGET,
        sha256=SOURCE_SHA256,
        prepare=prepare,
        record=RECORD,
    )


if __name__ == "__main__":
    main()
