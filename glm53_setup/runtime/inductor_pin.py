"""Keep Inductor's deterministic mode on through Dynamo's state restore (torch 2.12.1, 2.13).

``TORCHINDUCTOR_DETERMINISTIC=1`` sets ``torch._inductor.config.deterministic`` on
import, and the mode picks reduction configs without timing, the same on every
rank. But Dynamo restores global state after every frame it compiles with
``torch.use_deterministic_algorithms(prior)``, and that function also assigns
``inductor_config.deterministic = prior``. The global flag is off by default, so
the first compiled frame turns the mode off and every later kernel is generated
without it (a serving worker on 2026-09-25: fifteen such writes, the kernels'
meta ``deterministic: False``). torch reads an entry's ``env_value_force`` before
any override, so setting it keeps the mode on.

Imported at interpreter start through ``glm53-inductor-pin.pth``. It must not
import the config module itself: vLLM sets other TORCHINDUCTOR_* variables that
the module reads on import, later in start-up. So it waits for that import.
"""

import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path

TARGET = "torch._inductor.config"
# The .pth installed in the image's site directory (kept as .txt here: *.pth is
# ignored in this repository because PyTorch weights use that suffix).
PTH = Path(__file__).with_name("inductor_pin_pth.txt")


def pin(module):
    module._config["deterministic"].env_value_force = True


class PinAfterImport(importlib.abc.MetaPathFinder):
    """Finds nothing itself; for the target it wraps the real loader to pin after execution."""

    def __init__(self, target):
        self.target = target

    def find_spec(self, name, path, target=None):
        if name != self.target:
            return None
        sys.meta_path.remove(self)  # one shot, and not asked again below
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        original = loader.exec_module

        def exec_module(module):
            original(module)
            pin(module)

        loader.exec_module = exec_module
        return spec


def install(environ=os.environ, target=TARGET):
    """Pin now if the config module is loaded, else when it is; nothing unless the mode is asked for."""
    if environ.get("TORCHINDUCTOR_DETERMINISTIC") != "1":
        return "off"
    if target in sys.modules:
        pin(sys.modules[target])
        return "pinned"
    sys.meta_path.insert(0, PinAfterImport(target))
    return "waiting"


if __name__ != "__main__":
    install()
