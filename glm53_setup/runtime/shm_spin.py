"""Set how long a shared-memory broadcast reader spins before it sleeps (vLLM 385dce36).

A local reader of vLLM's ``MessageQueue`` waits through ``SpinCondition``: for ``busy_loop_s``
after its last read it loops on ``sched_yield()``, then sleeps on a zmq poll. The default is 1 s
(``shm_broadcast.py:134``), the only reader constructor passes no value
(``create_from_handle``, ``:583-585``), and no variable, config or flag reaches it, so a reader
that reads at least once a second never sleeps. When ``GLM53_SHM_SPIN_SECONDS`` is set
(``runtime.shm_spin_seconds``), this module replaces that default; the writer side sets its
own 0 and is unaffected. Only the waiting changes, not what is read.

Imported at interpreter start through ``glm53-shm-spin.pth``. It does not import vLLM itself;
it waits for that module's import.
"""

import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path

TARGET = "vllm.distributed.device_communicators.shm_broadcast"
ENV = "GLM53_SHM_SPIN_SECONDS"
# SpinCondition.__init__'s defaults as pinned: busy_loop_s = 1.
PINNED_DEFAULTS = (1,)
# The .pth installed in the image's site directory (kept as .txt here: *.pth is
# ignored in this repository because PyTorch weights use that suffix).
PTH = Path(__file__).with_name("shm_spin_pth.txt")


def set_spin(module, seconds):
    init = module.SpinCondition.__init__
    if init.__defaults__ != PINNED_DEFAULTS:
        raise RuntimeError(
            f"SpinCondition defaults {init.__defaults__} are not vLLM 385dce36's; "
            "refusing to set the spin"
        )
    init.__defaults__ = (seconds,)


class SetAfterImport(importlib.abc.MetaPathFinder):
    """Finds nothing itself; for the target it wraps the real loader to set after execution."""

    def __init__(self, target, seconds):
        self.target = target
        self.seconds = seconds

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
            set_spin(module, self.seconds)

        loader.exec_module = exec_module
        return spec


def install(environ=os.environ, target=TARGET):
    """Set now if the module is loaded, else when it is; nothing unless a spin is given."""
    if ENV not in environ:
        return "off"
    seconds = float(environ[ENV])
    if target in sys.modules:
        set_spin(sys.modules[target], seconds)
        return "set"
    sys.meta_path.insert(0, SetAfterImport(target, seconds))
    return "waiting"


if __name__ != "__main__":
    install()
