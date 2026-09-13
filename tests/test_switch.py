import copy
import unittest

from glm53_setup.switch import switch


class Backend:
    def __init__(self):
        self.calls = []
        self.fail_rank = None
        self.mismatch = False
        self.changed = False
        self.prepares = 0

    def current(self, rank):
        return {"launch": "old", "name": f"old-{rank}"}

    def prepare(self, rank, launch):
        self.prepares += 1
        if self.fail_rank == rank:
            raise ValueError("missing asset")
        return {
            "common": "bad" if self.mismatch and rank else "same",
            "local": self.prepares if self.changed else rank,
        }

    def stop(self, rank, identity):
        self.calls.append(("stop", rank, identity["name"]))

    def reserve(self, rank, launch):
        return {"launch": launch, "name": f"{launch}-{rank}"}

    def start(self, rank, identity):
        self.calls.append(("start", rank, identity["name"]))
        if identity["name"] == "new-0":
            raise RuntimeError("second rank launch failed")

    def ready(self, rows):
        self.calls.append(("ready", tuple(r["identity"]["name"] for r in rows)))


class SwitchTests(unittest.TestCase):
    def test_pre_stop_failure_never_stops_anything(self):
        for attribute, value in [
            ("fail_rank", 1),
            ("mismatch", True),
            ("changed", True),
        ]:
            backend = Backend()
            setattr(backend, attribute, value)
            with self.assertRaises(ValueError):
                switch(backend, "new", save=lambda r: None)
            self.assertEqual(backend.calls, [])

    def test_partial_launch_stops_only_new_identities_then_recovers_previous(self):
        backend = Backend()
        reports = []
        with self.assertRaises(RuntimeError):
            switch(backend, "new", save=lambda r: reports.append(copy.deepcopy(r)))
        self.assertEqual(
            backend.calls,
            [
                ("stop", 0, "old-0"),
                ("stop", 1, "old-1"),
                ("start", 1, "new-1"),
                ("start", 0, "new-0"),
                ("stop", 0, "new-0"),
                ("stop", 1, "new-1"),
                ("start", 1, "old-1"),
                ("start", 0, "old-0"),
                ("ready", ("old-1", "old-0")),
            ],
        )
        self.assertTrue(reports[-1]["recovered"])

    def test_cleanup_failure_never_starts_recovery_over_an_unknown_process(self):
        backend = Backend()
        original = backend.stop

        def stop(rank, identity):
            if identity["name"].startswith("new"):
                raise OSError("unreachable")
            original(rank, identity)

        backend.stop = stop
        reports = []
        with self.assertRaises(RuntimeError):
            switch(backend, "new", save=lambda r: reports.append(copy.deepcopy(r)))
        self.assertFalse(reports[-1]["recovery"])
        self.assertEqual(len(reports[-1]["cleanup_errors"]), 2)
