import copy
import unittest

from glm53_setup.switch import (
    DEGENERATE_ENGINE,
    READINESS_UNCONFIRMED,
    OperationFailure,
    resume,
    stop_owned,
    switch,
)


class Backend:
    def __init__(self):
        self.calls = []
        self.fail_rank = None
        self.mismatch = False
        self.changed = False
        self.prepares = 0
        self.marked = {}

    def current(self, rank):
        return {"launch": "old", "name": f"old-{rank}"}

    def prepare(self, rank, launch, *, recovery=False):
        self.prepares += 1
        self.marked[("prepare", launch)] = recovery
        if self.fail_rank == rank:
            raise ValueError("missing asset")
        return {
            "common": "bad" if self.mismatch and rank else "same",
            "local": self.prepares if self.changed else rank,
        }

    def stop(self, rank, identity):
        self.calls.append(("stop", rank, identity["name"]))

    def reserve(self, rank, launch, *, recovery=False):
        self.marked[("reserve", launch)] = recovery
        return {"launch": launch, "name": f"{launch}-{rank}"}

    def start(self, rank, identity):
        self.calls.append(("start", rank, identity["name"]))
        if identity["name"] == "new-0":
            raise RuntimeError("second rank launch failed")

    def ready(self, rows):
        self.calls.append(("ready", tuple(r["identity"]["name"] for r in rows)))

    def warmup(self, rows):
        self.calls.append(("warmup", tuple(r["identity"]["name"] for r in rows)))
        return {"skipped": True}

    def install(self, rank, identity, config):
        self.calls.append(("install", rank, identity["name"], config))
        return {"written": True}


def startable(backend):
    backend.start = lambda rank, identity: backend.calls.append(
        ("start", rank, identity["name"])
    )
    return backend


class StopOwnedTests(unittest.TestCase):
    def test_every_row_is_stopped_in_the_given_order_and_failures_are_kept(self):
        backend = Backend()
        stop = backend.stop

        def flaky(rank, identity):
            stop(rank, identity)
            if rank == 1:
                raise OperationFailure("stop", 1, "transport-timeout")

        backend.stop = flaky
        rows = [{"rank": r, "identity": {"name": f"new-{r}"}} for r in (1, 0)]
        report = {}
        self.assertFalse(stop_owned(backend, rows, report))
        self.assertEqual(backend.calls, [("stop", 1, "new-1"), ("stop", 0, "new-0")])
        self.assertEqual(
            report["cleanup_errors"],
            [
                {
                    "rank": 1,
                    "error": "OperationFailure",
                    "failure": {
                        "action": "stop",
                        "rank": 1,
                        "reason": "transport-timeout",
                        "exit_code": None,
                    },
                }
            ],
        )
        clean = {}
        self.assertTrue(stop_owned(Backend(), rows, clean))
        self.assertEqual(clean, {})


class SwitchTests(unittest.TestCase):
    def test_profile_text_reaches_both_ranks_after_the_pair_is_complete(self):
        backend = startable(Backend())
        reports = []
        result = switch(
            backend,
            "new",
            save=lambda r: reports.append(copy.deepcopy(r)),
            config="text",
        )
        names = [call[0] for call in backend.calls]
        self.assertEqual(
            [c for c in backend.calls if c[0] == "install"],
            [("install", 0, "new-0", "text"), ("install", 1, "new-1", "text")],
        )
        # The ladder gates the pair (Mia #268), so it runs before the profile
        # text is written: a degenerate candidate never gets its file.
        self.assertLess(names.index("ready"), names.index("warmup"))
        self.assertLess(names.index("warmup"), names.index("install"))
        self.assertEqual(
            result["config"],
            [{"rank": 0, "written": True}, {"rank": 1, "written": True}],
        )
        first = next(r for r in reports if r["status"] == "complete")
        self.assertNotIn("config", first)

    def test_no_profile_text_means_no_install(self):
        backend = startable(Backend())
        result = switch(backend, "new", save=lambda r: None)
        self.assertNotIn("install", [call[0] for call in backend.calls])
        self.assertNotIn("config", result)

    def test_a_failed_switch_never_installs_the_candidate_profile(self):
        backend = Backend()
        with self.assertRaises(RuntimeError):
            switch(backend, "new", save=lambda r: None, config="text")
        self.assertNotIn("install", [call[0] for call in backend.calls])

    def test_install_failure_is_recorded_and_never_rolls_back_a_complete_pair(self):
        backend = startable(Backend())

        def broken(rank, identity, config):
            raise OperationFailure("install", rank, "remote-operation-failed", 1)

        backend.install = broken
        result = switch(backend, "new", save=lambda r: None, config="text")
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["config"]["failed"])
        self.assertEqual(result["config"]["failure"]["action"], "install")
        self.assertEqual(result["recovery"], [])
        self.assertEqual(backend.calls[-1][0], "warmup")

    def test_warmup_failure_is_recorded_and_never_rolls_back_a_complete_pair(self):
        backend = Backend()
        backend.start = lambda rank, identity: backend.calls.append(
            ("start", rank, identity["name"])
        )

        def broken(rows):
            backend.calls.append(("warmup", "raised"))
            raise OperationFailure("warmup", 0, "remote-operation-failed", 1)

        backend.warmup = broken
        reports = []
        result = switch(backend, "new", save=lambda r: reports.append(copy.deepcopy(r)))
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["warmup"]["failed"])
        self.assertEqual(result["warmup"]["failure"]["action"], "warmup")
        self.assertEqual(backend.calls[-2:][0][0], "ready")
        self.assertEqual(backend.calls[-1], ("warmup", "raised"))
        self.assertNotIn("recovery_errors", result)
        self.assertEqual(result["recovery"], [])
        # A ladder that could not run is evidence, not a verdict.
        self.assertIn("complete", [r["status"] for r in reports])

    def test_lost_recovery_observation_does_not_destroy_the_recovering_pair(self):
        backend = Backend()

        def unavailable(rows):
            raise OperationFailure("poll", 1, "ssh-unavailable", 255)

        backend.ready = unavailable
        reports = []
        with self.assertRaises(RuntimeError):
            switch(backend, "new", save=lambda r: reports.append(copy.deepcopy(r)))
        self.assertEqual(reports[-1]["status"], "recovery-readiness-unconfirmed")
        self.assertEqual(
            reports[-1]["recovery_observation_failure"]["reason"], "ssh-unavailable"
        )
        self.assertEqual(
            backend.calls[-2:], [("start", 1, "old-1"), ("start", 0, "old-0")]
        )

    def test_lost_readiness_observation_preserves_supervised_new_attempts(self):
        backend = Backend()
        backend.start = lambda rank, identity: backend.calls.append(
            ("start", rank, identity["name"])
        )

        def unavailable(rows):
            raise OperationFailure("poll", 1, "ssh-unavailable", 255)

        backend.ready = unavailable
        reports = []
        with self.assertRaisesRegex(RuntimeError, "Readiness observation lost"):
            switch(backend, "new", save=lambda r: reports.append(copy.deepcopy(r)))
        self.assertEqual(reports[-1]["status"], "readiness-unconfirmed")
        self.assertEqual(
            backend.calls,
            [
                ("stop", 0, "old-0"),
                ("stop", 1, "old-1"),
                ("start", 1, "new-1"),
                ("start", 0, "new-0"),
            ],
        )

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

    def test_only_the_running_pair_is_prepared_and_reserved_as_a_recovery_target(self):
        # A new launch must meet today's image requirements; the pair that is
        # already serving only has to be restartable as it was.
        backend = Backend()
        with self.assertRaises(RuntimeError):
            switch(backend, "new", save=lambda r: None)
        self.assertEqual(
            backend.marked,
            {
                ("prepare", "new"): False,
                ("prepare", "old"): True,
                ("reserve", "new"): False,
                ("reserve", "old"): True,
            },
        )

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


def degenerate(backend):
    def warmup(rows):
        backend.calls.append(("warmup", tuple(r["identity"]["name"] for r in rows)))
        return {"degenerate": True, "canary": {"answer_ok": False}}

    backend.warmup = warmup
    return backend


class CanaryGateTests(unittest.TestCase):
    """Mia #268: only a degenerate verdict fails the pair; the old one returns."""

    def test_a_degenerate_candidate_is_stopped_and_the_old_pair_recovered(self):
        backend = degenerate(startable(Backend()))
        reports = []
        with self.assertRaises(RuntimeError):
            switch(
                backend,
                "new",
                save=lambda r: reports.append(copy.deepcopy(r)),
                config="text",
            )
        report = reports[-1]
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure"]["reason"], DEGENERATE_ENGINE)
        self.assertTrue(report["warmup"]["degenerate"])
        self.assertTrue(report["recovered"])
        self.assertNotIn("install", [call[0] for call in backend.calls])
        self.assertNotIn("complete", [r["status"] for r in reports])
        self.assertEqual(
            backend.calls[-6:],
            [
                ("warmup", ("new-1", "new-0")),
                ("stop", 0, "new-0"),
                ("stop", 1, "new-1"),
                ("start", 1, "old-1"),
                ("start", 0, "old-0"),
                ("ready", ("old-1", "old-0")),
            ],
        )

    def test_a_degenerate_first_launch_is_stopped_without_recovery(self):
        backend = degenerate(startable(Backend()))
        backend.current = lambda rank: None
        reports = []
        with self.assertRaises(RuntimeError):
            switch(backend, "new", save=lambda r: reports.append(copy.deepcopy(r)))
        self.assertEqual(reports[-1]["status"], "failed")
        self.assertEqual(reports[-1]["recovery"], [])
        self.assertEqual(
            backend.calls[-2:], [("stop", 0, "new-0"), ("stop", 1, "new-1")]
        )

    def test_a_resumed_degenerate_pair_is_stopped_and_reported(self):
        backend = degenerate(Backend())
        rows = [
            {
                "rank": 1,
                "identity": {"launch": "new", "name": "new-1", "fingerprint": "f"},
            },
            {
                "rank": 0,
                "identity": {"launch": "new", "name": "new-0", "fingerprint": "f"},
            },
        ]
        backend.current = lambda rank: {"name": f"new-{rank}", "fingerprint": "f"}
        report = {
            "status": READINESS_UNCONFIRMED,
            "new": rows,
            "assets": {0: backend.prepare(0, "new"), 1: backend.prepare(1, "new")},
            "failure": {"action": "poll"},
            "error": "OperationFailure",
            "recovery": [],
        }
        backend.prepare = lambda rank, launch, **kw: report["assets"][rank]
        saved = []
        with self.assertRaisesRegex(RuntimeError, "degenerate"):
            resume(
                backend,
                report,
                config="text",
                save=lambda r: saved.append(copy.deepcopy(r)),
            )
        self.assertEqual(saved[-1]["status"], "failed")
        self.assertEqual(saved[-1]["failure"]["reason"], DEGENERATE_ENGINE)
        self.assertEqual(
            backend.calls[-2:], [("stop", 0, "new-0"), ("stop", 1, "new-1")]
        )
        self.assertNotIn("install", [call[0] for call in backend.calls])
