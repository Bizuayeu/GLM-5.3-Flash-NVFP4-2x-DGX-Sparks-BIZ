"""Contracts for the server CLI's entry layer: the action table and its guards.

The launcher's preconditions used to live inside one 202-line ``main``: which
actions run away from the Linux host, which refuse an inherited allocator,
which need rank 0, and which must reach recovery without a loadable profile.
They are table entries here so each one can be read and exercised on its own.
"""

import contextlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from glm53_setup import server
from glm53_setup import server_config as config

ROOT = Path(__file__).resolve().parents[1]
INHERITED_ALLOCATOR = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def profile():
    return config.load(ROOT / "examples/server.example.toml")


@contextlib.contextmanager
def frozen_launch(manifest):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "launch.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        yield path


class HostView:
    """The os module as server sees it, with another host's name and environment.

    Patching os.name itself reaches pathlib, which then refuses to build a path
    of the other platform's flavour: the guard tests would pass only on the
    platform they happen to name.
    """

    def __init__(self, name, environ):
        self.name, self.environ = name, environ

    def __getattr__(self, attribute):
        return getattr(os, attribute)


def on_host(name="posix", environ=None):
    return patch.object(
        server, "os", HostView(name, {} if environ is None else environ)
    )


def dispatch(action, *extra, environ=None, name="posix"):
    """Run main() with this action's handler replaced; return the mock."""
    handler = MagicMock()
    entry = server.ACTIONS[action]._replace(handler=handler)
    with (
        patch.dict(server.ACTIONS, {action: entry}),
        on_host(name, environ),
        patch.object(server.settings, "load", return_value=profile()),
    ):
        server.main([action, *extra])
    return handler


class ActionTableTests(unittest.TestCase):
    def test_every_accepted_action_has_exactly_one_table_entry(self):
        choices = next(
            action.choices
            for action in server.parser()._actions
            if getattr(action, "dest", None) == "action"
        )
        self.assertEqual(sorted(choices), sorted(server.ACTIONS))
        for name, entry in server.ACTIONS.items():
            with self.subTest(action=name):
                self.assertTrue(callable(entry.handler))

    def test_the_launch_path_is_exactly_the_actions_that_can_start_a_rank(self):
        launching = {n for n, e in server.ACTIONS.items() if e.launch_path}
        self.assertEqual(launching, {"start", "preflight", "assets"})

    def test_only_planning_and_freezing_run_away_from_the_model_host(self):
        portable = {n for n, e in server.ACTIONS.items() if e.windows}
        self.assertEqual(portable, {"plan", "freeze"})

    def test_rank_zero_is_required_by_the_running_head_reports(self):
        # ``ask`` also needs rank 0 but refuses with its own message, because it
        # additionally requires the container to be Running.
        scoped = {n for n, e in server.ACTIONS.items() if e.rank0}
        self.assertEqual(scoped, {"capacity", "warmup", "mojibake", "agreement"})

    def test_stop_is_the_only_action_that_runs_before_the_profile_loads(self):
        detached = {n for n, e in server.ACTIONS.items() if not e.needs_profile}
        self.assertEqual(detached, {"stop"})


class ActionGuardTests(unittest.TestCase):
    def test_each_action_reaches_its_own_handler(self):
        for action in server.ACTIONS:
            with self.subTest(action=action):
                dispatch(action).assert_called_once()

    def test_actions_bound_to_the_model_host_refuse_to_run_on_windows(self):
        for action, entry in server.ACTIONS.items():
            # ``stop`` carries its own refusal so recovery reads one message.
            if entry.windows or not entry.needs_profile:
                continue
            with self.subTest(action=action):
                with self.assertRaises(SystemExit):
                    dispatch(action, name="nt")

    def test_stop_refuses_windows_with_its_own_recovery_message(self):
        with on_host("nt"):
            with self.assertRaises(SystemExit):
                server.main(["stop"])

    def test_planning_still_runs_on_windows(self):
        dispatch("plan", name="nt").assert_called_once()

    def test_an_inherited_allocator_is_refused_only_on_the_launch_path(self):
        for action, entry in server.ACTIONS.items():
            if not entry.needs_profile or not entry.launch_path:
                continue
            with self.subTest(action=action):
                with self.assertRaises(SystemExit):
                    dispatch(action, environ=INHERITED_ALLOCATOR)

    def test_actions_off_the_launch_path_ignore_the_hosts_allocator(self):
        for action, entry in server.ACTIONS.items():
            if not entry.needs_profile or entry.launch_path or entry.rank0:
                continue
            with self.subTest(action=action):
                dispatch(action, environ=INHERITED_ALLOCATOR).assert_called_once()

    def test_a_shared_frozen_launch_carries_its_own_allocator(self):
        # The freeze already recorded the launch-origin allocator, so the host's
        # environment is no longer a source of divergence between the ranks.
        handler = MagicMock()
        entry = server.ACTIONS["preflight"]._replace(handler=handler)
        with frozen_launch(server.freeze(profile(), {})) as path:
            with (
                patch.dict(server.ACTIONS, {"preflight": entry}),
                on_host("posix", INHERITED_ALLOCATOR),
            ):
                server.main(["preflight", "--launch", str(path)])
        handler.assert_called_once()

    def test_the_head_reports_refuse_a_peer_rank(self):
        for action, entry in server.ACTIONS.items():
            if not entry.rank0:
                continue
            with self.subTest(action=action):
                with self.assertRaises(SystemExit):
                    dispatch(action, "--rank", "1")

    def test_stop_never_reads_the_operators_settings(self):
        handler = MagicMock()
        entry = server.ACTIONS["stop"]._replace(handler=handler)
        with (
            patch.dict(server.ACTIONS, {"stop": entry}),
            patch.object(server.os, "name", "posix"),
            patch.object(
                server.settings, "load", side_effect=ValueError("bad TOML")
            ) as load,
        ):
            server.main(["stop"])
        load.assert_not_called()
        self.assertIsNone(handler.call_args.args[2])


if __name__ == "__main__":
    unittest.main()
