import contextlib
import io
import unittest
import urllib.error

from glm53_setup import model_http
from glm53_setup.validation import run_components


def sample(*token_ids):
    return {"seconds": 0.1, "response": {"choices": [{"token_ids": list(token_ids)}]}}


def captured(layer, position, indices):
    return {
        "request_id": "validation-2048-capture",
        "query_position": position,
        "coordinate_space": "logical_tokens",
        "layer": layer,
        "indices": indices,
    }


class ParserTests(unittest.TestCase):
    def test_the_three_paths_are_required(self):
        args = run_components.parser().parse_args(
            ["--config", "c.toml", "--corpus", "d.jsonl", "--output", "out"]
        )
        self.assertEqual(args.output.name, "out")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_components.parser().parse_args(["--config", "c.toml"])


class ModeAgreementTests(unittest.TestCase):
    def test_tokens_equal_needs_every_mode_and_native_repeatable_only_off_and_restored(
        self,
    ):
        modes = {
            "off": {"samples": [sample(1, 2), sample(1, 2)]},
            "fused": {"samples": [sample(1, 2)]},
            "restored": {"samples": [sample(1, 2)]},
        }
        self.assertEqual(
            run_components.mode_agreement(modes),
            {"tokens_equal": True, "native_repeatable": True},
        )
        modes["fused"]["samples"] = [sample(1, 3)]
        self.assertEqual(
            run_components.mode_agreement(modes),
            {"tokens_equal": False, "native_repeatable": True},
        )
        modes["restored"]["samples"] = [sample(9)]
        self.assertEqual(
            run_components.mode_agreement(modes),
            {"tokens_equal": False, "native_repeatable": False},
        )


class OverlapTests(unittest.TestCase):
    def test_pairs_each_query_position_across_layer_gaps_of_one_to_three(self):
        rows = [
            captured(3, 2047, [0, 1, 2]),
            captured(7, 2047, [1, 2, 3]),
            captured(11, 2047, [2, 3, 4]),
            captured(15, 2047, [0, 1]),
            captured(3, 4095, [5]),
            captured(7, 4095, [5]),
        ]
        overlap = run_components.overlap_rows(rows)
        pairs = [(r["source_layer"], r["target_layer"]) for r in overlap]
        self.assertEqual(
            pairs,
            [
                (3, 7),
                (7, 11),
                (11, 15),
                (3, 11),
                (7, 15),
                (3, 15),
                (3, 7),
            ],
        )
        self.assertEqual(overlap[0]["query_position"], 2047)
        self.assertAlmostEqual(overlap[0]["jaccard"], 2 / 4)
        self.assertEqual(overlap[-1]["query_position"], 4095)
        self.assertEqual(overlap[-1]["jaccard"], 1.0)
        self.assertEqual(run_components.overlap_rows([]), [])


class ReadinessTests(unittest.TestCase):
    def wait(self, answers, clock=None):
        calls = []
        sleeps = []
        ticks = iter(clock or range(100))

        @contextlib.contextmanager
        def probe(base_url, path, **kwargs):
            calls.append((base_url, path, kwargs))
            answer = answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            yield type("Response", (), {"status": answer})()

        run_components.wait_ready(
            {"api": {"port": 8000}},
            open_response=probe,
            sleep=sleeps.append,
            clock=lambda: next(ticks),
            deadline=30,
        )
        return calls, sleeps

    def test_retries_connection_and_timeout_failures_until_health_answers(self):
        calls, sleeps = self.wait(
            [urllib.error.URLError("refused"), TimeoutError(), 503, 200]
        )
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0][0], "http://127.0.0.1:8000")
        self.assertEqual(calls[0][1], "/health")
        self.assertEqual(calls[0][2], {"timeout": 5})
        # A reachable but not-yet-healthy server is asked again at once.
        self.assertEqual(sleeps, [5, 5])

    def test_an_authentication_failure_is_not_retried(self):
        with self.assertRaises(model_http.ModelHTTPError):
            self.wait([model_http.ModelHTTPError(403)])
        with self.assertRaises(model_http.ModelHTTPError):
            self.wait([model_http.ModelHTTPError(401)])

    def test_the_deadline_ends_the_wait(self):
        with self.assertRaisesRegex(TimeoutError, "readiness deadline"):
            self.wait([TimeoutError()] * 5, clock=[0, 10, 20, 30, 40, 50])


if __name__ == "__main__":
    unittest.main()
