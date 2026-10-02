import contextlib
import gzip
import importlib.metadata
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_setup.validation import profile_trace
from glm53_setup.validation.profile_trace import (
    decode_delta,
    graph_launches,
    package_versions,
    summarize_trace,
)


class KernelTraceTests(unittest.TestCase):
    def test_actual_kernel_events_are_not_cpu_launch_calls(self):
        events = [
            {"cat": "kernel", "ph": "X", "name": "gemm", "dur": 20},
            {"cat": "kernel", "ph": "X", "name": "ncclAllReduce", "dur": 30},
            {"cat": "cuda_runtime", "ph": "X", "name": "cudaLaunchKernel", "dur": 2},
            {"cat": "cpu_op", "ph": "X", "name": "aten::matmul", "dur": 100},
            {
                "cat": "cuda_runtime",
                "ph": "X",
                "name": "cudaStreamSynchronize",
                "dur": 11,
            },
        ]
        result = summarize_trace({"traceEvents": events})
        self.assertEqual(result["kernel_events"], 2)
        self.assertEqual(result["launch_api_events"], 1)
        self.assertEqual(result["nccl_kernel_events"], 1)
        self.assertEqual(result["synchronization_api_events"], 1)
        self.assertEqual(result["summed_synchronization_api_duration_us"], 11)
        self.assertEqual(result["summed_kernel_duration_us"], 50)
        self.assertEqual(
            result["kernel_duration_us_by_name"], {"gemm": 20, "ncclAllReduce": 30}
        )
        with self.assertRaises(ValueError):
            summarize_trace({"traceEvents": events[2:]})

    def test_prefill_control_is_subtracted_before_per_token_estimate(self):
        value = decode_delta(
            {"kernel_events": 310, "nccl_kernel_events": 60},
            {"kernel_events": 110, "nccl_kernel_events": 20},
            3,
            1,
        )
        self.assertEqual(value["kernel_event_delta_per_token"], 100)
        self.assertEqual(value["nccl_event_delta_per_token"], 20)

    def test_transfer_events_are_separate_and_unknown_bytes_are_not_zero_claims(self):
        result = summarize_trace(
            {
                "traceEvents": [
                    {"cat": "kernel", "ph": "X", "name": "gemm", "dur": 3},
                    {
                        "cat": "gpu_memcpy",
                        "ph": "X",
                        "name": "Memcpy HtoD",
                        "dur": 8,
                        "args": {"bytes": 4096},
                    },
                    {"cat": "gpu_memcpy", "ph": "X", "name": "Memcpy DtoD", "dur": 2},
                    {"cat": "gpu_memset", "ph": "X", "name": "Memset", "dur": 1},
                ]
            }
        )
        self.assertEqual(result["memcpy_events"], 2)
        self.assertEqual(result["memcpy_known_bytes"], 4096)
        self.assertEqual(result["memcpy_events_without_byte_count"], 1)
        self.assertEqual(result["memcpy_names"], {"Memcpy HtoD": 1, "Memcpy DtoD": 1})


class GraphLaunchTests(unittest.TestCase):
    def test_only_completed_cuda_graph_launch_calls_count(self):
        events = [
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch"},
            {"ph": "X", "cat": "cuda_driver", "name": "cuGraphLaunch_v2"},
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel"},
            {"ph": "B", "cat": "cuda_runtime", "name": "cudaGraphLaunch"},
            {"ph": "X", "cat": "kernel", "name": "GraphLaunch"},
            {"ph": "X", "cat": "cuda_runtime"},
        ]
        self.assertEqual(graph_launches({"traceEvents": events}), 2)
        self.assertEqual(graph_launches({}), 0)


class PackageVersionTests(unittest.TestCase):
    def test_a_missing_package_is_recorded_as_none(self):
        def version(name):
            if name == "torch":
                return "2.9.0"
            raise importlib.metadata.PackageNotFoundError(name)

        with patch.object(importlib.metadata, "version", side_effect=version):
            self.assertEqual(
                package_versions(("torch", "flashinfer-python")),
                {"torch": "2.9.0", "flashinfer-python": None},
            )


def trace(*kernels):
    return {
        "traceEvents": [
            {"cat": "kernel", "ph": "X", "name": name, "dur": 1} for name in kernels
        ]
    }


class MainTests(unittest.TestCase):
    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            profile_trace.main(list(argv))
        return out.getvalue(), err.getvalue()

    def test_a_gzipped_trace_is_summarized_and_paired_with_its_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            measured = Path(tmp) / "measured.json.gz"
            with gzip.open(measured, "wt", encoding="utf-8") as stream:
                json.dump(trace("gemm", "gemm", "ncclAllReduce", "gemm"), stream)
            control = Path(tmp) / "control.json"
            control.write_text(json.dumps(trace("gemm")), encoding="utf-8")
            out, _ = self.run_main(str(measured))
            self.assertEqual(json.loads(out)["kernel_events"], 4)
            self.assertNotIn("decode_delta", json.loads(out))
            out, _ = self.run_main(
                str(measured),
                "--prefill-control",
                str(control),
                "--output-tokens",
                "4",
                "--control-tokens",
                "1",
            )
        delta = json.loads(out)["decode_delta"]
        self.assertEqual(delta["additional_output_tokens"], 3)
        self.assertEqual(delta["kernel_event_delta_per_token"], 1)
        self.assertEqual(delta["nccl_event_delta_per_token"], 1 / 3)

    def test_a_prefill_control_without_both_token_counts_is_a_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            path.write_text(json.dumps(trace("gemm")), encoding="utf-8")
            for counts in ((), ("--output-tokens", "4"), ("--control-tokens", "1")):
                with self.subTest(counts=counts):
                    with self.assertRaises(SystemExit) as caught:
                        self.run_main(
                            str(path), "--prefill-control", str(path), *counts
                        )
                    self.assertEqual(caught.exception.code, 2)

    def test_a_trace_without_gpu_kernels_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            path.write_text(json.dumps({"traceEvents": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No GPU kernel events"):
                self.run_main(str(path))
