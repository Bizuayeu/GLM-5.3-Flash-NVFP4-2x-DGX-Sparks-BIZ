import unittest

from glm53_setup.validation.profile_trace import decode_delta, summarize_trace


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
