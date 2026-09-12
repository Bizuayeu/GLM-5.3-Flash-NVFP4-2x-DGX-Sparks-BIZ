"""Compare restricted scoring with the pinned native full-pool indexer."""

import argparse
import json
import statistics
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import torch
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

    from glm53_setup.runtime.indexer_reindex import candidate_scores_cuda
    from glm53_setup.runtime.indexer_shared_pool import shared_pool_scores

    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(73)
    report = {
        "scope": "synthetic indexer scoring only; candidate recall and model quality untested",
        "device": torch.cuda.get_device_name(),
        "cases": [],
    }
    for queries, pools, count in (
        (32, 2048, 1024),
        (512, 2048, 1024),
        (512, 8192, 1024),
    ):
        q = torch.randn((queries, 32, 128), device="cuda").to(torch.float8_e4m3fn)
        k = torch.randn((pools, 128), device="cuda").to(torch.float8_e4m3fn)
        scale = torch.rand(pools, device="cuda") + 0.1
        weights = torch.rand((queries, 32), device="cuda")
        pool = (
            torch.randperm(pools, device="cuda")[:count].sort().values.to(torch.int32)
        )
        ids = pool.repeat(queries, 1)
        starts = torch.zeros(queries, dtype=torch.int32, device="cuda")
        limits = torch.full((queries,), pools, dtype=torch.int32, device="cuda")

        def native():
            return fp8_fp4_mqa_logits(
                (q, None), (k, scale), weights, starts, limits, clean_logits=True
            )

        def restricted():
            return candidate_scores_cuda(q, k, scale, weights, ids, limits)

        def shared():
            return shared_pool_scores(q, k, scale, weights, pool, limits)

        expected = native().gather(1, ids.long())
        actual = restricted()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-4)
        torch.testing.assert_close(shared(), expected, rtol=2e-5, atol=2e-4)
        case = {
            "queries": queries,
            "native_pools": pools,
            "candidate_pools": count,
            "max_abs_error": (actual - expected).abs().max().item(),
            "paths": {},
        }
        for name, fn in (
            ("native_full", native),
            ("restricted_checked", restricted),
            ("shared_pool_checked", shared),
        ):
            for _ in range(10):
                fn()
            samples = []
            for _ in range(5):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                torch.cuda.synchronize()
                start.record()
                for _ in range(20):
                    fn()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / 20)
            case["paths"][name] = {
                "ms_samples": samples,
                "median_ms": statistics.median(samples),
            }
        report["cases"].append(case)
        (args.output / "result.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(case, flush=True)


if __name__ == "__main__":
    main()
