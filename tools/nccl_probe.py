"""Two-rank PyTorch/NCCL correctness and timing probe, not model qualification.

Run one process per GPU host with matching head/port and an explicit NCCL fabric
environment. Record both results and transport logs; timings include Python
dispatch and synchronization. No data download or network configuration occurs.
"""

import argparse
import datetime
import json
import os
import pathlib
import time


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--port", type=int, default=29653)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args(argv)
    if args.rank not in (0, 1):
        p.error("rank must be 0 or 1")
    if not 1024 <= args.port <= 65535:
        p.error("port must be 1024..65535")
    if args.output.exists():
        p.error("use a fresh output path; existing evidence is preserved")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(0)
    report = {
        "rank": args.rank,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "torch_reported_nccl": torch.cuda.nccl.version(),
        "device": torch.cuda.get_device_name(0),
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("NCCL_") or k == "GLOO_SOCKET_IFNAME"
        },
        "tests": [],
        "passed": False,
        "full_model_inference_validated": False,
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2))

    save()
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{args.head}:{args.port}",
        rank=args.rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=90),
        device_id=torch.device("cuda", 0),
    )
    try:
        for dtype in [torch.float32, torch.bfloat16]:
            for size in [1024, 1024**2, 16 * 1024**2, 256 * 1024**2]:
                n = size // torch.empty((), dtype=dtype).element_size()
                pattern = (torch.arange(n, device="cuda", dtype=torch.int64) % 7).to(
                    dtype
                )
                value = pattern + args.rank + 1
                dist.all_reduce(value)
                good = bool(torch.equal(value, pattern * 2 + 3))
                # Repeated sums remain exactly representable for this pattern and 10 iterations.
                for _ in range(3):
                    dist.all_reduce(value)
                value.copy_(pattern + args.rank + 1)
                dist.barrier()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(10):
                    dist.all_reduce(value)
                torch.cuda.synchronize()
                seconds = (time.perf_counter() - start) / 10
                good = good and bool(torch.equal(value, (pattern * 2 + 3) * 512))
                item = {
                    "operation": "all_reduce",
                    "dtype": str(dtype),
                    "bytes_per_rank": size,
                    "iterations": 10,
                    "seconds_per_operation": seconds,
                    "payload_GB_per_s": size / seconds / 1e9,
                    "passed": good,
                }
                report["tests"].append(item)
                save()
                print(json.dumps(item), flush=True)
                del value, pattern
        n = 1024 * 1024
        pattern = torch.arange(n, device="cuda", dtype=torch.int64) % 7
        source = (pattern + args.rank + 1).float()
        output = torch.empty(2 * n, device="cuda")
        dist.all_gather_into_tensor(output, source)
        good = all(
            torch.equal(output[r * n : (r + 1) * n], (pattern + r + 1).float())
            for r in (0, 1)
        )
        report["tests"].append(
            {"operation": "all_gather", "bytes_per_rank": n * 4, "passed": good}
        )
        whole = torch.arange(2 * n, device="cuda", dtype=torch.int64) % 7
        input_ = (whole + args.rank + 1).float()
        reduced = torch.empty(n, device="cuda")
        dist.reduce_scatter_tensor(reduced, input_)
        good = bool(
            torch.equal(
                reduced, (whole[args.rank * n : (args.rank + 1) * n] * 2 + 3).float()
            )
        )
        report["tests"].append(
            {
                "operation": "reduce_scatter",
                "input_bytes_per_rank": 2 * n * 4,
                "passed": good,
            }
        )
        source.copy_((pattern + args.rank + 1).float())
        dist.broadcast(source, src=0)
        report["tests"].append(
            {
                "operation": "broadcast",
                "bytes": n * 4,
                "passed": bool(torch.equal(source, (pattern + 1).float())),
            }
        )
        report["passed"] = all(t["passed"] for t in report["tests"])
        report["loaded_nccl_libraries"] = sorted(
            set(
                line.split()[-1]
                for line in pathlib.Path("/proc/self/maps").read_text().splitlines()
                if "libnccl" in line
            )
        )
        report["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
        dist.barrier()
    finally:
        dist.destroy_process_group()
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
