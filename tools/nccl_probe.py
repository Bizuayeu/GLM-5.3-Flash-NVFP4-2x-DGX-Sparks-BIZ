"""Two- or three-rank PyTorch/NCCL correctness and timing probe, not model qualification.

Run one process per GPU host with matching head/port/world size and an explicit NCCL
fabric environment. Record every result and transport log; timings include Python
dispatch and synchronization. No data download or network configuration occurs.
On a three-node ring, run it once with --world-size 3 over the launch's environment,
and once per link with --world-size 2: `server plan` lists each ring rank's links
with the pair's head, the rank in the pair and the environment naming that link.
"""

import argparse
import datetime
import json
import os
import pathlib
import time

# Up to three ranks every timed all-reduce step rounds once (a + a is exact, then
# one add rounds), so torch's step-by-step product is what NCCL returns in BF16;
# with more ranks the rounding count depends on the algorithm's order.
MAX_WORLD_SIZE = 3


def contribution(pattern, rank):
    """What a rank sends: the pattern plus its rank plus one."""
    return pattern + rank + 1


def reduced(pattern, world_size):
    """One all-reduce of every rank's contribution: N * pattern + N(N+1)/2."""
    return pattern * world_size + world_size * (world_size + 1) // 2


def repeated(value, world_size, times):
    """Further all-reduces of a value every rank holds: each multiplies it by N."""
    for _ in range(times):
        value = value * world_size
    return value


def shard(whole, rank, n):
    """The rank's slice of a tensor split into equal parts of n."""
    return whole[rank * n : (rank + 1) * n]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--head", required=True)
    p.add_argument("--port", type=int, default=29653)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args(argv)
    world = args.world_size
    if not 2 <= world <= MAX_WORLD_SIZE:
        p.error(f"world size must be 2 to {MAX_WORLD_SIZE}")
    if not 0 <= args.rank < world:
        p.error(f"rank must be 0 to {world - 1}")
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
        "world_size": world,
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
        world_size=world,
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
                value = contribution(pattern, args.rank)
                dist.all_reduce(value)
                good = bool(torch.equal(value, reduced(pattern, world)))
                for _ in range(3):
                    dist.all_reduce(value)
                value.copy_(contribution(pattern, args.rank))
                dist.barrier()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(10):
                    dist.all_reduce(value)
                torch.cuda.synchronize()
                seconds = (time.perf_counter() - start) / 10
                # Computed in the dtype, one rounding per step like the reduction
                # (MAX_WORLD_SIZE); exact for two ranks and in FP32.
                timed = repeated(reduced(pattern, world), world, 9)
                good = good and bool(torch.equal(value, timed))
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
        source = contribution(pattern, args.rank).float()
        output = torch.empty(world * n, device="cuda")
        dist.all_gather_into_tensor(output, source)
        good = all(
            torch.equal(shard(output, r, n), contribution(pattern, r).float())
            for r in range(world)
        )
        report["tests"].append(
            {"operation": "all_gather", "bytes_per_rank": n * 4, "passed": good}
        )
        whole = torch.arange(world * n, device="cuda", dtype=torch.int64) % 7
        input_ = contribution(whole, args.rank).float()
        scattered = torch.empty(n, device="cuda")
        dist.reduce_scatter_tensor(scattered, input_)
        good = bool(
            torch.equal(scattered, reduced(shard(whole, args.rank, n), world).float())
        )
        report["tests"].append(
            {
                "operation": "reduce_scatter",
                "input_bytes_per_rank": world * n * 4,
                "passed": good,
            }
        )
        source.copy_(contribution(pattern, args.rank).float())
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
