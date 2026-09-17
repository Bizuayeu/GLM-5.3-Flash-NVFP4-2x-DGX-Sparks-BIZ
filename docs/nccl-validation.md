# Two-host NCCL diagnostics

[日本語](nccl-validation.ja.md) · [Network preparation](qsfp-network.md) · [Validation limits](validation.md)

The repository supplies a small [PyTorch/NCCL probe](../tools/nccl_probe.py). It checks two ranks with rank-dependent patterns: FP32/BF16 AllReduce at 1 KiB, 1 MiB, 16 MiB and 256 MiB per rank, plus FP32 AllGather, ReduceScatter and Broadcast. It does not load model weights or generate a full-model qualification receipt.

## Run on both hosts

First verify fixed IPv4, HCA, RoCEv2 GID and peer routing. Inventory **host processes and active transfers**, not only Docker containers. Pause or wait for authorized competing work before claiming isolated bandwidth. Never stop an unrelated transfer automatically.

Use the same reviewed source and pinned base image on both hosts. Select fresh container names and output directories. In each host's Linux checkout, replace these illustrative values with observations (rank 1 uses its own interface/HCA/IP; `HEAD_IP` remains rank 0's address):

```sh
RANK=0
FABRIC_IF='REPLACE_WITH_OBSERVED_INTERFACE'
HCA='REPLACE_WITH_OBSERVED_HCA'
GID='REPLACE_WITH_OBSERVED_INDEX'
HEAD_IP='10.53.0.1'
RUN_ID='REPLACE_WITH_UNIQUE_RUN_ID'
IMAGE=$(python -c 'from glm53_setup.config import load_lock; print(load_lock()["image"])')
mkdir -p "records/$RUN_ID"
```

Confirm port 29653 is unused on rank 0. Start rank 1 and then rank 0 promptly (rendezvous timeout is 90 seconds). Keep an external five-minute experiment deadline; if exceeded, stop these specific test containers on both hosts and preserve their logs.

```sh
docker run --name "glm53-nccl-$RUN_ID-rank$RANK" \
  --gpus all --network host --memory 16g --memory-swap 16g --shm-size 1g \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  --device /dev/infiniband:/dev/infiniband \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e "NCCL_IB_HCA==$HCA" -e "NCCL_IB_GID_INDEX=$GID" \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_SOCKET_FAMILY=AF_INET -e "NCCL_SOCKET_IFNAME==$FABRIC_IF" \
  -e "GLOO_SOCKET_IFNAME=$FABRIC_IF" \
  -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH \
  -v "$PWD/tools/nccl_probe.py:/probe.py:ro" \
  -v "$PWD/records/$RUN_ID:/out" \
  --entrypoint python3 "$IMAGE" /probe.py \
  --rank "$RANK" --head "$HEAD_IP" --port 29653 \
  --output "/out/rank$RANK.json" >"records/$RUN_ID/nccl.log" 2>&1
```

The doubled equals signs in the Docker arguments are intentional: the environment value begins with `=` for an exact device-name match. See [NCCL's environment reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html). Keep the distributed ports within the trusted fabric.

## Accept and interpret

- Both processes must exit zero, both JSON reports must contain all 11 passing checks, and neither container may be OOM-killed.
- Inspect both transport logs: `Using network IB` and the intended HCA/RoCE and bootstrap interface must be present. A successful TCP rendezvous alone does not demonstrate the collective transport.
- Save the actual image ID and source revision. The NCCL runtime version printed during communicator initialization can differ from `torch.cuda.nccl.version()`; the report labels the latter `torch_reported_nccl` and records mapped libraries.
- Timings are ten-operation averages after warmup and include Python dispatch and synchronization. `payload_GB_per_s` is per-rank payload bytes divided by elapsed seconds, using decimal GB. It is not aggregate link bandwidth, an official `nccl-tests` result or model throughput.
- Record concurrent jobs and the MTU. No production bandwidth threshold is established by this probe. Repeat only when a changed condition or unresolved measurement concern warrants it.

DGX Spark's unified-memory platform does not support conventional GPUDirect RDMA via `nvidia-peermem`, DMA-BUF or GDRCopy according to [NVIDIA's porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html). Therefore `NET/IB` with `GDR 0` is not by itself a failed RoCE test. Do not load a kernel module or force GDR merely to change that log field.

The reviewed initial run passed all collective checks on two GB10 hosts at MTU 1500 using NCCL runtime 2.30.7. Large AllReduce measured about 1.2 GB/s while another transfer could contend for resources. A separate `NCCL_NET_GDR_LEVEL=SYS` comparison also passed but did not enable GDR or improve bandwidth; it is not part of the recommended command above. See [validation](validation.md) for the current evidence boundary.

The final run used the supplied probe after fabric transfers had ended. Both ranks passed all 11 checks; 256 MiB AllReduce measured 1.18–1.21 GB/s. An unrelated disk-checksum job remained active, so this is not an entirely idle-host benchmark. All test containers exited zero without OOM. Full-model TP=2 remains unqualified.

## Channel count

On its own, NCCL 2.30.7 opens 64 channels on this pair (identical on 2026-09-11 and 2026-09-17). The template sets [`runtime.nccl_channels = 8`](server-configuration.md). The measurements behind that value were taken on 2026-09-17 with the reference image and the launcher's fabric environment, changing only `NCCL_MIN_NCHANNELS`/`NCCL_MAX_NCHANNELS`.

**Collectives alone.** A two-rank BF16 AllReduce sweep, run twice in opposite orders, took the median of seven batches per size. Memory is the drop in `MemAvailable` across communicator setup and all sizes. The sizes the model uses are set by hidden size 4096 in BF16 (8,192 bytes per token): a decode verify with MTP k=3 is 32 KiB, a draft step 8 KiB, and a 512-token prefill chunk 4 MiB.

| Channels | Memory per rank | 32 KiB | 4 MiB | 256 MiB |
|---:|---:|---:|---:|---:|
| 64 (NCCL's choice) | 2.3 GiB | 0.022 ms | 0.41–0.45 ms | 20.0 ms (MTU 9000), 26.1 ms (MTU 1500) |
| 32 | 1.5 GiB | 0.022 ms | 0.35–0.39 ms | 19.7–23.5 ms |
| 16 | 1.0 GiB | 0.022 ms | 0.34–0.35 ms | 19.7–20.6 ms |
| 8 | 0.85 GiB | 0.022 ms | 0.33–0.36 ms | 19.4–19.5 ms |
| 4 | 0.75 GiB | 0.022 ms | 0.35–0.37 ms | 19.4 ms |

Decode-sized messages do not change, because NCCL already uses fewer channels for small messages. The memory does not depend on MTU. These timings are not comparable with the probe run above, which measured FP32 while another job contended for the host.

**Full model.** The same profile was started with only the channel count or the MTU changed. All runs followed one reboot, with 01's monitoring dashboard stopped. Prefill is the median of three fresh 38,961-token prompts. The lowest free memory is read from each rank's supervisor samples during startup, warmup and the measurement.

| MTU | Channels | Prefill tok/s | Head lowest free | Peer lowest free |
|---:|---:|---:|---:|---:|
| 1500 | 64 | 487.3 | 4.18 GiB | 6.48 GiB |
| 1500 | **8** | **492.0** | **6.93 GiB** | **9.43 GiB** |
| 9000 | 64 | 499.8 | 2.94 GiB | 4.76 GiB |
| 9000 | 8 | 503.2 | 5.81 GiB | 8.08 GiB |
| 9000 | 16 | 497.3 | 5.39 GiB | 7.57 GiB |

Each engine opens two communicators, so 8 channels return about 3 GiB per rank; prefill is within 1% either way. Decode varied more between runs of one setting (about ±15%) than between settings.

**MTU.** 9000 (RoCE active MTU 4096) raised prefill by 2.3–2.6% and lowered free memory by 1.1–1.7 GiB per host. An idle host without the model showed the same 1.4 GiB, which fits larger NIC receive buffers (four interfaces × 20 queues × 1,024 descriptors). The reference pair stays at MTU 1500. A prefill figure taken before the reboot (446 tok/s at MTU 1500) was lower from host state, not MTU, and is left out of the table.
