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
