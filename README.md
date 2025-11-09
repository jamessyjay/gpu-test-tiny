# GPU Cluster Acceptance (Tiny)
Tiny CUDA/cuBLAS/NCCL micro-bench suite for fast GPU node/cluster acceptance. No heavy deps: slim Docker image, Python + ctypes, direct .so calls.

## Why you may need it
You run one container, get three numbers (TFLOP/s, GB/s memory, GB/s NCCL), and immediately know if the node is suitable for ML.

**Resulting image size:** 2.71GB

## Repository layout
- **src/gpu_accept_cuda_so.py** — main script. Loads `libcudart`, `libcublas`, `libnccl` via ctypes and runs:
  - **SGEMM peak** (TFLOPs on a single GPU)
  - **(optional) Strided-Batched SGEMM**
  - **Bandwidth**: H2D/D2H/D2D memcpy on a single stream
  - **NCCL AllReduce** across all visible GPUs
- **Dockerfile.nv-tiny** — minimal Ubuntu 24.04 image with CUDA 13 runtime (cudart/compat/cuda-libraries) + NCCL + Python venv.
- **gpu-test-tiny.yaml** — example Kubernetes Pod (image `ghcr.io/jamessyjay/gpu-cluster-acceptance:tiny`).
- **reupload_image.sh** — build/push helper and quick sanity-check (.so loading and script run).
- **.github/workflows/build.yaml** — GitHub Actions: build/push the image and run the GPU test in a container.

## Requirements and environment
- **NVIDIA host driver** compatible with CUDA 13.x (see `NVIDIA_REQUIRE_CUDA` in the Dockerfile for ranges).
- **NVIDIA Container Toolkit** on the host to enable Docker `--gpus`.
- **Docker** 20.10+.
- For the **NCCL** test you need ≥2 visible GPUs inside the container.
- For Kubernetes — GPU-enabled nodes and a proper device plugin (resource `nvidia.com/gpu`).

### What the image installs
- CUDA runtime: `cuda-cudart-13-0`, `cuda-compat-13-0`, `cuda-libraries-13-0`
- NCCL runtime: `libnccl2`
- Python venv + package `cuda-python`
- Environment variables: `PATH`, `LD_LIBRARY_PATH`, `NVIDIA_VISIBLE_DEVICES=all`, `NVIDIA_DRIVER_CAPABILITIES=compute,utility`, `NCCL_DEBUG=INFO`

## Quick start

### Local (Docker)
1) Build and run (amd64):
```bash
docker build --platform=linux/amd64 -f Dockerfile.nv-tiny -t ghcr.io/<owner>/<repo>:tiny .
docker run --rm --gpus all --ipc=host ghcr.io/<owner>/<repo>:tiny \
  python /app/src/gpu_accept_cuda_so.py
```
Quick mode:
```bash
docker run --rm --gpus all --ipc=host ghcr.io/<owner>/<repo>:tiny \
  python /app/src/gpu_accept_cuda_so.py --quick
```

2) Verify .so libraries load inside the container:
```bash
docker run --rm --gpus all ghcr.io/<owner>/<repo>:tiny bash -lc 'python - <<PY
import ctypes
for n in ["libcudart.so.13","libcudart.so.12","libcudart.so",
          "libcublas.so.13","libcublas.so.12","libcublas.so",
          "libnccl.so.2","libnccl.so"]:
    try:
        ctypes.cdll.LoadLibrary(n); print(n, "-> OK")
    except OSError as e:
        print(n, "->", e)
PY'
```

### Kubernetes
Pod example is provided:
```bash
kubectl apply -f gpu-test-tiny.yaml
kubectl logs pod/gpu-acceptance -f
```
In the manifest: `nvidia.com/gpu: 2`, `--quick` mode, `/dev/shm` backed by `emptyDir: { medium: Memory }`.

## What to watch for in logs
- **[ENV] GPUs:** number of visible GPUs.
- **[COMPUTE/SGEMM]** r, iters, avg/best TF/s.
- **[COMPUTE/BATCHED]** printed if `cublasSgemmStridedBatched` is available.
- **[BW/H2D|D2H|D2D]** average/best GB/s.
- **[NCCL/AR]** with ≥2 GPUs: elems, bytes/gpu, avg_t, ring GB/s estimate.
- Final line — **[RESULT] SUCCESS** on success.


## What we test and how
- **Compute throughput (TFLOP/s):** large matrix multiply via cuBLAS SGEMM.
  - Why: GEMM is a canonical heavy GPU op. If it flies here — SMs/tensor blocks are alive and driver/runtime are fine.
- **Memory/bus bandwidth (GB/s):** large buffers with pinned host and async memcpy in three directions:
  - H2D (RAM→GPU), D2H (GPU→RAM), D2D (GPU→GPU).
  - Why: direct measurement of PCIe/NVLink (H2D/D2H) and HBM/DRAM (D2D). If this is narrow, everything else will choke.
- **Inter-GPU comms (NCCL AllReduce):** when ≥2 GPUs are visible, we estimate aggregate GB/s for collectives.
  - Why: AllReduce is the heart of DDP/ML. If it lags, training stalls even with stellar TFLOP/s.

Feature: all via ctypes and system .so (cudart/cublas/nccl) — no heavy user-space libs. The image stays slim while measurements are honest (warmups, large sizes, events/streams).

## Why this helps DevOps
- **Fast node/cluster acceptance:** "run the container → get numbers". Instantly see if the driver sees GPUs, /dev/shm is sufficient, NUMA/PCIe aren’t throttling.
- **Symptom-based diagnostics:**
  - Low H2D/D2H → PCIe/NVLink issues, IOMMU, BIOS settings, pinning, C-states.
  - Low D2D → throttling/power/cooling, clocks, bad memory.
  - Low NCCL → missing `--ipc=host`, poor NVLink/topology, wrong container runtime flags, cluster network/MTU/RoCE.
- **Compare against spec:** eyeball "expected/unexpected" for H100/L40S, quickly spot where the problem hides.
- **K8s/Slurm-friendly:** same image and commands; just grant GPUs and a sane /dev/shm.

## Future additions (without bloating the image)
- **FP16/BF16 GEMM (cuBLASLt)** to hit tensor cores → ML-real TFLOP/s.
- **Multi-node NCCL smoke:** use MASTER_ADDR/PORT, RANK, WORLD_SIZE for cross-node tests without PyTorch.
- **Auto-report JSON/CSV + short HTML** (driver/CUDA versions, GPU models, TFLOP/s, GB/s) — handy for CI and tickets.
- **Auto sniffing common pitfalls:** warnings for tiny /dev/shm, IPC off, single GPU, low H2D, etc.
- **MIG/Topology checks:** detect MIG profiles, simple NVLink topology sanity.
- **"60s stress" option** with frequency/temperature monitoring (shelling to nvidia-smi) — catch thermal throttling.


## Usage examples
- Full CI run (see `.github/workflows/build.yaml`):
  - Build and push image `:tiny`
  - Run: `docker run --gpus all ghcr.io/<owner>/<repo>:tiny python /app/src/gpu_accept_cuda_so.py`
- Quick local validation: `./reupload_image.sh` (build → push → probe .so → run → cleanup)

## Recommendations and possible improvements
- **Metrics/export:** add Prometheus push/JSON export for automated parsing.
- **Thresholds:** define minimal acceptable TFLOPs/GBps/NCCL BW and fail below thresholds.
- **Parameterization:** flags for matrix/buffer sizes, iteration counts, host pinning, number of streams.
- **Reporting:** store CI artifacts (log + JSON results).
- **ROCm/AMD support:** abstract the .so loading layer and implement ROCm equivalents.
- **Diagnostics:** richer CUDA/NCCL error decoding, and topology dump (nvidia-smi topo -m).

## Troubleshooting
- `No GPUs visible` — ensure `--gpus all` (Docker), proper NVIDIA runtime, permissions and host driver.
- .so loading errors — check driver vs CUDA 13.x compatibility and in-container `LD_LIBRARY_PATH`.
- NCCL hangs/fails — ensure ≥2 GPUs are available, shared IPC/SHM, `--ipc=host`, and no restrictive networking (in clusters).
