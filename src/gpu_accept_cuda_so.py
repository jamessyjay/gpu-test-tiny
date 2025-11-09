"""
GPU acceptance and micro-benchmark suite using CUDA Runtime, cuBLAS and NCCL via
Python ctypes.

What this script does
---------------------
- Dynamically loads NVIDIA shared libraries (`libcudart`, `libcublas`, `libnccl`) with
  fallbacks across common SONAMEs.
- Provides very thin ctypes bindings for a minimal subset of CUDA Runtime, cuBLAS v2,
  and NCCL needed for quick acceptance tests.
- Runs several focused checks/benchmarks:
  - SGEMM peak (single matrix multiply on one GPU) to estimate compute TFLOPs
  - Optional Strided-Batched SGEMM (if symbol is available) to validate batched path
  - H2D/D2H/D2D memcpy bandwidth using pinned host memory and a single stream
  - NCCL AllReduce across all visible GPUs to sanity-check multi-GPU comms

Usage
-----
- Local Docker (all GPUs, full run):
    docker run --rm --gpus all IMAGE:tag python src/gpu_accept_cuda_so.py

- Quick mode (faster, smaller buffers):
    python src/gpu_accept_cuda_so.py --quick

- In Kubernetes (example Pod spec provided in gpu-test-tiny.yaml):
    kubectl apply -f gpu-test-tiny.yaml

Prerequisites
-------------
- NVIDIA driver compatible with CUDA 13.x (see Dockerfile.nv-tiny for exact ranges)
- libcuda provided by the host driver
- Containers must be launched with `--gpus` and proper NVIDIA runtime hooks
- For NCCL test, at least 2 visible GPUs in the same container namespace

Notes
-----
- The bindings here are intentionally minimal and not a replacement for full Python
  CUDA/NCCL libraries. The goal is to reduce dependency surface and keep a tiny image
  footprint for acceptance checks.
- No function behavior is altered by the added documentation.
"""

import argparse, math, ctypes  # argparse: CLI flags; math: sizing math; ctypes: FFI bindings to CUDA/NCCL
from typing import Callable, Any  # typing helpers for clearer docstrings and signatures
from ctypes import (c_int, c_size_t, c_void_p, c_float, c_longlong,
                    POINTER, byref, c_char_p)  # low-level C types used across the bindings


# -------- helpers to load .so with fallbacks --------
def load_lib(names):
    """Try to load a shared library from a sequence of candidate names.

    Parameters
    ----------
    names : Iterable[str]
        Candidate SONAMEs to try in order, e.g. ("libcudart.so.13", "libcudart.so").

    Returns
    -------
    ctypes.CDLL
        The loaded library handle.

    Raises
    ------
    OSError
        Re-raises the last OSError if none of the candidate names could be loaded.
    """
    last = None
    for n in names:
        try:
            return ctypes.cdll.LoadLibrary(n)
        except OSError as e:
            last = e
    raise last

libcudart = load_lib(("libcudart.so.13","libcudart.so.12","libcudart.so"))  # CUDA Runtime API (streams, memcpy, events, allocs)
libcublas = load_lib(("libcublas.so.13","libcublas.so.12","libcublas.so"))  # cuBLAS v2 for SGEMM and (optionally) strided batched GEMM
libnccl   = load_lib(("libnccl.so.2","libnccl.so"))                         # NCCL collectives (AllReduce) for multi-GPU comms


# -------- CUDA runtime bindings --------
cudaGetErrorString     = libcudart.cudaGetErrorString;    cudaGetErrorString.argtypes    = [c_int];                                         cudaGetErrorString.restype  = c_char_p  # returns const char*

def check_cuda(st, where=""):
    """Raise a RuntimeError if a CUDA runtime call returned a non-success status.

    Parameters
    ----------
    st : int
        CUDA error code (0 is success).
    where : str
        Short context string to include into the exception for easier debugging.
    """
    if st != 0:
        msg = cudaGetErrorString(st)
        raise RuntimeError(f"CUDA error {int(st)} at {where}: {(msg or b'').decode()}")

# allocate page-locked ("pinned") host memory for faster H2D/D2H copies
cudaHostAlloc          = libcudart.cudaHostAlloc;         cudaHostAlloc.argtypes         = [POINTER(c_void_p), c_size_t, ctypes.c_uint];    cudaHostAlloc.restype  = c_int
# free pinned host memory allocated with cudaHostAlloc
cudaFreeHost           = libcudart.cudaFreeHost;          cudaFreeHost.argtypes          = [c_void_p];                                      cudaFreeHost.restype   = c_int
# how many CUDA devices are visible
cudaGetDeviceCount     = libcudart.cudaGetDeviceCount;    cudaGetDeviceCount.argtypes    = [POINTER(c_int)];                                cudaGetDeviceCount.restype   = c_int
# choose active device (by index) for subsequent API calls
cudaSetDevice          = libcudart.cudaSetDevice;         cudaSetDevice.argtypes         = [c_int];                                         cudaSetDevice.restype        = c_int
# query free/total memory info on the current device
cudaMemGetInfo         = libcudart.cudaMemGetInfo;        cudaMemGetInfo.argtypes        = [POINTER(c_size_t), POINTER(c_size_t)];          cudaMemGetInfo.restype = c_int
# allocate device (GPU) memory buffer
cudaMalloc             = libcudart.cudaMalloc;            cudaMalloc.argtypes            = [POINTER(c_void_p), c_size_t];                   cudaMalloc.restype     = c_int
# free device (GPU) memory buffer
cudaFree               = libcudart.cudaFree;              cudaFree.argtypes              = [c_void_p];                                      cudaFree.restype       = c_int
# synchronous memory copy (blocks host until finished)
cudaMemcpy             = libcudart.cudaMemcpy;            cudaMemcpy.argtypes            = [c_void_p, c_void_p, c_size_t, c_int];           cudaMemcpy.restype     = c_int
# asynchronous memory copy on a CUDA stream (non-blocking on host)
cudaMemcpyAsync        = libcudart.cudaMemcpyAsync;       cudaMemcpyAsync.argtypes       = [c_void_p, c_void_p, c_size_t, c_int, c_void_p]; cudaMemcpyAsync.restype = c_int
# wait until all work on the device is finished (global barrier)
cudaDeviceSynchronize  = libcudart.cudaDeviceSynchronize; cudaDeviceSynchronize.argtypes = [];                                              cudaDeviceSynchronize.restype = c_int
# create a CUDA stream (queue for async work)
cudaStreamCreate       = libcudart.cudaStreamCreate;      cudaStreamCreate.argtypes      = [POINTER(c_void_p)];                             cudaStreamCreate.restype = c_int
# block host until all prior work in the stream completes
cudaStreamSynchronize  = libcudart.cudaStreamSynchronize; cudaStreamSynchronize.argtypes = [c_void_p];                                      cudaStreamSynchronize.restype = c_int
# destroy a CUDA stream and release resources
cudaStreamDestroy      = libcudart.cudaStreamDestroy;     cudaStreamDestroy.argtypes     = [c_void_p];                                      cudaStreamDestroy.restype = c_int
# create an event (timestamp marker) for timing/synchronization
cudaEventCreate        = libcudart.cudaEventCreate;       cudaEventCreate.argtypes       = [POINTER(c_void_p)];                             cudaEventCreate.restype = c_int
# record (enqueue) an event into a stream
cudaEventRecord        = libcudart.cudaEventRecord;       cudaEventRecord.argtypes       = [c_void_p, c_void_p];                            cudaEventRecord.restype = c_int
# wait on an event to complete
cudaEventSynchronize   = libcudart.cudaEventSynchronize;  cudaEventSynchronize.argtypes  = [c_void_p];                                      cudaEventSynchronize.restype = c_int
# compute elapsed time between two events (ms)
cudaEventElapsedTime   = libcudart.cudaEventElapsedTime;  cudaEventElapsedTime.argtypes  = [POINTER(c_float), c_void_p, c_void_p];          cudaEventElapsedTime.restype = c_int
# destroy an event and release resources
cudaEventDestroy       = libcudart.cudaEventDestroy;      cudaEventDestroy.argtypes      = [c_void_p];                                      cudaEventDestroy.restype = c_int

# memcpy kinds (tell cudaMemcpy which direction the data moves)
cudaMemcpyHostToDevice = 1   # CPU RAM -> GPU VRAM
cudaMemcpyDeviceToHost = 2   # GPU VRAM -> CPU RAM
cudaMemcpyDeviceToDevice=3   # GPU VRAM -> GPU VRAM


# -------- cuBLAS bindings --------
cublasCreate   = libcublas.cublasCreate_v2;   cublasCreate.argtypes   = [POINTER(c_void_p)]; cublasCreate.restype = c_int
cublasDestroy  = libcublas.cublasDestroy_v2;  cublasDestroy.argtypes  = [c_void_p];          cublasDestroy.restype = c_int
cublasSgemm    = libcublas.cublasSgemm_v2;    cublasSgemm.restype     = c_int
# optional (if present) batched sgemm
try:
    cublasSgemmStridedBatched = libcublas.cublasSgemmStridedBatched
    cublasSgemmStridedBatched.restype = c_int
    HAS_STRIDED_BATCHED = True
except AttributeError:
    HAS_STRIDED_BATCHED = False

CUBLAS_OP_N = 0

# -------- NCCL bindings --------
class ncclUniqueId(ctypes.Structure):
    """ctypes representation of `ncclUniqueId`.

    The content is an opaque 128-byte blob produced by `ncclGetUniqueId` and consumed
    by `ncclCommInitRank`. We never interpret the bytes in Python; they are passed
    through directly to NCCL APIs.
    """
    _fields_ = [("internal", ctypes.c_char * 128)]

ncclGetUniqueId  = libnccl.ncclGetUniqueId;   ncclGetUniqueId.argtypes  = [POINTER(ncclUniqueId)]
ncclCommInitRank = libnccl.ncclCommInitRank;  ncclCommInitRank.argtypes = [POINTER(c_void_p), c_int, ncclUniqueId, c_int]
ncclCommDestroy  = libnccl.ncclCommDestroy;   ncclCommDestroy.argtypes  = [c_void_p]
ncclAllReduce    = libnccl.ncclAllReduce;     ncclAllReduce.argtypes    = [c_void_p, c_void_p, c_size_t, c_int, c_int, c_void_p, c_void_p]
ncclGroupStart   = libnccl.ncclGroupStart;    ncclGroupEnd              = libnccl.ncclGroupEnd
ncclFloat32      = 7
ncclSum          = 0


# -------- measuring helpers --------
def event_timer(stream: c_void_p | None = None) -> tuple[Callable[[Callable[[], Any]], float], Callable[[], Any]]:
    """Create a simple CUDA event-based timer around a callable.

    This utility places start/end events on the provided stream (or default stream)
    around a function that enqueues work, then synchronizes and reports the elapsed
    time in seconds.

    Parameters
    ----------
    stream : c_void_p or None
        CUDA stream handle. If None, uses default stream (0).

    Returns
    -------
    (measure, cleanup)
        - measure: Callable[[Callable[[], Any]], float]
          Wraps execution of a function that enqueues CUDA work and returns seconds.
        - cleanup: Callable[[], Any]
          Destroys created events; must be called to avoid leaks.
    """
    start = c_void_p(); end = c_void_p()
    check_cuda(cudaEventCreate(byref(start)), "evt create start")
    check_cuda(cudaEventCreate(byref(end)),   "evt create end")
    s = stream or c_void_p(0)
    def measure(fn):
        # events on the same stream where the work is happening
        check_cuda(cudaEventRecord(start, s))
        fn()
        check_cuda(cudaEventRecord(end, s))
        check_cuda(cudaEventSynchronize(end))
        ms = c_float()
        check_cuda(cudaEventElapsedTime(byref(ms), start, end))
        return ms.value / 1000.0
    return measure, lambda: (cudaEventDestroy(start), cudaEventDestroy(end))


# -------- tests --------
def pick_gemm_size(free_bytes: int, quick: bool) -> int:
    """Pick a square SGEMM size `r` based on available device memory.

    We budget a fraction of free GPU memory for three r×r float32 matrices (A, B, C),
    approximated by 12·r² bytes. In quick mode the budget is smaller.

    Parameters
    ----------
    free_bytes : int
        Free memory on the selected device (bytes).
    quick : bool
        If True, use conservative budgets for faster runs.

    Returns
    -------
    int
        Chosen matrix dimension r (clamped to reasonable ceilings).
    """
    # memory for A,B,C ~= 12*r^2 bytes (float32)
    budget = int(free_bytes * (0.50 if not quick else 0.20))  # 50% (full) / 20% (quick) of free memory
    r = int(math.sqrt(budget / 12.0))
    # clamp to sensible ceilings
    r = max(1024 if quick else 2048, min(r, 8192 if quick else 12288))
    return r


def gemm_peak(handle: c_void_p, r: int, iters: int) -> dict:
    """Run peak SGEMM on a single GPU and estimate TFLOPs.

    Parameters
    ----------
    handle : c_void_p
        cuBLAS handle (v2).
    r : int
        Matrix dimension (square) for A, B, C.
    iters : int
        Number of timed iterations (warm-up is done internally).
    stream : c_void_p or None
        Optional CUDA stream if associating cuBLAS handle externally.

    Returns
    -------
    dict
        Summary with r, iters, avg/best TFLOPs and average wall time.
    """
    # allocate
    bytesA = r*r*4; bytesB = r*r*4; bytesC = r*r*4
    dA = c_void_p(); dB = c_void_p(); dC = c_void_p()
    check_cuda(cudaMalloc(byref(dA), bytesA), "malloc A")
    check_cuda(cudaMalloc(byref(dB), bytesB), "malloc B")
    check_cuda(cudaMalloc(byref(dC), bytesC), "malloc C")

    # init zeros (ok for demo)
    hA = (c_float * (r*r))()
    hB = (c_float * (r*r))()
    check_cuda(cudaMemcpy(dA, ctypes.addressof(hA), bytesA, cudaMemcpyHostToDevice), "H2D A")
    check_cuda(cudaMemcpy(dB, ctypes.addressof(hB), bytesB, cudaMemcpyHostToDevice), "H2D B")

    # (optional) associate stream? cublasSgemm v2 uses handle's stream via cublasSetStream; we skip it — default.
    alpha = c_float(1.0); beta = c_float(0.0)

    # warmup
    for _ in range(2):
        ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, r, r, r,
                          byref(alpha), dA, r, dB, r, byref(beta), dC, r)
        assert ret == 0
    check_cuda(cudaDeviceSynchronize(), "warmup sync")

    measure, cleanup_evt = event_timer()
    times = []
    for _ in range(iters):
        t = measure(lambda: cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, r, r, r,
                                        byref(alpha), dA, r, dB, r, byref(beta), dC, r))
        times.append(t)
    cleanup_evt()

    check_cuda(cudaDeviceSynchronize(), "post runs")
    tflops_each = [(2.0*r*r*r)/t/1e12 for t in times]
    res = {
        "r": r,
        "iters": iters,
        "avg_tflops": round(sum(tflops_each)/len(tflops_each), 2),
        "best_tflops": round(max(tflops_each), 2),
        "avg_time_s": round(sum(times)/len(times), 4),
    }

    # free
    cudaFree(dA); cudaFree(dB); cudaFree(dC)
    return res


def gemm_batched(handle: c_void_p, r: int, batch: int, iters: int) -> dict:
    """Run strided-batched SGEMM when available.

    If the `cublasSgemmStridedBatched` symbol is not present in the cuBLAS library,
    the function returns `{ "supported": False }` without doing work.

    Parameters
    ----------
    handle : c_void_p
        cuBLAS handle.
    r : int
        Matrix dimension per batch item.
    batch : int
        Number of matrices in the batch.
    iters : int
        Number of timed iterations.

    Returns
    -------
    dict
        Summary with avg/best TFLOPs when supported, otherwise `{supported: False}`.
    """
    if not HAS_STRIDED_BATCHED:
        return {"supported": False}
    n = r
    bytes_per_mat = r*r*4
    stride = r*r  # elements between matrices
    totalA = bytes_per_mat * batch
    totalB = totalA
    totalC = totalA
    dA = c_void_p(); dB = c_void_p(); dC = c_void_p()
    check_cuda(cudaMalloc(byref(dA), totalA), "malloc Ab")
    check_cuda(cudaMalloc(byref(dB), totalB), "malloc Bb")
    check_cuda(cudaMalloc(byref(dC), totalC), "malloc Cb")

    # init zeros
    host = (c_float * (r*r))()
    check_cuda(cudaMemcpy(dA, ctypes.addressof(host), bytes_per_mat, cudaMemcpyHostToDevice), "H2Db A0")
    check_cuda(cudaMemcpy(dB, ctypes.addressof(host), bytes_per_mat, cudaMemcpyHostToDevice), "H2Db B0")

    alpha = c_float(1.0); beta = c_float(0.0)

    # warmup
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              r, r, r,
                              byref(alpha),
                              dA, r, c_longlong(stride),
                              dB, r, c_longlong(stride),
                              byref(beta),
                              dC, r, c_longlong(stride),
                              c_int(batch))
    check_cuda(cudaDeviceSynchronize(), "warmup batched")

    measure, cleanup_evt = event_timer()
    times = []
    for _ in range(iters):
        t = measure(lambda: cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                      r, r, r,
                                                      byref(alpha),
                                                      dA, r, c_longlong(stride),
                                                      dB, r, c_longlong(stride),
                                                      byref(beta),
                                                      dC, r, c_longlong(stride),
                                                      c_int(batch)))
        times.append(t)
    cleanup_evt()
    check_cuda(cudaDeviceSynchronize(), "post batched")

    # ops per iter = batch * 2*r^3
    tflops = [ (batch*2.0*r*r*r)/t/1e12 for t in times ]
    res = {
        "supported": True,
        "r": r, "batch": batch, "iters": iters,
        "avg_tflops": round(sum(tflops)/len(tflops), 2),
        "best_tflops": round(max(tflops), 2),
        "avg_time_s": round(sum(times)/len(times), 4),
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC)
    return res


def bandwidth_test(kind: str, total_bytes: int, reps: int = 10) -> dict:
    """Measure memcpy bandwidth for H2D/D2H/D2D using a single CUDA stream.

    Parameters
    ----------
    kind : {"H2D","D2H","D2D"}
        Copy direction.
    total_bytes : int
        Transfer size in bytes.
    reps : int
        Number of timed repetitions (one warm-up is done internally).

    Returns
    -------
    dict
        Summary with average and best GB/s and the tested size.
    """
    # single stream for measurement
    s = c_void_p()
    check_cuda(cudaStreamCreate(byref(s)), "bw stream")

    # pinned host (for H2D/D2H)
    hptr = c_void_p()
    if kind in ("H2D", "D2H"):
        check_cuda(cudaHostAlloc(byref(hptr), total_bytes, 0), "host alloc (pinned)")

    # device buffers
    d0 = c_void_p(); d1 = c_void_p()
    check_cuda(cudaMalloc(byref(d0), total_bytes), "malloc d0")
    if kind == "D2D":
        check_cuda(cudaMalloc(byref(d1), total_bytes), "malloc d1")

    # helper to run copy
    def do_copy():
        if kind == "H2D":
            check_cuda(cudaMemcpyAsync(d0, hptr, total_bytes, cudaMemcpyHostToDevice, s))
        elif kind == "D2H":
            check_cuda(cudaMemcpyAsync(hptr, d0, total_bytes, cudaMemcpyDeviceToHost, s))
        else:  # D2D
            check_cuda(cudaMemcpyAsync(d1, d0, total_bytes, cudaMemcpyDeviceToDevice, s))

    # warmup
    do_copy()
    check_cuda(cudaStreamSynchronize(s), "bw warmup")

    measure, cleanup_evt = event_timer(s)
    times = []
    for _ in range(reps):
        t = measure(do_copy)
        times.append(t)
    cleanup_evt()
    check_cuda(cudaStreamSynchronize(s), "bw post")

    gb = total_bytes / 1e9
    bw = [ gb/t for t in times ]
    res = {
        "kind": kind,
        "size_gb": round(gb, 3),
        "reps": reps,
        "avg_gbps": round(sum(bw)/len(bw), 2),
        "best_gbps": round(max(bw), 2)
    }

    # free
    if kind in ("H2D", "D2H"):
        cudaFreeHost(hptr)
    if kind == "D2D":
        cudaFree(d1)
    cudaFree(d0)
    cudaStreamDestroy(s)
    return res


def nccl_allreduce(world: int, elems: int, iters: int = 20) -> dict:
    """Run NCCL AllReduce across `world` GPUs and estimate effective ring bandwidth.

    Parameters
    ----------
    world : int
        Number of ranks/GPUs (assumed to be device IDs 0..world-1).
    elems : int
        Number of float32 elements per GPU buffer.
    iters : int
        Timed iterations (includes light warm-up before).

    Returns
    -------
    dict
        Summary with average time and a coarse GB/s ring estimate.
    """
    uid = ncclUniqueId(); assert ncclGetUniqueId(byref(uid)) == 0
    comms   = [c_void_p() for _ in range(world)]
    streams = [c_void_p() for _ in range(world)]
    bufs    = [c_void_p() for _ in range(world)]

    # setup per rank
    for r in range(world):
        check_cuda(cudaSetDevice(r), f"setdev {r}")
        s = c_void_p(); check_cuda(cudaStreamCreate(byref(s)), f"stream {r}")
        streams[r] = s
        check_cuda(cudaMalloc(byref(bufs[r]), elems*4), f"malloc {r}")
        # fill ones
        host = (c_float * elems)(*([1.0]*elems))
        check_cuda(cudaMemcpy(bufs[r], ctypes.addressof(host), elems*4, cudaMemcpyHostToDevice), f"H2D {r}")
        assert ncclCommInitRank(byref(comms[r]), world, uid, r) == 0

    # warmup
    ncclGroupStart()
    for r in range(world):
        assert ncclAllReduce(bufs[r], bufs[r], elems, ncclFloat32, ncclSum, comms[r], streams[r]) == 0
    ncclGroupEnd()
    for r in range(world):
        check_cuda(cudaSetDevice(r)); check_cuda(cudaStreamSynchronize(streams[r]))

    # measure
    measure, cleanup_evt = event_timer()  # simple "around group call"
    times = []
    for _ in range(iters):
        t = measure(lambda: (
            ncclGroupStart(),
            [ncclAllReduce(bufs[r], bufs[r], elems, ncclFloat32, ncclSum, comms[r], streams[r]) for r in range(world)],
            ncclGroupEnd(),
            [cudaStreamSynchronize(streams[r]) for r in range(world)]
        ))
        times.append(t)
    cleanup_evt()

    # cleanup
    for r in range(world):
        ncclCommDestroy(comms[r])
        cudaFree(bufs[r])
        cudaStreamDestroy(streams[r])

    avg_t = sum(times)/len(times)
    bytes_per_gpu = elems*4
    # rough estimate of "effective" bandwidth in ring:
    eff = (2.0*(world-1)/world) * bytes_per_gpu / avg_t / 1e9  # GB/s
    return {
        "elems": elems,
        "bytes_per_gpu": bytes_per_gpu,
        "iters": iters,
        "avg_time_s": round(avg_t, 4),
        "eff_ring_gbps_est": round(eff, 2),
    }


def main():
    """CLI entry point: run acceptance checks and print human-readable summaries.

    Flags
    -----
    --quick : reduce matrix sizes, buffer sizes and iteration counts to finish fast.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    # devices
    ndev = c_int()
    check_cuda(cudaGetDeviceCount(byref(ndev)), "getDeviceCount")
    print(f"[ENV] GPUs: {ndev.value}")
    assert ndev.value >= 1, "No GPUs visible"

    # memory info (GPU0)
    free_b = c_size_t(); total_b = c_size_t()
    check_cuda(cudaSetDevice(0)); check_cuda(cudaMemGetInfo(byref(free_b), byref(total_b)), "meminfo")
    # cuBLAS handle
    handle = c_void_p(); assert cublasCreate(byref(handle)) == 0

    # --- SGEMM peak ---
    r = pick_gemm_size(free_b.value, quick=args.quick)
    iters = 6 if args.quick else 16
    g = gemm_peak(handle, r, iters)
    print(f"[COMPUTE/SGEMM] r={g['r']} iters={g['iters']}  avg={g['avg_tflops']} TF/s  best={g['best_tflops']} TF/s  avg_t={g['avg_time_s']} s")

    # --- Batched GEMM (если поддерживается) ---
    if HAS_STRIDED_BATCHED:
        batch = 16 if args.quick else 64
        ib = 4 if args.quick else 12
        gb = gemm_batched(handle, min(1024, r//2), batch, ib)
        if gb.get("supported"):
            print(f"[COMPUTE/BATCHED] r={gb['r']} batch={gb['batch']} iters={gb['iters']}  avg={gb['avg_tflops']} TF/s  best={gb['best_tflops']} TF/s")
        else:
            print("[COMPUTE/BATCHED] not supported (symbol not found)")
    else:
        print("[COMPUTE/BATCHED] not supported (symbol not found)")

    # destroy handle
    cublasDestroy(handle)

    # --- Memory bandwidth ---
    # take buffer 1–4 GB (512 MB in quick), but <= 20% of free memory
    max_for_bw = int(free_b.value * (0.20 if not args.quick else 0.10))
    target = (512<<20) if args.quick else (1<<30)  # 512MB / 1GB
    sz = min(max_for_bw, target*4)  # cap at 4GB in full mode
    sz = max((256<<20), sz)  # minimum 256MB
    reps = 6 if args.quick else 12
    for kind in ("H2D","D2H","D2D"):
        bw = bandwidth_test(kind, sz, reps=reps)
        print(f"[BW/{kind}] size={bw['size_gb']} GB reps={bw['reps']}  avg={bw['avg_gbps']} GB/s  best={bw['best_gbps']} GB/s")

    # --- NCCL all-reduce (если >=2 GPU) ---
    if ndev.value >= 2:
        # take buffer ~256MB (quick) / 1GB (full), but <= 15% of free memory
        max_nccl = int(free_b.value * (0.15 if not args.quick else 0.08))
        tgt = (256<<20) if args.quick else (1<<30)
        bsz = min(max_nccl, tgt)
        elems = bsz//4
        it_nccl = 12 if args.quick else 24
        n = nccl_allreduce(ndev.value, elems, iters=it_nccl)
        print(f"[NCCL/AR] elems={n['elems']} bytes/gpu={n['bytes_per_gpu']} iters={n['iters']} avg_t={n['avg_time_s']} s  eff_ring≈{n['eff_ring_gbps_est']} GB/s")
    else:
        print("[NCCL] skipped (need >=2 GPUs)")

    print("[RESULT] SUCCESS")

if __name__ == "__main__":
    main()
