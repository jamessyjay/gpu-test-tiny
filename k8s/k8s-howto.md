# K8s GPU Acceptance: How-To

This folder contains Kubernetes manifests to run the GPU acceptance tests across your cluster using the image `ghcr.io/jamessyjay/gpu-cluster-acceptance:tiny`.

## Files
- `gpu-test-tiny.yaml` (DaemonSet)
  - Runs one pod per GPU node (by default `nvidia.com/gpu: 1`).
  - Good for quickly verifying each node has working CUDA/NCCL and to collect per-node reports.
- `gpu-test-job.yaml` (Job)
  - Lets you control how many nodes run in parallel via `spec.parallelism` and total runs via `spec.completions`.
  - Spreads pods one-per-node with `podAntiAffinity` + `topologySpreadConstraints`.

Both manifests:
- Pass `--log-level` and `--report-dir` to the Python test.
- Emit JSON reports into `/reports`.
- Set sensible NCCL envs; IB is enabled by default (tune as needed).

## Prerequisites
- GPU nodes labeled/tainted appropriately (examples assume:
  - `nodeSelector: nvidia.com/gpu.present: "true"`
  - toleration for taint key `nvidia.com/gpu`)
- NVIDIA device plugin / GPU Operator installed.
- Nodes have `/var/log/gpu-acceptance` (created automatically via `hostPath: DirectoryOrCreate`).

## Usage
### 1) Run on ALL GPU nodes (one pod per node)
DaemonSet variant:
```bash
kubectl apply -f k8s/gpu-test-tiny.yaml
# Observe
kubectl -n default get pods -l app=gpu-acceptance -o wide
# Fetch a report from a node (example)
NODE=<your-node-name>
ssh $NODE 'sudo ls -1 /var/log/gpu-acceptance'
```
Delete when done:
```bash
kubectl delete -f k8s/gpu-test-tiny.yaml
```

### 2) Run on N nodes in parallel (Job, adjustable)
Edit `k8s/gpu-test-job.yaml` and set:
- `spec.parallelism: <N>` (pods running at once)
- `spec.completions: <N>` (total pods to complete)

Apply:
```bash
kubectl apply -f k8s/gpu-test-job.yaml
kubectl get jobs
kubectl get pods -l app=gpu-acceptance-job -o wide
```
Delete when done:
```bash
kubectl delete -f k8s/gpu-test-job.yaml
```

## Tuning
- GPU per pod:
  - Change `resources.requests/limits: nvidia.com/gpu`.
  - DS: set to `8` to test all GPUs on a node at once.
  - Job: set to `1` for light spread, or `8` for full-node tests.
- Logging/report:
  - `LOG_LEVEL` env (`INFO`|`DEBUG` etc.).
  - `REPORT_DIR` (default `/reports`). Reports saved as `gpu_accept_${NODE_NAME}[ _${POD_NAME}].json`.
- NCCL over IB:
  - `NCCL_IB_DISABLE=0` (enable IB).
  - Optionally set `NCCL_SOCKET_IFNAME` (e.g. `eth0`) and `NCCL_IB_HCA` (e.g. `mlx5_0,mlx5_1`).

## Collecting Reports
- DaemonSet: reports land on each node under `/var/log/gpu-acceptance`.
- Job: same path via hostPath. Aggregate with a simple ssh/rsync or a daemonset-sidecar collector (optional).

## Troubleshooting
- Pod pending: check `nodeSelector`/`tolerations` and GPU availability.
- NCCL hangs/timeouts:
  - Ensure IB fabric is healthy and ports are up.
  - Try setting `NCCL_DEBUG=INFO` or `TRACE`.
  - Specify interfaces via `NCCL_SOCKET_IFNAME`/`NCCL_IB_HCA`.
- OOM or low perf: ensure `--ipc=host` is not required in your environment (we use `/dev/shm` via `emptyDir`).

## Cleanup
```bash
kubectl delete -f k8s/gpu-test-tiny.yaml || true
kubectl delete -f k8s/gpu-test-job.yaml  || true
```
