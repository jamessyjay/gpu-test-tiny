export IMAGE=ghcr.io/jamessyjay/gpu-cluster-acceptance

echo "Building docker image locally"
docker build --no-cache --platform=linux/amd64 -f Dockerfile.nv-tiny -t "${IMAGE}:tiny" .
echo "Uploading docker image to $IMAGE"
docker push "${IMAGE}:tiny"

# sanity: just check that python is there and .so libraries are loaded
echo "Validate importability of NVidia and CUDA so libraries"
docker run --rm --gpus all "${IMAGE}:tiny" bash -lc 'python - <<PY
import ctypes
def probe(n): 
    try: 
        print(n, "-> OK"); ctypes.cdll.LoadLibrary(n)
    except OSError as e: 
        print(n, "->", e)
[probe(n) for n in ["libcudart.so.13","libcudart.so.12","libcudart.so",
                    "libcublas.so.13","libcublas.so.12","libcublas.so",
                    "libnccl.so.2","libnccl.so"]]
PY'

# real run (IPC is important)
echo "Running python test for GPU"
docker run --rm --gpus all --ipc=host "${IMAGE}:tiny" \
  python src/gpu_accept_cuda_so.py

# clean up
echo "Cleaning docker artefacts"
docker image ls
docker system prune
docker image ls

echo "[Done] Completed"