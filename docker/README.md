# Docker build for vllm-kvnorm

Produces a reproducible image containing:

- **vLLM 0.28.0** (base `vllm/vllm-openai`) — version 0.28.0 is required; the connector uses the Hybrid Memory Allocator (HMA) connector API introduced in vLLM ≥ 0.21, and 0.28.0 is the validated version.
- **Flowcept fork** from `https://github.com/spotter-ai-genesis/flowcept.git` — carries `flowcept.flowceptor.adapters.vllm`, which is not present in released flowcept packages.
- **vllm-kvnorm** (this repo) — the KVNormConnector.

## Building

**x86 / amd64:**

```bash
docker build -f docker/Dockerfile -t vllm-kvnorm:0.28.0 .
```

**GH200 / aarch64 (CUDA 12.9):**

```bash
docker build -f docker/Dockerfile \
  --build-arg VLLM_TAG=v0.28.0-aarch64-cu129 \
  -t vllm-kvnorm:0.28.0-gh200 .
```

The build runs a sanity import (`KVNormConnector` + `vllm.__version__`) and will fail fast if anything is missing.

## HPC / Apptainer (DeltaAI)

Compute nodes lack a Docker daemon. On Delta we build the `.sif` directly via Apptainer from the base image and the same two pip installs (see `deploy/build.slurm` in the sibling experiments/deploy area). Key points:

- OCI→SIF extraction must target node-local tmpfs (e.g. `/dev/shm`), **not** Lustre `/work` — extracting the ~10 GB vLLM image on Lustre timed out due to slow metadata and no xattr support.
- If you have already built the Docker image locally, you can also convert it:

  ```bash
  apptainer build vllm-kvnorm.sif docker-daemon://vllm-kvnorm:0.28.0-gh200
  ```

  from a machine where the Docker daemon is running and has the image loaded.

## Runtime dependencies

- **Redis** — a live Redis instance is required for the default multiprocess / served path (flowcept uses it as a message queue). Set the connection details in your flowcept settings file.
- Set `FLOWCEPT_SETTINGS_PATH` to point at your flowcept settings YAML/TOML.
- To skip Redis for single-process use, set `VLLM_ENABLE_V1_MULTIPROCESSING=0`.
