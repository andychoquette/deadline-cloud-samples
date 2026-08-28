#!/usr/bin/env bash
#
# Run one command inside the Isaac Sim / Isaac Lab SO-101 workshop container.
#
# All three steps of this job bundle (Datagen, Train, Render) need the same
# ~50 lines of `docker run` boilerplate: GPU runtime flags, the Omniverse EULA
# env vars, cache bind mounts, permission fixes for Deadline Cloud's asset
# staging, an inner wall-clock timeout, and exit-code propagation through two
# `tee` pipelines. Rather than repeat that in each step's embedded script, it
# lives here once and each step calls it with the command to run.
#
# Invoked as `bash run_in_container.sh ...` -- it never relies on the execute
# bit, because Deadline Cloud's asset staging can strip +x from
# job-attachment files.
#
# Usage:
#   bash run_in_container.sh \
#     --image URI --ecr-login true|false --cache-dir DIR --timeout SECONDS \
#     --script-dir DIR --output-dir DIR --phase NAME \
#     -- COMMAND [ARG ...]
#
# COMMAND and its arguments are forwarded into the container as an argv array
# and executed via "$@". They are never re-parsed by a shell, so values
# containing spaces, quotes, `$(...)` or backticks are inert.
set -euo pipefail

IMAGE=""
ECR_LOGIN="false"
CACHE_DIR="/tmp/isaacsim-so101-cache"
TIMEOUT_S="3600"
SCRIPT_DIR=""
OUTPUT_DIR=""
PHASE="run"
SHM_SIZE="8g"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --image)      IMAGE="$2"; shift 2 ;;
    --ecr-login)  ECR_LOGIN="$2"; shift 2 ;;
    --cache-dir)  CACHE_DIR="$2"; shift 2 ;;
    --timeout)    TIMEOUT_S="$2"; shift 2 ;;
    --script-dir) SCRIPT_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --phase)      PHASE="$2"; shift 2 ;;
    --shm-size)   SHM_SIZE="$2"; shift 2 ;;
    --)           shift; break ;;
    *) echo "run_in_container.sh: unknown option '$1'" >&2; exit 64 ;;
  esac
done

[ -n "$IMAGE" ]      || { echo "run_in_container.sh: --image is required" >&2; exit 64; }
[ -n "$SCRIPT_DIR" ] || { echo "run_in_container.sh: --script-dir is required" >&2; exit 64; }
[ -n "$OUTPUT_DIR" ] || { echo "run_in_container.sh: --output-dir is required" >&2; exit 64; }
[ "$#" -gt 0 ]       || { echo "run_in_container.sh: no command given after --" >&2; exit 64; }

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not on PATH. This bundle needs a fleet whose workers have" >&2
  echo "       Docker and the NVIDIA Container Toolkit installed. Deadline Cloud" >&2
  echo "       service-managed Linux GPU fleets have both." >&2
  exit 70
fi

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Deadline Cloud's asset staging can strip the execute bit and tighten read
# permissions on staged inputs. Make the bundled scripts readable and their
# parent directories traversable so the container can read them through the
# bind mount. Harmless when the container runs as root; required if someone
# swaps in a rootless image.
chmod a+r "$SCRIPT_DIR"/* 2>/dev/null || true
chmod a+rx "$SCRIPT_DIR" 2>/dev/null || true
chmod a+rx "$(dirname "$SCRIPT_DIR")" 2>/dev/null || true

# --- Registry login ---------------------------------------------------------
# The ECR region is parsed out of the image URI rather than accepted as a
# separate free-text parameter, and the only user-facing switch is a
# true/false dropdown. A "registry login command" string parameter would be a
# shell-injection hole by construction.
if [ "$ECR_LOGIN" = "true" ]; then
  REGISTRY="${IMAGE%%/*}"
  REGION="$(printf '%s' "$REGISTRY" \
    | sed -n 's/^[0-9]\{12\}\.dkr\.ecr\.\([a-z0-9-]\{1,\}\)\.amazonaws\.com.*$/\1/p')"
  if [ -z "$REGION" ]; then
    echo "ERROR: EcrLogin=true but '$IMAGE' is not an Amazon ECR URI of the form" >&2
    echo "       <account-id>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>" >&2
    exit 64
  fi
  echo "[bundle] docker login $REGISTRY (region $REGION)"
  aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$REGISTRY"
fi

echo "[bundle] docker pull $IMAGE"
# ~50 GB image: 5-10 min on a worker that has never pulled it, seconds after.
time docker pull "$IMAGE"

# --- Caches -----------------------------------------------------------------
# Without the Omniverse/shader caches, every task pays a cold shader compile.
# TORCH_HOME matters too: ACT's ResNet-18 vision backbone downloads ImageNet
# weights on first use. Keeping all of this under /tmp lets it survive across
# tasks that land on the same worker.
# ShaderCacheDir defaults to a path on the fleet's persistent volume, which
# survives worker replacement and so turns a multi-minute cold shader compile
# into seconds on every worker after the first. Fleets without a persistent
# volume configured have no such mount point, so fall back to /tmp rather than
# failing the task: the cache is a cold-start optimisation, never a correctness
# requirement. Caches then live only as long as the worker.
if ! mkdir -p "$CACHE_DIR" 2>/dev/null; then
  echo "[bundle] WARNING: cannot create '$CACHE_DIR'." >&2
  echo "[bundle]   Its parent does not exist or is not writable. If this path is on a" >&2
  echo "[bundle]   fleet persistent volume, this fleet has none configured." >&2
  echo "[bundle]   Falling back to /tmp -- caches will NOT survive this worker." >&2
  CACHE_DIR="/tmp/isaacsim-so101-cache"
  mkdir -p "$CACHE_DIR"
fi

mkdir -p "$CACHE_DIR"/{kit,ov,pip,glcache,computecache,logs,data,hf,torch}
chmod -R 777 "$CACHE_DIR" 2>/dev/null || true

# Report cache state up front -- a warm cache is the difference between a
# ~30 s and a ~15 min Kit boot, so it is worth seeing in the task log.
# Do NOT test glcache here. Headless Kit never initialises OpenGL -- it logs
# "GLFW initialization failed" by design when there is no display -- so
# $CACHE_DIR/glcache stays empty forever and is a permanent false negative. It
# reported COLD through a measured 9.9x warm-boot speedup. The caches that
# actually make a warm boot fast are Kit's DerivedDataCache (kit/) and the
# Omniverse texture cache (ov/); on a warm volume those hold ~63M and ~81M
# while glcache holds 0 bytes.
if [ -n "$(find "$CACHE_DIR/kit" "$CACHE_DIR/ov" -type f 2>/dev/null | head -1)" ]; then
  echo "[bundle] shader cache: WARM ($CACHE_DIR, $(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1))"
else
  echo "[bundle] shader cache: COLD ($CACHE_DIR) -- expect a slow first Kit boot"
fi

# /isaac-sim/kit/cache is the target the workshop's own `docker run` command
# uses, so it is the documented one. In the isaac-lab image Isaac Sim actually
# lives at /workspace/isaaclab/_isaac_sim, so that mount only has an effect if
# /isaac-sim is a symlink to it. If a profiling run shows Kit still recompiling
# shaders on a warm worker, set KIT_CACHE_TARGET to
# /workspace/isaaclab/_isaac_sim/kit/cache. Mounting only one of the two avoids
# a duplicate-mount-target error should /isaac-sim turn out to be a symlink.
# The remaining caches below are HOME-relative and correct either way, and
# GLCache + ComputeCache are where most of the warm-start win comes from.
KIT_CACHE_TARGET="${KIT_CACHE_TARGET:-/isaac-sim/kit/cache}"

# --- Run --------------------------------------------------------------------
# Flags that look removable but are not:
#
#  * The image ENTRYPOINT is KEPT, not replaced with `--entrypoint bash`.
#    (`--entrypoint bash` is the documented workaround for the plain
#    nvcr.io/nvidia/isaac-sim image, whose entrypoint swallows stdout.) This
#    image's entrypoint is the workshop's: it sources setup_python_env.sh,
#    sets CARB_APP_PATH / ISAAC_PATH / EXP_PATH, installs the `python` shim
#    that forwards to Isaac Sim's python.sh, and ends in `exec "$@"`. It does
#    not swallow stdout, and dropping it makes every Isaac Lab import fail.
#
#  * stdbuf -o0 -e0 disables stdio buffering inside the container. Without it
#    a long silent Kit boot looks like a hang and a traceback arrives only at
#    exit.
#
#  * The inner `timeout` makes the container exit on its own before the
#    worker's task timeout SIGKILLs it. A killed container never lets Deadline
#    Cloud's output sync run, so partial results and logs are lost.
#
#  * --shm-size: PyTorch DataLoader workers communicate through /dev/shm.
#    Docker's 64 MB default makes `lerobot-train` die with "DataLoader worker
#    killed" on image batches.
#
#  * HOST_UID/HOST_GID and the chown trap: this image runs as root, so files
#    it writes into the bind-mounted output directory are root-owned. Read
#    access is enough for output upload, but a later re-run of the step could
#    not `rm -rf` a root-owned subdirectory. The trap also runs on failure.
echo "[bundle] === $PHASE === $*"
set +e
time docker run --rm \
  --runtime=nvidia \
  --gpus all \
  --shm-size="$SHM_SIZE" \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e HEADLESS=1 \
  -e ENABLE_CAMERAS=1 \
  -e "INNER_TIMEOUT_S=$TIMEOUT_S" \
  -e "HOST_UID=$(id -u)" \
  -e "HOST_GID=$(id -g)" \
  -e "PHASE=$PHASE" \
  -e WANDB_MODE=disabled \
  -e HF_HOME=/root/.cache/huggingface \
  -e TORCH_HOME=/root/.cache/torch \
  -v "$CACHE_DIR/kit:$KIT_CACHE_TARGET:rw" \
  -v "$CACHE_DIR/ov:/root/.cache/ov:rw" \
  -v "$CACHE_DIR/pip:/root/.cache/pip:rw" \
  -v "$CACHE_DIR/glcache:/root/.cache/nvidia/GLCache:rw" \
  -v "$CACHE_DIR/computecache:/root/.nv/ComputeCache:rw" \
  -v "$CACHE_DIR/logs:/root/.nvidia-omniverse/logs:rw" \
  -v "$CACHE_DIR/data:/root/.local/share/ov/data:rw" \
  -v "$CACHE_DIR/hf:/root/.cache/huggingface:rw" \
  -v "$CACHE_DIR/torch:/root/.cache/torch:rw" \
  -v "$SCRIPT_DIR:/job_scripts:ro" \
  -v "$OUTPUT_DIR:/outputs" \
  "$IMAGE" \
  bash -c '
    set -uo pipefail
    mark() { printf "%s %s\n" "$(date -u +%FT%TZ)" "$1" \
               > "/outputs/logs/_marker_${PHASE}_$1" 2>/dev/null || true; }
    # Hand the outputs back to the submitting user however we exit.
    trap "chown -R \"${HOST_UID}:${HOST_GID}\" /outputs 2>/dev/null || true" EXIT
    mark 01_container_started
    echo "[inner] $(date -u +%FT%TZ) phase=${PHASE} timeout=${INNER_TIMEOUT_S}s"
    echo "[inner] cmd: $*"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
    mark 02_gpu_probed
    timeout --kill-after=30 "${INNER_TIMEOUT_S}" stdbuf -o0 -e0 "$@" \
      2>&1 | tee "/outputs/logs/${PHASE}.log"
    INNER_EXIT="${PIPESTATUS[0]}"
    mark 03_command_returned
    echo "[inner] exit=${INNER_EXIT} at $(date -u +%FT%TZ)"
    printf "%s\n" "${INNER_EXIT}" > "/outputs/logs/_inner_exit_${PHASE}"
    exit "${INNER_EXIT}"
  ' -- "$@" \
  2>&1 | tee "$LOG_DIR/docker_${PHASE}.log"
DOCKER_EXIT=${PIPESTATUS[0]}
set -e

# Belt and braces: if the container's chown trap could not run (e.g. the
# container was SIGKILLed), at least leave everything readable so the output
# upload still works.
chmod -R a+rX "$OUTPUT_DIR" 2>/dev/null || true

echo "[bundle] $PHASE docker exit: $DOCKER_EXIT"
if [ "$DOCKER_EXIT" -eq 124 ] || [ "$DOCKER_EXIT" -eq 137 ]; then
  echo "[bundle] exit $DOCKER_EXIT is the inner ${TIMEOUT_S}s timeout firing." >&2
  echo "[bundle] Raise StepTimeoutSeconds, or read logs/${PHASE}.log to find the hang." >&2
fi
exit "$DOCKER_EXIT"
