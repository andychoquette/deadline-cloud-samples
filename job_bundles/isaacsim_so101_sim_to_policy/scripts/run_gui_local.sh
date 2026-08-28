#!/usr/bin/env bash
#
# LOCAL-ONLY launcher: run render_rollout.py --gui so the Isaac Sim viewport is
# VISIBLE on the local X display (e.g. an Amazon DCV desktop on a GPU dev box).
#
# This is deliberately NOT part of run_in_container.sh. That script is farm
# code: every Deadline Cloud worker runs it, workers are headless, and its
# `docker run` block must not grow X11 mounts, a DISPLAY, or an Xauthority
# that only make sense on a workstation. Keeping them apart means a mistake
# here cannot change what a worker does.
#
# Usage (on the box, as the user who owns the X session):
#   CHECKPOINT=/path/to/checkpoint bash run_gui_local.sh [--episodes 2] ...
# Any arguments are forwarded to render_rollout.py unchanged.
#
# CHECKPOINT is a HOST path and is bind-mounted at /checkpoint inside the
# container, which is then what --checkpoint is set to. Do not pass
# --checkpoint yourself: a host path that happens to exist outside the few
# directories mounted below is invisible to the container, and because
# AppLauncher runs at import time the "no config.json" error only surfaces
# AFTER a full ~350 s Kit boot. Mounting it removes that whole failure mode.
#
# Requirements, all of which are checked below:
#   * an X server you own, with its socket in /tmp/.X11-unix
#   * `xauth` on the host
#   * docker + the NVIDIA Container Toolkit
#
# Why each Docker flag exists -- these were each a failed run:
#
#  * -e HEADLESS=0 -- the workshop image sets HEADLESS=1 in its own ENV, and
#    Isaac Lab's AppLauncher._resolve_headless_settings() only lets the CLI
#    flag *raise* headless, never lower it: with headless=False it falls
#    through to `self._headless = bool(headless_env)`. So --gui alone is
#    silently ignored inside this image. This is the flag that matters most.
#
#  * -e DISPLAY + -v /tmp/.X11-unix -- without a display Kit never even tries
#    to create a window, and the run is headless no matter what else is set.
#
#  * FamilyWild Xauthority (the `sed -e 's/^..../ffff/'` below) -- the host's
#    MIT-MAGIC-COOKIE-1 entry is scoped to the host's *hostname*, and the
#    container has a different one, so mounting ~/.Xauthority verbatim gives
#    "Authorization required, but no authorization protocol specified".
#    Rewriting the address family to ffff (FamilyWild) makes the cookie match
#    any hostname. This is preferred over `xhost +local:`, which disables
#    access control for every local process for as long as you forget to undo
#    it. Nothing here has to be revoked: the temp file is deleted on exit.
#
#  * --gpus all --runtime=nvidia + NVIDIA_DRIVER_CAPABILITIES=all -- `all`
#    (not `compute,utility`) is what injects libGLX_nvidia / libnvidia-glcore
#    and makes /etc/vulkan/icd.d/nvidia_icd.json resolvable. Note the ICD in
#    this image is under /etc/vulkan, NOT /usr/share/vulkan.
#
#  * --shm-size -- same reason as the farm path: torch DataLoader workers.
set -euo pipefail

IMAGE="${IMAGE:-isaacsim-so101-workshop:2.3.2}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/isaac-gui/out}"
CACHE_DIR="${CACHE_DIR:-/tmp/isaacsim-so101-cache}"
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SHM_SIZE="${SHM_SIZE:-8g}"
TIMEOUT_S="${TIMEOUT_S:-1800}"
DISPLAY_ARG="${DISPLAY:-:0}"
CHECKPOINT="${CHECKPOINT:-}"
export DISPLAY="$DISPLAY_ARG"

# --- preflight --------------------------------------------------------------
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not on PATH" >&2; exit 70; }
[ -n "$CHECKPOINT" ] || { echo "ERROR: set CHECKPOINT=<host dir with config.json>" >&2; exit 64; }
[ -f "$CHECKPOINT/config.json" ] || {
  echo "ERROR: '$CHECKPOINT/config.json' not found -- that is not a LeRobot checkpoint." >&2
  exit 64
}
CHECKPOINT="$(cd "$CHECKPOINT" && pwd)"
command -v xauth  >/dev/null 2>&1 || { echo "ERROR: xauth not on PATH (apt-get install xauth)" >&2; exit 70; }
[ -d /tmp/.X11-unix ] || { echo "ERROR: /tmp/.X11-unix missing -- no local X server" >&2; exit 70; }
if ! xauth nlist "$DISPLAY_ARG" >/dev/null 2>&1; then
  echo "ERROR: cannot read an X authority entry for DISPLAY=$DISPLAY_ARG." >&2
  echo "       Run this as the user who owns the X session (on a DCV box that is" >&2
  echo "       the autologin user), not via sudo from a remote shell -- \$XAUTHORITY" >&2
  echo "       would point at the wrong home directory." >&2
  exit 70
fi

# --- FamilyWild cookie ------------------------------------------------------
XAUTH_TMP="$(mktemp /tmp/.docker.xauth.XXXXXX)"
cleanup() { rm -f "$XAUTH_TMP"; }
trap cleanup EXIT
: > "$XAUTH_TMP"
xauth nlist "$DISPLAY_ARG" | sed -e 's/^..../ffff/' | xauth -f "$XAUTH_TMP" nmerge -
chmod 644 "$XAUTH_TMP"

mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$CACHE_DIR"/{kit,ov,pip,glcache,computecache,logs,data,hf,torch}
chmod -R 777 "$CACHE_DIR" 2>/dev/null || true

echo "[gui] image=$IMAGE display=$DISPLAY_ARG out=$OUTPUT_DIR"
echo "[gui] Kit's first boot on a cold shader cache takes ~350 s before a window"
echo "[gui]   appears. Watch $OUTPUT_DIR/logs/gui.log; do not assume a hang."

# NOT `exec docker run`: exec replaces this shell, the EXIT trap never fires,
# and the FamilyWild cookie is left in /tmp for anyone on the box to reuse.
set +e
docker run --rm \
  --runtime=nvidia \
  --gpus all \
  --shm-size="$SHM_SIZE" \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e HEADLESS=0 \
  -e ENABLE_CAMERAS=1 \
  -e "DISPLAY=$DISPLAY_ARG" \
  -e "XAUTHORITY=$XAUTH_TMP" \
  -e WANDB_MODE=disabled \
  -e HF_HOME=/root/.cache/huggingface \
  -e TORCH_HOME=/root/.cache/torch \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$XAUTH_TMP:$XAUTH_TMP:ro" \
  -v "$CACHE_DIR/kit:/isaac-sim/kit/cache:rw" \
  -v "$CACHE_DIR/ov:/root/.cache/ov:rw" \
  -v "$CACHE_DIR/pip:/root/.cache/pip:rw" \
  -v "$CACHE_DIR/glcache:/root/.cache/nvidia/GLCache:rw" \
  -v "$CACHE_DIR/computecache:/root/.nv/ComputeCache:rw" \
  -v "$CACHE_DIR/logs:/root/.nvidia-omniverse/logs:rw" \
  -v "$CACHE_DIR/data:/root/.local/share/ov/data:rw" \
  -v "$CACHE_DIR/hf:/root/.cache/huggingface:rw" \
  -v "$CACHE_DIR/torch:/root/.cache/torch:rw" \
  -v "$SCRIPT_DIR:/job_scripts:ro" \
  -v "$CHECKPOINT:/checkpoint:ro" \
  -v "$OUTPUT_DIR:/outputs" \
  "$IMAGE" \
  bash -c '
    set -uo pipefail
    timeout --kill-after=30 "'"$TIMEOUT_S"'" stdbuf -o0 -e0 \
      python /job_scripts/render_rollout.py --gui \
        --checkpoint /checkpoint --output-dir /outputs "$@" \
      2>&1 | tee /outputs/logs/gui.log
    exit "${PIPESTATUS[0]}"
  ' -- "$@"
RC=$?
set -e
echo "[gui] exit: $RC (log: $OUTPUT_DIR/logs/gui.log)"
exit "$RC"
