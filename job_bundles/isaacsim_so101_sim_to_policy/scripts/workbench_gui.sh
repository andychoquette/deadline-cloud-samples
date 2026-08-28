#!/usr/bin/env bash
#
# LOCAL-ONLY launcher: ONE containerised Isaac Sim GUI session that hosts BOTH
# the Deadline Cloud submitter panel AND an in-session policy rollout. The
# "single-instance workbench".
#
# Derived from run_gui_local.sh, which is left untouched. Read that file's header
# first: every Docker flag it justifies is repeated here for the same reasons and
# is NOT re-derived. The differences, and only the differences, are documented
# below.
#
# Usage (on the box, as the user who owns the X session -- a DCV terminal, or
# `sudo -u ubuntu -H` from a remote shell):
#
#   bash workbench_gui.sh
#
# It blocks until you close the Isaac Sim window. Then, inside that window:
#
#   Window -> Script Editor, and paste two lines:
#       import workbench_rollout as wr
#       wr.run()
#
#   Tools -> Submit to AWS Deadline Cloud   (the submitter panel)
#
# --- what this does that run_gui_local.sh does not ---------------------------
#
#  1. It runs NO rollout and does NOT exit. The container command is
#     workbench_session.py, which launches Kit through Isaac Lab's AppLauncher
#     and then idles in `while simulation_app.is_running(): simulation_app.update()`.
#     The rollout is a module the user imports from the Script Editor later, in
#     THIS app, so there is only ever one Isaac Sim process and one window.
#
#  2. enable_cameras is set at LAUNCH time (workbench_session.py, before
#     AppLauncher). The policy's observation is two 480x640 TiledCameras and
#     `enable_cameras` is a launch-time setting: a Script Editor cannot turn it
#     on afterwards. That is the whole reason this launcher exists rather than
#     "just open Isaac Sim and paste some code".
#
#  3. It mounts what the SUBMITTER needs in order to run inside the container:
#
#     * the extension source, at its host path, and enables
#       isaacsim.deadline.submitter in the same Kit session.
#
#     * a `deadline` CLI that actually runs in the container. VERIFIED: the
#       host's own venv at $DEADLINE_VENV does, provided you also mount the
#       host's python3.12 -- the venv's bin/python3.12 is a symlink to
#       /usr/bin/python3.12, which the image does NOT have (its only python is
#       Kit's 3.11, at /workspace/isaaclab/_isaac_sim/kit/python/bin/python3).
#       Host and image are both Ubuntu 24.04 / glibc 2.39, so the host
#       interpreter runs as-is. Three read-only mounts, no install, and the
#       user's already-configured `deadline 0.60.5` with pyside6 6.10.3 is on
#       PATH inside the container. The alternative -- pip-installing deadline
#       into Kit's python -- would put PySide6 in Kit's site-packages, which the
#       extension's own rules forbid.
#
#     * the six Qt6 xcb runtime libraries the image is MISSING (verified with
#       `ldconfig -p` inside it: libxcb-cursor.so.0, libxcb-icccm.so.4,
#       libxcb-image.so.0, libxcb-keysyms.so.1, libxcb-render-util.so.0,
#       libxcb-xinerama.so.0). Without them `deadline bundle gui-submit` dies
#       with `could not load the Qt platform plugin "xcb"` -- the single most
#       common cause of "the dialog never appears", and it would happen on
#       stage during the demo, not here. They cannot be supplied via
#       LD_LIBRARY_PATH because the extension deliberately SCRUBS
#       LD_LIBRARY_PATH out of the child process (that scrub is what makes the
#       CLI survive being spawned from Kit at all). So the closure is staged
#       from the host and copied into the container's own lib dir at startup,
#       followed by ldconfig. Only libraries the image LACKS are copied;
#       nothing the image already ships is shadowed.
#
#     * PyYAML. CONFIRMED MISSING from this image's Kit python
#       (`python3 -c "import yaml"` -> ModuleNotFoundError), which is the open
#       risk the extension's BUILD-NOTES flags: bundle_filter.py cannot read
#       template.yaml without it, so the panel would load and then refuse to
#       filter. A pure-Python copy is staged from the host and appended to
#       sys.path INSIDE KIT ONLY, and only if `import yaml` actually fails. The
#       staged copy has its cp312 C extension removed so 3.11 uses the pure
#       Python parser.
#
#     * $HOME/.aws (read-only) and the Monitor credential DIRECTORY. The
#       profile's credential_process is literally
#         cat "$HOME/.cache/com.amazonaws.deadline.monitor/credentials_<profile>.json"
#       so no Monitor binary is needed in the container -- just that path. The
#       DIRECTORY is mounted, not the file: the Monitor refreshes credentials by
#       writing a temp file and renaming it, and a bind-mounted FILE would pin
#       the old inode and the session would appear to expire an hour in.
#
#     * a deadline home at $DEADLINE_HOME, mounted rw on /root/.deadline. It
#       defaults to a COPY of $HOME/.deadline rather than the real thing,
#       because the container is root: `gui-submit` writes job history and a
#       `job_id` default back into that tree, and doing so as root would leave
#       root-owned files -- and possibly a root-owned config -- in the user's
#       home. Set DEADLINE_HOME=$HOME/.deadline to use the real one.
#
#  4. It mounts the job bundle directory and the job's OutputDir at IDENTICAL
#     paths inside the container. This is not cosmetic. When the submitter runs
#     in the container, the paths it records into the job -- OutputDir,
#     JobScriptDir -- are CONTAINER paths, and job attachments then upload and
#     download against them. Identical paths are what make the host and the
#     container agree about where the bundle and the outputs are.
#
#  5. It does NOT `exec docker run`. Same bug as run_gui_local.sh calls out, and
#     it now matters twice as much: exec replaces this shell, the EXIT trap
#     never fires, and the FamilyWild cookie AND two staging directories are
#     left in /tmp.
#
# Everything else -- HEADLESS=0 (the image sets HEADLESS=1 and AppLauncher only
# lets the flag RAISE headless), ENABLE_CAMERAS=1, the FamilyWild Xauthority
# rewrite, --gpus all --runtime=nvidia, NVIDIA_DRIVER_CAPABILITIES=all,
# --shm-size, and the cache mounts -- is carried over from run_gui_local.sh
# unchanged, for the reasons documented there.
set -euo pipefail

IMAGE="${IMAGE:-isaacsim-so101-workshop:2.3.2}"
NAME="${NAME:-isaac-workbench}"
CHECKPOINT="${CHECKPOINT:-$HOME/isaac-gui/checkpoint}"
BUNDLE_DIR="${BUNDLE_DIR:-$HOME/isaac-gui/bundle}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/isaac-gui/out}"
CACHE_DIR="${CACHE_DIR:-/tmp/isaacsim-so101-cache}"
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SUBMITTER_SRC="${SUBMITTER_SRC:-$HOME/deadline-cloud-for-isaacsim/src}"
DEADLINE_VENV="${DEADLINE_VENV:-$HOME/deadline-venv}"
DEADLINE_HOME="${DEADLINE_HOME:-$HOME/isaac-gui/workbench-deadline-home}"
AWS_DIR="${AWS_DIR:-$HOME/.aws}"
MONITOR_CACHE="${MONITOR_CACHE:-$HOME/.cache/com.amazonaws.deadline.monitor}"
SHM_SIZE="${SHM_SIZE:-8g}"
DISPLAY_ARG="${DISPLAY:-:0}"
# Empty = no timeout, which is the point of a persistent session. Set it to a
# number of seconds for an unattended test run so nothing is left on the GPU.
TIMEOUT_S="${TIMEOUT_S:-}"
# Free VRAM floor. Below this, abort rather than OOM whatever else the user has
# on the GPU (their own Isaac Sim, the Monitor's compositor).
MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-8000}"
export DISPLAY="$DISPLAY_ARG"

LOG_DIR="$OUTPUT_DIR/logs"
LOG="$LOG_DIR/workbench.log"

# --- preflight --------------------------------------------------------------
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not on PATH" >&2; exit 70; }
command -v xauth  >/dev/null 2>&1 || { echo "ERROR: xauth not on PATH (apt-get install xauth)" >&2; exit 70; }
[ -d /tmp/.X11-unix ] || { echo "ERROR: /tmp/.X11-unix missing -- no local X server" >&2; exit 70; }
if ! xauth nlist "$DISPLAY_ARG" >/dev/null 2>&1; then
  echo "ERROR: cannot read an X authority entry for DISPLAY=$DISPLAY_ARG." >&2
  echo "       Run this as the user who owns the X session (on a DCV box that is" >&2
  echo "       the autologin user), not via sudo from a remote shell -- \$XAUTHORITY" >&2
  echo "       would point at the wrong home directory." >&2
  exit 70
fi
[ -f "$CHECKPOINT/config.json" ] || {
  echo "ERROR: '$CHECKPOINT/config.json' not found -- that is not a LeRobot checkpoint." >&2
  echo "       Set CHECKPOINT=<host dir with config.json>." >&2
  exit 64
}
CHECKPOINT="$(cd "$CHECKPOINT" && pwd)"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "ERROR: a container named '$NAME' already exists. Another workbench is" >&2
  echo "       probably running. Deliberately NOT removing it -- check with" >&2
  echo "       'docker ps -a' and either use it or 'docker rm -f $NAME' yourself." >&2
  exit 69
fi

# Never OOM someone else's session. The workbench needs ~6-8 GiB for Kit plus
# the policy; below the floor, say so and stop.
FREE_VRAM="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)"
if [ "${FREE_VRAM:-0}" -lt "$MIN_FREE_VRAM_MIB" ]; then
  echo "ERROR: only ${FREE_VRAM} MiB of VRAM free, need >= ${MIN_FREE_VRAM_MIB} MiB." >&2
  echo "       Something else is on the GPU. Refusing to launch rather than OOM it." >&2
  nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv >&2 || true
  exit 75
fi
echo "[wb] ${FREE_VRAM} MiB VRAM free"

# --- staging ----------------------------------------------------------------
XAUTH_TMP="$(mktemp /tmp/.docker.xauth.XXXXXX)"
STAGE_LIBS="$(mktemp -d /tmp/.wb-hostlibs.XXXXXX)"
STAGE_PY="$(mktemp -d /tmp/.wb-pypath.XXXXXX)"
cleanup() { rm -rf "$XAUTH_TMP" "$STAGE_LIBS" "$STAGE_PY"; }
trap cleanup EXIT

# FamilyWild cookie -- see run_gui_local.sh for why the address family is
# rewritten to ffff rather than using `xhost +local:`.
: > "$XAUTH_TMP"
xauth nlist "$DISPLAY_ARG" | sed -e 's/^..../ffff/' | xauth -f "$XAUTH_TMP" nmerge -
chmod 644 "$XAUTH_TMP"

# Qt6 xcb closure. Taken from libqxcb.so itself rather than from a hand-written
# list of six names, so a PySide6 upgrade that needs a seventh library cannot
# silently break the submit dialog. Only the libraries the IMAGE lacks are
# installed, container-side; staging extra copies here is harmless.
VENV_SITE="$(echo "$DEADLINE_VENV"/lib/python3.*/site-packages)"
QT_PLUGIN="$VENV_SITE/PySide6/Qt/plugins/platforms/libqxcb.so"
QT_LIBDIR="$VENV_SITE/PySide6/Qt/lib"
if [ -f "$QT_PLUGIN" ]; then
  LD_LIBRARY_PATH="$QT_LIBDIR" ldd "$QT_PLUGIN" 2>/dev/null \
    | awk '{print $3}' | grep -E '^/(usr/)?lib' | sort -u \
    | while read -r so; do cp -Lf "$so" "$STAGE_LIBS/" 2>/dev/null || true; done
  echo "[wb] staged $(ls -1 "$STAGE_LIBS" | wc -l) host libs for the Qt xcb plugin"
else
  echo "[wb] WARNING: $QT_PLUGIN not found -- 'deadline bundle gui-submit' will"
  echo "[wb]          probably fail to load its Qt xcb platform plugin."
fi

# SQLite for the `deadline` CLI. The image has no python3.12, so the CLI runs
# against the host's /usr/lib/python3.12 mount -- which carries
# _sqlite3.cpython-312-*.so but NOT libsqlite3.so.0, which lives in
# /usr/lib/x86_64-linux-gnu and was never mounted. `ldd` on the extension then
# reports `libsqlite3.so.0 => not found`, `import sqlite3` raises ImportError,
# and deadline-cloud's job-attachments cache catches it, sets enabled=False,
# leaves db_connection as None -- and the submit dies on
# `'NoneType' object has no attribute 'execute'`, a message that names neither
# SQLite nor the cache. Cost an entire debugging round; do not remove this.
# Closure via ldd rather than a name list, same reasoning as the Qt block.
SQLITE_EXT="$(echo /usr/lib/python3.12/lib-dynload/_sqlite3.cpython-*.so)"
if [ -f "$SQLITE_EXT" ]; then
  before="$(ls -1 "$STAGE_LIBS" 2>/dev/null | wc -l)"
  ldd "$SQLITE_EXT" 2>/dev/null \
    | awk '{print $3}' | grep -E '^/(usr/)?lib' | sort -u \
    | while read -r so; do cp -Lf "$so" "$STAGE_LIBS/" 2>/dev/null || true; done
  echo "[wb] staged the _sqlite3 closure ($(( $(ls -1 "$STAGE_LIBS" | wc -l) - before )) new) --"
  echo "[wb]   without it the submit dialog dies on a NoneType .execute error"
else
  echo "[wb] WARNING: no _sqlite3 extension under /usr/lib/python3.12/lib-dynload."
  echo "[wb]          Submitting will fail with \"'NoneType' object has no"
  echo "[wb]          attribute 'execute'\" -- that error means SQLite, not the panel."
fi

# Pure-Python PyYAML for Kit's 3.11. The cp312 accelerator is deleted from the
# COPY (never from the venv) so `import yaml` falls back to the Python parser.
if [ -d "$VENV_SITE/yaml" ]; then
  cp -a "$VENV_SITE/yaml" "$STAGE_PY/"
  rm -rf "$STAGE_PY/yaml/__pycache__"
  rm -f "$STAGE_PY"/yaml/_yaml*.so
  echo "[wb] staged pure-Python PyYAML for Kit's python"
else
  echo "[wb] WARNING: no PyYAML found under $VENV_SITE -- if Kit's python also"
  echo "[wb]          lacks it, the submitter panel cannot read template.yaml."
fi

# A private deadline home, so a root container cannot leave root-owned files in
# the user's ~/.deadline. Copied once; afterwards it is the workbench's own.
if [ ! -d "$DEADLINE_HOME" ]; then
  mkdir -p "$DEADLINE_HOME"
  [ -d "$HOME/.deadline" ] && cp -a "$HOME/.deadline/." "$DEADLINE_HOME/" || true
  echo "[wb] created $DEADLINE_HOME from $HOME/.deadline"
fi

mkdir -p "$LOG_DIR" "$BUNDLE_DIR"
mkdir -p "$CACHE_DIR"/{kit,ov,pip,glcache,computecache,logs,data,hf,torch}
chmod -R 777 "$CACHE_DIR" 2>/dev/null || true

[ -f "$BUNDLE_DIR/template.yaml" ] || {
  echo "[wb] WARNING: no template.yaml in $BUNDLE_DIR. The panel will open but"
  echo "[wb]          will have no bundle to load. Copy the job bundle there,"
  echo "[wb]          or set BUNDLE_DIR."
}
[ -d "$SUBMITTER_SRC/isaacsim.deadline.submitter" ] || {
  echo "ERROR: $SUBMITTER_SRC does not contain isaacsim.deadline.submitter." >&2
  echo "       --ext-folder must point at the PARENT of the extension dir." >&2
  exit 64
}

echo "[wb] image=$IMAGE display=$DISPLAY_ARG"
echo "[wb] bundle=$BUNDLE_DIR  outputs=$OUTPUT_DIR  (both mounted at the SAME path inside)"
echo "[wb] log=$LOG"
echo "[wb] Kit's first boot on a cold shader cache takes ~350 s before a window"
echo "[wb]   appears. Watch $LOG; do not assume a hang."

# Optional mounts: only added when the host actually has them, so a box without
# the host python3.12 still gets a GUI + rollout (it just has no working CLI).
OPT_MOUNTS=()
add_ro() { [ -e "$1" ] && OPT_MOUNTS+=(-v "$1:$1:ro") || echo "[wb] WARNING: missing $1"; }
add_ro "$DEADLINE_VENV"
add_ro /usr/bin/python3.12
add_ro /usr/lib/python3.12
add_ro /usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0
add_ro "$AWS_DIR"
add_ro "$MONITOR_CACHE"
add_ro "$SUBMITTER_SRC"
add_ro "$BUNDLE_DIR"
# Fonts. The image has NO /usr/share/fonts and no fc-list at all, so Qt draws
# every glyph in the submit dialog as a placeholder box. libfontconfig itself IS
# in the image -- it is the config and the font files that are absent, which is
# why the only hint is `Fontconfig error: Cannot load default config file`.
# Mounted rather than apt-installed: no network dependency, nothing to redo on
# each launch. Note a Qt smoke test that only opens a display connection passes
# on a fontless image, so this cannot be verified without rendering text.
add_ro /etc/fonts
add_ro /usr/share/fonts

# NOT `exec docker run`: exec replaces this shell, the EXIT trap never fires,
# and the cookie plus both staging dirs are left in /tmp.
set +e
docker run --rm \
  --name "$NAME" \
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
  -e "AWS_CONFIG_FILE=$AWS_DIR/config" \
  -e "OUTPUT_DIR=$OUTPUT_DIR" \
  -e "WORKBENCH_BUNDLE_DIR=$BUNDLE_DIR" \
  -e "WORKBENCH_DEADLINE_VENV=$DEADLINE_VENV" \
  -e "WORKBENCH_SUBMITTER_SRC=$SUBMITTER_SRC" \
  -e "WORKBENCH_TIMEOUT_S=$TIMEOUT_S" \
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
  -v "$OUTPUT_DIR:$OUTPUT_DIR:rw" \
  -v "$DEADLINE_HOME:/root/.deadline:rw" \
  -v "$STAGE_LIBS:/opt/wb-hostlibs:ro" \
  -v "$STAGE_PY:/opt/wb-pypath:ro" \
  "${OPT_MOUNTS[@]}" \
  "$IMAGE" \
  bash -c '
    set -uo pipefail

    # The Qt xcb libraries the image lacks. Copied (not LD_LIBRARY_PATHed): the
    # submitter scrubs LD_LIBRARY_PATH out of the `deadline` child on purpose.
    if [ -d /opt/wb-hostlibs ]; then
      have="$(ldconfig -p | awk "{print \$1}")"
      n=0
      for so in /opt/wb-hostlibs/*.so*; do
        b="$(basename "$so")"
        if ! printf "%s\n" "$have" | grep -Fxq "$b"; then
          cp "$so" /usr/lib/x86_64-linux-gnu/ 2>/dev/null && n=$((n+1))
        fi
      done
      ldconfig
      echo "[wb] installed $n host libs the image was missing"
    fi

    # The CLI on PATH. Kit inherits this, so the extension resolves it with
    # shutil.which("deadline") and needs no setting.
    #
    # APPENDED, NOT PREPENDED. The venv also contains `python`, `python3` and
    # `python3.12`, and this image runs Isaac Sim through a `python` SHIM that
    # exports PYTHONHOME=<Kit python 3.11>. Prepending the venv makes `python`
    # resolve to the 3.12 binary in that venv under a 3.11 PYTHONHOME, and Kit dies
    # before it starts with:
    #     AssertionError: SRE module mismatch
    # from re/_compiler.py -- a 3.12 interpreter reading a 3.11 stdlib. Observed,
    # not theorised. Appending leaves the shim first and still resolves
    # `deadline`, which is the only name needed from that venv.
    if [ -x "${WORKBENCH_DEADLINE_VENV:-}/bin/deadline" ]; then
      export PATH="$PATH:${WORKBENCH_DEADLINE_VENV}/bin"
      # Scrubbed for the same reason the extension scrubs them before spawning
      # the real submit: the entrypoint of this image exports PYTHONHOME / PYTHONPATH
      # for Kit 3.11, and the CLI is a 3.12 venv. Inherited, they produce the
      # same SRE mismatch. This echo is a smoke test of the exact contract
      # submit_runner.py implements, so if it prints a version the panel can
      # spawn the CLI too.
      echo "[wb] deadline: $(command -v deadline) -> $(env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH deadline --version 2>&1 | tail -1)"
    else
      echo "[wb] WARNING: no deadline CLI on PATH inside the container"
    fi

    # No timeout by default -- a workbench is meant to outlive one rollout. A
    # value is only set for unattended test runs.
    TMO=()
    if [ -n "${WORKBENCH_TIMEOUT_S:-}" ]; then
      TMO=(timeout --kill-after=30 "$WORKBENCH_TIMEOUT_S")
      echo "[wb] session will be killed after ${WORKBENCH_TIMEOUT_S}s (TIMEOUT_S)"
    fi

    "${TMO[@]}" stdbuf -o0 -e0 python /job_scripts/workbench_session.py \
      --checkpoint /checkpoint \
      --output-dir "$OUTPUT_DIR" \
      --ext-folder "$WORKBENCH_SUBMITTER_SRC" \
      --enable isaacsim.deadline.submitter \
      --pypath /opt/wb-pypath \
      "$@"
  ' -- "$@" 2>&1 | tee "$LOG"
RC="${PIPESTATUS[0]}"
set -e
echo "[wb] exit: $RC (log: $LOG)"
exit "$RC"
