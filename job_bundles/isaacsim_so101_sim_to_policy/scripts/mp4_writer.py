#!/usr/bin/env python3
"""One MP4 encoder, shared by the rollout render and the datagen debug video.

Extracted from render_rollout.py so `generate_dataset.py` can record footage
without carrying a second, subtly-different encoder. The hard-won part is
`pick_encoder`: the image's FFmpeg is an LGPL build with no libx264, and a
hardcoded `-c:v libx264` fails at runtime while the task still reports success.
Duplicating that probe would mean duplicating the chance of getting it wrong.

Import this AFTER AppLauncher has started Kit. It only needs numpy and
subprocess, but both callers keep their non-Isaac imports below the launcher and
this file follows that convention rather than fighting it.
"""

from __future__ import annotations

import re
import subprocess

import numpy as np

import video_policy as vp

_ENCODER = None


def _log(msg):
    print(f"[video] {msg}", flush=True)


def pick_encoder():
    """Choose a video encoder this ffmpeg build actually has, and its quality flags.

    Do NOT hardcode libx264. The image installs an **LGPL** FFmpeg build (BtbN
    `ffmpeg-n7.1.x-linux64-lgpl-shared`) because torchcodec has to link ffmpeg's
    shared libraries -- and LGPL builds deliberately omit libx264, which is GPL.
    Asking for it fails at runtime with:

        Unknown encoder 'libx264'
        Error opening output files: Encoder not found

    and because this writer is intentionally non-fatal (a lost video must not
    cost us the success metrics), the task still SUCCEEDS while silently
    producing no MP4. Measured on a real 8-shard fan-out: 8/8 shards wrote their
    PNG and their JSON, and 0/8 wrote an MP4 -- with nothing but a line in the
    log to say so. That is exactly the failure mode a probe prevents.

    Preference order: libx264 (in case a GPL build is swapped in), then
    libopenh264 (Cisco's; LGPL builds do ship it; real H.264 in an .mp4), then
    mpeg4 (native to every ffmpeg, always present, chunkier but universally
    playable).
    """
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER
    quality = {
        # -crf is x264/x265-specific; openh264 wants a bitrate; mpeg4 wants -q:v.
        "libx264": ["-preset", "medium", "-crf", "20"],
        "libopenh264": ["-b:v", "4M"],
        "mpeg4": ["-q:v", "3"],
    }
    available = ""
    try:
        available = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        _log(f"could not probe ffmpeg encoders ({exc!r}); assuming mpeg4")
    for cand in ("libx264", "libopenh264", "mpeg4"):
        # Encoder lines look like " V....D libx264   H.264 ...", so match the
        # name as a standalone token to avoid 'libx264' matching 'libx264rgb'.
        if re.search(r"^\s*V\S*\s+" + re.escape(cand) + r"\s", available, re.M):
            _ENCODER = (cand, quality[cand])
            _log(f"video encoder: {cand}")
            return _ENCODER
    _ENCODER = ("mpeg4", quality["mpeg4"])
    _log("no preferred encoder found in `ffmpeg -encoders`; falling back to mpeg4")
    return _ENCODER


class Mp4Writer:
    """Stream uint8 HWC frames straight into the ffmpeg binary in the image.

    Streaming rather than buffering: at 480x640x3 a single 15 s episode is ~200
    MB of frames, so a 100-episode run would need tens of gigabytes of RAM
    before encoding anything.

    Piping rawvideo to ffmpeg rather than going through imageio: the image ships
    a shared-library ffmpeg build (torchcodec needs it) but not imageio-ffmpeg,
    so `imageio.mimwrite` has no encoder plugin available.
    """

    # Machine-readable reasons, recorded in the summary JSON as "video_status".
    # The point of separating these is that only ONE of them is the rollout's
    # fault, and only that one should ever fail the task. See render_rollout's
    # main().
    OK = vp.VIDEO_OK
    DISABLED = vp.VIDEO_DISABLED
    NO_FRAMES = vp.VIDEO_NO_FRAMES
    NOT_FOUND = vp.VIDEO_NOT_FOUND
    NO_ENCODER = vp.VIDEO_NO_ENCODER
    ENC_FAILED = vp.VIDEO_ENC_FAILED

    def __init__(self, path, fps):
        self.path = path
        self.fps = fps
        self.proc = None
        self.count = 0
        self.failed = False
        # None means "nothing has gone wrong yet".
        self.reason = None
        self.detail = None

    def _fail(self, reason, detail):
        self.failed = True
        # Keep the FIRST reason: it is the root cause. A dead encoder produces a
        # broken pipe on the next write, and reporting the pipe would bury the
        # actual "Unknown encoder" message.
        if self.reason is None:
            self.reason, self.detail = reason, detail
        _log(f"VIDEO {reason}: {detail}")

    _classify = staticmethod(vp.classify_ffmpeg_error)

    def _start(self, height, width):
        encoder, quality = pick_encoder()
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "rgb24", "-r", str(self.fps),
            "-i", "-",
            "-c:v", encoder, *quality,
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            self.path,
        ]
        try:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            self._fail(self.NOT_FOUND,
                       "no ffmpeg binary on PATH in the container")

    def add(self, frame):
        if self.failed:
            return
        if self.proc is None:
            self._start(frame.shape[0], frame.shape[1])
            if self.failed:
                return
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            # ffmpeg died mid-stream. Report it once and keep the caller going:
            # a lost video must not cost us the run's actual results.
            err = self.proc.stderr.read().decode(errors="replace")[:600]
            self._fail(self._classify(err), f"ffmpeg exited early: {err}")
            return
        self.count += 1

    def close(self):
        if self.proc is None:
            # Either never started (no frames offered) or the binary was missing.
            return False
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.wait()
        if self.proc.returncode != 0 or self.failed:
            err = b""
            try:
                err = self.proc.stderr.read()
            except (OSError, ValueError):
                pass
            text = err.decode(errors="replace")[:600]
            self._fail(self._classify(text),
                       f"ffmpeg exited {self.proc.returncode}: {text}")
            return False
        _log(f"wrote {self.path} ({self.count} frames at {self.fps} fps)")
        return True


def write_png(frame, path):
    try:
        from PIL import Image
        Image.fromarray(np.asarray(frame)).save(path)
        _log(f"wrote {path}")
    except Exception as exc:  # noqa: BLE001 - a missing thumbnail must not fail the task
        _log(f"thumbnail skipped ({exc!r})")
