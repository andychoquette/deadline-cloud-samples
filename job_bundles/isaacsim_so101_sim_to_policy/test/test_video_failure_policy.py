#!/usr/bin/env python3
"""Prove that a video failure cannot fail a rollout task that produced episodes.

Run with no arguments and no dependencies:

    python3 test/test_video_failure_policy.py

Lives outside `scripts/` on purpose: `scripts/` is the `JobScriptDir` job
attachment, uploaded to every worker on every submit, so tests do not belong in
it. Mirrors the layout of `job_bundles/job_dev_progression/.../test/`.

Why this file exists at all
--------------------------
The regression it guards is not hypothetical. `render_rollout.py` used to end
with:

    if want_video and frames == 0:
        return 4

`frames` is what the *encoder* accepted, so when the container's LGPL ffmpeg
build turned out to have no libx264, a shard that had run all 8 of its episodes
and written valid success data was one frame away from exiting 4 -- which would
have failed the task, cancelled `5 - Aggregate` by dependency, and destroyed the
run's headline number over a missing artifact. On a real 8-shard farm run it
survived only because exactly one frame reached ffmpeg before it died
(`frames == 1`). That is luck, not a safety property, and reading the code did
not reveal it. Hence a test.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "scripts"))

import video_policy as vp  # noqa: E402

FAILS = []
CHECKS = [0]


def check(label, got, want):
    CHECKS[0] += 1
    if got != want:
        FAILS.append(f"{label}\n      got:  {got!r}\n      want: {want!r}")
        print(f"  FAIL  {label}")
    else:
        print(f"  ok    {label}")


# ---------------------------------------------------------------------------
# 1. THE HEADLINE GUARANTEE: every environment-side video failure still exits 0
#    when episodes ran. This is the regression.
# ---------------------------------------------------------------------------
print("\n[1] a dead encoder must NOT fail a shard that produced episodes")
for status in sorted(vp.ENVIRONMENT_VIDEO_FAILURES):
    # 8 episodes ran; the rollout offered 3600 frames; the encoder took 0 or 1.
    for frames_accepted_irrelevant in (0, 1):
        code = vp.task_exit_code(episode_count=8, want_video=True,
                                 video_status=status)
        check(f"task_exit_code(8 episodes, want_video, {status}) == EXIT_OK",
              code, vp.EXIT_OK)

# The exact historical scenario: LGPL ffmpeg, "Unknown encoder 'libx264'".
status, err = vp.classify_video(
    want_video=True, video_ok=False, frames_offered=3600,
    writer_reason=vp.classify_ffmpeg_error("Unknown encoder 'libx264'"),
    writer_detail="ffmpeg exited early: Unknown encoder 'libx264'",
    camera="external_D455",
)
check("LGPL-ffmpeg scenario classifies as encoder_unavailable",
      status, vp.VIDEO_NO_ENCODER)
check("LGPL-ffmpeg scenario exits 0 with 8 episodes",
      vp.task_exit_code(8, True, status), vp.EXIT_OK)
check("LGPL-ffmpeg scenario is flagged as an environment failure",
      vp.is_environment_video_failure(status), True)
check("LGPL-ffmpeg scenario keeps a human-readable reason",
      bool(err), True)

# The one-frame accident that masked the bug must now be irrelevant.
check("frames_offered=3600 & encoder took 1 frame -> still exits 0",
      vp.task_exit_code(8, True, vp.VIDEO_NO_ENCODER), vp.EXIT_OK)

# ---------------------------------------------------------------------------
# 2. But a genuinely broken ROLLOUT must still be visible as a failure.
# ---------------------------------------------------------------------------
print("\n[2] a broken rollout must STILL fail")
status, err = vp.classify_video(want_video=True, video_ok=False,
                                frames_offered=0, camera="external_D455")
check("no frames offered -> no_frames_captured", status, vp.VIDEO_NO_FRAMES)
check("no frames offered -> EXIT_NO_FRAMES(4)",
      vp.task_exit_code(8, True, status), vp.EXIT_NO_FRAMES)
check("no_frames_captured is NOT an environment failure",
      vp.is_environment_video_failure(vp.VIDEO_NO_FRAMES), False)
check("the error message names the camera",
      "external_D455" in (err or ""), True)
check("zero episodes -> EXIT_NO_EPISODES(5)",
      vp.task_exit_code(0, True, vp.VIDEO_OK), vp.EXIT_NO_EPISODES)
check("zero episodes outranks a fine video",
      vp.task_exit_code(0, False, vp.VIDEO_DISABLED), vp.EXIT_NO_EPISODES)

# ---------------------------------------------------------------------------
# 3. "not requested" must be distinguishable from "requested and broken".
#    Both leave video=null, which is what made the failure invisible.
# ---------------------------------------------------------------------------
print("\n[3] disabled must be distinguishable from broken")
disabled, derr = vp.classify_video(want_video=False, video_ok=False,
                                   frames_offered=0)
check("--no-video -> disabled", disabled, vp.VIDEO_DISABLED)
check("--no-video has no error text", derr, None)
check("--no-video exits 0 even with 0 frames offered",
      vp.task_exit_code(8, False, disabled), vp.EXIT_OK)
check("disabled is NOT an environment failure",
      vp.is_environment_video_failure(vp.VIDEO_DISABLED), False)
check("disabled != encoder_unavailable", disabled == vp.VIDEO_NO_ENCODER, False)

# ---------------------------------------------------------------------------
# 4. Happy path and stderr classification.
# ---------------------------------------------------------------------------
print("\n[4] happy path and error classification")
ok, oerr = vp.classify_video(want_video=True, video_ok=True, frames_offered=3600)
check("video written -> written", ok, vp.VIDEO_OK)
check("video written has no error", oerr, None)
check("video written exits 0", vp.task_exit_code(8, True, ok), vp.EXIT_OK)

check("'Unknown encoder' -> encoder_unavailable",
      vp.classify_ffmpeg_error("Unknown encoder 'libx264'"), vp.VIDEO_NO_ENCODER)
check("'Encoder not found' -> encoder_unavailable",
      vp.classify_ffmpeg_error("Error opening output files: Encoder not found"),
      vp.VIDEO_NO_ENCODER)
check("case-insensitive match",
      vp.classify_ffmpeg_error("UNKNOWN ENCODER 'libx264'"), vp.VIDEO_NO_ENCODER)
check("other stderr -> encoder_failed",
      vp.classify_ffmpeg_error("Invalid argument"), vp.VIDEO_ENC_FAILED)
check("empty stderr -> encoder_failed",
      vp.classify_ffmpeg_error(""), vp.VIDEO_ENC_FAILED)
check("None stderr does not raise",
      vp.classify_ffmpeg_error(None), vp.VIDEO_ENC_FAILED)

# A writer that failed without setting a reason must still classify, not crash.
fallback, _ = vp.classify_video(want_video=True, video_ok=False,
                                frames_offered=10, writer_reason=None)
check("failed writer with no reason -> encoder_failed",
      fallback, vp.VIDEO_ENC_FAILED)
check("...and still exits 0", vp.task_exit_code(8, True, fallback), vp.EXIT_OK)

# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
if FAILS:
    print(f"{len(FAILS)} of {CHECKS[0]} checks FAILED:\n")
    for f in FAILS:
        print("  - " + f)
    sys.exit(1)
print(f"all {CHECKS[0]} checks passed")
print("GUARANTEE HELD: no environment-side video failure can fail a task that")
print("produced episodes; a rollout that captured nothing still fails loudly.")
sys.exit(0)
