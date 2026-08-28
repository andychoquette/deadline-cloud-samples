#!/usr/bin/env python3
"""What a video failure may and may not do to a rollout task.

Lives in its own module, with **no third-party imports at all** (no numpy, no
torch, no Isaac Lab), for one reason: it makes the safety property directly
unit-testable. `render_rollout.py` cannot be imported outside the container --
it launches Omniverse Kit at module scope -- so any policy expressed inline
there can only ever be reviewed by eye. That is exactly how the bug this module
exists to prevent survived a full 8-shard farm run.

See `test/test_video_failure_policy.py`.

The rule
--------
The summary JSON is the deliverable; the MP4 is an artifact. A shard that ran
its episodes and produced valid success data MUST report success even if no
video could be written. The only video-shaped condition that may fail a task is
the one that means the *rollout* is broken.

Concretely, `frames_offered` (what the rollout produced) is tracked separately
from `frames written` (what the encoder accepted), because those two diverge
precisely when the encoder is at fault. The original code branched on the
encoder's count and so treated a missing codec as a rollout failure:

    if want_video and frames == 0:
        return 4          # -> task FAILED -> Aggregate CANCELED by dependency

With an LGPL ffmpeg build (no libx264), that turns every shard of a fan-out into
a failure and destroys the run's headline number over a cosmetic artifact. It
escaped notice only because one frame happened to reach ffmpeg before it died,
leaving `frames == 1`. A one-frame margin is not a safety property.
"""

from __future__ import annotations

# --- video_status values ------------------------------------------------------
VIDEO_OK = "written"
VIDEO_DISABLED = "disabled"            # --no-video: not requested
VIDEO_NO_FRAMES = "no_frames_captured"  # the ROLLOUT produced nothing: a DEFECT
VIDEO_NOT_FOUND = "ffmpeg_not_found"    # no ffmpeg binary:      environment
VIDEO_NO_ENCODER = "encoder_unavailable"  # build lacks the codec: environment
VIDEO_ENC_FAILED = "encoder_failed"     # ffmpeg errored:        environment

#: Statuses that are the ENVIRONMENT's fault, never the rollout's. These must
#: never fail a task on their own.
ENVIRONMENT_VIDEO_FAILURES = frozenset(
    {VIDEO_NOT_FOUND, VIDEO_NO_ENCODER, VIDEO_ENC_FAILED}
)

# --- exit codes ---------------------------------------------------------------
EXIT_OK = 0
EXIT_NO_FRAMES = 4    # video requested, rollout captured nothing
EXIT_NO_EPISODES = 5  # no episodes ran at all


def classify_ffmpeg_error(stderr_text):
    """A codec absent from the build is an environment fact, not an encode error."""
    low = (stderr_text or "").lower()
    if "unknown encoder" in low or "encoder not found" in low:
        return VIDEO_NO_ENCODER
    return VIDEO_ENC_FAILED


def classify_video(want_video, video_ok, frames_offered,
                   writer_reason=None, writer_detail=None, camera=None):
    """Resolve why there is (or is not) a video, as one explicit value.

    Returns ``(video_status, video_error)``. Recorded in the summary JSON so a
    missing MP4 is *reported* rather than inferred from an absent file --
    inferring it is what let a broken encoder go unnoticed. ``video`` alone
    cannot distinguish "not requested" from "requested and broken": both are
    null.
    """
    if not want_video:
        return VIDEO_DISABLED, None
    if video_ok:
        return VIDEO_OK, None
    if frames_offered == 0:
        return VIDEO_NO_FRAMES, (
            "the rollout captured no frames at all; the requested camera "
            f"{camera!r} produced nothing"
        )
    # Frames existed, so the rollout worked and the writer did not.
    return (writer_reason or VIDEO_ENC_FAILED), writer_detail


def task_exit_code(episode_count, want_video, video_status):
    """The task's exit code, given what actually happened.

    Only two things fail: no episodes at all, and a video that was requested
    but got no frames out of the rollout. Every environment-side video failure
    returns success, because the episode data is intact and is the deliverable.
    """
    if not episode_count:
        return EXIT_NO_EPISODES
    if want_video and video_status == VIDEO_NO_FRAMES:
        return EXIT_NO_FRAMES
    return EXIT_OK


def is_environment_video_failure(video_status):
    return video_status in ENVIRONMENT_VIDEO_FAILURES
