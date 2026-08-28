#!/usr/bin/env python3
"""Record a LeRobot dataset of a SCRIPTED SO-101 vial-into-rack pick in Isaac Sim.

Isaac Lab counterpart of `mujoco_sim_to_policy/scripts/generate_dataset.py`, and
it keeps that script's contract: a scripted joint-space expert drives the arm,
every episode is VERIFIED by the environment's own success predicate, failed
attempts are DISCARDED and retried, and under-production exits non-zero so a
short dataset cannot masquerade as a complete one.

What is reused unmodified from the Sim-to-Real-SO-101-Workshop:
  * the environment (`Lerobot-So101-Teleop-Vials-To-Rack[-DR]-Eval`), including
    all five domain-randomization terms and the gripper contact sensors;
  * `LeRobotRecorder`, which writes a LeRobot v3.0 dataset natively (no HDF5
    intermediate) and is driven by two carb events -- STOP keeps the episode,
    CANCEL discards it;
  * `LeRobotSO101Interface`, whose `get_raw_actions_from_radians()` is the
    sim-radians -> servo-space inverse the dataset's `action` column needs;
  * `vial_placed_on_rack_termination()`, the success predicate (grasp history +
    verticality + rack-local bounds + a 25-frame confirmation window).

What this script replaces:
  * the teleop action source. The workshop's `lerobot_agent.py` reads a physical
    SO-101 leader arm (`robot_iface.robot.get_action()`) and gates recording on
    keyboard events acquired from `omni.appwindow` -- neither exists on a render
    farm. A scripted waypoint routine takes its place and never calls
    `init_device()` / `connect()`, so there is no hardware dependency.
  * the process lifetime. `lerobot_agent.py` ends in `while True:
    simulation_app.update()` with `simulation_app.close()` unreachable behind
    it; as-is that hangs a Deadline Cloud task forever. This script runs a
    bounded attempt loop, drains the recorder, closes the app, and returns a
    status code.

Why a joint-space waypoint routine and not Isaac Lab's `lift_cube_sm.py`:
  that state machine commands end-effector poses, i.e. it assumes a
  differential-IK action space. This environment uses `JointPositionActionCfg`,
  so the recorded `action` column is six absolute joint targets. Adopting an IK
  action space would change what `action` means, break parity with the MuJoCo
  sample, and break the real-SO-101 deployment story (the follower consumes
  joint positions).

------------------------------------------------------------------------------
CALIBRATION REQUIRED -- READ THIS BEFORE YOU EXPECT EPISODES
------------------------------------------------------------------------------
The waypoint routine needs a handful of geometric constants that can only be
measured against the actual SO-ARM101 USD on a GPU:

    --grasp-z-offset        gripper-body origin -> pinch point, in metres
    --gripper-pitch-offset  correction to the home gripper pitch to point down
    --reorient-offset       gripper pitch change that stands a lying vial up
    --wrist-roll            forearm roll that aligns the jaw across the vial
    --jaw-open / --jaw-closed

Run `--probe` FIRST. It boots the env, resets once, and prints joint names and
limits, the home pose, the end-effector frame pose, every body position, the
vial and rack poses, the rack slot poses, and a Jacobian self-test that commands
a known displacement and reports the measured one. Everything the constants
above need is in that output.

------------------------------------------------------------------------------
MEASURED GEOMETRY -- do not re-derive this, and do not trust the obvious reading
------------------------------------------------------------------------------
All of the following came off the collision meshes in the shipped USD assets and
was confirmed against a farm run. It is written down here because every one of
these facts is counter-intuitive and cost real debugging time.

GRIPPER. The `gripper` body origin IS the ee_frame. Both fingers -- the static
one (`wrist_roll_follower_so101_v1`, part of the `gripper` body) and the moving
one (`moving_jaw_so101_v1`, the `jaw` body) -- run out to gripper-local
z = -0.105, and their faces are ~33 mm apart over z in [-0.05, -0.105] at
Jaw ~ 0.22, i.e. a near-parallel grip. The pinch centre is therefore about
(0.005, 0, -0.075) in the gripper frame: **75 mm from the ee_frame**, which is
why --grasp-z-offset defaults to 0.075.
  THE TRAP: the `jaw` body position that --probe prints is only ~5 mm below and
  ~34 mm ahead of the ee_frame, and it is tempting to read --grasp-z-offset off
  that. It is the jaw HINGE (the Jaw joint's localPos0), not the pinch. Using it
  gives a value 15x too small.

JAW. Jaw=0.6 opens the faces to ~55 mm, comfortably clear of the 34 mm vial.
Jaw=-0.1 is effectively hard shut: the vial forces the joint to ~0.22 rad, so
0.32 rad of position error against stiffness 4 gives ~25 N at the pads, far
above the 2 N contact-sensor threshold the success predicate uses.

VIAL (`Vial_opaque.usda`). Length 116 mm, collider approximation `sdf` (so it is
mesh-faithful -- NOT a capsule, and the two ends genuinely differ). THE VIAL IS
NOT SYMMETRIC, and this drives everything:
  * the TUBE is r=0.01565, and the CAP at the top (local z 0.085..0.0977) is
    r=0.01703. Quoting "vial radius 0.017" is the cap, not the body.
  * the BOTTOM (local z -0.017..0) is a DOME/taper: r falls 0.01525 -> 0.0036 ->
    0.002 over the last 17 mm. The bottom is a rounded point, not a foot.
  * the TOP is the flat face of the cap at local z ~ +0.0997.
  * `root_pos_w` is the origin at local z=0, i.e. the top of the dome / bottom of
    the tube. It is NOT the centre of mass, and it is 17 mm above the lowest
    point. So `target["pos"]` names the DOMED end, and a z-only grasp offset can
    only ever grasp the vial near that end.
  * local +z runs from the dome toward the flat cap. So up_z > 0 means DOME DOWN,
    CAP UP.

RACK (`Vial_rack_simple.usda`). Four things worth knowing:
  * The `top_*` slot xforms are at rack-local z=0.1 (world 0.16), which is 27 mm
    ABOVE the rack's physical top surface (rack-local z=0.073, world 0.133).
    They are markers, not the slot entry.
  * The bore admits the vial DOME-FIRST and it goes deep. Bore r=0.0171 from
    rack-local z=0.014 to 0.051 against a TUBE radius of 0.01565 -- 1.45 mm of
    clearance -- above an internal shelf where r drops to 0.0132 at z~0.0135.
    A dome-down vial descends until the dome jams on that shelf, which puts the
    vial ROOT at rack-local ~0.0166, i.e. world z ~ 0.077. The cap (r=0.01703)
    never enters and never needs to; it ends up ~28 mm clear of the rack top.
    Above the bore a funnel flares to r=0.022 at the rim, giving ~6 mm of
    lateral capture to guide the vial in.
  * That seated pose is genuinely stable: 1.45 mm of clearance over a 37 mm bore
    limits tilt to ~4.5 deg, so up_z stays above 0.99 and the 25-frame
    confirmation window is easy to hold.
  * The predicate wants the vial vertical (|up_z| > 0.7), its root inside the
    rack's 12x12 cm footprint, and its root below rack-local z=0.1 (world 0.16).
    Seated in the bore, root ~0.077, all three hold comfortably.
  * DO NOT aim to stand the vial on the rack's flat top instead. It is
    geometrically inside the bound (root would be ~0.15) but physically absurd:
    dome-down means balancing a 116 mm body on a ~4 mm rounded tip, which cannot
    survive 25 confirming frames. The bore is load-bearing, not decorative.
  * Corollary: an upside-down (cap-down) vial does NOT count, even though the
    predicate takes abs(up_z). Cap-down is the STABLE resting pose on a flat
    surface, but it puts the root ~0.0997 ABOVE the contact point, so the root
    lands at 0.2325 on the rack top -- outside the z bound. Cap-down also cannot
    enter the bore, because the cap is 0.07 mm wider than it. So the predicate
    accepts exactly one final pose: DOME-DOWN, SEATED IN A BORE.

The defaults here are structurally correct but NOT calibrated. Uncalibrated, the
routine will fail every attempt and this script will exit non-zero rather than
write a bad dataset. That failure mode is deliberate -- and note that it only
works because the status checks at the end of main() run BEFORE Kit is closed;
see the comment there before moving them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

# Joint order of the SO-ARM101 USD articulation. Also the order the
# `joint_positions` action term resolves to, and the order the LeRobot `action`
# and `observation.state` columns are written in.
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
ROT, PITCH, ELBOW, WPITCH, WROLL, JAW = range(6)

# The three joints the position servo solves for. Wrist_Pitch is slaved to them
# (see `slaved_pose`) and Wrist_Roll follows the base yaw.
SERVO_JOINTS = (ROT, PITCH, ELBOW)

UNIT_Z = (0.0, 0.0, 1.0)


def parse_args():
    p = argparse.ArgumentParser(description="Scripted SO-101 vial-to-rack datagen.")
    # --- what to record ---
    p.add_argument("--task", default="Lerobot-So101-Teleop-Vials-To-Rack-DR-Eval",
                   help="Registered Isaac Lab gym id. Must be an -Eval variant: the "
                        "non-Eval variants set terminations=None, so there is no "
                        "success signal to verify episodes against.")
    p.add_argument("--episodes", type=int, default=50,
                   help="Target count of VERIFIED episodes.")
    p.add_argument("--max-attempt-factor", type=int, default=6,
                   help="Give up after episodes * this many attempts.")
    p.add_argument("--repo-id", default="local/so101_isaac_vials")
    p.add_argument("--dataset-root", default="")
    p.add_argument("--instruction", default="pick up the vial and place it in the rack",
                   help="Recorded as the language annotation on every frame.")
    p.add_argument("--fps", type=int, default=30,
                   help="Dataset frame rate. One env.step() is sim.dt*decimation = "
                        "1/60 s, so frames are pushed every other step to make this "
                        "rate true rather than nominal.")
    p.add_argument("--episode-length-s", type=float, default=15.0,
                   help="Overrides the env cfg. The -Eval variants ship 7.5 s, which "
                        "is too short for reach + grasp + reorient + insert + the "
                        "25-step success confirmation.")
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "output"))

    # --- geometry that needs calibrating (see the module docstring) ---
    p.add_argument("--grasp-z-offset", type=float, default=0.075,
                   help="Metres added to the vial height to get the servo target "
                        "for the gripper frame origin, i.e. the vertical gap "
                        "between the ee_frame and the PINCH CENTRE BETWEEN THE "
                        "FINGERTIPS. Default measured off the collision meshes in "
                        "SO-ARM101-USD.usd -- see the module docstring. Do NOT "
                        "read this off the `jaw` body position that --probe "
                        "prints: that body's origin is the jaw HINGE, only ~5 mm "
                        "below the ee_frame, and using it gives a value 15x too "
                        "small and a grasp that closes on thin air.")
    p.add_argument("--place-z-offset", type=float, default=None,
                   help="Metres added to the rack slot height to get the servo "
                        "target for the gripper frame origin during traverse and "
                        "insert. Default: the same value as --grasp-z-offset. "
                        "This exists because the ee_frame -> pinch offset applies "
                        "just as much when PLACING as when grasping: without it "
                        "the insert waypoint drives the vial 48 mm into the solid "
                        "top of the rack (rack top is at world z=0.133, the slot "
                        "marker at 0.16, and the pinch sits 75 mm below the "
                        "ee_frame), the fingertips reach z=0.055, and the servo "
                        "stalls against the geometry instead of arriving.")
    p.add_argument("--gripper-pitch-offset", type=float, default=0.0,
                   help="Radians added to the home gripper pitch sum. 0 keeps "
                        "whatever orientation the home pose gives.")
    p.add_argument("--reorient-offset", type=float, default=1.5708,
                   help="Radians of gripper pitch change applied during transfer to "
                        "stand a lying vial upright. The sign is geometry-dependent; "
                        "--probe prints the gripper quaternion so you can pick it.")
    p.add_argument("--wrist-roll", type=float, default=None,
                   help="Absolute Wrist_Roll for the grasp. Default: the home value.")
    p.add_argument("--no-wrist-roll-track-base", dest="wrist_roll_track_base",
                   action="store_false", default=True,
                   help="By default Wrist_Roll counter-rotates with Rotation so the "
                        "jaw keeps a world-fixed yaw while the base swings. When the "
                        "gripper points down those two axes are collinear, which is "
                        "exactly the grasp case. Pass this to hold Wrist_Roll fixed.")
    p.add_argument("--jaw-open", type=float, default=0.6)
    p.add_argument("--jaw-closed", type=float, default=-0.1)
    p.add_argument("--hover-height", type=float, default=0.09,
                   help="Metres above the grasp point for the pre-grasp hover.")
    p.add_argument("--lift-height", type=float, default=0.14,
                   help="Metres above the grasp point to lift to after closing.")
    p.add_argument("--insert-clearance", type=float, default=0.07,
                   help="Metres above the target slot to line up before descending.")

    # --- servo tuning ---
    p.add_argument("--servo-tol", type=float, default=0.006,
                   help="Position tolerance (m) that ends a servo segment.")
    p.add_argument("--servo-max-steps", type=int, default=140,
                   help="Control-step budget per servo segment.")
    p.add_argument("--servo-step-m", type=float, default=0.004,
                   help="Max commanded end-effector displacement per control step.")
    p.add_argument("--servo-damping", type=float, default=1e-4,
                   help="Damped-least-squares lambda^2. MUST be small relative to "
                        "the SQUARED singular values of the measured Jacobian, "
                        "which on this arm are 3.8e-3 .. 6.1e-2 m^2/rad^2. The "
                        "old default of 0.02 was larger than the smallest of "
                        "those, and attenuated the three singular directions to "
                        "0.75 / 0.47 / 0.16 of the commanded step -- that is the "
                        "'2x undershoot' --probe's self-test reports. It is not "
                        "harmful on its own (the servo still converges, just "
                        "slower), but it becomes fatal in combination with two "
                        "other defaults: --servo-step-m clamps the error to 4 mm "
                        "BEFORE the solve, so the approach is rate-limited to "
                        "~1.4 mm per iteration rather than closing a fixed "
                        "fraction of the remaining error, and --servo-max-steps "
                        "then caps the segment at 140 iterations. Measured on a "
                        "farm worker: 'hover' burned all 140 iterations and "
                        "stopped 12.8 mm short of the 6 mm tolerance, on a target "
                        "that was fully reachable. At 1e-4 the gains are "
                        "0.998 / 0.994 / 0.975 and the whole plan completes in "
                        "~430 of the 900 available steps.")
    p.add_argument("--fd-delta", type=float, default=0.03,
                   help="Joint perturbation (rad) for the finite-difference Jacobian.")
    p.add_argument("--fd-settle", type=int, default=6,
                   help="Control steps to settle after each Jacobian perturbation.")
    p.add_argument("--jacobian-refresh", type=int, default=35,
                   help="Re-measure the Jacobian every N control steps -- but only "
                        "while the jaw is open. Perturbing the arm mid-carry can "
                        "shake the vial loose, so a carrying segment reuses the last "
                        "Jacobian and relies on damping plus step clamping.")

    p.add_argument("--probe", action="store_true",
                   help="Print calibration data and exit without recording.")

    # --- diagnostics -------------------------------------------------------
    p.add_argument("--debug-video", default="",
                   help="Write an MP4 of EVERY attempt, including the ones that are "
                        "discarded, to this path. Empty (default) writes nothing.\n"
                        "This exists because datagen's whole integrity model is to "
                        "throw failed attempts away -- which also throws away the "
                        "only footage that could explain WHY they failed. Until this "
                        "option existed the sole diagnostic for a failed run was one "
                        "line per attempt ('discard attempt 1 (70 frames): hover did "
                        "not complete'), with no way to see whether the gripper was "
                        "in the wrong place, the wrong attitude, or holding nothing. "
                        "The recorder's own videos are no help: CANCEL deletes them.\n"
                        "Every control step is captured, including the Jacobian "
                        "finite-difference probes, so the video is a complete record "
                        "of what the arm did rather than only the frames the dataset "
                        "kept. Encoding is non-fatal: a broken encoder logs loudly "
                        "and the run continues.")
    p.add_argument("--debug-video-camera", default="external_D455",
                   help="Which env camera feeds --debug-video. external_D455 is the "
                        "third-person view (right for anything about the arm's "
                        "relationship to the rack or the mat); ego is the wrist "
                        "camera (right for what the fingers are doing).")

    # --- scene overrides ---------------------------------------------------
    # Mutating the parsed env cfg rather than editing the task config: the task
    # config lives inside a 28.9 GB container image, so anything that required
    # rebuilding it would cost an image build and push per experiment.
    # parse_env_cfg() hands back a plain configclass, so the same change costs
    # three lines here and takes effect before gym.make().
    p.add_argument("--vial-spawn", default="lying", choices=("lying", "capdown"),
                   help="Vial spawn attitude. 'lying' (default) is the shipped "
                        "behaviour: rot = euler(0,90,0), i.e. horizontal on the mat. "
                        "'capdown' stands each vial on its FLAT CAP FACE, which is "
                        "its only stable upright pose -- the other end is a rounded "
                        "dome that cannot balance.\n"
                        "'capdown' also has to move the spawn height. The shipped "
                        "`fixed_vial_z` of 0.05 is the resting root height of a LYING "
                        "vial (root is on the cylinder axis, 15.65 mm above the mat at "
                        "0.033). Standing cap-down the root is 99.7 mm above the "
                        "contact face instead, so 0.05 would spawn the vial ~50 mm "
                        "INSIDE the mat and physics would eject it. See "
                        "--capdown-spawn-z.")
    p.add_argument("--capdown-spawn-z", type=float, default=0.136,
                   help="Root height (m) for --vial-spawn capdown. A resting cap-down "
                        "vial has its root at mat 0.033 + 0.0997 = 0.1327, so this "
                        "default drops it under 1 mm onto the mat.")
    p.add_argument("--deterministic-scene", action="store_true",
                   help="Zero the reset event's vial and rack pose ranges and disable "
                        "the 0.33 chance of pre-placing a vial in a rack slot. Makes "
                        "the scene reproducible, which is what you want when the point "
                        "of the run is to demonstrate a geometric property rather than "
                        "to measure robustness to placement.")

    # --- alternative waypoint plans ---------------------------------------
    p.add_argument("--plan", default="reorient", choices=("reorient", "flip"),
                   help="Which scripted waypoint routine to run.\n"
                        "'reorient' (default) is the shipped plan: grasp a lying vial, "
                        "lift, rotate the gripper to stand it up, insert.\n"
                        "'flip' grasps an already-STANDING cap-down vial from directly "
                        "above, rotates ~180 deg about the wrist-pitch axis so the dome "
                        "faces down, moves over an empty bore and tries to lower it in. "
                        "Use it with --vial-spawn capdown. Stages are NON-FATAL in this "
                        "plan (see run_flip_attempt) because the interesting output is "
                        "the video, and a stage that aborts the attempt early is a "
                        "stage whose obstruction never gets filmed.")
    p.add_argument("--flip-preset", default="",
                   help="Comma-separated Rotation,Pitch,Elbow to ramp to before the "
                        "grasp, as a starting branch for the servo. Empty means skip.\n"
                        "Why this is needed: gripper elevation is exactly -(Pitch + "
                        "Elbow + Wrist_Pitch), and Wrist_Pitch is limited to +/-1.658, "
                        "so pointing the gripper straight DOWN (which needs the sum = "
                        "+1.5708) requires Pitch+Elbow >= -0.087. At the home pose "
                        "Pitch+Elbow = -0.685, so Wrist_Pitch saturates and the gripper "
                        "sits at -55.7 deg however large --gripper-pitch-offset is. The "
                        "position servo optimises POSITION ONLY, so it has no reason to "
                        "leave that branch. Starting it on the correct side of the "
                        "Wrist_Pitch limit is what makes the grasp attitude reachable.")
    p.add_argument("--flip-hold-steps", type=int, default=60,
                   help="Control steps to keep pressing at the final insert target "
                        "after the servo has stopped making progress. This is what "
                        "leaves the obstruction on screen long enough to see, instead "
                        "of one ambiguous frame.")

    # Isaac Lab's own launcher flags (--device, --headless, --enable_cameras,
    # --experience, ...). Added to this parser rather than a separate namespace
    # so AppLauncher gets a fully-populated argument set, which is how the
    # workshop's scripts do it.
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    # Placing needs the same ee_frame -> pinch correction as grasping, so default
    # it to the grasp value rather than to zero. Resolved here, once, so every
    # reader of args.place_z_offset sees a float.
    if args.place_z_offset is None:
        args.place_z_offset = args.grasp_z_offset
    return args


ARGS = parse_args()

# Cameras are mandatory: the dataset's image columns come from the env's
# `visual` observation group, and Isaac Lab only builds offscreen render
# products when enable_cameras is set (it then selects the
# isaaclab.python.headless.rendering.kit experience). `lerobot_agent.py` forces
# this the same way.
ARGS.enable_cameras = True
ARGS.headless = True

app_launcher = AppLauncher(ARGS)
simulation_app = app_launcher.app

# --- everything below needs Kit running -------------------------------------
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import omni.kit.app  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaacsim.core.prims import XFormPrim  # noqa: E402

import sim_to_real_so101.tasks  # noqa: F401,E402
from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface  # noqa: E402
from sim_to_real_so101.utils.lerobot_recorder import LeRobotRecorder  # noqa: E402

# The debug video shares render_rollout.py's encoder rather than carrying a
# second one. That matters specifically because of the ffmpeg-encoder probe: the
# image's ffmpeg is an LGPL build with no libx264, and a hardcoded encoder fails
# at runtime while the task still reports success. See mp4_writer.pick_encoder.
#
# sys.path is set explicitly rather than relying on sys.path[0]: the container
# runs this through a `python` shim that forwards to Isaac Sim's python.sh, and a
# shim that exec's differently need not leave the script directory first on the
# path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mp4_writer import Mp4Writer  # noqa: E402


def log(msg):
    print(f"[datagen] {msg}", flush=True)


def mark(out_dir, name):
    """Filesystem progress marker, uploaded with the job outputs.

    Cheap insurance: if Kit wedges or the task is killed, the markers say
    exactly how far the script got, which stdout alone often does not.
    """
    try:
        d = os.path.join(out_dir, "logs")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"_marker_datagen_{name}"), "w") as fh:
            fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + f" {name}\n")
    except OSError:
        pass


def joint_pos_limits(robot, joint_ids):
    """Per-joint (lower, upper) position limits, across Isaac Lab versions.

    The attribute has been spelled `joint_pos_limits` and `soft_joint_pos_limits`
    in different releases. Fail loudly rather than fall back to an unbounded
    range, because unclamped joint targets are how a scripted expert ends up
    driving the arm through the table.
    """
    for attr in ("joint_pos_limits", "soft_joint_pos_limits"):
        limits = getattr(robot.data, attr, None)
        if limits is not None:
            return limits[0, joint_ids].clone()
    raise RuntimeError(
        "Articulation data exposes neither `joint_pos_limits` nor "
        "`soft_joint_pos_limits`; cannot clamp joint targets safely."
    )


def to_frame(pos_w, frame_pos_w, frame_quat_w):
    """Express a world point in a frame, given that frame's world pose.

    Written out rather than using `subtract_frame_transforms` because the
    workshop's own `terms.py` uses exactly this idiom
    (`quat_apply(quat_inv(q), p - t)`) and it has no optional-argument
    behaviour that varies across releases.
    """
    q_inv = math_utils.quat_inv(frame_quat_w.unsqueeze(0))
    return math_utils.quat_apply(q_inv, (pos_w - frame_pos_w).unsqueeze(0))[0]


class ScriptedExpert:
    """Joint-space waypoint expert with a finite-difference position servo.

    The arm is driven by absolute joint targets, so the recorded `action`
    column keeps the same meaning as in the MuJoCo sample. Positions are
    reached by a damped-least-squares servo over three joints (Rotation, Pitch,
    Elbow), with Wrist_Pitch slaved to hold gripper pitch and Wrist_Roll
    counter-rotating with the base.

    The Jacobian is measured by finite differences rather than read from
    `root_physx_view.get_jacobians()`. That is a deliberate trade. The analytic
    Jacobian is free and exact, but it is expressed in the WORLD frame while
    this robot's root is rotated 90 degrees about Z, so it needs a rotation into
    the root frame; the accessor moved between Isaac Lab 2.x and 3.x; and the
    fixed-base body-index offset is easy to get silently wrong -- a wrong index
    yields a plausible matrix and a servo that drifts. Finite differences cost
    roughly 20 sim steps per refresh and are correct by construction, because
    the displacement is measured in the same frame the target is expressed in.
    """

    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.device = env.unwrapped.device
        self.robot = env.unwrapped.scene["robot"]

        # Assert the action vector really is in JOINT_NAMES order. The action
        # term resolves joint names through find_joints(), which returns
        # articulation order, not the order written in the cfg. If the USD ever
        # reorders its joints, every recorded action would be silently
        # permuted, so fail loudly here instead.
        idx, names = self.robot.find_joints(JOINT_NAMES, preserve_order=False)
        if list(names) != JOINT_NAMES:
            raise RuntimeError(
                f"SO-101 articulation joint order is {list(names)}, expected "
                f"{JOINT_NAMES}. The action vector and the recorded action/state "
                "columns would be permuted; refusing to record."
            )
        self.joint_ids = list(idx)
        self.limits = joint_pos_limits(self.robot, self.joint_ids)
        self.home = self.robot.data.default_joint_pos[0, self.joint_ids].clone()

        # Holding (Pitch + Elbow + Wrist_Pitch) constant holds the gripper's
        # pitch constant, because those three joints rotate about parallel
        # axes. This is what lets the servo work without knowing link lengths
        # or the gripper's zero orientation -- only a delta from home.
        self.pitch_sum_home = float(
            self.home[PITCH] + self.home[ELBOW] + self.home[WPITCH]
        )
        self.wrist_roll_ref = (
            float(self.home[WROLL]) if args.wrist_roll is None else args.wrist_roll
        )
        self.rot_home = float(self.home[ROT])

        self.q = self.home.clone()          # last commanded joint target
        self.jacobian = None                # 3x3, d(ee_pos_base) / d(servo joints)
        self.steps_since_jacobian = 10 ** 9
        self.obs = None
        self.terminated = False
        self.truncated = False
        self.control_steps_recorded = 0
        self._push = None
        # Debug video state. Deliberately NOT reset per attempt: one MP4 spans
        # every attempt in the run, so a multi-attempt failure reads as one
        # continuous recording instead of N files to correlate by hand.
        self._video = None
        self._video_key = None
        self.video_steps = 0
        self.video_frames_offered = 0

    # -- plumbing ------------------------------------------------------------

    def bind_recorder(self, push_fn):
        self._push = push_fn

    def bind_video(self, writer, camera):
        """Record every control step to `writer`, regardless of episode outcome."""
        self._video = writer
        self._video_key = f"rgb_{camera}"

    def _capture(self):
        """Push one frame to the debug video.

        Captures on the same every-other-step cadence as the dataset so the MP4
        plays at --fps in real time, but is deliberately INDEPENDENT of the
        `record` flag: Jacobian probes and discarded attempts are exactly the
        footage this option exists to keep.
        """
        if self._video is None or self.obs is None:
            return
        self.video_steps += 1
        if self.video_steps % 2:
            return
        visual = self.obs.get("visual") or {}
        img = visual.get(self._video_key)
        if img is None:
            return
        frame = img[0].detach().to("cpu").numpy().astype("uint8")[..., :3]
        self.video_frames_offered += 1
        self._video.add(frame)

    def reset(self):
        self.obs, _ = self.env.reset()
        self.q = self.home.clone()
        self.jacobian = None
        self.steps_since_jacobian = 10 ** 9
        self.terminated = False
        self.truncated = False
        self.control_steps_recorded = 0

    def slaved_pose(self, q_servo, jaw, pitch_offset):
        """Expand three servo joints into a full six-joint target."""
        q = self.q.clone()
        q[ROT], q[PITCH], q[ELBOW] = q_servo[0], q_servo[1], q_servo[2]
        # Keep gripper pitch at (home + offset) regardless of Pitch/Elbow.
        q[WPITCH] = (self.pitch_sum_home + pitch_offset) - q[PITCH] - q[ELBOW]
        q[WROLL] = self.wrist_roll_ref
        if self.args.wrist_roll_track_base:
            # With the gripper pointing down, Wrist_Roll's axis is vertical and
            # so is Rotation's, so counter-rotating keeps the jaw's world yaw
            # fixed as the base swings to aim at the vial.
            q[WROLL] = q[WROLL] - (q[ROT] - self.rot_home)
        q[JAW] = jaw
        return torch.clamp(q, self.limits[:, 0], self.limits[:, 1])

    def step(self, q_full, record):
        """Apply one control step. Returns False once the episode has ended."""
        self.obs, _, terminated, truncated, _ = self.env.step(q_full.unsqueeze(0))
        self.q = q_full
        self.terminated = bool(terminated.any().item())
        self.truncated = bool(truncated.any().item())
        self._capture()
        # One env.step() advances sim.dt * decimation = 1/60 s while the
        # recorder declares 30 fps, so push every other step. The workshop's
        # teleop loop pushes every step, which labels 60 Hz data as 30 fps.
        if record and self._push is not None:
            self.control_steps_recorded += 1
            if self.control_steps_recorded % 2 == 0:
                self._push(self.obs, self.q)
        self.steps_since_jacobian += 1
        return not (self.terminated or self.truncated)

    # -- state readers -------------------------------------------------------

    def ee_pos_b(self):
        """End-effector position in the robot ROOT frame (metres)."""
        return self.obs["policy"]["ee_frame_state"][0, 0:3].clone()

    def ee_quat_b(self):
        return self.obs["policy"]["ee_frame_state"][0, 3:7].clone()

    def to_base(self, pos_w):
        d = self.robot.data
        return to_frame(pos_w, d.root_pos_w[0], d.root_quat_w[0])

    # -- servo ---------------------------------------------------------------

    def _q_servo(self):
        return torch.stack([self.q[ROT], self.q[PITCH], self.q[ELBOW]])

    def measure_jacobian(self, jaw, pitch_offset):
        """Finite-difference d(ee_pos_base)/d(servo joint), 3x3."""
        base_q = self._q_servo()
        # Settle first so the reference reading is not taken mid-transient.
        for _ in range(self.args.fd_settle):
            if not self.step(self.slaved_pose(base_q, jaw, pitch_offset), record=False):
                return False
        ref = self.ee_pos_b()

        cols = []
        for k in range(3):
            lo, hi = self.limits[SERVO_JOINTS[k], 0], self.limits[SERVO_JOINTS[k], 1]
            delta = self.args.fd_delta
            # Perturb away from whichever limit is nearer, so the probe is not
            # silently clamped to zero displacement.
            if base_q[k] + delta > hi:
                delta = -delta
            probe = base_q.clone()
            probe[k] = torch.clamp(probe[k] + delta, lo, hi)
            applied = float(probe[k] - base_q[k])
            for _ in range(self.args.fd_settle):
                if not self.step(self.slaved_pose(probe, jaw, pitch_offset), False):
                    return False
            if abs(applied) < 1e-6:
                log(f"WARNING: {JOINT_NAMES[SERVO_JOINTS[k]]} is pinned at a limit; "
                    f"Jacobian column {k} is zero.")
                cols.append(torch.zeros(3, device=self.device))
            else:
                cols.append((self.ee_pos_b() - ref) / applied)
            # Return to the reference pose before probing the next axis.
            for _ in range(self.args.fd_settle):
                if not self.step(self.slaved_pose(base_q, jaw, pitch_offset), False):
                    return False

        self.jacobian = torch.stack(cols, dim=1)  # 3x3, columns are joints
        self.steps_since_jacobian = 0
        return True

    def _dls(self, err):
        jt = self.jacobian.transpose(0, 1)
        reg = self.args.servo_damping * torch.eye(3, device=self.device)
        return jt @ torch.linalg.solve(self.jacobian @ jt + reg, err)

    def servo_to(self, target_b, jaw, pitch_offset, record, label=""):
        """Drive the end-effector to `target_b` (root frame). True on arrival."""
        a = self.args
        may_probe = jaw >= a.jaw_open - 1e-6  # only perturb when not carrying
        for _ in range(a.servo_max_steps):
            if self.jacobian is None or (
                may_probe and self.steps_since_jacobian >= a.jacobian_refresh
            ):
                if not self.measure_jacobian(jaw, pitch_offset):
                    return False
            err = target_b - self.ee_pos_b()
            dist = float(torch.linalg.vector_norm(err))
            if dist < a.servo_tol:
                return True
            if dist > a.servo_step_m:
                err = err * (a.servo_step_m / dist)
            try:
                dq = self._dls(err)
            except RuntimeError:
                log(f"servo[{label}]: singular Jacobian, re-measuring")
                self.jacobian = None
                continue
            if not self.step(
                self.slaved_pose(self._q_servo() + dq, jaw, pitch_offset), record
            ):
                return False
        return False

    def hold(self, jaw, pitch_offset, steps, record):
        q_servo = self._q_servo()
        for _ in range(steps):
            if not self.step(self.slaved_pose(q_servo, jaw, pitch_offset), record):
                return False
        return True

    def ramp_jaw(self, start, end, segments, pitch_offset, record):
        """Move the jaw over several recorded frames, not in one jump.

        Same reason as the MuJoCo sample: a single-frame step in the jaw channel
        is exactly what an open-loop ACT chunk mis-times, closing late or on
        air. A ramp makes the close timing learnable.
        """
        q_servo = self._q_servo()
        for k in range(1, segments + 1):
            jaw = start + (end - start) * (k / segments)
            for _ in range(3):
                if not self.step(self.slaved_pose(q_servo, jaw, pitch_offset), record):
                    return False
        return True

    def ramp_servo_joints(self, target, jaw, pitch_offset, steps, record):
        """Interpolate the three servo joints straight to `target`, no IK.

        Used to put the arm on a chosen kinematic branch before handing over to
        the position servo. The servo cannot do this itself: it optimises
        POSITION only, so when two branches reach the same point it has no
        preference between them -- and on this arm the two branches differ in
        whether Wrist_Pitch is saturated, i.e. in the gripper's ATTITUDE. See
        --flip-preset.
        """
        start = self._q_servo()
        tgt = torch.as_tensor(target, device=self.device, dtype=start.dtype)
        for k in range(1, steps + 1):
            q_servo = start + (tgt - start) * (k / steps)
            if not self.step(self.slaved_pose(q_servo, jaw, pitch_offset), record):
                return False
        return True

    def hold_ramp_pitch(self, start, end, steps, jaw, record):
        """Sweep the gripper pitch offset from `start` to `end`, holding position.

        The shipped `reorient` stage jumps the offset in one step and holds. That
        is fine for a 90 deg change of a lightly-held vial, but a ~180 deg flip
        commanded as a step makes Wrist_Pitch chase a target 3 rad away and the
        vial leaves the jaws. Ramping keeps the commanded change per control step
        small.

        Position is NOT re-servoed here: the servo would perturb the arm mid-carry
        and shake the vial loose (the same reason jacobian_refresh is gated on the
        jaw being open). The pinch drifts during the flip as a result, which the
        following `traverse` stage corrects.
        """
        q_servo = self._q_servo()
        for k in range(1, steps + 1):
            off = start + (end - start) * (k / steps)
            if not self.step(self.slaved_pose(q_servo, jaw, off), record):
                return False
        return True

    def gripper_elevation_deg(self):
        """Elevation of the gripper's approach axis (-z of the gripper frame).

        -90 deg is straight down, +90 straight up. Logged per stage in the flip
        plan because attitude, not position, is what the Wrist_Pitch limit
        silently takes away -- and a stage can arrive at its target position
        with completely the wrong attitude.
        """
        q = self.ee_quat_b()
        m = math_utils.matrix_from_quat(q.unsqueeze(0))[0]
        az = -float(m[2, 2])
        return math.degrees(math.asin(max(-1.0, min(1.0, az))))


# -- scene queries -----------------------------------------------------------

def vial_states(env, vial_names):
    out = []
    for name in vial_names:
        data = env.unwrapped.scene[name].data
        pos = data.root_pos_w[0].clone()
        quat = data.root_quat_w[0].clone()
        up = math_utils.quat_apply(
            quat.unsqueeze(0), torch.tensor([UNIT_Z], device=pos.device)
        )[0]
        out.append({"name": name, "pos": pos, "quat": quat, "up": up})
    return out


def rack_slot_positions(env, rack_name):
    """World positions of the rack's slots, recomputed from the live rack pose.

    Mirrors `reset_vials_rack`: the slot xforms are read as LOCAL poses and
    composed with the rack's current root pose, because `write_root_pose_to_sim`
    does not refresh USD world transforms until the next `sim.step()`.
    """
    rack = env.unwrapped.scene[rack_name]
    view = XFormPrim(prim_paths_expr=f"{rack.cfg.prim_path}/Body1/Mesh/top_*")
    local_pos, local_quat = view.get_local_poses()
    device = rack.data.root_pos_w.device
    local_pos = torch.as_tensor(local_pos, device=device, dtype=torch.float32)
    local_quat = torch.as_tensor(local_quat, device=device, dtype=torch.float32)
    n = local_pos.shape[0]
    world_pos, _ = math_utils.combine_frame_transforms(
        rack.data.root_pos_w[0:1].repeat(n, 1),
        rack.data.root_quat_w[0:1].repeat(n, 1),
        local_pos,
        local_quat,
    )
    return world_pos


def pick_target_vial(vials):
    """The lowest vial is the one lying on the mat.

    `reset_vials_rack` pre-places one vial in a rack slot with probability 0.33;
    a slotted vial sits well above the mat, so minimum z picks a loose one.
    """
    return min(vials, key=lambda v: float(v["pos"][2]))


def pick_empty_slot(slot_pos_w, vials, min_clear=0.03):
    """Nearest slot with no vial in it, measured in the XY plane."""
    best, best_d = None, None
    for i in range(slot_pos_w.shape[0]):
        s = slot_pos_w[i]
        occupied = any(
            float(torch.linalg.vector_norm((v["pos"] - s)[0:2])) < min_clear
            and float(v["pos"][2]) > float(s[2]) - 0.04
            for v in vials
        )
        if occupied:
            continue
        d = float(torch.linalg.vector_norm(s[0:2]))
        if best_d is None or d < best_d:
            best, best_d = s.clone(), d
    return best


def _event_terms(env_cfg):
    """Every event term on the cfg, without assuming how configclass stores them.

    dir() rather than vars(): isaaclab's @configclass has changed between a plain
    dataclass and a slotted one across releases, and vars() is empty on the
    slotted form. A silently-empty iteration here would skip the fixed_vial_z fix
    and spawn the vials inside the mat.
    """
    out = []
    for name in dir(env_cfg.events):
        if name.startswith("_"):
            continue
        term = getattr(env_cfg.events, name, None)
        if getattr(term, "params", None) is not None:
            out.append((name, term))
    return out


def apply_scene_overrides(env_cfg, args):
    """Mutate the parsed env cfg in place, BEFORE gym.make().

    Everything here could equally be done by editing the workshop's
    `vials_to_rack_env_cfg.py` -- except that file lives inside a 28.9 GB
    container image, so every change would cost an image build and push. The cfg
    that `parse_env_cfg` returns is a plain configclass, so the same override
    costs a few lines and no rebuild.

    Two things have to move together for --vial-spawn capdown, and missing the
    second one produces a convincing-looking but completely misleading scene:

      * the ORIENTATION, `init_state.rot`. The shipped value is euler(0,90,0),
        i.e. the vial lying on its side. euler(0,180,0) turns vial-local +z to
        world -z, which stands the vial on the flat face of its cap. That face is
        the only stable upright pose it has -- the other end is a rounded dome
        tapering to a ~4 mm tip, which cannot balance a 116.5 mm body.
      * the HEIGHT, the reset event's `fixed_vial_z`. The shipped 0.05 is the
        resting root height of a LYING vial: the root sits on the cylinder axis,
        15.65 mm above the mat at z=0.033. Standing cap-down the root is instead
        99.7 mm above the contact face, so leaving 0.05 in place spawns the vial
        ~50 mm INSIDE the mat, and physics resolves that by flinging it. The
        video would then show an arm failing for reasons that have nothing to do
        with the geometry under test.
    """
    if args.vial_spawn == "capdown":
        # quat (w,x,y,z) for a 180 deg rotation about Y: vial-local +z -> world -z.
        rot = (0.0, 0.0, 1.0, 0.0)
        n = 0
        for name in sorted(k for k in dir(env_cfg.scene) if k.startswith("vial_")):
            vial_cfg = getattr(env_cfg.scene, name)
            vial_cfg.init_state.rot = rot
            n += 1
        log(f"vial spawn: CAP-DOWN standing, rot={rot} applied to {n} vials")
        hits = 0
        for name, term in _event_terms(env_cfg):
            if "fixed_vial_z" in term.params:
                term.params["fixed_vial_z"] = args.capdown_spawn_z
                hits += 1
                log(f"vial spawn height: {name}.fixed_vial_z -> "
                    f"{args.capdown_spawn_z} (a resting cap-down vial has its "
                    "root at 0.1327)")
        if not hits:
            raise RuntimeError(
                "no reset event exposes fixed_vial_z, so the cap-down spawn "
                "height could not be corrected. Refusing to run: at the shipped "
                "0.05 the vial spawns ~50 mm inside the mat and the resulting "
                "video would be meaningless.")

    if args.deterministic_scene:
        for _name, term in _event_terms(env_cfg):
            params = term.params
            for key in ("pose_range", "rack_pose_range"):
                if isinstance(params.get(key), dict):
                    params[key] = {k: (0.0, 0.0) for k in params[key]}
            if "rack_placement_prob" in params or "fixed_vial_z" in params:
                # Explicit rather than relying on the 0.33 default: a vial
                # pre-placed in a bore changes which bore pick_empty_slot picks,
                # which is exactly the kind of run-to-run variation that makes a
                # demonstration hard to reproduce.
                params["rack_placement_prob"] = 0.0
        log("deterministic scene: vial and rack pose ranges zeroed, "
            "rack_placement_prob=0")


# -- probe -------------------------------------------------------------------

def run_probe(env, expert, vial_names, rack_name, out_dir):
    """Print everything the calibration constants need."""
    robot = env.unwrapped.scene["robot"]
    log(f"articulation joint order : {list(robot.joint_names)}")
    log(f"servo joint ids          : {expert.joint_ids}")
    for k, name in enumerate(JOINT_NAMES):
        lo, hi = expert.limits[k].tolist()
        log(f"  {name:<12} home={float(expert.home[k]):+.4f} "
            f"limits=[{lo:+.4f}, {hi:+.4f}]")
    log(f"pitch sum at home        : {expert.pitch_sum_home:+.4f} rad "
        "(Pitch+Elbow+Wrist_Pitch; holding this holds gripper pitch)")
    log(f"ee_frame pos (root)      : {expert.ee_pos_b().tolist()}")
    log(f"ee_frame quat (root)     : {expert.ee_quat_b().tolist()}")
    log("  -> choose --gripper-pitch-offset so this quaternion points the jaw at "
        "the mat, and --reorient-offset for the pitch change that stands a vial up.")

    log("body positions (world):")
    for i, name in enumerate(robot.body_names):
        log(f"  {name:<24} {robot.data.body_pos_w[0, i].tolist()}")
    log("  -> --grasp-z-offset is the z gap between the gripper body origin (which "
        "is the ee_frame) and the pinch point between the jaws.")

    for v in vial_states(env, sorted(vial_names)):
        log(f"{v['name']}: pos={v['pos'].tolist()} up_axis={v['up'].tolist()} "
            f"(|up_z|={abs(float(v['up'][2])):.3f}; the success term needs > 0.7)")
    rack = env.unwrapped.scene[rack_name]
    log(f"{rack_name}: pos={rack.data.root_pos_w[0].tolist()} "
        f"quat={rack.data.root_quat_w[0].tolist()}")
    slots = rack_slot_positions(env, rack_name)
    for i in range(slots.shape[0]):
        log(f"  slot[{i}] world={slots[i].tolist()}")

    log("Jacobian self-test ...")
    measured = None
    if expert.measure_jacobian(ARGS.jaw_open, ARGS.gripper_pitch_offset):
        log(f"  J (rows x,y,z; cols Rotation,Pitch,Elbow) =\n{expert.jacobian}")
        want = torch.tensor([0.01, 0.0, 0.0], device=expert.device)
        before = expert.ee_pos_b()
        dq = expert._dls(want)
        q_servo = expert._q_servo() + dq
        for _ in range(12):
            expert.step(
                expert.slaved_pose(q_servo, ARGS.jaw_open, ARGS.gripper_pitch_offset),
                record=False,
            )
        measured = (expert.ee_pos_b() - before).tolist()
        log(f"  commanded dx={want.tolist()} -> measured {measured}")
        # NOT "these should agree": a damped solve is SUPPOSED to undershoot, and
        # reading a shortfall as a broken Jacobian sends you hunting the wrong
        # bug. What matters is the DIRECTION and the size of the shortfall
        # relative to what the damping predicts.
        sv = torch.linalg.svdvals(expert.jacobian)
        gains = (sv ** 2) / (sv ** 2 + ARGS.servo_damping)
        ratio = float(torch.linalg.vector_norm(torch.tensor(measured))) / float(
            torch.linalg.vector_norm(want)
        )
        log(f"  Jacobian singular values {sv.tolist()}")
        log(f"  with --servo-damping {ARGS.servo_damping:g} (= lambda^2) the solve "
            f"attenuates the three singular directions to {gains.tolist()}")
        log(f"  so |measured|/|commanded| = {ratio:.3f} is EXPECTED, not a fault. "
            "Judge this line on direction, not magnitude: a result pointing the "
            "wrong way, or one far below the attenuation above, means the servo "
            "will not converge. A magnitude at or below ~0.5 means --servo-damping "
            "is too large for this Jacobian and long segments will exhaust "
            "--servo-max-steps before arriving.")
    else:
        log("  aborted (episode ended during probing) -- raise --episode-length-s.")

    summary = {
        "joint_order": list(robot.joint_names),
        "home": expert.home.tolist(),
        "joint_limits": expert.limits.tolist(),
        "pitch_sum_home": expert.pitch_sum_home,
        "ee_pos_root": expert.ee_pos_b().tolist(),
        "ee_quat_root": expert.ee_quat_b().tolist(),
        "body_names": list(robot.body_names),
        "jacobian": None if expert.jacobian is None else expert.jacobian.tolist(),
        "jacobian_selftest_measured_dx": measured,
        "rack_slots_world": slots.tolist(),
        "max_episode_length_steps": int(env.unwrapped.max_episode_length),
    }
    path = os.path.join(out_dir, "probe_summary.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"wrote {path}")


# -- attempt -----------------------------------------------------------------

def run_attempt(expert, vial_names, rack_name):
    """One scripted attempt. Returns (success, reason)."""
    a = expert.args
    env = expert.env
    expert.reset()

    vials = vial_states(env, vial_names)
    target = pick_target_vial(vials)
    slot = pick_empty_slot(rack_slot_positions(env, rack_name), vials)
    if slot is None:
        return False, "no empty rack slot"

    z = lambda h: torch.tensor([0.0, 0.0, h], device=expert.device)  # noqa: E731
    grasp_b = expert.to_base(target["pos"]) + z(a.grasp_z_offset)
    hover_b = grasp_b + z(a.hover_height)
    lift_b = grasp_b + z(a.lift_height)
    # Both the grasp target and the slot target are positions for the GRIPPER
    # FRAME ORIGIN, but the thing that has to arrive is the pinch point between
    # the fingertips, ~75 mm further along the gripper's approach axis. So the
    # slot target needs the same correction the grasp target gets. Omitting it
    # here (as this script originally did) is not a small error: it puts the
    # vial 48 mm inside the solid top of the rack and the fingertips 78 mm
    # inside it, so `insert` can never reach tolerance.
    slot_b = expert.to_base(slot) + z(a.place_z_offset)
    above_slot_b = slot_b + z(a.insert_clearance)

    down = a.gripper_pitch_offset                          # jaw at the mat
    upright = a.gripper_pitch_offset + a.reorient_offset   # vial held vertical

    # Recording starts from the reset pose so the policy learns the whole reach,
    # not just the lift -- the same choice as the MuJoCo sample.
    plan = [
        ("hover", lambda: expert.servo_to(hover_b, a.jaw_open, down, True, "hover")),
        ("descend", lambda: expert.servo_to(grasp_b, a.jaw_open, down, True, "descend")),
        ("close", lambda: expert.ramp_jaw(a.jaw_open, a.jaw_closed, 6, down, True)),
        ("settle", lambda: expert.hold(a.jaw_closed, down, 6, True)),
        ("lift", lambda: expert.servo_to(lift_b, a.jaw_closed, down, True, "lift")),
        # Stand the vial up while clear of both the mat and the rack.
        ("reorient", lambda: expert.hold(a.jaw_closed, upright, 24, True)),
        ("traverse", lambda: expert.servo_to(above_slot_b, a.jaw_closed, upright, True,
                                             "traverse")),
        ("insert", lambda: expert.servo_to(slot_b, a.jaw_closed, upright, True,
                                           "insert")),
        ("release", lambda: expert.ramp_jaw(a.jaw_closed, a.jaw_open, 4, upright, True)),
        ("retreat", lambda: expert.servo_to(above_slot_b, a.jaw_open, upright, True,
                                            "retreat")),
    ]
    for name, fn in plan:
        if expert.terminated:
            return True, f"success during {name}"
        if not fn():
            if expert.terminated:
                return True, f"success during {name}"
            return False, f"{name} did not complete"

    # The success predicate needs 25 consecutive confirming frames after the
    # placement event. Hold still and let it confirm; these frames ARE recorded,
    # because the policy should learn to hold position after releasing.
    for _ in range(60):
        if expert.terminated:
            return True, "success confirmed after release"
        if not expert.hold(a.jaw_open, upright, 1, True):
            break
    return expert.terminated, "success" if expert.terminated else "not confirmed"


def run_flip_attempt(expert, vial_names, rack_name):
    """Grasp a STANDING cap-down vial, flip it dome-down, try to insert it.

    Differs from run_attempt in two deliberate ways.

    1. STAGES ARE NON-FATAL. run_attempt returns as soon as a stage misses its
       tolerance, which is right when the goal is a clean dataset. It is wrong
       when the goal is to SEE what stops the arm: the stage that fails is
       precisely the one worth filming, and aborting on it leaves the arm frozen
       wherever the servo gave up, usually with the interesting geometry never
       reached. So every stage here runs, its outcome is logged, and the plan
       continues to the next one.
    2. IT LOGS ATTITUDE, not just position. Gripper elevation is exactly
       -(Pitch + Elbow + Wrist_Pitch) radians, and Wrist_Pitch saturates at
       +/-1.658, so the arm can reach a target POSITION while silently holding
       the wrong ATTITUDE. A position-only log cannot tell those apart.

    The plan itself: preset the arm onto the gripper-down branch, descend onto
    the vial's tube, close, lift, rotate ~180 deg about the wrist-pitch axis so
    the dome faces down, traverse over an empty bore, and drive down until
    something stops it.
    """
    a = expert.args
    env = expert.env
    expert.reset()

    vials = vial_states(env, vial_names)
    target = pick_target_vial(vials)
    slot = pick_empty_slot(rack_slot_positions(env, rack_name), vials)
    if slot is None:
        return False, "no empty rack slot"

    up_z = float(target["up"][2])
    log(f"flip: target {target['name']} root_pos_w={target['pos'].tolist()} "
        f"up_z={up_z:+.3f} ({'CAP-DOWN' if up_z < -0.7 else 'DOME-DOWN' if up_z > 0.7 else 'LYING'})")
    log(f"flip: aiming at bore/slot marker {slot.tolist()}")

    z = lambda h: torch.tensor([0.0, 0.0, h], device=expert.device)  # noqa: E731
    grasp_b = expert.to_base(target["pos"]) + z(a.grasp_z_offset)
    hover_b = grasp_b + z(a.hover_height)
    lift_b = grasp_b + z(a.lift_height)
    slot_b = expert.to_base(slot) + z(a.place_z_offset)
    above_slot_b = slot_b + z(a.insert_clearance)

    down = a.gripper_pitch_offset
    flipped = a.gripper_pitch_offset + a.reorient_offset

    preset = None
    if a.flip_preset.strip():
        preset = [float(v) for v in a.flip_preset.split(",")]
        if len(preset) != 3:
            raise ValueError("--flip-preset needs exactly Rotation,Pitch,Elbow")

    plan = []
    if preset is not None:
        plan.append(("preset", lambda: expert.ramp_servo_joints(
            preset, a.jaw_open, down, 40, True)))
    plan += [
        ("hover", lambda: expert.servo_to(hover_b, a.jaw_open, down, True, "hover")),
        ("descend", lambda: expert.servo_to(grasp_b, a.jaw_open, down, True, "descend")),
        ("close", lambda: expert.ramp_jaw(a.jaw_open, a.jaw_closed, 6, down, True)),
        ("settle", lambda: expert.hold(a.jaw_closed, down, 10, True)),
        ("lift", lambda: expert.servo_to(lift_b, a.jaw_closed, down, True, "lift")),
        # The flip. Ramped over many steps rather than one jump: the pitch chain
        # has to swing ~180 deg while carrying the vial, and a step change in the
        # joint targets throws it out of the jaws.
        ("flip", lambda: expert.hold_ramp_pitch(down, flipped, 90, a.jaw_closed, True)),
        ("traverse", lambda: expert.servo_to(above_slot_b, a.jaw_closed, flipped, True,
                                             "traverse")),
        ("insert", lambda: expert.servo_to(slot_b, a.jaw_closed, flipped, True,
                                           "insert")),
        # Keep pressing after the servo gives up, so the obstruction is on screen
        # long enough to see rather than for one frame.
        ("press", lambda: expert.hold(a.jaw_closed, flipped, a.flip_hold_steps, True)),
    ]

    outcomes = []
    for name, fn in plan:
        if expert.terminated:
            return True, f"success during {name}"
        ok = fn()
        ee = expert.ee_pos_b()
        vial_now = vial_states(env, [target["name"]])[0]
        log(f"flip[{name}]: {'reached' if ok else 'STALLED'} "
            f"ee_root=({ee[0]:+.4f},{ee[1]:+.4f},{ee[2]:+.4f}) "
            f"elev={expert.gripper_elevation_deg():+.1f}deg "
            f"vial_z={float(vial_now['pos'][2]):.4f} "
            f"vial_up_z={float(vial_now['up'][2]):+.3f}")
        outcomes.append((name, ok))
        if expert.terminated:
            return True, f"success during {name}"
        if expert.truncated:
            log(f"flip: episode ran out of time during {name}; raise "
                "--episode-length-s to film the whole plan")
            break
    stalled = [n for n, ok in outcomes if not ok]
    reason = ("all stages reached" if not stalled
              else "stalled at " + ",".join(stalled))
    return expert.terminated, reason


# -- main --------------------------------------------------------------------

def main():
    args = ARGS
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    mark(out_dir, "01_main_started")
    log(f"task={args.task} episodes={args.episodes} fps={args.fps}")

    if not args.task.endswith("-Eval"):
        log("ERROR: --task must be an -Eval variant. The non-Eval configs set "
            "terminations=None, so env.step() never reports success and every "
            "episode would be discarded.")
        return 2

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = args.episode_length_s
    apply_scene_overrides(env_cfg, args)
    env = gym.make(args.task, cfg=env_cfg)
    mark(out_dir, "02_env_created")
    log(f"action space {env.action_space}  max_episode_length="
        f"{env.unwrapped.max_episode_length} steps")

    cameras = {}
    for obj in env.unwrapped.scene.keys():
        if obj.startswith("camera_"):
            cam_cfg = getattr(env.unwrapped.scene.cfg, obj)
            cameras[obj.replace("camera_", "")] = {
                "height": cam_cfg.height, "width": cam_cfg.width,
            }
    log(f"cameras: {sorted(cameras)}")
    if not cameras:
        log("ERROR: the task exposes no cameras, so the dataset would have no image "
            "columns and the policy would have nothing to see.")
        env.close()
        return 3

    vial_names = sorted(k for k in env.unwrapped.scene.keys() if k.startswith("vial_"))
    rack_name = next(
        (k for k in sorted(env.unwrapped.scene.keys()) if k.startswith("rack_")), None
    )
    if not vial_names or rack_name is None:
        log("ERROR: expected vial_* and rack_* in the scene, found "
            f"{sorted(env.unwrapped.scene.keys())}")
        env.close()
        return 3
    log(f"vials={vial_names} rack={rack_name}")

    expert = ScriptedExpert(env, args)

    # Bound the debug video BEFORE the probe branch as well: a probe run drives
    # the arm through a Jacobian self-test, and being able to watch that is
    # exactly as useful as watching an attempt.
    debug_writer = None
    if args.debug_video:
        if args.debug_video_camera not in cameras:
            log(f"ERROR: --debug-video-camera {args.debug_video_camera!r} is not one "
                f"of {sorted(cameras)}")
            env.close()
            return 3
        path = os.path.abspath(args.debug_video)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        debug_writer = Mp4Writer(path, args.fps)
        expert.bind_video(debug_writer, args.debug_video_camera)
        log(f"debug video -> {path} (camera {args.debug_video_camera}, every "
            "attempt including discarded ones)")

    def finish_debug_video():
        """Close the debug MP4 and say plainly whether it exists.

        Non-fatal on purpose -- a lost video must not change the run's exit code
        -- but LOUD, because a silently missing MP4 is the failure mode this
        whole option exists to eliminate.
        """
        if debug_writer is None:
            return
        ok = debug_writer.close()
        mark(out_dir, "07_debug_video_closed")
        if ok:
            log(f"debug video: {debug_writer.path} "
                f"({debug_writer.count} frames from "
                f"{expert.video_frames_offered} offered)")
        else:
            log("=" * 72)
            log(f"WARNING: no debug MP4 was written (reason="
                f"{debug_writer.reason}).")
            log(f"  detail: {debug_writer.detail}")
            log(f"  the run offered {expert.video_frames_offered} frames and the "
                f"encoder accepted {debug_writer.count}.")
            log("  This is an ENVIRONMENT problem, not a datagen problem. See "
                "mp4_writer.pick_encoder: the image's ffmpeg is an LGPL build "
                "with no libx264, so libopenh264 is what actually works.")
            log("=" * 72)

    if args.probe:
        expert.reset()
        run_probe(env, expert, vial_names, rack_name, out_dir)
        finish_debug_video()
        env.close()
        # Deliberately no simulation_app.close() here either: it does not return
        # (see the long comment near the end of main()), so any code added after
        # it would be silently unreachable. The __main__ finally block closes the
        # app under a watchdog and exits with this return value.
        return 0

    # `init_device()` and `connect()` are deliberately NOT called: they reach
    # for a physical SO-101 leader arm. Everything the datagen path needs
    # (get_raw_actions_from_radians, sim_to_real_dataset_processor) touches only
    # joint_mins/joint_maxs/cameras/device.
    robot_iface = LeRobotSO101Interface(
        device=env.unwrapped.device, port=None, id="scripted",
        cameras=cameras, fps=args.fps, kind="follower",
    )

    dataset_root = args.dataset_root or os.path.join(out_dir, "dataset")
    recorder = LeRobotRecorder(
        task_name=args.instruction,
        repo_id=args.repo_id,
        dataset_root=dataset_root,
        fps=args.fps,
        device=env.unwrapped.device,
        cameras=cameras,
    )
    try:
        recorder.init_dataset()
    except ValueError as exc:
        log(f"ERROR: {exc}. The step script clears {dataset_root} before running, so "
            "this usually means a stale, partially-written dataset is in the way.")
        finish_debug_video()
        env.close()
        return 4
    mark(out_dir, "03_recorder_ready")

    def push(obs, q_cmd):
        # `action` is the COMMANDED joint target in servo space -- the same
        # quantity teleop recorded as `real_action`, so a policy trained on this
        # dataset emits values a real SO-101 follower can consume.
        action = robot_iface.get_raw_actions_from_radians(q_cmd)
        real_obs, visual, depth, seg = robot_iface.sim_to_real_dataset_processor(
            obs["policy"]["joint_pos_obs"][0], obs["visual"]
        )
        recorder.push_frame_to_buffer(action, real_obs, visual, depth, seg)

    expert.bind_recorder(push)

    def dispatch(event_name):
        """Queue a carb recorder event and pump Kit so it is delivered.

        `LeRobotRecorder` subscribes through the carb event dispatcher and
        `omni.kit.app.queue_event` is deferred, so without pumping updates the
        handler would not have run by the time the next attempt starts.
        """
        omni.kit.app.queue_event(event_name, payload={})
        for _ in range(5):
            simulation_app.update()

    saved = 0
    attempts = 0
    max_attempts = max(1, args.episodes * args.max_attempt_factor)
    t0 = time.time()
    attempt_fn = run_flip_attempt if args.plan == "flip" else run_attempt
    log(f"waypoint plan: {args.plan}")
    while saved < args.episodes and attempts < max_attempts:
        attempts += 1
        ok, reason = attempt_fn(expert, vial_names, rack_name)
        frames = recorder.current_frame
        if ok and frames > 0:
            dispatch(LeRobotRecorder.STOP_RECORDING_EVENT)
            saved += 1
            log(f"SAVED {saved}/{args.episodes} (attempt {attempts}, {frames} frames) "
                f"{reason}")
        else:
            # Discarding is the integrity guarantee. Without it the next
            # attempt's frames append to this one and save_episode() writes an
            # over-length episode that begins with a failed grasp.
            dispatch(LeRobotRecorder.CANCEL_RECORDING_EVENT)
            log(f"discard attempt {attempts} ({frames} frames): {reason}"
                + (f" -- footage kept in {debug_writer.path}" if debug_writer else ""))
    mark(out_dir, "04_attempts_done")
    log(f"{saved}/{args.episodes} verified episodes from {attempts} attempts in "
        f"{time.time() - t0:.0f}s")

    # LeRobotRecorder.save_episode() only ENQUEUES; a DAEMON thread does the
    # parquet and video writing. Daemon threads die at interpreter exit, so
    # exiting here without draining silently drops episodes. Poll rather than
    # queue.join(): the processor swallows exceptions with `continue` and skips
    # task_done(), so join() can hang forever.
    deadline = time.time() + 900
    while recorder.num_recorded_episodes < saved and time.time() < deadline:
        simulation_app.update()
        time.sleep(0.25)
    written = recorder.num_recorded_episodes
    mark(out_dir, "05_recorder_drained")
    log(f"recorder wrote {written}/{saved} enqueued episodes")

    # Flush the debug video HERE, not after the guards below. Those guards
    # `return` non-zero on an under-produced dataset -- which is the normal
    # outcome of a run whose purpose is to film a failure -- and an unclosed
    # ffmpeg pipe at that point means a truncated or absent MP4. The footage has
    # to survive the run being judged a failure, because that is the case it
    # exists to explain.
    finish_debug_video()

    env.close()
    # DECIDE THE STATUS BEFORE CLOSING KIT, and do not close Kit here at all.
    #
    # simulation_app.close() DOES NOT RETURN: Kit's fast-shutdown path
    # terminates the process with status 0. This used to sit right here, above
    # the guards below, which made every one of them unreachable -- so a run
    # that verified ZERO episodes printed "0/1 verified episodes", never printed
    # the ERROR line, exited 0, and the Deadline Cloud step reported SUCCEEDED
    # with nothing in the dataset but meta/info.json. Train would then have
    # finetuned on an empty dataset.
    #
    # Verified on a farm worker (job-59c41621017a4a9fae92defc2f356c6f):
    # _marker_datagen_05_recorder_drained was written, _marker_datagen_06_
    # simapp_closed never was, and _inner_exit_datagen contained 0.
    #
    # Every check below touches only the filesystem, so it is safe -- and
    # necessary -- to run them while Kit is still up. The __main__ finally block
    # closes the app under a 45 s watchdog and then os._exit(_code)s with the
    # code this function returns, so nothing is leaked by not closing here.
    # Proof-of-life marker for the guards themselves: if this file exists, the
    # status checks below really ran. Its absence next to a present
    # 05_recorder_drained is the exact signature of the close()-swallows-status
    # bug described above.
    mark(out_dir, "06_status_decided")

    info = os.path.join(dataset_root, "meta", "info.json")
    if not os.path.isfile(info):
        log(f"ERROR: no dataset metadata at {info}")
        return 5
    if written < saved:
        log(f"ERROR: only {written} of {saved} episodes reached disk before the drain "
            "timeout; the dataset is short of what was verified.")
        return 6
    if saved < args.episodes:
        # Same guard as the MuJoCo sample: under-production must not look like
        # success, or Train silently finetunes on a truncated dataset.
        log(f"ERROR: only {saved}/{args.episodes} episodes verified after {attempts} "
            "attempts. Failing so a partial dataset is not mistaken for a complete "
            "one. Re-run with --probe and re-check the calibration constants, or "
            "raise --max-attempt-factor.")
        return 1
    log(f"Done -> {dataset_root}")
    return 0


if __name__ == "__main__":
    # See the identical guard in render_rollout.py. Kit's non-daemon threads mean
    # an uncaught exception hangs the process instead of ending it, so the task
    # sits until StepTimeoutSeconds SIGKILLs it rather than failing in seconds.
    # Always close the app, flush, and hard-exit past Kit's teardown.
    _code = 1
    try:
        _code = main() or 0
    except BaseException:  # noqa: BLE001 - never leave the process wedged
        import traceback

        traceback.print_exc()
        _code = 1
    finally:
        # See render_rollout.py: simulation_app.close() can itself hang, so bound
        # it with a watchdog rather than trusting Kit's teardown to return.
        import threading

        _t = threading.Timer(45.0, lambda: os._exit(_code))
        _t.daemon = True
        _t.start()
        try:
            simulation_app.close()
        except BaseException:  # noqa: BLE001
            pass
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except BaseException:  # noqa: BLE001
            pass
        os._exit(_code)
