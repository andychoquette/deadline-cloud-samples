#!/usr/bin/env python3
"""Drive the finetuned LeRobot policy in Isaac Sim and record an MP4 + final PNG.

Isaac Lab counterpart of `mujoco_sim_to_policy/scripts/render_scene.py`. It runs
the trained checkpoint in the same environment the dataset was generated from --
the point of the whole pipeline, since a policy trained on real-robot images
cannot drive a simulator (the real->sim appearance gap) and vice versa.

It also prints an OBJECTIVE metric per episode, not just a pretty video: the
environment's own `vial_placed_on_rack_termination()` predicate. `terminated`
means the success term fired (vial upright, inside the rack bounds, released,
confirmed for 25 consecutive frames); `truncated` means the episode ran out of
time. So `success_rate` in the summary is a real number you can regress on
across runs, the same role `grasped=True` plays in the MuJoCo sample.

Why this script exists rather than reusing the workshop's `lerobot_eval.py`:
that script is a ZMQ *client* of a GR00T inference server
(`run_gr00t_server.py`) that lives in a second, separate container. Running it
on a worker would mean a daemon-process job environment plus another ~50 GB
image pull, to reach an inference server that can just as well be in-process.
Meanwhile `LeRobotSO101Interface` already carries a complete LOCAL LeRobot
inference path -- `make_policy` / `sim_obs_to_policy_processor` /
`prediction_to_sim_processor` -- that no workshop script calls. This script uses
those, so one image covers datagen, training and rollout.

The one method it does NOT reuse is `LeRobotSO101Interface.predict_action()`,
which hardcodes `task="Pick up the vial and place it in the tray"`. Calling
LeRobot's `predict_action` directly is what lets the `Instruction` job parameter
actually reach the policy, and keeps the instruction identical between training
and rollout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

# Pure, dependency-free decision logic for what a video failure may do to a
# task. Kept in its own module so it can be unit-tested without Omniverse
# Kit -- see test/test_video_failure_policy.py.
#
# Put this file's own directory on sys.path explicitly rather than relying on
# sys.path[0]. The container does not run this script with plain CPython: the
# image's entrypoint installs a `python` shim that forwards to Isaac Sim's
# python.sh, and a shim that exec's differently (or a future `python -m`
# invocation) need not leave the script directory first on the path. A failed
# import here would kill the Render step AND every Evaluate shard at once, so it
# is worth two lines to not depend on that.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import video_policy as vp  # noqa: E402

# Episode accounting for num_envs > 1, plus the env-cfg surgery
# --deterministic-scene performs. Same reasoning as video_policy: pure, no
# third-party imports, so the logic that decides the job's headline number is
# unit-testable without Omniverse Kit. See test/test_vector_rollout.py.
import rollout_vec as rv  # noqa: E402

HOME_ACTION = [-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, -0.1465]


def parse_args():
    p = argparse.ArgumentParser(description="Isaac Sim SO-101 policy rollout render.")
    p.add_argument("--task", default="Lerobot-So101-Teleop-Vials-To-Rack-Eval",
                   help="Registered Isaac Lab gym id. Must be an -Eval variant so "
                        "success is measurable. Defaults to the NON-randomized "
                        "variant: the video should show the policy, not the "
                        "randomizer, and the success rate stays comparable run to run.")
    p.add_argument("--checkpoint", required=True,
                   help="Directory holding the LeRobot pretrained_model "
                        "(config.json + weights).")
    p.add_argument("--instruction", default="pick up the vial and place it in the rack",
                   help="Must match the instruction the dataset was recorded with.")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--num-envs", type=int, default=1,
                   help="How many Isaac Lab environments to simulate IN PARALLEL in "
                        "this one process. Episodes are run in batches of this size, "
                        "exactly as LeRobot's own lerobot_eval.py does with "
                        "--eval.batch_size: n_batches = ceil(episodes / num_envs). "
                        "Total episodes reported is always --episodes, whatever this "
                        "is set to; a last partial batch has its surplus envs "
                        "discarded (highest env index first), so make --episodes a "
                        "multiple of this to waste nothing.\n"
                        "PIN THIS AND REPORT IT. Isaac Lab's renderer is documented "
                        "as stochastic, and its anti-aliasing mode switches on the "
                        "COMBINED tiled resolution -- which is a function of the env "
                        "count -- so a success rate measured at one value is not "
                        "strictly comparable to one measured at another. The value is "
                        "recorded in the summary JSON and Aggregate refuses to mix "
                        "shards that disagree on it.")
    p.add_argument("--video-env", type=int, default=0,
                   help="Which env index the MP4 and the PNG are recorded from. Only "
                        "ONE env is recorded, deliberately: the per-step GPU->CPU "
                        "frame copy is the dominant per-env host cost, and a tiled "
                        "grid of N shrunken views cannot show a 4 mm vial tip. So the "
                        "video is a SAMPLE, not the measurement -- success is counted "
                        "over ALL envs and all episodes, and the MP4 contains only "
                        "the ceil(episodes/num_envs) episodes this env ran.")
    p.add_argument("--deterministic-scene", action="store_true",
                   help="Zero EVERY range on every reset-mode event term (object "
                        "poses, light exposure, mat rotation, camera focal length, "
                        "camera pose, sky light) and set rack_placement_prob=0, so "
                        "every env in every episode starts from an identical scene.\n"
                        "This is the control for a --num-envs A/B. Isaac Lab has no "
                        "per-env seed API (gym.vector, which LeRobot uses, does), so "
                        "reset randomization is drawn from one stage-global RNG and "
                        "the initial conditions episode k sees are NOT invariant "
                        "under num_envs. With this flag there is nothing left to "
                        "vary, so any difference in outcome between two num_envs "
                        "values is attributable to rendering and physics ordering "
                        "rather than to a different scene.\n"
                        "Broader than generate_dataset.py's flag of the same name, "
                        "which zeroes object poses only because the scripted expert "
                        "cares about geometry and not appearance.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--camera", default="external_D455",
                   help="Which env camera to record the video from (without the "
                        "`camera_` prefix). Use `ego` for the wrist view.")
    p.add_argument("--warmup-steps", type=int, default=10,
                   help="Steps held at the home pose before the policy takes over, so "
                        "the first observation is not mid-reset. Mirrors "
                        "lerobot_eval.py.")
    p.add_argument("--episode-length-s", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=1984)
    p.add_argument("--rename-map", default="",
                   help="Optional JSON mapping sim camera name -> policy feature name, "
                        "for a checkpoint trained with different camera keys.")
    p.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "output"))
    # --- fan-out support (the Evaluate step) ---------------------------------
    # These exist so the parallel Evaluate step can reuse this exact rollout
    # loop instead of carrying a second copy of it. A second loop would have to
    # re-derive the no_grad-not-inference_mode rule and the hard-exit guard
    # below, and would drift from them. See README "4 - Evaluate".
    p.add_argument("--summary-path", default="",
                   help="Where to write the JSON summary. Empty (default) means "
                        "<output-dir>/render_summary.json, which is what the Render "
                        "step wants. The Evaluate step points each shard at its own "
                        "eval/shard_NN.json so concurrent tasks never write the "
                        "same file.")
    p.add_argument("--video-path", default="",
                   help="Where to write the MP4. Empty (default) means "
                        "<output-dir>/so101_policy_rollout.mp4 with the PNG at "
                        "<output-dir>/so101_final_frame.png. When set, the PNG goes "
                        "beside it using the same stem.")
    p.add_argument("--no-video", action="store_true",
                   help="Skip the MP4 and the PNG entirely, and skip the per-frame "
                        "GPU->CPU copy that feeds them. Only the success metrics are "
                        "produced. Useful for a wide Evaluate fan-out where the "
                        "aggregate number is the deliverable and N videos are not.")
    p.add_argument("--shard", type=int, default=-1,
                   help="Shard index recorded in the summary and in every "
                        "[render-metric] log line. -1 (default) means 'not sharded'. "
                        "At fan-out width the worker logs of every shard land in one "
                        "place, so an unlabelled metric line is unattributable.")
    p.add_argument("--condition", default="",
                   help="Free-text label for this shard's evaluation condition (the "
                        "Evaluate step passes 'nominal' or 'randomized'). Recorded in "
                        "the summary so Aggregate can break the success rate down by "
                        "condition.")
    # --- local interactive viewing only -------------------------------------
    # OFF BY DEFAULT, and the default path must stay byte-identical to the
    # headless one: every Deadline Cloud worker runs this with no display at
    # all, where a windowed Kit dies in vkCreateSwapchainKHR. So this is opt-in
    # and the farm never passes it. See README "Watching a rollout locally".
    p.add_argument("--gui", action="store_true",
                   help="Open a visible Isaac Sim viewport instead of running "
                        "headless. LOCAL DEV ONLY -- requires an X display "
                        "(DISPLAY, /tmp/.X11-unix and an Xauthority reachable "
                        "from wherever this process runs). Never pass this on a "
                        "Deadline Cloud worker: they are headless and Kit will "
                        "fail to create a Vulkan swapchain. Implies "
                        "--num-envs 1, see below.")
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


ARGS = parse_args()
ARGS.enable_cameras = True
# `not gui` rather than a conditional assignment, so the no-flag case is
# exactly the unconditional `True` this line used to be.
ARGS.headless = not ARGS.gui
if ARGS.gui and ARGS.num_envs != 1:
    # A GUI run is for watching one robot, and Isaac Lab renders N envs as a
    # tiled grid in which a 4 mm vial tip is invisible -- the same reason
    # --video-env records only one env. Success rates are also not comparable
    # across num_envs (see --num-envs), so a GUI run must not double as a
    # measurement at a different width than the farm used. Force 1 and say so
    # loudly rather than silently producing a summary nobody can compare.
    print(f"[render] WARNING: --gui forces --num-envs 1 (was {ARGS.num_envs}). "
          "A tiled multi-env viewport cannot be watched, and its success rate "
          "would not be comparable to the farm's. Drop --gui to fan out.",
          flush=True)
    ARGS.num_envs = 1

app_launcher = AppLauncher(ARGS)
simulation_app = app_launcher.app

# --- everything below needs Kit running -------------------------------------
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import sim_to_real_so101.tasks  # noqa: F401,E402
from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface  # noqa: E402

# The MP4 encoder lives in its own module so generate_dataset.py's debug video
# uses the SAME ffmpeg-encoder probe. See mp4_writer.pick_encoder for why that
# probe must not be duplicated.
from mp4_writer import Mp4Writer, write_png  # noqa: E402

from lerobot.policies.utils import make_robot_action  # noqa: E402
from lerobot.utils.control_utils import predict_action  # noqa: E402
from lerobot.utils.utils import get_safe_torch_device  # noqa: E402


def log(msg):
    print(f"[render] {msg}", flush=True)


def mark(out_dir, name):
    # Shard-qualified so N concurrent Evaluate tasks do not all write the same
    # marker filenames into the shared OutputDir and lose each other's progress
    # trail when the job-attachment outputs are merged.
    tag = "render" if ARGS.shard < 0 else f"eval{ARGS.shard:02d}"
    try:
        d = os.path.join(out_dir, "logs")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"_marker_{tag}_{name}"), "w") as fh:
            fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + f" {name}\n")
    except OSError:
        pass


def main():
    args = ARGS
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    mark(out_dir, "01_main_started")

    ckpt_cfg = os.path.join(args.checkpoint, "config.json")
    if not os.path.isfile(ckpt_cfg):
        log(f"ERROR: no LeRobot checkpoint at {args.checkpoint} (missing config.json). "
            "The Train step must run first.")
        return 2
    if not args.task.endswith("-Eval"):
        log("ERROR: --task must be an -Eval variant, otherwise terminations=None and "
            "there is no success signal to report.")
        return 2

    n_envs = args.num_envs
    if n_envs < 1:
        log(f"ERROR: --num-envs must be >= 1, got {n_envs}")
        return 2
    if not 0 <= args.video_env < n_envs:
        log(f"ERROR: --video-env {args.video_env} is outside 0..{n_envs - 1}")
        return 2
    if args.episodes < n_envs:
        log(f"WARNING: --episodes {args.episodes} < --num-envs {n_envs}, so "
            f"{n_envs - args.episodes} env(s) will be simulated and discarded. "
            "Lower --num-envs or raise --episodes.")
    elif args.episodes % n_envs:
        log(f"WARNING: --episodes {args.episodes} is not a multiple of --num-envs "
            f"{n_envs}; the last batch discards "
            f"{n_envs - (args.episodes % n_envs)} env(s) of otherwise-good work.")

    # num_envs goes through parse_env_cfg, NOT a direct assignment after it: the
    # workshop base cfg forces `self.scene.num_envs = 1  # Always 1 env for
    # teleoperation` in SO101TeleopEnvCfg.__post_init__, and parse_env_cfg
    # assigns cfg.scene.num_envs AFTER instantiating the cfg (so after
    # __post_init__ has run). Setting it before gym.make any other way is racing
    # that line. Verified below rather than assumed.
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=n_envs)
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = args.episode_length_s
    if args.deterministic_scene:
        changed = rv.zero_reset_randomization(env_cfg)
        if not changed:
            log("ERROR: --deterministic-scene found no reset event ranges to zero, "
                "so it would silently be a no-op and the scene would still vary. "
                "Refusing to run, because the caller asked for a fixed scene.")
            return 2
        log(f"deterministic scene: pinned {len(changed)} range(s): "
            f"{', '.join(sorted(changed))}")
    env = gym.make(args.task, cfg=env_cfg)
    # The cfg says n_envs; this is what the ENV says. A mismatch means something
    # overrode it after parse_env_cfg and every per-env index below would be
    # wrong, so it is a hard stop rather than a warning.
    actual_envs = int(env.unwrapped.num_envs)
    if actual_envs != n_envs:
        log(f"ERROR: asked for {n_envs} envs but the environment reports "
            f"{actual_envs}. Something overrode scene.num_envs after "
            "parse_env_cfg; per-env accounting would be wrong.")
        env.close()
        return 2
    log(f"num_envs={n_envs} (episodes are run in batches of this size)")
    mark(out_dir, "02_env_created")

    cameras = {}
    for obj in env.unwrapped.scene.keys():
        if obj.startswith("camera_"):
            cam_cfg = getattr(env.unwrapped.scene.cfg, obj)
            cameras[obj.replace("camera_", "")] = {
                "height": cam_cfg.height, "width": cam_cfg.width,
            }
    log(f"cameras: {sorted(cameras)}")
    if args.camera not in cameras:
        log(f"ERROR: --camera {args.camera!r} is not one of {sorted(cameras)}")
        env.close()
        return 3

    rename_map = json.loads(args.rename_map) if args.rename_map.strip() else None
    robot_iface = LeRobotSO101Interface(
        device=env.unwrapped.device, port=None, id="render",
        cameras=cameras, fps=args.fps, kind="follower", rename_map=rename_map,
    )
    # init_device() builds the LeRobot robot object so its observation_features /
    # action_features can describe the policy's inputs. connect() is NOT called --
    # that is the only part that touches a serial port. This is exactly what the
    # workshop's own lerobot_eval.py does.
    robot_iface.init_device(visualize=False)
    robot_iface.make_policy(args.checkpoint)
    try:
        pol_queues = rv.PerEnvPolicyQueues(robot_iface.policy, n_envs)
    except RuntimeError as exc:
        log(f"ERROR: {exc}")
        env.close()
        return 6
    mark(out_dir, "03_policy_loaded")

    device = get_safe_torch_device(robot_iface.policy.config.device)
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    home = torch.tensor(HOME_ACTION, device=env.unwrapped.device)

    # Default paths are the Render step's historical ones, unchanged. The
    # Evaluate step overrides them per shard.
    if args.video_path:
        video = os.path.abspath(args.video_path)
        png = os.path.splitext(video)[0] + ".png"
        os.makedirs(os.path.dirname(video), exist_ok=True)
    else:
        video = os.path.join(out_dir, "so101_policy_rollout.mp4")
        png = os.path.join(out_dir, "so101_final_frame.png")
    want_video = not args.no_video
    writer = Mp4Writer(video, args.fps) if want_video else None
    last_frame = None
    episode_results = []
    # Frames the ROLLOUT produced, counted independently of whether the encoder
    # accepted any. writer.count only counts frames ffmpeg actually took, so the
    # two diverge exactly when the encoder is broken -- and that difference is
    # what tells "the rollout captured nothing" (a real defect) apart from "the
    # rollout was fine, the encoder was not" (an environment problem). Collapsing
    # them is the bug this separation exists to prevent.
    frames_offered = 0
    # One env.step() is sim.dt * decimation = 1/60 s, so capture every other step
    # to make the MP4 play at the requested rate in real time.
    capture_every = 2

    # Batched episodes, LeRobot's own scheme: lerobot_eval.py computes
    #   n_batches = n_episodes // env.num_envs + ((n_episodes % env.num_envs) != 0)
    # over a gym.vector.VectorEnv. One batch = one env.reset() + one run to
    # completion of all N envs, giving N episodes.
    #
    # Why batches rather than a continuous stream that starts a fresh episode in
    # whichever env just finished: an Isaac Lab env cannot be removed from the
    # batch, so a streaming scheme still pays to render every env; batching costs
    # only the tail of the slowest env in each batch (in this task nearly every
    # episode runs the full budget, so the tail is ~0), and in exchange the
    # episode -> (batch, env) mapping is exact and the accounting is trivial.
    n_batches, n_simulated, n_surplus = rv.plan_batches(args.episodes, n_envs)
    if n_surplus:
        log(f"WARNING: {n_batches} batches x {n_envs} envs simulates "
            f"{n_simulated} episodes to report {args.episodes}; "
            f"{n_surplus} will be discarded. Make --episodes a multiple of "
            "--num-envs to waste nothing.")
    video_episodes = 0
    # Rollout-only wall clock and env-step count, recorded in the summary. The
    # whole point of --num-envs is throughput, and the task's duration cannot
    # measure it: that number also contains the image pull and the Kit boot,
    # which is exactly the confusion that makes a fan-out speedup and an
    # in-process speedup impossible to tell apart. sim_steps counts env.step()
    # calls; env_steps = sim_steps x num_envs is the comparable quantity, and it
    # is what Isaac Lab's own benchmarks report.
    sim_steps = 0
    t_rollout = time.time()

    for batch in range(n_batches):
        obs, _ = env.reset()
        pol_queues.reset_all()
        step = 0
        # Per-env bookkeeping, in a unit-tested module. `terminated =
        # bool(term_t.any().item())` -- what this used to be -- is correct only
        # at num_envs == 1; at N it ends every env's episode the instant the
        # FIRST one finishes, silently reporting N episodes that all ended when
        # the luckiest one did. test/test_vector_rollout.py has a mutation check
        # that goes red under the old semantics.
        tracker = rv.EpisodeTracker(n_envs)
        ep_final_jp = [None] * n_envs
        while not tracker.all_done():
            # torch.no_grad(), NOT torch.inference_mode(). Tensors *created*
            # inside an inference_mode block are permanently flagged as inference
            # tensors, and Isaac Lab allocates some articulation buffers lazily
            # (joint_acc among them). If such a buffer is first touched in here,
            # the next env.reset() dies in write_joint_velocity_to_sim with:
            #   RuntimeError: Inplace update to inference tensor outside
            #   InferenceMode is not allowed.
            # Observed for real: episode 0 completed 900 steps, then episode 1's
            # reset raised. no_grad gives the same inference speedup without
            # tainting tensors.
            with torch.no_grad():
                if step < args.warmup_steps:
                    actions[:] = home
                else:
                    for i in range(n_envs):
                        if tracker.done[i]:
                            # This env's episode is over. Isaac Lab auto-reset it
                            # inside step(), so it is now running an episode we
                            # are NOT recording -- hold it at the home pose so it
                            # cannot influence anything and does not need
                            # inference. It still renders; that is the cost of
                            # the batch tail.
                            actions[i] = home
                            continue
                        # A 1-env slice positioned at i, so the workshop's
                        # sim_obs_to_policy_processor -- which indexes its visual
                        # argument with a hardcoded [0] -- reads env i without
                        # being forked. Still the real-robot batch-of-1 path.
                        frame = robot_iface.sim_obs_to_policy_processor(
                            obs["policy"]["joint_pos_obs"][i],
                            rv.env_visual_slice(obs["visual"], i),
                        )
                        # ACT ships ONE action queue; make env i's the live one.
                        pol_queues.select(i)
                        values = predict_action(
                            observation=frame,
                            policy=robot_iface.policy,
                            device=device,
                            preprocessor=robot_iface.preprocessor,
                            postprocessor=robot_iface.postprocessor,
                            use_amp=robot_iface.policy.config.use_amp,
                            # The instruction the dataset was recorded and trained
                            # with. LeRobotSO101Interface.predict_action() hardcodes
                            # a different string, which is why it is not used here.
                            task=args.instruction,
                            robot_type=robot_iface.robot.robot_type,
                        )
                        robot_action = make_robot_action(values, robot_iface.dataset_features)
                        sent = robot_iface.robot_action_processor((robot_action, None))
                        motor = {k: v for k, v in sent.items() if k.endswith(".pos")}
                        actions[i] = robot_iface.get_mapped_actions_vectorized(
                            robot_iface.get_raw_actions_tensor(motor)
                        )
                # The joint state as it is BEFORE this step. Recorded on
                # termination as the episode's final pose: env.step() auto-resets
                # terminated envs and recomputes observations, so the post-step
                # obs of a terminated env is already its NEXT episode's initial
                # state and reading it there would report the wrong pose.
                prev_jp = obs["policy"]["joint_pos_obs"].detach().clone()
                was_done = list(tracker.done)
                obs, _, term_t, trunc_t, _ = env.step(actions)
                for i in tracker.observe(term_t.detach().to("cpu").tolist(),
                                         trunc_t.detach().to("cpu").tolist(),
                                         step):
                    ep_final_jp[i] = [round(float(v), 6)
                                      for v in prev_jp[i].tolist()]
                # The GPU->CPU frame copy is the per-step cost of having a video
                # at all, so --no-video skips it rather than just discarding the
                # result. Only ONE env is copied however many are simulated --
                # see --video-env. `not was_done[...]` stops the recording at the
                # end of that env's episode instead of filming the auto-reset
                # episode that follows it.
                if want_video and not was_done[args.video_env] \
                        and step % capture_every == 0:
                    img = obs["visual"][f"rgb_{args.camera}"][args.video_env]
                    last_frame = img.detach().to("cpu").numpy().astype(np.uint8)[..., :3]
                    frames_offered += 1
                    writer.add(last_frame)
                step += 1
                sim_steps += 1

        if want_video:
            video_episodes += 1
        # Episode index is batch-major so it is stable and reproducible: episode
        # e is always (batch e // n_envs, env e % n_envs). The last batch's
        # surplus envs are dropped from the HIGH end, so the episodes that ARE
        # reported are the same set whether or not a partial batch happened --
        # `episodes` stays exactly --episodes long and
        # EvalShards x EvalEpisodesPerShard remains the true total.
        surplus = rv.surplus_envs(batch, n_envs, args.episodes)
        for i in range(n_envs):
            ep = rv.episode_index(batch, i, n_envs)
            if i in surplus:
                log(f"[render-metric] shard={args.shard} "
                    f"condition={args.condition or '-'} DISCARDED batch={batch} "
                    f"env={i} steps={tracker.steps[i]} "
                    f"success={tracker.success[i]} "
                    "(surplus of a partial last batch, not counted)")
                continue
            episode_results.append({
                "episode": ep,
                "batch": batch,
                "env": i,
                "steps": tracker.steps[i],
                "success": tracker.success[i],
                # A continuous observable, so a num_envs A/B has something with
                # more statistical power than a handful of binary outcomes: under
                # --deterministic-scene every episode is the same scene, so any
                # difference in this vector between two num_envs values is
                # measurable trajectory divergence.
                "final_joint_pos": ep_final_jp[i],
            })
            # Mirrors the MuJoCo sample's [render-metric] line so a multi-run sweep
            # can be graded by grepping the logs. shard= and condition= are what make
            # the line attributable when N shards log concurrently.
            log(f"[render-metric] shard={args.shard} condition={args.condition or '-'} "
                f"episode={ep} batch={batch} env={i} steps={tracker.steps[i]} "
                f"success={tracker.success[i]}")

    rollout_s = time.time() - t_rollout
    env_steps = sim_steps * n_envs
    steps_per_s = env_steps / rollout_s if rollout_s > 0 else 0.0

    successes = sum(1 for r in episode_results if r["success"])
    rate = successes / max(1, len(episode_results))
    log(f"[render-metric] shard={args.shard} condition={args.condition or '-'} "
        f"success_rate={successes}/{len(episode_results)} ({100 * rate:.1f}%)")
    # One grep-able line carrying everything the in-process half of the speedup
    # decomposition needs, so it does not have to be reconstructed from task
    # durations that also contain image pull and Kit boot.
    log(f"[render-throughput] shard={args.shard} num_envs={n_envs} "
        f"batches={n_batches} sim_steps={sim_steps} env_steps={env_steps} "
        f"rollout_s={rollout_s:.1f} env_steps_per_s={steps_per_s:.2f} "
        f"s_per_sim_step={rollout_s / max(1, sim_steps):.4f}")
    mark(out_dir, "04_rollouts_done")

    video_ok = False
    frames = 0
    if writer is not None:
        video_ok = writer.close()
        frames = writer.count
        if last_frame is not None:
            write_png(last_frame, png)

    # Resolve WHY there is or is not a video, as one explicit value. Recorded in
    # the summary so a missing MP4 is *reported* rather than inferred from an
    # absent file -- inferring it is what let a broken encoder go unnoticed
    # through a whole 8-shard run.
    video_status, video_error = vp.classify_video(
        want_video=want_video,
        video_ok=video_ok,
        frames_offered=frames_offered,
        writer_reason=writer.reason if writer is not None else None,
        writer_detail=writer.detail if writer is not None else None,
        camera=args.camera,
    )

    summary_path = os.path.abspath(args.summary_path) if args.summary_path \
        else os.path.join(out_dir, "render_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    # Written LAST and in one json.dump: Aggregate treats the presence of this
    # file as the shard's completion signal, so it must never exist in a
    # half-written state.
    with open(summary_path, "w") as fh:
        json.dump({
            "task": args.task,
            "instruction": args.instruction,
            "checkpoint": args.checkpoint,
            "camera": args.camera,
            "fps": args.fps,
            # Fan-out metadata. Aggregate groups on "condition" and keys on
            # "shard"; "seed" is here so a surprising shard can be reproduced
            # exactly with one openjd run.
            "shard": args.shard,
            "condition": args.condition,
            "seed": args.seed,
            "episode_length_s": args.episode_length_s,
            # The parallelism knob, recorded because it is part of the
            # EXPERIMENTAL CONDITION and not merely a performance setting: Isaac
            # Lab's anti-aliasing mode switches on the combined tiled resolution
            # (a function of the env count) and its renderer is documented as
            # stochastic, so success rates measured at different num_envs are not
            # guaranteed comparable. Aggregate refuses to average shards that
            # disagree on this.
            "num_envs": n_envs,
            "episode_batches": n_batches,
            "deterministic_scene": bool(args.deterministic_scene),
            "rollout_seconds": round(rollout_s, 2),
            "sim_steps": sim_steps,
            "env_steps": env_steps,
            "env_steps_per_second": round(steps_per_s, 3),
            "episodes": episode_results,
            "successes": successes,
            "success_rate": rate,
            "frames": frames,
            "video": video if video_ok else None,
            # Which env the video shows, and how many episodes are in it. At
            # num_envs > 1 the MP4 is NOT the measurement: it holds one env's
            # n_batches episodes out of the `episodes` total, so an MP4 of one
            # failure next to a nonzero success_rate is expected, not a
            # contradiction.
            "video_env": args.video_env if want_video else None,
            "video_episodes": video_episodes,
            # "video" alone cannot distinguish "not requested" from "requested
            # and broken" -- both are null. These two fields can, and
            # aggregate_eval.py surfaces them.
            "video_status": video_status,
            "video_error": video_error,
            "frames_offered": frames_offered,
        }, fh, indent=2)
    log(f"wrote {summary_path}")

    env.close()
    # DECIDE THE STATUS BEFORE CLOSING KIT, and do not close Kit here at all.
    #
    # simulation_app.close() DOES NOT RETURN: Kit's fast-shutdown path terminates
    # the process with status 0. This call used to sit right here, above every
    # check below, which made all of them -- and mark("05_simapp_closed") --
    # unreachable. Consequence: this script could not fail. EXIT_NO_EPISODES and
    # EXIT_NO_FRAMES were dead code, so a Render or Evaluate shard that ran zero
    # episodes still reported SUCCEEDED, and Aggregate happily averaged it in.
    # The same bug was found and fixed in generate_dataset.py, where it let
    # Datagen report SUCCEEDED with an empty dataset; the marker evidence is in
    # that file's comment.
    #
    # Everything below is pure Python and filesystem work, so it is safe to run
    # with Kit still up. The __main__ finally block closes the app under a 45 s
    # watchdog and then os._exit()s with the code this function returns.
    mark(out_dir, "05_status_decided")

    # --- what may and may not fail this task ---------------------------------
    # The summary JSON is the deliverable; the MP4 is an artifact. So the ONLY
    # video-related condition that fails the task is the one that means the
    # ROLLOUT is broken.
    #
    # This used to read `if want_video and frames == 0: return 4`, which
    # conflated two unrelated things, because `frames` counts what the ENCODER
    # accepted. With a codec missing from the image, a shard that ran all its
    # episodes and produced perfectly good success data would exit 4 -> task
    # FAILED -> `5 - Aggregate` CANCELED by dependency -> the whole run's
    # headline number destroyed, by a video problem. It only escaped notice on a
    # real 8-shard run because exactly one frame happened to land before ffmpeg
    # died, leaving frames == 1. A one-frame margin is not a safety property.
    code = vp.task_exit_code(len(episode_results), want_video, video_status)
    if code == vp.EXIT_NO_EPISODES:
        log("ERROR: no episodes ran, so there is no result to report.")
        return code
    if code == vp.EXIT_NO_FRAMES:
        log(f"ERROR: {video_error}")
        return code
    if want_video and not video_ok:
        # Loud, because the opposite failure -- a silent missing video -- is what
        # cost a full day of debugging. Never fatal, because the metrics are fine.
        log("=" * 72)
        log(f"WARNING: no MP4 was written (video_status={video_status}).")
        log(f"  reason: {video_error}")
        log(f"  the rollout produced {frames_offered} frames and the encoder "
            f"accepted {frames}.")
        log("  This is an ENVIRONMENT problem, not a rollout problem: all "
            f"{len(episode_results)} episodes ran and their success data is in "
            "the summary, so this task is reporting SUCCESS.")
        log("  Fix the container's ffmpeg (see pick_encoder) to get videos back.")
        log("=" * 72)
    # A zero success rate is a legitimate result to report, not a task failure:
    # the video and the summary are exactly what you need in order to see why.
    # This matters more under fan-out: one shard failing the TASK on a low score
    # would cancel Aggregate by dependency and destroy the run's headline number.
    log(f"Done -> {video if want_video else summary_path}")
    return 0


if __name__ == "__main__":
    # Kit keeps non-daemon threads alive, so an uncaught exception here does NOT
    # end the process -- CPython waits on those threads forever and the task
    # hangs until StepTimeoutSeconds (default 5400s) SIGKILLs it. A 30-second
    # validation error then burns 90 minutes of GPU time, multiplied by
    # maxRetriesPerTask. Observed for real: a LeRobot camera-key mismatch printed
    # its traceback and then wedged with no further output.
    #
    # So: always close the app, flush the pipes the bundle is tee-ing, and
    # os._exit() past Kit's teardown (which is itself a known hang). os._exit
    # skips atexit/GC on purpose -- there is nothing left worth cleaning up, and
    # a clean exit is exactly what doesn't work here.
    _code = 1
    try:
        _code = main() or 0
    except BaseException:  # noqa: BLE001 - never leave the process wedged
        import traceback

        traceback.print_exc()
        _code = 1
    finally:
        # simulation_app.close() can ITSELF hang -- Kit teardown is a known hang,
        # and Isaac Sim 6.0 added a shutdown watchdog precisely because of it. So
        # arm our own watchdog before calling it: if close() has not returned in
        # 45s, force the exit anyway. Without this, an exception whose traceback
        # printed fine still leaves the task wedged until StepTimeoutSeconds.
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
