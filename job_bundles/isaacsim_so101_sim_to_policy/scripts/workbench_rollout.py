#!/usr/bin/env python3
"""One policy episode, run INSIDE an already-running Isaac Sim GUI session.

Paste into **Window -> Script Editor** of the window ``workbench_gui.sh`` opens::

    import workbench_rollout as wr
    wr.run()

and watch the arm in the viewport. ``wr.run()`` prints one line::

    [rollout] EPISODE RESULT: success=False steps=900 ...

**``success=False`` is the expected outcome.** This policy's measured success rate
is about 1 in 48, so a single failed attempt is the correct result and not a bug
to be chased. That is also the point of the demo: one attempt tells you almost
nothing, which is why the next thing you do is submit N of them to the farm from
the same window.

Relationship to ``render_rollout.py``
------------------------------------
This is the same rollout, re-expressed for a live app. It deliberately does NOT
import render_rollout: that module calls ``AppLauncher(ARGS)`` at import time
(~line 199) and its ``__main__`` block ends in ``simulation_app.close()`` +
``os._exit()``. Importing it here would try to launch a second Kit inside the
first, and any code path that reached its exit handling would take the submitter
panel down with the process. The episode bookkeeping is NOT re-implemented: it
comes from ``rollout_vec.py``, the unit-tested module render_rollout uses.

Constraints this file exists to respect
---------------------------------------
* ``torch.no_grad()``, never ``torch.inference_mode()``. Tensors *created* inside
  an inference_mode block are permanently flagged as inference tensors, Isaac Lab
  allocates some articulation buffers lazily (``joint_acc`` among them), and if
  one is first touched in such a block the NEXT ``env.reset()`` dies inside
  ``write_joint_velocity_to_sim`` with "Inplace update to inference tensor
  outside InferenceMode is not allowed". In a workbench this is worse than in a
  batch job: the poisoned buffer outlives the call, so a second ``wr.run()``
  fails and nothing short of restarting the app fixes it. See render_rollout.py
  ~line 406.
* No ``AppLauncher``, no ``simulation_app.close()``, no ``os._exit()``. The app
  already exists and must survive this.
* ``num_envs`` goes through ``parse_env_cfg`` and is 1. The workshop base config
  forces ``self.scene.num_envs = 1`` in ``SO101TeleopEnvCfg.__post_init__`` and
  derives from it, so assigning it afterwards races that; and a tiled multi-env
  viewport cannot be watched, which is the entire reason to run one here.
* Cameras must already be on. ``enable_cameras`` is launch-time only, which is
  why ``workbench_session.py`` owns it. If the app was launched without it,
  ``build()`` says so instead of producing a policy fed with empty images.
* ``env.step()`` auto-resets a terminated env and recomputes observations, so the
  post-step observation of a finished env already belongs to its NEXT episode.
  Nothing here reads state after termination for that reason.

Why the viewport keeps moving while this runs
--------------------------------------------
``env.step()`` renders, and Isaac Lab's ``SimulationContext.render()`` pumps the
app -- so the blocking loop in ``run()`` is also what draws the frames, and Kit
keeps processing input. If a future Isaac Lab release stops pumping and the UI
freezes, use ``wr.start()`` instead: identical episode, one ``env.step()`` per
Kit update event, so the app owns the loop. ``start()`` guards against
reentrancy, because the nested ``app.update()`` inside ``env.step()`` would
otherwise re-enter its own callback.
"""

from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import rollout_vec as rv  # noqa: E402

#: Same home pose render_rollout.py holds during warmup.
HOME_ACTION = [-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, -0.1465]

DEFAULT_TASK = os.environ.get(
    "WORKBENCH_TASK", "Lerobot-So101-Teleop-Vials-To-Rack-Eval")
DEFAULT_CHECKPOINT = os.environ.get("WORKBENCH_CHECKPOINT", "/checkpoint")
DEFAULT_INSTRUCTION = "pick up the vial and place it in the rack"

#: Built once, reused by every later call. Building the env and loading a 206 MB
#: ACT checkpoint takes 1-2 minutes; a demo cannot pay that per attempt, and
#: Isaac Lab does not support two live envs in one process anyway.
_SESSION = None
#: start()'s state, and the flag that makes its callback non-reentrant.
_ASYNC = None
#: How many times run() has been called in this session, logged per call.
_RUN_CALLS = 0


def log(msg):
    print(f"[rollout] {msg}", flush=True)
    try:
        # Also into Kit's Console window: prints issued from an update callback
        # do not reach the Script Editor's output pane, and the result line must
        # be visible wherever the user happens to be looking.
        import carb
        carb.log_warn(f"[rollout] {msg}")
    except Exception:  # noqa: BLE001
        pass


def cameras_enabled():
    """Whether the app was launched with cameras. Launch-time, unfixable here."""
    try:
        import carb
        settings = carb.settings.get_settings()
        for key in ("/isaaclab/cameras_enabled", "/app/omni.replicator/captureOnPlay"):
            if settings.get(key):
                return True
    except Exception:  # noqa: BLE001
        pass
    return None  # unknown -- do not block on a setting name that may have moved


class Session:
    """A live env + policy pair, plus the tensors the step loop reuses."""

    def __init__(self, env, iface, queues, policy_device, actions, home, task,
                 checkpoint, instruction):
        self.env = env
        self.iface = iface
        self.queues = queues
        self.policy_device = policy_device
        self.actions = actions
        self.home = home
        self.task = task
        self.checkpoint = checkpoint
        self.instruction = instruction


def build(task=None, checkpoint=None, instruction=None, episode_length_s=15.0,
          seed=1984, fps=30, device="cuda:0", camera_check=True):
    """Build the Isaac Lab env and load the policy. Idempotent: returns the cache.

    Safe to call directly if you want the cost paid up front (e.g. before the
    audience is watching) rather than inside the first ``run()``.
    """
    global _SESSION
    if _SESSION is not None:
        return _SESSION

    task = task or DEFAULT_TASK
    checkpoint = checkpoint or DEFAULT_CHECKPOINT
    instruction = instruction or DEFAULT_INSTRUCTION

    if not task.endswith("-Eval"):
        raise ValueError(
            f"task {task!r} must be an -Eval variant, otherwise terminations=None "
            "and there is no success signal to report.")
    if not os.path.isfile(os.path.join(checkpoint, "config.json")):
        raise FileNotFoundError(
            f"{checkpoint}/config.json not found -- that is not a LeRobot "
            "checkpoint. Pass checkpoint=... or check the container mount.")

    t0 = time.time()
    log(f"building env: task={task} num_envs=1 device={device}")

    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401  (registers the Isaac Lab task ids)
    from isaaclab_tasks.utils import parse_env_cfg

    import sim_to_real_so101.tasks  # noqa: F401  (registers the workshop task ids)
    from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface

    from lerobot.utils.utils import get_safe_torch_device

    # num_envs through parse_env_cfg, never assigned afterwards -- see the module
    # docstring. parse_env_cfg sets cfg.scene.num_envs AFTER __post_init__ has
    # run, which is the only ordering that survives the workshop base config.
    env_cfg = parse_env_cfg(task, device=device, num_envs=1)
    env_cfg.seed = seed
    env_cfg.episode_length_s = episode_length_s

    env = gym.make(task, cfg=env_cfg)
    actual = int(env.unwrapped.num_envs)
    if actual != 1:
        env.close()
        raise RuntimeError(
            f"asked for 1 env but the environment reports {actual}; something "
            "overrode scene.num_envs after parse_env_cfg.")

    cameras = {}
    for obj in env.unwrapped.scene.keys():
        if obj.startswith("camera_"):
            cam_cfg = getattr(env.unwrapped.scene.cfg, obj)
            cameras[obj.replace("camera_", "")] = {
                "height": cam_cfg.height, "width": cam_cfg.width,
            }
    log(f"cameras: {sorted(cameras)}")
    if camera_check and not cameras:
        env.close()
        raise RuntimeError(
            "the env exposes no camera_* sensors, so the policy has no visual "
            "observation. The app was almost certainly launched without "
            "enable_cameras -- that is launch-time only and cannot be fixed from "
            "the Script Editor. Relaunch with workbench_gui.sh.")

    iface = LeRobotSO101Interface(
        device=env.unwrapped.device, port=None, id="workbench",
        cameras=cameras, fps=fps, kind="follower", rename_map=None,
    )
    # init_device() builds the LeRobot robot object so its observation_features /
    # action_features can describe the policy's inputs. connect() is NOT called --
    # that is the only part that touches a serial port. Same as the workshop's own
    # lerobot_eval.py, and as render_rollout.py.
    iface.init_device(visualize=False)
    iface.make_policy(checkpoint)
    queues = rv.PerEnvPolicyQueues(iface.policy, 1)

    policy_device = get_safe_torch_device(iface.policy.config.device)
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    home = torch.tensor(HOME_ACTION, device=env.unwrapped.device)

    _SESSION = Session(env, iface, queues, policy_device, actions, home,
                       task, checkpoint, instruction)
    log(f"ready in {time.time() - t0:.0f}s -- call wr.run()")
    return _SESSION


def _episode_setup(session, instruction, warmup_steps):
    """Reset the env and the policy queue, and return the per-episode state."""
    import torch  # noqa: F401  (imported so a missing torch fails here, not mid-loop)

    obs, _ = session.env.reset()
    session.queues.reset_all()
    tracker = rv.EpisodeTracker(1)
    return {
        "obs": obs,
        "tracker": tracker,
        "step": 0,
        "instruction": instruction or session.instruction,
        "warmup_steps": warmup_steps,
        "t0": time.time(),
    }


def _episode_step(session, state):
    """One env.step(), policy inference included. Returns True when the episode ends.

    torch.no_grad(), NOT torch.inference_mode(). See the module docstring: an
    inference tensor allocated in here poisons the articulation buffers for the
    rest of the app's life, so in a persistent session the cost is not one failed
    episode but every subsequent one.
    """
    import torch

    from lerobot.policies.utils import make_robot_action
    from lerobot.utils.control_utils import predict_action

    iface = session.iface
    obs = state["obs"]
    tracker = state["tracker"]

    with torch.no_grad():
        if state["step"] < state["warmup_steps"]:
            # Held at the home pose so the policy's first observation is not a
            # mid-reset one. Mirrors lerobot_eval.py and render_rollout.py.
            session.actions[:] = session.home
        else:
            frame = iface.sim_obs_to_policy_processor(
                obs["policy"]["joint_pos_obs"][0],
                rv.env_visual_slice(obs["visual"], 0),
            )
            session.queues.select(0)
            values = predict_action(
                observation=frame,
                policy=iface.policy,
                device=session.policy_device,
                preprocessor=iface.preprocessor,
                postprocessor=iface.postprocessor,
                use_amp=iface.policy.config.use_amp,
                # The instruction the dataset was recorded and trained with.
                # LeRobotSO101Interface.predict_action() hardcodes a different
                # string, which is why it is not used.
                task=state["instruction"],
                robot_type=iface.robot.robot_type,
            )
            robot_action = make_robot_action(values, iface.dataset_features)
            sent = iface.robot_action_processor((robot_action, None))
            motor = {k: v for k, v in sent.items() if k.endswith(".pos")}
            session.actions[0] = iface.get_mapped_actions_vectorized(
                iface.get_raw_actions_tensor(motor)
            )

        # This step's render is also what pumps the app, which is why the
        # viewport animates during a blocking run().
        obs, _, term_t, trunc_t, _ = session.env.step(session.actions)
        state["obs"] = obs
        tracker.observe(term_t.detach().to("cpu").tolist(),
                        trunc_t.detach().to("cpu").tolist(),
                        state["step"])
        state["step"] += 1

    return tracker.all_done()


def _report(state):
    """The one line the demo is judged on."""
    tracker = state["tracker"]
    success = bool(tracker.success[0])
    steps = tracker.steps[0]
    secs = time.time() - state["t0"]
    log("=" * 68)
    log(f"EPISODE RESULT: success={success} steps={steps} "
        f"wall={secs:.0f}s task={state.get('task', '')}")
    if not success:
        log("success=False is the EXPECTED result for a single attempt: this "
            "policy succeeds about 1 time in 48. That is the argument for "
            "submitting N attempts to the farm, not a bug.")
    log("=" * 68)
    return {"success": success, "steps": steps, "seconds": round(secs, 1)}


def run(episodes=1, instruction=None, warmup_steps=10, build_kwargs=None):
    """Run ``episodes`` complete episodes now, blocking, and print the result.

    Blocking is fine and is the documented path: ``env.step()`` renders, and
    rendering pumps Kit, so the viewport animates and the UI stays responsive
    while this runs. It returns a list of result dicts.
    """
    session = build(**(build_kwargs or {}))
    global _RUN_CALLS
    _RUN_CALLS += 1
    # Call-count in the log, because "two episodes appeared and I asked for one"
    # is otherwise unattributable between a caller that ran twice and a loop that
    # iterated twice.
    log(f"run(episodes={episodes}) call #{_RUN_CALLS}")
    if os.environ.get("WORKBENCH_TRACE_CALLERS"):
        import traceback
        log("run() called from:\n" + "".join(traceback.format_stack()[:-1]))
    results = []
    for episode in range(episodes):
        state = _episode_setup(session, instruction, warmup_steps)
        state["task"] = session.task
        log(f"episode {episode + 1}/{episodes} running (call #{_RUN_CALLS}, "
            f"instruction={state['instruction']!r})")
        while not _episode_step(session, state):
            pass
        results.append(_report(state))
    return results


def start(episodes=1, instruction=None, warmup_steps=10, build_kwargs=None):
    """Same episode, driven from Kit's update loop. Returns immediately.

    Use this if a future Isaac Lab release stops pumping the app inside
    ``env.step()`` and ``run()`` freezes the UI. The result line appears in the
    Console window when the episode finishes, not in the Script Editor.
    """
    global _ASYNC
    if _ASYNC is not None and not _ASYNC.get("done"):
        log("a rollout is already running -- ignoring start()")
        return None

    import omni.kit.app

    session = build(**(build_kwargs or {}))
    state = _episode_setup(session, instruction, warmup_steps)
    state["task"] = session.task
    _ASYNC = {"state": state, "episode": 0, "episodes": episodes, "done": False,
              "busy": False, "results": [], "sub": None}

    def on_update(_event):
        ctx = _ASYNC
        # Reentrancy guard, and it is not optional: env.step() calls
        # SimulationContext.render(), which calls app.update(), which pops this
        # very event stream -- so without the flag this callback re-enters itself
        # mid-step and interleaves two half-steps.
        if ctx["busy"] or ctx["done"]:
            return
        ctx["busy"] = True
        try:
            if _episode_step(session, ctx["state"]):
                ctx["results"].append(_report(ctx["state"]))
                ctx["episode"] += 1
                if ctx["episode"] >= ctx["episodes"]:
                    ctx["done"] = True
                    ctx["sub"] = None  # unsubscribe
                else:
                    ctx["state"] = _episode_setup(session, instruction,
                                                  warmup_steps)
                    ctx["state"]["task"] = session.task
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            log(f"ERROR: rollout aborted: {exc}")
            ctx["done"] = True
            ctx["sub"] = None
        finally:
            ctx["busy"] = False

    _ASYNC["sub"] = (
        omni.kit.app.get_app()
        .get_update_event_stream()
        .create_subscription_to_pop(on_update, name="workbench_rollout.step")
    )
    log(f"started {episodes} episode(s) on Kit's update loop; "
        "the result line will print when it finishes")
    return _ASYNC


def status():
    """Progress of a start() rollout, for pasting into the Script Editor."""
    if _ASYNC is None:
        return {"running": False, "results": []}
    return {"running": not _ASYNC["done"],
            "episode": _ASYNC["episode"],
            "step": _ASYNC["state"]["step"],
            "results": _ASYNC["results"]}


def close():
    """Drop the env and the policy. The APP stays up; only the rollout goes away.

    Not called automatically anywhere: closing the env is exactly what a
    workbench must not do behind the user's back.
    """
    global _SESSION, _ASYNC
    if _ASYNC is not None:
        _ASYNC["done"] = True
        _ASYNC["sub"] = None
        _ASYNC = None
    if _SESSION is not None:
        try:
            _SESSION.env.close()
        except Exception as exc:  # noqa: BLE001
            log(f"env.close() raised (ignored): {exc}")
        _SESSION = None
        log("env closed; the app is still running")
