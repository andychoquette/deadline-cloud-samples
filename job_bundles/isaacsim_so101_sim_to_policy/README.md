# Isaac Sim SO-101 Sim-to-Policy Pipeline (5-step)

This sample trains and renders a learned robot-manipulation policy on
[AWS Deadline Cloud](https://docs.aws.amazon.com/deadline-cloud/) using
[NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim) /
[Isaac Lab](https://isaac-sim.github.io/IsaacLab/), on the vials-into-rack task
from NVIDIA's
[Sim-to-Real-SO-101-Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop).
It is a single submitted job that runs five dependent steps on managed GPU
workers, and it **fans out** the evaluation stage across as many workers as the
fleet will give it:

```
                                        ┌───────────────┐
                                   ┌──▶ │   3. Render   │
                                   │    │ policy rollout│
                                   │    │ → MP4 + PNG   │
                                   │    └───────────────┘
 ┌───────────────┐   ┌───────────┐  │
 │  1. Datagen   │──▶│ 2. Train  │──┤    ┌───────────────┐     ┌──────────────┐
 │  Isaac Lab    │   │ ACT       │  │    │  4. Evaluate  │     │ 5. Aggregate │
 │  scripted     │   │ (LeRobot) │  └──▶ │ ╭───────────╮ │ ──▶ │ eval_summary │
 │  expert       │   │ finetune  │       │ │ shard 1   │ │     │    .json     │
 │  → LeRobot ds │   │ → ckpt    │       │ │ shard 2   │ │     │ + log table  │
 └───────────────┘   └───────────┘       │ │  ...      │ │     │ (no GPU)     │
                                         │ │ shard N   │ │     └──────────────┘
                                         │ ╰───────────╯ │
                                         │  N PARALLEL   │
                                         │     TASKS     │
                                         └───────────────┘
 └────────────── shared OutputDir (job attachment) ──────────────┘
 └───── one container image for steps 1-4 (step 5 needs none) ────┘
```

A scripted expert picks up a vial lying on the mat, stands it upright, and
places it in a rack slot. Every episode is verified; the verified ones become a
LeRobot dataset, an ACT policy is finetuned on it, and the policy is then
replayed in the same simulator with an objective success rate.

Steps 3 and 4 both depend only on step 2, so **Render and the whole Evaluate
fan-out become schedulable at the same moment** rather than one after the other.
Whether they then actually run *at the same time* is a property of the fleet, not
of the template — see "What actually bounds observed concurrency".

It is the Isaac Sim counterpart of
[`mujoco_sim_to_policy`](../mujoco_sim_to_policy), and it keeps that sample's
contract: verify each episode, discard failures, and never let a partial dataset
look like a complete one.

## Parallel evaluation: what fans out and why it can

`4 - Evaluate` is the embarrassingly-parallel step, and it is the reason this
sample is more than a three-stage pipeline. Policy evaluation decomposes
perfectly:

- every episode is an **independent sample** of one fixed, frozen checkpoint;
- no shard reads another shard's output, and no shard writes a path another
  shard writes;
- there is no shared mutable state, no cross-shard communication, and no
  ordering requirement — only a final reduction.

So the step declares a `parameterSpace` over a shard index and Deadline Cloud
turns it into `EvalShards` independent tasks. `5 - Aggregate` then reduces them
into a single number.

The task count is deterministic and is purely a function of the parameter:

```
total episodes  = EvalShards × EvalEpisodesPerShard
total job tasks = 4 + EvalShards    # 1 Datagen + 1 Train + 1 Render + N Evaluate + 1 Aggregate
```

### Two axes of parallelism, and fan-out is the smaller one

Read this before quoting any speedup from this sample.

| Axis | Parameter | What it is | Published scale |
|------|-----------|------------|-----------------|
| **In-process** | `EvalNumEnvs` | Isaac Lab environments vectorized in one process on one GPU, sharing one tiled render pass | NVIDIA's Isaac Lab-Arena eval speedup is **40×** — 4096 envs on 8 GPUs. Their *eval-specific* guidance is **10–30 envs** (`IsaacLabEvalTasks`); `IsaacLab-Arena` ships `num_envs: 25`. |
| **Fleet** | `EvalShards` | Independent tasks on separate workers | **~5×** measured here at 8 shards — bounded by `scaleOutWorkersPerMinute` (default 10/min) scaling from zero, not by the rollout. |

They compose, and the honest framing is **vectorize the rollouts, distribute the
sweep**:

```
EvalShards × EvalNumEnvs = concurrent environments
8          × 12          = 96
```

**What genuinely does not vectorize is the *condition*, not the seed.** Seeds and
episodes batch fine — LeRobot's `lerobot_eval.py` takes `--eval.batch_size` and
assigns per-episode seeds inside one process, and `EvalNumEnvs` now does the same
here. But the `-DR-Eval` variant's dome light is at `prim_path="/World/sky_light"`,
a **stage-global** prim, while every other asset in the scene is under
`{ENV_REGEX_NS}`. Nominal and randomized lighting therefore cannot coexist in one
simulation at any `num_envs`. That is why `EvalRandomize` splits across *shards*
and the episodes do not. NVIDIA's own `IsaacLab-Arena` OSMO package independently
shards per **Run** and vectorizes the episodes within it.

### `EvalNumEnvs` is a correctness knob, not only a speed knob

Pin it for a whole campaign and quote it with every success rate. Three cited
reasons:

- NVIDIA on [issue #3505](https://github.com/isaac-sim/IsaacLab/issues/3505):
  "The current rendering pipeline is unfortunately **stochastic in nature**. We
  don't yet have a good way to guarantee determinism across the rendered
  outputs."
- [PR #1246](https://github.com/isaac-sim/IsaacLab/pull/1246): the tiled image is
  `(width × cols, height × rows)` with `cols = ceil(sqrt(N))`, and anti-aliasing
  switches between DLAA and DLSS-performance on the **combined** resolution — so
  "there may be visible quality differences when comparing render outputs of
  different numbers of environments."
- [Issue #1031](https://github.com/isaac-sim/IsaacLab/issues/1031), still **open**:
  "Increasing the number of parallel environments results in severe degradation in
  TiledCamera rendering quality."

For a sim-to-real *vision* policy the image **is** the observation, so a
parallelism knob that changes anti-aliasing changes the observation distribution.
Every shard therefore records `num_envs` in its JSON, and `5 - Aggregate`
**splits the success rate by `num_envs` and warns loudly** rather than averaging
shards that disagree — which is exactly what a stale `eval/shard_NN.json` from an
earlier run at a different `EvalNumEnvs` would otherwise cause, since `OutputDir`
is `dataFlow: INOUT`. **Use a fresh `OutputDir` per experiment.**

Why the default is **12**, and not 25 or 1: NVIDIA's eval band starts at 10, and
this task's observation spec is far heavier than the configs those numbers come
from — two 640×480 `TiledCamera`s with three annotators each (`rgb`, `depth`,
colorized `instance_id_segmentation_fast`) at `rendering_mode="quality"`, plus a
lightbox studio USD, an articulated SO-101, a mat, three vials and a rack **per
env**. The two published fits that bracket it disagree by ~4×: extrapolating
ManiSkill3's Isaac Lab 640×480 sweep gives ~38 envs on 24 GB, while Isaac Lab's
own COMPASS + NuRec empirical fit (`9 GB + 1.3 GB × num_envs`) gives ~10–11. A
service-managed fleet offering `a10g`/`l4` (both 24 GB) *and* `l40s` (48 GB) may
hand a shard any of them, so the default has to be safe on the smaller fit.
**Measure your own ceiling** — every shard logs
`[render-throughput] … env_steps_per_s=` and records `env_steps_per_second` in its
JSON, so a two-value A/B settles it in one submit.

### What the MP4 shows at `EvalNumEnvs > 1`

**Only env 0**, and only `ceil(EvalEpisodesPerShard / EvalNumEnvs)` episodes of
it. Recording all N would multiply the per-step GPU→CPU frame copy — the dominant
per-env host cost — by N, and a tiled grid of N shrunken views cannot show a 4 mm
vial tip, which is what the video is for. So **the video is a sample, not the
measurement**: the success rate is over *all* envs and all episodes. An MP4 of a
single failure next to a nonzero `success_rate` is expected, not a contradiction.
`shard_NN.json` records `video_env` and `video_episodes` so this is never left to
inference.

**How many of those N tasks run simultaneously is a separate question**, and it is
not something the template can guarantee. Given N workers that are already up, N
shards finish in roughly the wall-clock time of one; given one worker, they run
back-to-back and you get no speedup at all. The next section is about what
decides which of those you get.

Each shard gets its own seed (`EvalBaseSeed + (shard-1) × 1000`) and its own
domain-randomization condition, and writes:

| Path | Contents |
|------|----------|
| `OutputDir/eval/shard_NN.json` | Per-episode success flags — each tagged with its `batch`, `env` and final joint pose — plus that shard's seed, condition, task id, `num_envs`, rollout seconds and `env_steps_per_second`. |
| `OutputDir/eval/shard_NN.mp4` | **Env 0's** rollouts only (unless `EvalVideo=false`) — see "What the MP4 shows" above. |
| `OutputDir/eval_summary.json` | Written by Aggregate: overall success rate, `by_shard`, `by_condition`, **`by_num_envs`**, `num_envs_consistent`, episode count, summed rollout seconds and env-steps, video list. |

> **The video encoder is a property of the pinned container image, and a missing
> one fails silently.** `Mp4Writer` is deliberately non-fatal — a lost video must
> not cost you the success metrics — so if ffmpeg cannot encode, the shard logs one
> line and still exits 0 with correct metrics and **no MP4**. This is not
> hypothetical: an 8-shard run produced JSON + PNG for 8/8 shards and an MP4 for
> 0/8, because the image installs an **LGPL** FFmpeg build
> (`ffmpeg-...-lgpl-shared`, required since torchcodec links ffmpeg's shared
> libraries) and LGPL builds are compiled `--disable-libx264` — x264 is GPL.
>
> `render_rollout.py` therefore **probes** `ffmpeg -encoders` at runtime and takes
> the first of `libx264` → `libopenh264` → `mpeg4` that exists, with per-encoder
> quality flags (`libopenh264` has no `-crf`, so it uses `-b:v`). Verified inside
> the actual image rather than inferred: `libx264 present=False`,
> `libopenh264 present=True`, `mpeg4 present=True`, and piping rawvideo to
> `libopenh264 -b:v 4M -pix_fmt yuv420p -movflags +faststart` exits 0 producing
> `codec_name=h264 640x480 nb_frames=30`.
>
> If you rebuild the image with a different ffmpeg tarball, this can change again.
> **`eval_summary.json` always keeps its `videos` key** (an empty list, never a
> dropped key) and an empty list never fails Aggregate — so check that key, or grep
> a shard log for `ffmpeg`, rather than assuming videos exist.

Aggregate also prints a table to its task log, so the headline result can be read
straight off one log without downloading anything. This is **real output** from an
8-shard run on a service-managed GPU fleet:

```
=== Policy evaluation: 8 shards, 64 episodes ===

shard  condition   seed   episodes  successes  success%  mean steps
-----  ----------  -----  --------  ---------  --------  ----------
1      nominal     20001  8         0          0.0       900
2      randomized  21001  8         0          0.0       900
3      nominal     22001  8         0          0.0       900
4      randomized  23001  8         0          0.0       900
5      nominal     24001  8         0          0.0       900
6      randomized  25001  8         0          0.0       900
7      nominal     26001  8         0          0.0       900
8      randomized  27001  8         0          0.0       900

condition   shards  episodes  successes  success%
----------  ------  --------  ---------  --------
nominal     4       32        0          0.0
randomized  4       32        0          0.0

OVERALL SUCCESS RATE: 0/64 = 0.0%
```

The `nominal` vs `randomized` gap is the interesting number: a policy that scores
well nominal and badly randomized has not generalized.

> The 0% above is **correct and expected for that particular run** — it evaluated
> a checkpoint trained on an unrelated public dataset, purely to exercise the
> harness. `mean steps 900` is the episode-length cap, i.e. every episode ran to
> truncation without the success term ever firing. It is shown here rather than a
> flattering invented number because it is what the pipeline actually printed, and
> because it demonstrates the intended behaviour: **a 0% success rate is reported,
> not treated as a failure.** All 8 shards and the Aggregate step exited 0.

## What problem it solves

A policy trained on real-robot camera images cannot drive a simulator. The sim
renders don't look like the real world (the real→sim appearance gap), so the
policy flails. This pipeline closes that gap by training on images rendered from
the same simulator the policy will run in — including Isaac Sim's RTX
path-traced translucency, which is what makes a glass vial look like a glass
vial.

The five steps are wired with Open Job Description (OpenJD) `dependsOn`
dependencies and share one `OutputDir` that flows `INOUT`:

| Step | Reads | Writes | What it does |
|------|-------|--------|--------------|
| Datagen | — | `OutputDir/dataset/`, `OutputDir/probe_summary.json` | Scripted joint-space pick of a vial in Isaac Lab, standing it upright in a rack slot, recorded as a LeRobot v3.0 dataset. Each episode is verified by the environment's own `vial_placed_on_rack_termination()` predicate and discarded if it fails. |
| Train | `OutputDir/dataset/` | `OutputDir/train/`, `OutputDir/checkpoint/` | Finetunes a LeRobot ACT policy on the generated dataset (`lerobot-train`, CUDA). |
| Render | `OutputDir/checkpoint/` | `OutputDir/*.mp4`, `*.png`, `render_summary.json` | Drives the environment with the finetuned policy and records the result, plus a per-episode success flag. |
| Evaluate (×N) | `OutputDir/checkpoint/` | `OutputDir/eval/shard_NN.json`, `eval/shard_NN.mp4` | **Fans out.** `EvalShards` independent tasks each score the same checkpoint over `EvalEpisodesPerShard` episodes at their own seed and domain-randomization condition, with `EvalNumEnvs` environments vectorized in-process per task. |
| Aggregate | `OutputDir/eval/shard_*.json` | `OutputDir/eval_summary.json` | Reduces every shard into one success rate with per-shard and per-condition breakdowns, and prints a table to the task log. **Needs no GPU and no container.** |

Every step also writes `OutputDir/logs/` — the raw docker stream, the
in-container stdout, `_inner_exit_<phase>`, and `_marker_<phase>_NN` progress
files. Evaluate's are per-shard (`docker_eval01.log`, `eval01.log`,
`_inner_exit_eval01`, `_marker_eval01_NN_*`), so N concurrent shards do not
overwrite each other's trail.

Because the steps are independent and share the work directory, you can re-run a
single step — for example, re-render from a new camera without re-generating data
or re-training, or re-evaluate an existing checkpoint at a different shard width
without touching anything upstream.

## Why these design choices

- **Reuse the NVIDIA workshop, replace only what cannot run on a farm.** The
  environment (USD assets, lightbox, mat, both cameras, gripper contact sensors,
  all five domain-randomization terms, the gym registrations), `LeRobotRecorder`,
  `LeRobotSO101Interface`, and the success predicate are used **unmodified**.
  Two things are replaced, because they have no farm analogue:
  - *The action source.* The workshop's `lerobot_agent.py` reads a physical
    SO-101 leader arm over `/dev/ttyACM0` and gates recording on keyboard events
    acquired from `omni.appwindow`. A scripted waypoint expert takes its place
    and never calls `init_device()` / `connect()`.
  - *The process lifetime.* `lerobot_agent.py` ends in
    `while True: simulation_app.update()` with `simulation_app.close()`
    unreachable behind it. As-is that hangs a Deadline Cloud task forever.
- **Joint-space waypoints, not Isaac Lab's `lift_cube_sm.py`.** That state
  machine commands end-effector poses, i.e. it assumes a differential-IK action
  space. This environment uses `JointPositionActionCfg`, so the recorded
  `action` column is six absolute joint targets. Adopting an IK action space
  would change what `action` means, break parity with the MuJoCo sample, and
  break the real-SO-101 deployment story — the physical follower consumes joint
  positions.
- **Keep the workshop's vials-into-rack task; don't author a simpler cube
  scene.** A cube lift would be an easier scripted target, but it would mean
  shipping a new Isaac Lab extension (scene, events, terminations, gym
  registration, a new success predicate) and duplicating the workshop's
  lightbox / camera / mat setup — a fork of vendor content this repo would then
  have to maintain. Keeping vials reuses
  `vial_placed_on_rack_termination()` (grasp history + verticality + rack-local
  bounds + a 25-frame confirmation window) for free. **The tradeoff is real and
  is not hidden:** the vials task is materially harder to script, so expect a
  higher attempt-to-saved-episode ratio than MuJoCo's ~1.3, which is why
  `MaxAttemptFactor` defaults to 6. It is bought back two ways — the expert gets
  *privileged state* (exact vial and rack-slot poses read straight out of the
  scene, so alignment is analytic rather than perceptual), and the
  discard-and-retry loop converts a mediocre success rate into a clean dataset
  at the cost of wall-clock rather than data quality.
- **A container, not a conda environment.** The `isaacsim` wheels are
  `manylinux_2_35` (glibc ≥ 2.35) and Amazon Linux 2023 ships glibc 2.34, so
  `pip install isaacsim` cannot run on the default service-managed fleet worker
  OS — and glibc is the system loader, so conda cannot substitute it at runtime.
  The runtime is ~50 GB either way, so a conda package would win nothing on
  size, and NVIDIA recommends Docker for headless and cloud use.
- **All four GPU steps in the *same* container.** The image already carries LeRobot
  at the workshop's pinned commit with `lerobot-train` on `PATH`. Running Train
  in a separate conda environment would put a *different* LeRobot in front of the
  dataset than the one that wrote it, and different again from the one that loads
  the checkpoint at render time. That is precisely the config-schema drift that
  makes public checkpoints fail to load; one image makes it impossible.
- **Local inference, not the workshop's GR00T server.** The workshop's
  `lerobot_eval.py` is a ZMQ *client* of `run_gr00t_server.py`, which lives in a
  second container. On a worker that would mean a daemon-process job environment
  plus another ~50 GB image pull. Meanwhile `LeRobotSO101Interface` already
  contains a complete local LeRobot inference path (`make_policy`,
  `sim_obs_to_policy_processor`, `prediction_to_sim_processor`) that no workshop
  script calls. Render and Evaluate use it, so one image covers all four GPU
  steps. (`5 - Aggregate` deliberately uses no container at all.)
- **Frames are pushed every *other* control step.** One `env.step()` advances
  `sim.dt * decimation` = 1/60 s, but the recorder declares 30 fps. Recording
  every step would label 60 Hz data as 30 fps — a mislabelling the workshop's own
  teleop loop has, and one that would make every trained policy run at half
  speed on the real robot.
- **The recorder's queue is drained before exit.** `LeRobotRecorder.save_episode()`
  only *enqueues*; a **daemon** thread writes the parquet files. Daemon threads
  are killed at interpreter exit, so returning straight after the last episode
  would silently drop it. Datagen polls `num_recorded_episodes` with a timeout
  and fails the task if fewer episodes reached disk than were verified.
  (`queue.join()` is not used: the processor swallows exceptions with `continue`
  and skips `task_done()`, so `join()` can hang forever.)
- **Exit codes: `simulation_app.close()`, then `sys.exit(code)`.** Passing a code
  to `close(exit_code=N)` is unreliable — Kit's fast-shutdown path can reset it to
  0, which would make a failed task report success. The code then propagates
  through two `tee` pipelines via `PIPESTATUS[0]` and is also written to
  `logs/_inner_exit_<phase>`.
- **The container ENTRYPOINT is kept, not replaced with `--entrypoint bash`.**
  That override is the documented workaround for the plain
  `nvcr.io/nvidia/isaac-sim` image, whose entrypoint swallows stdout. This
  image's entrypoint is the workshop's: it sources `setup_python_env.sh`, sets
  `CARB_APP_PATH` / `ISAAC_PATH` / `EXP_PATH`, installs the `python` shim that
  forwards to Isaac Sim's `python.sh`, and ends in `exec "$@"`. It does not
  swallow stdout, and dropping it makes every Isaac Lab import fail.
- **`docker run` boilerplate lives in one shell script, not four step bodies.**
  [`scripts/run_in_container.sh`](scripts/run_in_container.sh) carries the GPU
  flags, EULA env vars, cache mounts, permission fixes, inner timeout, and
  exit-code plumbing once. Each step passes the command to run after `--`, as an
  argv array that is never re-parsed by a shell. It is invoked as
  `bash run_in_container.sh` and never relies on the execute bit, because
  Deadline Cloud's asset staging can strip `+x` from job-attachment files.
- **Evaluate reuses `render_rollout.py`; it is not a second rollout loop.** The
  Evaluate step calls the same script the Render step does, with
  `--summary-path` / `--video-path` / `--shard` / `--condition` added. A separate
  `evaluate_policy.py` would need its own copy of two hard-won details and would
  drift from them: the `torch.no_grad()`-not-`torch.inference_mode()` rule (see
  below — it fails on *episode 1 of every multi-episode run*, which is precisely
  what an eval shard is), and the `BaseException` + 45 s watchdog + `os._exit()`
  guard that stops a Kit exception from wedging the task until
  `StepTimeoutSeconds`. At fan-out width both matter N times over.
- **One `parameterSpace` dimension, 1-based, not `CHUNK[INT]`.** The shard index
  is `type: INT`, `range: '1-{{Param.EvalShards}}'` — the same idiom as
  [`tile_render_with_maya_arnold`](../tile_render_with_maya_arnold)'s
  `'1-{{Param.NumXTiles}}'`. It is 1-based because OpenJD 2023-09 has no
  arithmetic in format strings, so a 0-based space would need an
  `EvalShards - 1` that cannot be written; 1-based makes the task count exactly
  `EvalShards`. This sample deliberately does **not** use `CHUNK[INT]` and the
  `TASK_CHUNKING` extension the way
  [`monte_carlo_simulation`](../monte_carlo_simulation) does: chunking exists to
  *batch together* tasks that are individually too small, and it picks the batch
  size at runtime from an observed runtime target. Here the batch size is the
  physically meaningful quantity, it is chosen for the startup-cost reason
  documented under "Sizing the shards", and it is already an explicit parameter.
  Chunking would also let the scheduler collapse the fan-out into fewer, wider
  tasks — the opposite of what this step is for.
- **The domain-randomization condition is derived from shard parity, not a second
  parameter-space dimension.** A second dimension would make the task count
  `shards × conditions`, so `EvalShards` would no longer mean "how many parallel
  tasks". Instead `EvalRandomize=alternate` gives odd shards the nominal `-Eval`
  variant and even shards the randomized `-DR-Eval` one, which still yields a
  real per-condition breakdown from a single-dimension space of exactly N tasks.
- **Every shard writes distinct filenames, and no shard ever clears `eval/`.**
  This is what makes the fan-out safe. Per-shard paths mean the job-attachment
  output manifests of N concurrent tasks merge without conflict. And an
  `rm -rf OutputDir/eval` inside a shard would be actively destructive: shards
  run concurrently, and because `OutputDir` is `dataFlow: INOUT` each worker holds
  a local copy of whatever its peers have already uploaded. Stale files from an
  earlier, wider run are filtered by Aggregate instead — the reduce step is the
  only place that can distinguish "stale" from "not finished yet".
- **Aggregate declares no GPU and does not enter the container.** It reads a
  handful of small JSON files, so it runs `aggregate_eval.py` on the worker's own
  `python3` — stdlib-only and Python 3.9-safe, which is what Amazon Linux 2023
  ships. That skips the ~9 GB image pull entirely on a step whose actual work is
  milliseconds. Be clear about what dropping `amount.worker.gpu` buys, though:
  `hostRequirements` are a compatibility **filter**, not a request, so on a queue
  whose only fleet is the GPU fleet this step still lands on a GPU worker. The
  win is portability — associate a cheap CPU fleet with the queue and Aggregate
  schedules there while Evaluate cannot. Same shape as
  [`vllm_lm_eval_leaderboard`](../vllm_lm_eval_leaderboard)'s aggregate step.
- **A low success rate is a result, not a task failure.** Evaluate never fails a
  task over a bad score. Under fan-out this is load-bearing rather than merely
  tidy: one shard failing would cancel Aggregate by dependency and destroy the
  run's headline number.
- **Free-text parameters go to data files, never into the shell.** OpenJD
  substitutes `{{Param.*}}` as literal text *before* bash parses the line, so
  interpolating a value directly into a script — even inside quotes — lets a
  value containing a quote, `$(...)`, or a backtick execute on the worker, and a
  benign apostrophe breaks the script. Values are written to `embeddedFiles` and
  read back with `$(cat ...)`. This matters more here than in the MuJoCo sample,
  because these values go on to form a `docker run` command line. `INT` /
  `FLOAT` parameters and `STRING` parameters constrained by `allowedValues` are
  interpolated directly — the schema already guarantees they cannot contain shell
  metacharacters.

## Prerequisites

1. **A Deadline Cloud farm and queue with a Linux x86_64 GPU fleet** whose
   workers have Docker and the NVIDIA Container Toolkit. Deadline Cloud
   service-managed Linux GPU fleets have both. Raise the fleet's root volume to
   at least **500 GiB** — the image is ~50 GB and the shader cache adds ~10 GB.
2. **Build the container image and push it to a registry you control.**

   ```bash
   cd ../../containers/isaacsim-so101-workshop
   docker build -t isaacsim-so101-workshop:2.3.2 .
   ```

   Then push to your own private Amazon ECR repository — see
   [`containers/isaacsim-so101-workshop/README.md`](../../containers/isaacsim-so101-workshop/README.md)
   for the exact commands and the fleet-role permissions.

   > **This bundle does not, and cannot, ship a working default image.** The
   > built image contains Omniverse Kit, which may not be redistributed, so
   > there is no public URI to point at. The base image
   > `nvcr.io/nvidia/isaac-lab:2.3.2` is anonymously pullable from NVIDIA — no
   > NGC account or API key needed — so building it yourself is a one-command
   > prerequisite, not a licensing negotiation. Keep the built image private.
3. **The Deadline Cloud CLI** configured (`deadline config show` resolves your
   default farm and queue).

A [Conda queue environment](https://docs.aws.amazon.com/deadline-cloud/latest/developerguide/conda-queue-environment.html)
is *not* required. The container carries the whole Python runtime, so
`CondaPackages` defaults to empty; the parameter exists only so this bundle
still submits cleanly to a queue that has one attached.

## Calibrate the scripted expert first

The scripted expert reaches positions with a damped-least-squares servo whose
Jacobian is measured by finite differences, so it needs **no link lengths and no
analytic IK**. What it does need is a handful of geometric constants that can
only be measured against the actual SO-ARM101 USD on a GPU.

**Run the probe first.** It boots the environment, resets once, prints the joint
names and limits, the home pose, the end-effector frame pose and quaternion,
every body position, the vial / rack / slot poses, and a Jacobian self-test that
commands a known 1 cm displacement and reports the measured one — then exits
without recording:

```bash
OUT="$(pwd)/output"; mkdir -p "$OUT"
deadline bundle submit isaacsim_so101_sim_to_policy \
  -p "ContainerImage=<your-ecr-uri>" -p EcrLogin=true \
  -p ProbeOnly=true -p "OutputDir=$OUT" --yes
```

Read `OutputDir/probe_summary.json` and `OutputDir/logs/datagen.log`, then set:

| Parameter | How to read it off the probe |
|-----------|------------------------------|
| `GraspZOffset` | The z gap between the `gripper` body origin (which *is* the end-effector frame) and the pinch point between the jaws. |
| `GripperPitchOffset` | The correction to the home gripper pitch that points the jaw at the mat. The expert holds `Pitch + Elbow + Wrist_Pitch` constant, which holds gripper pitch constant, so this one scalar is the whole grasp-orientation calibration. |
| `ReorientOffset` | The gripper pitch change that stands a lying vial upright. The vials spawn horizontal but the success term requires `abs(vial up z) > 0.7`, so this rotation is what makes the task solvable at all. The sign depends on the gripper's frame convention. |
| `JawOpen` / `JawClosed` | From the `Jaw` joint limits. |

Uncalibrated, Datagen discards every attempt and **exits non-zero** rather than
writing a bad dataset. That failure mode is deliberate: a miscalibrated expert
produces a failed task, never a garbage dataset that Train would silently
finetune on.

Finer tuning (`--hover-height`, `--lift-height`, `--insert-clearance`,
`--servo-tol`, `--servo-step-m`, `--servo-damping`, `--fd-delta`,
`--jacobian-refresh`) is available on
[`scripts/generate_dataset.py`](scripts/generate_dataset.py) but is not surfaced
as job parameters; the defaults are reasonable and promoting them would clutter
the submitter UI.

## Submit

```bash
# Submit with an absolute, known output path (so outputs upload correctly):
OUT="$(pwd)/output"; mkdir -p "$OUT"
deadline bundle submit isaacsim_so101_sim_to_policy \
  -p "ContainerImage=<account>.dkr.ecr.<region>.amazonaws.com/isaacsim-so101-workshop:2.3.2" \
  -p EcrLogin=true \
  -p "OutputDir=$OUT" \
  -p EvalShards=8 -p EvalEpisodesPerShard=12 --yes
```

That submits `4 + 8 = 12` tasks, of which eight are Evaluate shards that all
become schedulable at once. How many actually run concurrently is
`min(EvalShards, workers the fleet has running)` — **match `EvalShards` to the
worker count you expect** for the shortest wall clock. More shards than workers
still works, they just queue.

Or review parameters in a GUI before sending:

```bash
deadline bundle gui-submit isaacsim_so101_sim_to_policy
```

Watch progress and collect the videos and the aggregate number:

```bash
deadline job get --job-id <job-id>
deadline job download-output --job-id <job-id>
cat "$OUT/eval_summary.json"          # overall + per-shard + per-condition
ls "$OUT/eval/"                       # shard_NN.json + shard_NN.mp4 grid
```

Or just read the `5 - Aggregate` task log, which contains the same table.

### Recommended first real run: Train + Render on an existing dataset

Datagen is the only step that depends on the calibration above. Train and Render
do not. So the fastest way to prove the image, the fleet, the GPU wiring, and
two thirds of the pipeline is to **skip Datagen** and run Train + Render against
a LeRobot v3.0 dataset you already have — for example one of the workshop
course's published SO-101 teleop datasets, or a dataset from the MuJoCo sample:

```bash
OUT="$(pwd)/output"; mkdir -p "$OUT/dataset"
# place a LeRobot v3.0 dataset (meta/info.json + data/ + videos/) in $OUT/dataset
deadline bundle submit isaacsim_so101_sim_to_policy \
  -p "ContainerImage=<your-ecr-uri>" -p EcrLogin=true \
  -p "DatasetRepoId=<the dataset's repo id>" \
  -p "Instruction=<the instruction it was recorded with>" \
  -p "OutputDir=$OUT" --yes
```

Then mark the Datagen step **`SUCCEEDED`** — do **not** cancel it:

```bash
aws deadline update-step --farm-id "$FARM" --queue-id "$QUEUE" --job-id "$JOB" \
  --step-id "<datagen-step-id>" --target-task-run-status SUCCEEDED
```

> **Do not cancel Datagen.** Cancelling (or failing) a step transitively cancels
> everything downstream of it, so `Train` and `Render` would be `CANCELED` too and
> the whole job would end up doing nothing. Per the Deadline Cloud docs: "If StepA
> fails, or if StepA is canceled, StepB moves to the CANCELED state." `SUCCEEDED`
> is the only status that *resolves* a dependency edge and lets `Train` start.

`Instruction` and `DatasetRepoId` must match the dataset, because ACT is
conditioned on the instruction and `lerobot-train` resolves the dataset by repo
id.

Also note `OutputDir` is declared `dataFlow: INOUT`, so it is uploaded as an
**input** as well as collected as an output. For a Train- or Render-only run,
whatever the earlier step produced has to exist in your local `OutputDir` before
you submit — run `deadline job download-output --job-id <previous-job>` first, or
the step will fail its own "no dataset" / "no checkpoint" guard.

## Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `Instruction` | `pick up the vial and place it in the rack` | Recorded on every frame, conditions training, and passed to the policy at render and evaluation time. One parameter for every step that uses it (Datagen, Render, Evaluate) so they cannot drift. |
| `Episodes` | `50` | Target count of **verified** episodes. |
| `MaxAttemptFactor` | `6` | Datagen gives up after `Episodes × this`. |
| `Randomize` | `true` | Selects the `-DR-Eval` task for Datagen (robot colour, HDRI sky light, mat rotation, camera focal length, camera pose). Render always uses the non-randomized variant. |
| `Fps` | `30` | Dataset and video frame rate. |
| `EpisodeLengthSeconds` | `15.0` | Overrides the env cfg. The workshop's `-Eval` variants ship 7.5 s, too short for reach + grasp + reorient + insert + the 25-frame confirmation. |
| `Seed` | `101` | Environment seed. |
| `DatasetRepoId` | `local/so101_isaac_vials` | LeRobot repo id in the dataset metadata. |
| `ProbeOnly` | `false` | Run Datagen in calibration-probe mode and record nothing. |
| `GraspZOffset` | `0.0` | **Needs calibration.** See above. |
| `GripperPitchOffset` | `0.0` | **Needs calibration.** |
| `ReorientOffset` | `1.5708` | **Needs calibration** (sign especially). |
| `JawOpen` / `JawClosed` | `0.6` / `-0.1` | **Needs calibration.** |
| `TrainSteps` | `20000` | ACT finetuning steps. |
| `TrainBatchSize` | `8` | Training batch size. |
| `ActNActionSteps` | `20` | ACT actions executed open-loop per observation. Below `chunk_size` (100) to run closed-loop. |
| `RenderEpisodes` | `3` | Policy rollouts, all in one MP4. |
| `RenderCamera` | `external_D455` | `external_D455` is the third-person lightbox view; `ego` is the wrist camera. Also used by Evaluate. |
| `EvalShards` | `8` | **Number of Evaluate tasks created.** Total job tasks = `4 + EvalShards`, deterministically. Observed *concurrency* is a separate matter — capped by `min(EvalShards, running workers)` and by whether the tasks last long enough for autoscaling to react. More shards than workers still works, they just queue. |
| `EvalEpisodesPerShard` | `12` | Episodes per shard. **Keep it chunky** — see "Sizing the shards" below. Total episodes = `EvalShards × EvalEpisodesPerShard`. Keep it a multiple of `EvalNumEnvs`. |
| `EvalNumEnvs` | `12` | **Environments each shard simulates in parallel, in one process, on one GPU** — the *other* parallelism axis, and on published numbers the bigger one. Episodes run in `ceil(EvalEpisodesPerShard / EvalNumEnvs)` batches, the scheme LeRobot's own `lerobot_eval.py` uses. **Pin it and report it** — see "Two axes of parallelism" below. |
| `EvalDeterministicScene` | `false` | Pin *every* reset-event range (object poses, light exposure, mat yaw, focal length, camera pose, sky light) so every env in every episode starts from an identical scene. The control for an `EvalNumEnvs` A/B; not for a real measurement, because every episode becomes the same episode. |
| `EvalBaseSeed` | `20001` | Seed of shard 1; shard *i* gets `EvalBaseSeed + (i-1) × 1000`. Deliberately distinct from `Seed`: evaluating on the seeds the data was generated from measures memorisation, not generalisation. |
| `EvalRandomize` | `alternate` | `alternate` = odd shards nominal, even shards randomized, so one submit yields both numbers and the gap between them. `on` / `off` put every shard in one condition. Prefer an **even** `EvalShards` so the two conditions get equal episode counts. |
| `EvalVideo` | `true` | Write `eval/shard_NN.mp4` per shard. `false` skips the encode *and* the per-step GPU→CPU frame copy that feeds it — the cheapest way to speed up a wide fan-out when only the success rate matters. |
| `ContainerImage` | `isaacsim-so101-workshop:2.3.2` | **No working default is possible** — see Prerequisites. |
| `EcrLogin` | `false` | `docker login` to Amazon ECR before pulling. The region is parsed from the image URI. |
| `ShaderCacheDir` | `/mnt/deadline-persistent/isaacsim-so101-cache` | Host path for the Omniverse / pip / torch-hub caches, reused across tasks on the same worker. Defaults to a **fleet persistent volume** so it survives worker replacement; falls back to `/tmp` with a warning if the path is not writable, so it is safe on fleets without one. |
| `StepTimeoutSeconds` | `5400` | In-container wall-clock budget, **shared by all steps**. Keep below the queue's task timeout so the container exits cleanly and outputs still upload. Raise it if you raise `EvalEpisodesPerShard` — see below. |
| `CondaPackages` | `''` | Intentionally empty; the container carries the runtime. |
| `CondaChannels` | `deadline-cloud` | Only read by a Conda queue environment, if one is attached. |
| `OutputDir` | `output` | Shared work dir. Pass an **absolute** path on submit. |
| `JobScriptDir` | `scripts` | Hidden. Holds the two Python entrypoints and `run_in_container.sh`, staged as a job-attachment input. |

## Sizing the shards, and what the first fan-out costs

### Shards must be chunky. Do not set `EvalEpisodesPerShard=1`.

The instinct with an embarrassingly parallel problem is to make the unit of work
as small as possible. That is wrong here, because **every task pays a large fixed
startup cost that an episode does not amortize**:

| Cost | Cold | Warm |
|------|------|------|
| Container image pull (~9 GB) | ~4 min | seconds (already in `/var/lib/docker`) |
| Omniverse shader / Kit boot | **347 s** | **35 s** (measured, 9.9×) |
| One episode of rollout | ~15 s of sim | ~15 s of sim |

An episode is ~15 seconds. A cold task's startup is ~10 minutes. So one-episode
shards would spend the overwhelming majority of every task's wall clock on
startup — the fan-out would look spectacular in the monitor while delivering
almost no throughput, and would cost far more GPU-minutes per episode than a
serial run.

The default `EvalEpisodesPerShard=12` puts a *warm* shard at roughly 3–6 minutes
of rollout against ~35 s of boot, i.e. startup is under ~20% of the task. Rules
of thumb:

- **Raise it** if your shards are startup-dominated (compare `eval NN.log`'s Kit
  boot time against the rollout time).
- **Lower it** only when you are demonstrating *width* rather than measuring a
  policy.
- **Widen `EvalShards`, not the episode count**, to get a shorter wall clock —
  that is the knob that buys parallelism. `EvalEpisodesPerShard` buys efficiency.

Also note `StepTimeoutSeconds` (default `5400`) is **shared by every step**, and
it is the *in-container* budget for one task. A large `EvalEpisodesPerShard` can
trip it: the container then exits on its own and the task reports exit code
`124` (or `137` after the `--kill-after` SIGKILL). Budget roughly
`boot + EvalEpisodesPerShard × episode wall-clock` and raise
`StepTimeoutSeconds` if that approaches 5400 s.

### What actually bounds observed concurrency

Creating N tasks is not the same as running N tasks at once. Three things bound
how wide the fan-out actually gets, and only the first is under the template's
control:

1. **`EvalShards`** — the number of tasks created. Deterministic.
2. **Running workers** — concurrency is at most `min(EvalShards, workers up)`.
   `maxWorkerCount` is the ceiling on that, not a promise; a service-managed
   fleet scales up towards it but starts from whatever is currently running.
3. **Task duration** — autoscaling needs time to react. **A worker cannot boot,
   pull a ~9 GB image and join the fleet faster than a short task finishes**, so
   if tasks are short the existing worker simply drains the queue alone.

Point 3 is counter-intuitive and worth stating plainly: **short tasks do not just
fail to demonstrate parallelism, they actively hide it.**

#### Measured: 8 shards, 8 workers — cold run and warm run

Two runs of the **identical** configuration (`EvalShards=8`,
`EvalEpisodesPerShard=8`, same checkpoint) on a service-managed **Spot** GPU fleet
with `maxWorkerCount=8`, each starting from a scaled-down fleet. Run 2 followed run
1, so its persistent volumes were already populated. Both achieved full width:

| | Run 1 (cold caches) | Run 2 (warm caches) |
|---|---|---|
| Peak distinct workers | **8** | **8** |
| **MAX SIMULTANEOUS TASKS** | **8** | **8** |
| Evaluate step wall-clock | 1479 s (24.6 min) | **1040 s (17.3 min)** |
| Sum of per-shard durations | 7314 s (121.9 min) | 5631 s (93.8 min) |
| **Effective speedup** | **4.95×** | **5.42×** |
| Mean Kit boot (in-container) | 215 s | **52 s** (4.09× faster) |
| Mean rollout (8 episodes) | 377 s | 376 s |
| `shader cache:` reported | **COLD 8/8** | **WARM 8/8** |
| Container image | fresh pull 8/8 | fresh pull 8/8 |
| Scale-up stagger (shard 1→8 start) | 585 s (~9.8 min) | **344 s (~5.7 min)** |
| Spot reclamations / retries | 0 | 0 |

Run 1's per-shard detail, showing the staggered ramp that costs the speedup:

```
shard  start     end       secs   boot   worker (suffix)
-----  --------  --------  ----   ----   ---------------
1      14:02:20  14:19:42  1042   284s   ...5bfafd7a
2      14:06:35  14:22:07   932   201s   ...d779b08a
3      14:08:27  14:24:08   941   202s   ...65efa0c15
4      14:10:25  14:24:48   863   203s   ...85b4abb4
5      14:10:56  14:25:12   857   200s   ...a18708db
6      14:11:42  14:26:48   906   213s   ...4f4739978
7      14:11:56  14:26:35   879   201s   ...91668321c
8      14:12:05  14:26:59   893   213s   ...6fbabb71d
```

**Why 5.42× and not 8×, even warm.** Look at the start column: the shards do not
start together. The fleet scales up one instance at a time, so shard 8 begins ~5.7
minutes (warm) or ~9.8 minutes (cold) after shard 1. That ramp — not the rollout —
is the entire gap to the theoretical 8×. On a fleet whose 8 workers are *already
running*, the same job should land close to 8×.

**The controlled variable is boot, and only boot.** Mean rollout time was 377 s vs
376 s across the two runs — statistically identical, as it must be, since the same
checkpoint ran the same 8 episodes. The whole 24.6 → 17.3 min improvement comes
from warm shader caches cutting mean boot from 215 s to 52 s. That is a clean
demonstration that `ShaderCacheDir` is doing what it claims.

**Two honest caveats on these numbers:**

- **The container image was re-pulled on all 8 workers in *both* runs.** Docker
  stores images under `/var/lib/docker` on the instance root volume, *not* on the
  persistent volume, so "warm" never includes the ~9 GB pull whenever the fleet has
  created new instances. This is why run 2's wall-clock did not improve as much as
  its 4× boot improvement suggests.
- **One shard in run 2 was an outlier: 193 s boot against ~30 s for the other
  seven**, despite reporting `WARM` and running on an identical GPU and driver
  (`NVIDIA L4`, `580.159.03` — checked, because a GPU-architecture mismatch would
  have been a plausible cause and turned out not to be it). Cause not determined.
  Expect this kind of per-instance variance and prefer medians over single-shard
  timings.

Reproduce these tables with `list-sessions` + `list-session-actions`; note the
taskId lives at `definition.taskRun.taskId`, not at the top level of a session
action, and **dedupe to one interval per shard** before computing overlap or a
Spot-retried shard will read as phantom concurrency.

A measured example. A `ProbeOnly=true` run of this exact template
(`job-aac7a87bf6754c8285e46f9e995dcef1`, 3 shards) succeeded on all 7 tasks, and
the fan-out produced exactly the expected `4 + 3 = 7` tasks — but every task ran
on **one** worker, strictly sequentially:

```
sessions: 5        distinct workers: 1
4 - Evaluate   shard=1   23:05:52 → 23:05:55   3s
               shard=2   23:05:55 → 23:05:58   3s
               shard=3   23:05:59 → 23:06:01   3s
3 - Render               23:06:10 → ...        (started only after Evaluate finished)
1 - Datagen                                    389 s  (cold Kit boot)
```

Under `ProbeOnly=true` every downstream task hits its `exit 0` guard and lasts
~3 seconds, so there was never any queue depth for the fleet to scale into. Note
also that `3 - Render` became READY at the same time as the Evaluate shards — the
dependency graph was correct — and a single worker still serialised it after them.

So that run proves the **DAG and the task count**, and proves nothing whatsoever
about parallelism. To observe real concurrency you need `ProbeOnly=false`, shards
long enough to keep a queue non-empty while workers boot (minutes, not seconds),
and a fleet that is either already wide or given time to widen.

Measure it from the service, not from a stopwatch:

```bash
# distinct workers that ran the job, and per-task start/end
aws deadline list-sessions --farm-id "$F" --queue-id "$Q" --job-id "$J" \
  --query 'sessions[].{worker:workerId,start:startedAt,end:endedAt}' --output table
```

If `distinct workers == 1`, there was no parallelism regardless of how fast the
job looked.

### The first wide fan-out is mostly cold

`ShaderCacheDir` defaults to a path on a **fleet persistent volume**, which is what
turns a multi-minute Kit boot into seconds. But persistent volumes are **per-AZ and
attached exclusively to one worker at a time**, so a *first* wide fan-out has no
warm volumes to inherit and every worker pays the full cold shader compile, on top
of a ~4 minute image pull.

Measured, on the two runs above:

- **Run 1 (first wide fan-out): `shader cache: COLD` on 8/8 shards**, mean boot
  215 s. Exactly as predicted — nothing was warm.
- **Run 2 (same config, immediately after): `WARM` on 8/8**, mean boot 52 s.
- The container image was **re-pulled on 8/8 workers in both runs.** The image is
  *never* on the persistent volume — Docker keeps images under `/var/lib/docker` on
  the instance root volume — so any newly created worker re-pulls regardless of how
  warm its cache is.

> ### Rule: warm up at the width you intend to run
>
> **The number of workers that can come up warm is bounded by the widest previous
> run, not by 1.** Exclusive attachment means a volume serves one worker *at a
> time* — it does **not** mean only one worker can ever be warm. Run 1 created 8
> volumes; when run 2 scaled back up to 8 workers those 8 volumes were reattached
> and **all 8** came up warm.
>
> The practical consequences:
>
> - **A narrow warm-up does not warm a wide fan-out.** An `EvalShards=1` warm-up
>   run populates exactly one volume, so a subsequent 8-wide run gets 1 warm worker
>   and 7 cold ones. To warm 8 workers you must previously have *run* 8 workers.
> - **Warm up with the same `EvalShards` you intend to demo**, and accept that the
>   warm-up run itself pays the full cold cost.
> - Raising `maxWorkerCount` beyond your last run's width re-introduces cold
>   workers at the margin: the new instances have no volume to inherit.
>
> (An earlier draft of this README claimed "at most one worker inherits an
> already-warm volume". Run 2 disproved it; the rule above replaces it.)

**Consequence: measure the speedup on the second run, not the first.** Run 1 of a
wide fan-out is dominated by N cold starts. For a demo, submit the same job once as
a warm-up so the volumes exist and are populated at the width you intend to show —
a narrow `EvalShards=1` warm-up is *not* enough, because it only warms one volume.

This is also why `maxWorkerCount` matters more than it looks: it is a scaling
knob and raising it does **not** replace workers, but every *newly created*
worker is a cold start.

### Suggested starting points

| Goal | `EvalShards` | `EvalEpisodesPerShard` | `EvalVideo` | Notes |
|------|--------------|------------------------|-------------|-------|
| Smoke-test the harness | 3 | 2 | `true` | Cheapest run that proves the wiring and aggregation. **Will not show parallelism** — the shards are too short for the fleet to scale into, and will likely all land on one worker. That is expected; do not read it as a failure of the fan-out. |
| Demo the fan-out | = workers you expect to be **running** | 10–12 | `true` | Shards long enough (minutes) that queue depth persists while workers boot, so the fleet actually widens. Warm the fleet first, and confirm afterwards with `list-sessions` that distinct workers > 1. |
| Actually measure a policy | 8–16 | 25+ | `false` | 200–400 episodes gives a success rate with a usable confidence interval; skipping video removes the per-step frame copy. Check `StepTimeoutSeconds`. |

Keep `EvalShards` **even** whenever `EvalRandomize=alternate`, so the nominal and
randomized conditions get equal episode counts and the gap between them is a fair
comparison.

## Validate the bundle

```bash
openjd check   template.yaml
openjd summary template.yaml -p OutputDir=output
```

`openjd summary` is the quickest way to confirm the fan-out width before spending
any GPU time. Task count is always `4 + EvalShards`:

```bash
$ openjd summary template.yaml -p OutputDir=output -p EvalShards=3 | grep Total
Total steps: 5
Total tasks: 7

$ openjd summary template.yaml -p OutputDir=output -p EvalShards=16 | grep Total
Total steps: 5
Total tasks: 20
```

You can also dry-run a single shard's shell logic locally without a GPU by
pinning the task parameter:

```bash
openjd run template.yaml --step "4 - Evaluate" -tp ShardIndex=2 \
  -p OutputDir=output -p EvalShards=4
```

## Running a subset

The five steps share the work directory, so you can run part of the flow. Be
aware of the mechanics first: **a job bundle always submits all of its steps** —
`dependsOn` controls the order, not whether a step runs. OpenJD 2023-09 has no
conditional-step construct, and `CreateJob` has no per-step targeting (its
`targetTaskRunStatus` is job-wide and only accepts `READY` or `SUSPENDED`). So
"run one step" means one of:

1. **Copy the bundle and delete the steps you don't want** — remembering to also
   remove any now-dangling `dependsOn` entries, or the template fails validation
   with `Unknown step '1 - Datagen'`. This is the only approach that truly does
   not submit the other steps.
2. **Submit the whole job, then mark the steps you want skipped as `SUCCEEDED`**
   via `aws deadline update-step` (see the Train-only example above). Never
   *cancel* them — cancelling propagates to everything downstream.

   If you are skipping *several* steps, **space the calls out**. Marking a step
   `SUCCEEDED` starts an asynchronous update that also touches its dependents, so
   a second `update-step` issued immediately fails with:

   ```
   ConflictException: Failed to update tasks in step-... to 'SUCCEEDED'
   because another update is in progress.
   ```

   Wait for the next step to reach `READY` (a few seconds) before marking it, or
   retry on `ConflictException`.

The useful subsets:

- **Probe only** — `-p ProbeOnly=true`. Datagen prints calibration data and
  records nothing, and Train, Render, **every Evaluate shard** and Aggregate then
  skip themselves with success. The job comes back green and no step surgery is
  needed. (Earlier revisions let the downstream steps fail on their missing-input
  guards, which marked the whole job FAILED, retried an unclearable error
  `maxRetriesPerTask` times, and made a working probe look broken. This matters
  most for Evaluate: without the guard a probe run becomes `EvalShards` failed
  tasks, each retried `maxRetriesPerTask` times against an error that can never
  clear.)
- **Re-render only** — once `OutputDir/checkpoint/` exists locally, mark Datagen
  and Train `SUCCEEDED` and let Render run (e.g. `-p RenderCamera=ego`).
- **Re-train only** — mark Datagen `SUCCEEDED` and run Train + Render against an
  existing `OutputDir/dataset/`.
- **Datagen only** — submit normally and let Train fail on the missing
  checkpoint, or use approach 1 if you want a clean job.
- **Evaluate + Aggregate only** — the most useful subset, and the one worth
  knowing by heart. See below.

### Evaluate an existing checkpoint standalone

Evaluate needs nothing but `OutputDir/checkpoint/`, so you can score a checkpoint
you already have without re-generating data or re-training. This is how to
iterate on shard width, seeds, or the randomization condition cheaply.

Put the checkpoint at `OutputDir/checkpoint/` locally first — `OutputDir` is
`dataFlow: INOUT`, so it is uploaded as an *input*, and Evaluate's guard fails
fast if `checkpoint/config.json` is absent:

```bash
OUT=/path/to/work            # must already contain checkpoint/config.json + weights
deadline bundle submit isaacsim_so101_sim_to_policy \
  -p "ContainerImage=<your-ecr-uri>" -p EcrLogin=true \
  -p "OutputDir=$OUT" \
  -p EvalShards=4 -p EvalEpisodesPerShard=12 \
  --yes
```

Then mark `1 - Datagen` **`SUCCEEDED`**, wait for `2 - Train` to reach `READY`,
and mark it `SUCCEEDED` too. Render, the four Evaluate shards and Aggregate then
run:

```bash
FARM=...; QUEUE=...; JOB=...
aws deadline update-step --farm-id "$FARM" --queue-id "$QUEUE" --job-id "$JOB" \
  --step-id "<datagen-step-id>" --target-task-run-status SUCCEEDED
# WAIT for '2 - Train' to become READY before the next call, or retry on
# ConflictException -- marking a step SUCCEEDED starts an async update that also
# touches its dependents, and a second immediate call is rejected.
aws deadline update-step --farm-id "$FARM" --queue-id "$QUEUE" --job-id "$JOB" \
  --step-id "<train-step-id>" --target-task-run-status SUCCEEDED
```

Two things to get right:

- **Never mark a step `CANCELED` to skip it.** Cancel propagates transitively, so
  it would kill Render, all N Evaluate shards and Aggregate.
- **If the checkpoint was trained elsewhere, set `PolicyCameraRenameMap`.** A
  checkpoint is tied to the camera keys it trained on, and the map is indexed
  directly per camera so it must cover *every* active camera or the shard dies
  with a `KeyError`. For a checkpoint trained on `lerobot/svla_so101_pickplace`:

  ```bash
  -p 'PolicyCameraRenameMap={"ego": "up", "external_D455": "side"}'
  ```

  Evaluate plumbs this through exactly as Render does. Expect `success=False`
  everywhere from a checkpoint trained on unrelated data — that exercises the
  harness, not the policy.

To skip Render as well (it is not on Evaluate's dependency path, so it only
competes for workers), mark it `SUCCEEDED` too — same spacing rule.

In every case, because `OutputDir` is `dataFlow: INOUT`, the artifacts a skipped
step would have produced must already be present locally before you submit —
`deadline job download-output` from the previous job first.

> Note: the `Datagen` step regenerates from scratch each time it runs — it
> clears `OutputDir/dataset/` first, because `LeRobotRecorder.init_dataset()`
> *appends* to an existing dataset root and would otherwise silently mix runs.
> To finetune on a dataset you supply yourself, skip `Datagen` and place your
> dataset at `OutputDir/dataset/` before running `Train`.

## Watching a rollout locally in a visible viewport

The farm only ever produces an MP4. To watch the policy drive the arm live — on a
GPU workstation, or a cloud box you reach over Amazon DCV — `render_rollout.py`
takes a **`--gui`** flag and `scripts/run_gui_local.sh` supplies the Docker
invocation:

```bash
CHECKPOINT=$HOME/work/checkpoint \
OUTPUT_DIR=$HOME/work/gui-out \
IMAGE=isaacsim-so101-workshop:2.3.2 \
  bash scripts/run_gui_local.sh --episodes 2 --seed 101
```

Run it **as the user who owns the X session** (on a DCV box, the autologin user),
with `DISPLAY` and `XAUTHORITY` set. It bind-mounts `CHECKPOINT` at `/checkpoint`
inside the container, so you do not pass `--checkpoint` yourself.

`--gui` is off by default and the farm never passes it, so the headless path is
unchanged. Three things had to be true at once, and each one alone silently
produces a headless run:

1. **`--gui` exists at all.** `ARGS.headless` used to be an unconditional `True`.
2. **`HEADLESS=0` in the environment.** The workshop image sets `HEADLESS=1` in
   its own `ENV`, and Isaac Lab's `AppLauncher._resolve_headless_settings()` only
   lets the flag *raise* headless: with `headless=False` it falls through to
   `self._headless = bool(headless_env)`. This is the one that wastes an
   afternoon, because nothing warns you.
3. **A display and a working cookie.** `-e DISPLAY` plus
   `-v /tmp/.X11-unix:/tmp/.X11-unix`, and an Xauthority whose address family is
   rewritten to `ffff` (FamilyWild) — the host's cookie is scoped to the host's
   hostname, which the container does not share. `run_gui_local.sh` builds that
   temp file and deletes it on exit. Do not reach for `xhost +local:`; it drops
   access control for every local process until you remember to undo it.

What is *not* required, despite appearances: `libxcb-keysyms.so.1` and
`libxcb-cursor.so.0` are absent from the image, and Kit does not need them. Its
window comes up on the X11/xcb libraries already present. **No derived image is
needed** — the same `:2.3.2` tag the farm runs serves the GUI path.

`--gui` forces `--num-envs 1` and says so. A tiled multi-env viewport cannot
usefully be watched (the same reason `--video-env` records one env), and a
success rate measured at a different width is not comparable to the farm's —
see `--num-envs`.

Practical notes:

- **Boot takes ~350 s on a cold shader cache**, ~110 s once `CACHE_DIR/kit` and
  `ov` are warm. The window appears well before the first physics step; a silent
  gap is not a hang.
- Measured **12 env-steps/s** for one env at 1440x779 with full RTX rendering, so
  a 15 s episode of 900 steps takes ~75 s of wall clock. Budget for that in a
  live demo, or lower `--episode-length-s`.
- **Expect `success=False`.** Measured success rate is ~2%, so a two-episode
  viewing will almost certainly show two failures. Seeing the arm move under the
  policy is the point.
- Screenshot the window with `xwd -id <window-id> -silent | convert xwd:- out.png`
  (`wmctrl -l` finds the id; the title is `Isaac Sim <version>`). If the capture
  is a single uniform colour, the screen is blanked by a locker, not broken —
  check `convert xwd:- -format %k info:`.

## Reading a failed task

Every step leaves a trail under `OutputDir/logs/`:

| File | What it tells you |
|------|-------------------|
| `docker_<phase>.log` | The raw, unbuffered `docker run` stream, including the image pull and `nvidia-smi`. |
| `<phase>.log` | Just the in-container command's stdout/stderr. |
| `_inner_exit_<phase>` | The Python process's exit code, independent of how docker reported it. |
| `_marker_<phase>_NN_*` | Phase boundaries the script actually reached. If Kit wedges, these say where. |

For the Evaluate step, `<phase>` is `eval NN` per shard (`eval01`, `eval02`, …),
so each shard has its own `docker_eval01.log`, `eval01.log`,
`_inner_exit_eval01` and `_marker_eval01_NN_*`. `[render-metric]` lines carry
`shard=` and `condition=` so they stay attributable when N shards log
concurrently.

Exit code `124` or `137` means the in-container `StepTimeoutSeconds` fired —
raise it, or read `<phase>.log` to find the hang. Datagen's own codes: `1`
under-production, `2` bad task id, `3` scene missing cameras or assets, `4`
stale dataset in the way, `5` no dataset metadata written, `6` episodes verified
but not flushed to disk before the drain timeout.

Aggregate's codes: `2` one or more expected shards missing (it refuses to report
a success rate over a subset — check the corresponding Evaluate tasks), `3` no
`eval/` directory at all, `4` a shard file present but unparseable, `70` no
`python3` on the worker's PATH.

Render / Evaluate codes: `2` no checkpoint or a non-`-Eval` task, `3` the
requested camera is not in the scene, `4` video was requested but **the rollout**
captured no frames, `5` no episodes ran at all.

### A missing video never fails a shard

`video_status` in each `eval/shard_NN.json` (and in `render_summary.json`) says in
one word why there is or is not an MP4:

| `video_status` | Meaning | Task outcome |
|----------------|---------|--------------|
| `written` | MP4 produced. | success |
| `disabled` | `EvalVideo=false` / `--no-video`; none requested. | success |
| `encoder_unavailable` | ffmpeg has no usable H.264 encoder (e.g. an LGPL build without libx264). | **success** |
| `encoder_failed` | ffmpeg started and then errored. | **success** |
| `ffmpeg_not_found` | No ffmpeg binary in the container. | **success** |
| `no_frames_captured` | The **rollout** produced no frames. | **failure, exit 4** |

The distinction is the point. The first five are environment or configuration
facts and **must not** fail a shard whose episodes ran — under fan-out, one failed
shard cancels `5 - Aggregate` by dependency and destroys the run's headline
number, so failing a task over a missing artifact is expensive. Only
`no_frames_captured` means the rollout itself is broken, and that still fails
loudly.

`aggregate_eval.py` collects any non-`disabled` status into
`eval_summary.json`'s `video_problems` and prints the affected shards and the
reason in its log, so a broken encoder is *reported* rather than inferred from a
short `videos` list. Video *presence* is still decided by the filesystem, which
is the more robust source of truth.

The guarantee is unit-tested — the logic lives in
[`scripts/video_policy.py`](scripts/video_policy.py) with no third-party imports
precisely so it can be tested without launching Omniverse Kit:

```bash
python3 test/test_video_failure_policy.py    # 33 checks, no dependencies, no GPU
```

## The vectorized episode accounting is unit-tested too

At `EvalNumEnvs > 1` the episode bookkeeping decides the number the whole job
exists to produce, and **every way of getting it wrong is silent** — no
exception, just a plausible success rate:

- `terminated = bool(term_t.any().item())`, which is what the single-env rollout
  used, ends *every* env's episode when the **first** one finishes. At
  `num_envs=12` one early success would be reported as 12 successes — a 100%
  success rate where the truth is 8.3%.
- ACT ships **one** `self._action_queue` and pops one action per call, so driving
  N envs through one policy object feeds env *i* the plan made for env *j*. The
  policy simply appears to have forgotten the task.
- A partial last batch inflates the episode count past `EvalEpisodesPerShard`, so
  `EvalShards × EvalEpisodesPerShard` stops being the total and Aggregate reports
  a rate over a denominator nobody asked for.

So that logic lives in [`scripts/rollout_vec.py`](scripts/rollout_vec.py) — pure,
no torch, no Isaac Lab — for the same reason `video_policy.py` does:
`render_rollout.py` launches Kit at module scope and cannot be imported outside
the container, so anything inline in it can only ever be reviewed by eye, which
is exactly how the `frames == 0` video bug survived a full 8-shard farm run.

```bash
python3 test/test_vector_rollout.py         # 131 checks, no dependencies, no GPU
```

Section 4 of that file is a **mutation check**: it reinstates the old `any()`
semantics and asserts the result is wrong, because a green test that cannot go
red proves nothing. It also pins the invariants that matter:

- the reported episode set is *exactly* `0..episodes-1` for every
  `(episodes, num_envs)` pair — surplus comes off the high end, so the reported
  set does not depend on whether a partial batch happened;
- envs finish independently, and a re-terminating env (Isaac Lab auto-resets
  inside `step()`) cannot overwrite its own record;
- `terminated` **and** `truncated` on the same step is *not* a success;
- N envs get N distinct ACT queues, and a policy using temporal ensembling — whose
  state is a batch tensor, not a queue — is **refused** at `num_envs > 1` rather
  than silently sharing one ensembler;
- `--num-envs 1` still works for a policy with no `_action_queue` at all, so the
  `3 - Render` step is not newly restricted to ACT.

Exit codes added by this work: **6** = the checkpoint's policy cannot be driven at
`num_envs > 1` (temporal ensembling, or no swappable action queue), and **2** now
also covers `--deterministic-scene` finding no reset ranges to pin — a silently
ineffective determinism flag is worse than no flag, so it is a hard stop.

## Notes for engineers running datagen at scale

Generating data in a batch on a farm surfaces bugs that one-off local testing
hides. Two that are already handled here, and worth knowing about:

- **The recorder's writer is a daemon thread.** A local run that ends with
  Ctrl-C looks fine because you never check whether the last episode landed. In
  a batch, exiting promptly after the final episode drops it. Datagen drains and
  then *verifies* that `num_recorded_episodes` matches what it saved.
- **A discard must clear the frame buffer.** Without the CANCEL event, the next
  attempt's frames append to the failed one and `save_episode()` writes an
  over-length episode that *begins with a failed grasp* — so the policy learns
  to fail first. Failure integrity is not a nicety; it is what makes the dataset
  trainable.

The sibling lesson from the MuJoCo sample applies here too: optimize the grasp
recipe for the **imperfect learned policy**, not for the scripted demo. A tight,
perfectly-centred grip looks better in the scripted rollout but is a smaller
target once a policy with real positional error is the one driving.
