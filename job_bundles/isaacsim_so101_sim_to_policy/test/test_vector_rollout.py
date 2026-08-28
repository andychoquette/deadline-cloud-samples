#!/usr/bin/env python3
"""Prove the vectorized episode accounting is right, without a GPU.

Run with no arguments and no dependencies:

    python3 test/test_vector_rollout.py

Lives outside `scripts/` on purpose: `scripts/` is the `JobScriptDir` job
attachment, uploaded to every worker on every submit, so tests do not belong in
it. Same reasoning as `test_video_failure_policy.py`.

Why this file exists
--------------------
At `num_envs > 1` the episode bookkeeping decides the number the whole job
exists to produce, and every way of getting it wrong is SILENT -- no exception,
just a plausible success rate:

  * `terminated = bool(term_t.any().item())`, the single-env code, ends every
    env's episode when the FIRST one finishes;
  * one ACT action queue shared across N envs feeds env i env j's actions;
  * a partial last batch inflates the episode count past
    `EvalEpisodesPerShard`, so `EvalShards x EvalEpisodesPerShard` stops being
    the total.

`render_rollout.py` cannot be imported outside the container (it launches
Omniverse Kit at module scope), so without extracting this logic the only
available review is by eye -- which is exactly how the `frames == 0` video bug
survived a full 8-shard farm run. Section 4 is a MUTATION CHECK: it reinstates
the old `any()` semantics and asserts the result is wrong, because a test that
cannot go red proves nothing.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "scripts"))

import rollout_vec as rv  # noqa: E402

FAILS = []
CHECKS = [0]


def check(label, got, want):
    CHECKS[0] += 1
    if got != want:
        FAILS.append(f"{label}\n      got:  {got!r}\n      want: {want!r}")
        print(f"  FAIL  {label}")
    else:
        print(f"  ok    {label}")


def raises(label, fn, exc=Exception):
    CHECKS[0] += 1
    try:
        fn()
    except exc:
        print(f"  ok    {label}")
        return
    except BaseException as other:  # noqa: BLE001
        FAILS.append(f"{label}: raised {other!r}, wanted {exc.__name__}")
        print(f"  FAIL  {label}")
        return
    FAILS.append(f"{label}: did not raise")
    print(f"  FAIL  {label}")


# ---------------------------------------------------------------------------
print("\n[1] batch plan matches LeRobot's ceil(n_episodes / num_envs)")
# LeRobot: n_batches = n // N + int((n % N) != 0)
for episodes, num_envs in [(12, 1), (12, 12), (12, 8), (12, 5), (1, 12),
                           (0, 4), (48, 12), (100, 25), (7, 3)]:
    want = episodes // num_envs + int((episodes % num_envs) != 0)
    got, sim, surplus = rv.plan_batches(episodes, num_envs)
    check(f"plan_batches({episodes},{num_envs}) batches", got, want)
    check(f"plan_batches({episodes},{num_envs}) simulated", sim, want * num_envs)
    check(f"plan_batches({episodes},{num_envs}) surplus",
          surplus, want * num_envs - episodes)

# The single-env path must be untouched: one batch per episode, nothing wasted.
check("num_envs=1 -> one batch per episode, zero surplus",
      rv.plan_batches(12, 1), (12, 12, 0))
# The default pairing in template.yaml: 12 episodes / 12 envs is exactly 1 batch.
check("default 12 episodes / 12 envs -> a single batch, no waste",
      rv.plan_batches(12, 12), (1, 12, 0))
raises("plan_batches rejects num_envs=0", lambda: rv.plan_batches(4, 0), ValueError)
raises("plan_batches rejects negative episodes",
       lambda: rv.plan_batches(-1, 4), ValueError)

# ---------------------------------------------------------------------------
print("\n[2] THE TOTAL IS EXACT: reported episodes are precisely 0..episodes-1")
# This is the invariant aggregate_eval.py and the README both depend on:
# total episodes == EvalShards x EvalEpisodesPerShard, whatever num_envs is.
for episodes in (1, 5, 12, 13, 48, 100):
    for num_envs in (1, 2, 3, 8, 12, 25, 64):
        n_batches, _, _ = rv.plan_batches(episodes, num_envs)
        reported = []
        for b in range(n_batches):
            surplus = rv.surplus_envs(b, num_envs, episodes)
            for i in range(num_envs):
                if i in surplus:
                    continue
                reported.append(rv.episode_index(b, i, num_envs))
        check(f"episodes={episodes} num_envs={num_envs}: reported set",
              reported, list(range(episodes)))

# Surplus comes off the HIGH end, so the set of reported episodes does not
# depend on whether a partial batch happened.
check("12 episodes / 8 envs: batch 1 discards envs 4..7",
      rv.surplus_envs(1, 8, 12), [4, 5, 6, 7])
check("12 episodes / 8 envs: batch 0 discards nothing",
      rv.surplus_envs(0, 8, 12), [])
check("batch-major index: episode 14 at num_envs=4 is (batch 3, env 2)",
      rv.episode_index(3, 2, 4), 14)

# ---------------------------------------------------------------------------
print("\n[3] EpisodeTracker: envs finish INDEPENDENTLY")
t = rv.EpisodeTracker(3)
check("nothing done at the start", t.all_done(), False)
check("all three envs active", t.active(), [0, 1, 2])
# Step 5: env 0 succeeds (terminated, not truncated). Envs 1 and 2 continue.
newly = t.observe([True, False, False], [False, False, False], 5)
check("env 0 finishing is reported once", newly, [0])
check("env 0 recorded 6 steps", t.steps[0], 6)
check("env 0 recorded as a success", t.success[0], True)
check("env 1 has NOT been given env 0's step count", t.steps[1], 0)
check("the batch is not over", t.all_done(), False)
check("only envs 1 and 2 remain active", t.active(), [1, 2])
# Step 99: env 0 terminates AGAIN (Isaac Lab auto-reset it and it ran another,
# unrecorded episode). Its record must not move.
newly = t.observe([True, False, True], [False, False, False], 99)
check("a re-terminating env is not reported again", newly, [2])
check("env 0's step count is sticky at 6", t.steps[0], 6)
check("env 2 recorded its own 100 steps", t.steps[2], 100)
# Step 899: env 1 times out.
newly = t.observe([False, False, False], [False, True, False], 899)
check("env 1 finishing on truncation is reported", newly, [1])
check("env 1 recorded 900 steps", t.steps[1], 900)
check("a timeout is NOT a success", t.success[1], False)
check("the batch is over", t.all_done(), True)
check("final step counts are per-env", t.steps, [6, 900, 100])
check("final success flags are per-env", t.success, [True, False, True])

# terminated AND truncated on the same step: the clock won, so not a success.
t2 = rv.EpisodeTracker(1)
t2.observe([True], [True], 899)
check("terminated+truncated together -> not a success", t2.success[0], False)
check("...but the episode is still recorded", t2.steps[0], 900)

raises("EpisodeTracker rejects num_envs=0", lambda: rv.EpisodeTracker(0), ValueError)
raises("observe() rejects a wrong-length flag list",
       lambda: rv.EpisodeTracker(3).observe([True], [False], 0), ValueError)

# The single-env path must behave exactly as the old code did.
t3 = rv.EpisodeTracker(1)
check("num_envs=1: not done before termination", t3.all_done(), False)
t3.observe([False], [False], 0)
check("num_envs=1: still not done", t3.all_done(), False)
t3.observe([True], [False], 41)
check("num_envs=1: done, 42 steps, success", (t3.all_done(), t3.steps, t3.success),
      (True, [42], [True]))

# ---------------------------------------------------------------------------
print("\n[4] MUTATION CHECK: the old any() semantics must produce a WRONG answer")


class AnyTracker:
    """The pre-vectorization logic, verbatim, generalized to N envs.

    `terminated = bool(term_t.any().item())` / `truncated = bool(...)` with one
    shared step counter -- i.e. what render_rollout.py did before
    EpisodeTracker existed. Reproduced here so the test can demonstrate that it
    is wrong, rather than merely asserting that the new code is right.
    """

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.finished = False
        self.steps = [0] * num_envs
        self.success = [False] * num_envs

    def observe(self, terminated, truncated, step):
        term = any(terminated)
        trunc = any(truncated)
        if term or trunc:
            self.finished = True
            self.steps = [step + 1] * self.num_envs
            self.success = [term and not trunc] * self.num_envs
        return []

    def all_done(self):
        return self.finished


# Same event stream as section 3, up to the point the old code would have quit.
old = AnyTracker(3)
old.observe([True, False, False], [False, False, False], 5)
CHECKS[0] += 1
if old.all_done() and old.steps == [6, 6, 6] and old.success == [True, True, True]:
    print("  ok    old any() logic ends ALL 3 episodes at env 0's step 6 "
          "and marks all 3 successes -- the bug is real and this test sees it")
else:
    FAILS.append("mutation check did not reproduce the old wrong behaviour; "
                 "the test may no longer be able to detect the regression")
    print("  FAIL  mutation check could not reproduce the bug")

new = rv.EpisodeTracker(3)
new.observe([True, False, False], [False, False, False], 5)
check("new logic does NOT end the batch on env 0", new.all_done(), False)
check("new logic does NOT copy env 0's step count", new.steps, [6, 0, 0])
check("new logic does NOT copy env 0's success", new.success, [True, False, False])
# The consequence, stated as a number: at num_envs=12 the old logic would report
# 12 successes from 1, i.e. 100% instead of 8.3%.
CHECKS[0] += 1
old12 = AnyTracker(12)
old12.observe([True] + [False] * 11, [False] * 12, 5)
new12 = rv.EpisodeTracker(12)
new12.observe([True] + [False] * 11, [False] * 12, 5)
if sum(old12.success) == 12 and sum(new12.success) == 1:
    print("  ok    at num_envs=12 the old logic reports 12/12 successes where "
          "the truth is 1 -- a 100% success rate instead of 8.3%")
else:
    FAILS.append("the 12-env consequence of the old bug did not reproduce")
    print("  FAIL  12-env consequence")

# ---------------------------------------------------------------------------
print("\n[5] PerEnvPolicyQueues: N envs get N queues, never one shared queue")


class FakeCfg:
    def __init__(self, ensemble=None, n_action_steps=20):
        self.temporal_ensemble_coeff = ensemble
        self.n_action_steps = n_action_steps


class FakePolicy:
    """Mimics ACT: reset() installs a FRESH _action_queue object."""

    def __init__(self, ensemble=None):
        self.config = FakeCfg(ensemble)
        self._action_queue = None
        self.resets = 0

    def reset(self):
        self.resets += 1
        self._action_queue = ["queue-%d" % self.resets]


pol = FakePolicy()
q = rv.PerEnvPolicyQueues(pol, 4)
q.reset_all()
check("one policy.reset() per env", pol.resets, 4)
check("four DISTINCT queue objects were captured",
      len({id(x) for x in q.queues}), 4)
q.select(2)
check("select(2) installs env 2's queue", pol._action_queue, q.queues[2])
q.select(0)
check("select(0) installs env 0's queue", pol._action_queue, q.queues[0])
check("...which is not env 2's", pol._action_queue is q.queues[2], False)

# Temporal ensembling has no queue to swap, so it must refuse rather than
# silently share one ensembler across the batch.
raises("temporal ensembling + num_envs>1 is refused",
       lambda: rv.PerEnvPolicyQueues(FakePolicy(ensemble=0.01), 8), RuntimeError)
check("temporal ensembling at num_envs=1 is FINE",
      isinstance(rv.PerEnvPolicyQueues(FakePolicy(ensemble=0.01), 1),
                 rv.PerEnvPolicyQueues), True)


class NoQueuePolicy:
    def __init__(self):
        self.config = FakeCfg()

    def reset(self):
        pass


raises("a policy with no _action_queue is refused at num_envs>1",
       lambda: rv.PerEnvPolicyQueues(NoQueuePolicy(), 2), RuntimeError)
# Critical: the single-env path must not newly fail for such a policy, or this
# change would break the Render step for anything but ACT.
check("a policy with no _action_queue is accepted at num_envs=1",
      isinstance(rv.PerEnvPolicyQueues(NoQueuePolicy(), 1),
                 rv.PerEnvPolicyQueues), True)
one = rv.PerEnvPolicyQueues(NoQueuePolicy(), 1)
one.reset_all()
one.select(0)  # must be a no-op, not an AttributeError
check("select() on a queue-less policy is a harmless no-op", one.queues, [None])

# ---------------------------------------------------------------------------
print("\n[6] zero_reset_randomization pins every range it can find")


class Term:
    def __init__(self, params, mode="reset"):
        self.params = params
        self.mode = mode


class Events:
    pass


class Cfg:
    pass


cfg = Cfg()
cfg.events = Events()
# The shapes that actually occur in the workshop task cfgs.
cfg.events.reset_vials_setup = Term({
    "pose_range": {"x": (-0.04, 0.04), "y": (-0.01, 0.01), "roll": (-0.3, 0.3)},
    "rack_pose_range": {"x": (-0.04, 0.04), "yaw": (-0.5, 0.5)},
    "fixed_vial_z": 0.05,
    "rack_placement_prob": 0.33,
})
cfg.events.reset_lightbox_light_exposure = Term({"exposure_range": (-3.0, 1.0)})
cfg.events.reset_camera_ego_fov = Term({"focal_length_range": (12.0, 15.0)})
cfg.events.reset_camera_external_pose = Term({
    "pos_range": {"x": (-0.02, 0.02), "z": (-0.01, 0.01)},
    "rot_range": {"roll": (-0.05, 0.05)},
})
cfg.events.startup_thing = Term({"exposure_range": (-9.0, 9.0)}, mode="startup")

changed = rv.zero_reset_randomization(cfg)
check("every reset range and the placement prob were pinned", len(changed), 7)
check("pose_range collapsed to a constant",
      cfg.events.reset_vials_setup.params["pose_range"],
      {"x": (-0.04, -0.04), "y": (-0.01, -0.01), "roll": (-0.3, -0.3)})
check("rack_placement_prob zeroed",
      cfg.events.reset_vials_setup.params["rack_placement_prob"], 0.0)
check("a bare (lo,hi) exposure range collapsed",
      cfg.events.reset_lightbox_light_exposure.params["exposure_range"],
      (-3.0, -3.0))
# The reason it is (lo, lo) and not (0, 0): an ABSOLUTE focal length of 0 is a
# degenerate camera, so zeroing would change what is being measured.
check("focal length pinned to 12.0, NOT to 0.0",
      cfg.events.reset_camera_ego_fov.params["focal_length_range"], (12.0, 12.0))
check("non-reset (startup) terms are left alone",
      cfg.events.startup_thing.params["exposure_range"], (-9.0, 9.0))
check("fixed_vial_z (not a *_range) is untouched",
      cfg.events.reset_vials_setup.params["fixed_vial_z"], 0.05)

# An empty result is what render_rollout.py turns into a hard error, because a
# silently-ineffective --deterministic-scene is worse than no flag at all.
empty = Cfg()
empty.events = Events()
check("a cfg with no ranges reports that nothing changed",
      rv.zero_reset_randomization(empty), [])
noev = Cfg()
check("a cfg with no events at all does not crash",
      rv.zero_reset_randomization(noev), [])

# ---------------------------------------------------------------------------
print("\n[7] env_visual_slice positions a 1-env view at env i")
# Lists stand in for tensors: both slice the same way, which is the whole point
# -- the workshop's sim_obs_to_policy_processor indexes [0] on what it is given.
visual = {"rgb_ego": ["e0", "e1", "e2"], "rgb_external_D455": ["x0", "x1", "x2"]}
for i in range(3):
    sl = rv.env_visual_slice(visual, i)
    check(f"slice at env {i} keeps the leading dim", [len(v) for v in sl.values()],
          [1, 1])
    check(f"[0] of the slice at env {i} is env {i}'s frame",
          [v[0] for v in sl.values()], [f"e{i}", f"x{i}"])
check("every camera key survives slicing",
      sorted(rv.env_visual_slice(visual, 1)), sorted(visual))

# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
if FAILS:
    print(f"{len(FAILS)} of {CHECKS[0]} checks FAILED:\n")
    for f in FAILS:
        print("  - " + f)
    sys.exit(1)
print(f"all {CHECKS[0]} checks passed")
print("ACCOUNTING HELD: episodes are reported exactly once, envs finish")
print("independently, the reported total is always --episodes, and each env")
print("gets its own ACT action queue. The old any() logic is demonstrably wrong.")
sys.exit(0)
