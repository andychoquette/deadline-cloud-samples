#!/usr/bin/env python3
"""Episode accounting and env-config surgery for a VECTORIZED rollout.

Lives in its own module with **no third-party imports at all** (no torch, no
numpy, no Isaac Lab), for the same reason `video_policy.py` does:
`render_rollout.py` launches Omniverse Kit at module scope and therefore cannot
be imported outside the container, so any logic expressed inline there can only
ever be reviewed by eye. That is how the `frames == 0` video bug survived a full
8-shard farm run.

What is at stake here is worse than a missing MP4. At `num_envs > 1` the episode
bookkeeping decides the number the whole job exists to produce, and every way of
getting it wrong is SILENT:

  * `terminated = bool(term_t.any().item())` -- the single-env code -- ends every
    env's episode when the FIRST one finishes, so N episodes are all reported
    with the luckiest env's step count and the unluckiest envs are never
    scored;
  * one ACT action queue shared across N envs feeds env i the actions planned
    for env j;
  * a partial last batch silently inflates the episode count above
    `EvalEpisodesPerShard`, so `EvalShards x EvalEpisodesPerShard` stops being
    the total and `aggregate_eval.py` reports a rate over a denominator nobody
    asked for.

None of those raise. All of them produce a plausible success rate.

See `test/test_vector_rollout.py`.
"""

from __future__ import annotations


# --- episode plan -------------------------------------------------------------

def plan_batches(episodes, num_envs):
    """Return (n_batches, n_simulated, n_surplus) for `episodes` in groups of N.

    LeRobot's own vectorized eval harness computes exactly this
    (`src/lerobot/scripts/lerobot_eval.py`)::

        n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    `n_surplus` is the work that gets simulated and then DISCARDED because the
    last batch is partial. It is returned rather than hidden so the caller can
    warn: at `episodes=12, num_envs=8` a quarter of the second batch's rendering
    is thrown away, which looks like an unexplained throughput loss otherwise.
    """
    if episodes < 0:
        raise ValueError("episodes must be >= 0")
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")
    n_batches = -(-episodes // num_envs)  # ceil without floats
    n_simulated = n_batches * num_envs
    return n_batches, n_simulated, n_simulated - episodes


def episode_index(batch, env, num_envs):
    """Global episode index of (batch, env). Batch-major, so it is stable.

    Batch-major matters for reproducibility: episode e is always
    (e // num_envs, e % num_envs), so re-running one surprising episode does not
    require knowing how the batches happened to be scheduled.
    """
    return batch * num_envs + env


def surplus_envs(batch, num_envs, episodes):
    """Env indices in `batch` whose episodes are beyond `episodes`.

    Dropped from the HIGH end deliberately: the episodes that ARE reported are
    then the same set whether or not a partial batch happened, so
    `EvalShards x EvalEpisodesPerShard` stays the true total and a shard's
    `episodes` list is always exactly that long.
    """
    return [i for i in range(num_envs)
            if episode_index(batch, i, num_envs) >= episodes]


# --- per-env episode bookkeeping ---------------------------------------------

class EpisodeTracker:
    """Per-env done / steps / success for one batch of N concurrent episodes.

    Replaces `terminated = bool(term_t.any().item())`, which is correct only at
    N == 1. Isaac Lab auto-resets a terminated env inside `step()` and recomputes
    observations, so an env that has finished is immediately running an episode
    the caller is NOT recording -- hence `done` is sticky and `finish()` must be
    called exactly once per env per batch.
    """

    def __init__(self, num_envs):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = num_envs
        self.done = [False] * num_envs
        self.steps = [0] * num_envs
        self.success = [False] * num_envs

    def all_done(self):
        return all(self.done)

    def active(self):
        return [i for i in range(self.num_envs) if not self.done[i]]

    def observe(self, terminated, truncated, step):
        """Record the outcome of one env.step().

        `terminated` / `truncated` are per-env booleans as returned by Isaac Lab
        (`reset_terminated` / `reset_time_outs`). `step` is the 0-based index of
        the step just taken, so the recorded length is `step + 1`.

        Returns the list of env indices that finished on THIS step, so the caller
        can capture per-env state (e.g. the pre-step joint positions) exactly
        once.
        """
        if len(terminated) != self.num_envs or len(truncated) != self.num_envs:
            raise ValueError(
                f"expected {self.num_envs} flags, got "
                f"{len(terminated)} terminated / {len(truncated)} truncated")
        newly = []
        for i in range(self.num_envs):
            if self.done[i]:
                # Sticky. Without this, an env that terminated at step 100 and
                # was auto-reset would terminate AGAIN later in the same batch
                # and overwrite its own episode record with the second, unwanted
                # episode's step count.
                continue
            if terminated[i] or truncated[i]:
                self.done[i] = True
                self.steps[i] = step + 1
                # `terminated and not truncated` is the success predicate the
                # single-env path already used: the env's own
                # vial_placed_on_rack_termination() fired, as opposed to the
                # episode running out of time. Both flags can be set on the same
                # step, and in that case the timeout wins -- a success confirmed
                # only by the clock running out is not a success.
                self.success[i] = bool(terminated[i]) and not bool(truncated[i])
                newly.append(i)
        return newly


# --- ACT action queues, one per env ------------------------------------------

class PerEnvPolicyQueues:
    """One ACT action queue per env, because the policy only ships with one.

    ACT's `select_action()` keeps a single `self._action_queue` and pops one
    action per call, so driving N envs through one policy object interleaves N
    envs' plans into one queue and every env receives another env's actions.
    Nothing raises; the policy just appears to have forgotten the task.

    Why not batch the policy instead, which ACT natively supports? Because ACT
    re-plans only when the queue empties (every `n_action_steps` steps), so a
    single batched queue has to be flushed whenever ANY env resets -- which
    re-plans the other N-1 envs early and makes each env's behaviour depend on
    its neighbours' episode boundaries. Swapping per-env queues keeps each env's
    inference sequence identical to what it would be at `num_envs=1`, which is
    the property that makes a vectorized success rate comparable with a serial
    one at all.

    It also preserves the sample's stated reason for driving the policy through
    LeRobot's real-robot control path: `predict_action()` is a batch-of-1 API
    because the physical SO-101 is one arm. Vectorizing by batching the POLICY
    forks that path; vectorizing by batching the SIMULATOR does not.
    """

    def __init__(self, policy, num_envs):
        self.policy = policy
        self.num_envs = num_envs
        if num_envs > 1:
            # Fail loudly and early rather than produce a plausible wrong number.
            if getattr(getattr(policy, "config", None),
                       "temporal_ensemble_coeff", None) is not None:
                raise RuntimeError(
                    "this checkpoint uses ACT temporal ensembling "
                    "(temporal_ensemble_coeff is set), whose state is a tensor "
                    "shared across the batch rather than a queue, so it cannot "
                    "be swapped per env. Run with --num-envs 1, or retrain with "
                    "temporal_ensemble_coeff=None (LeRobot's default).")
            if not hasattr(policy, "_action_queue"):
                raise RuntimeError(
                    f"{type(policy).__name__} does not expose the _action_queue "
                    "that this per-env swap relies on, so --num-envs > 1 cannot "
                    "be made correct for it. Run with --num-envs 1.")
        self.queues = [None] * num_envs

    def reset(self, i):
        """Reset env i's policy state and remember the fresh queue object."""
        self.policy.reset()
        self.queues[i] = getattr(self.policy, "_action_queue", None)

    def reset_all(self):
        for i in range(self.num_envs):
            self.reset(i)

    def select(self, i):
        """Make env i's queue the live one, immediately before predict_action."""
        if self.queues[i] is not None:
            self.policy._action_queue = self.queues[i]


# --- env cfg surgery ----------------------------------------------------------

def event_terms(env_cfg):
    """Every event term on the cfg, without assuming how configclass stores them.

    `dir()` rather than `vars()`: isaaclab's `@configclass` has changed between a
    plain dataclass and a slotted one across releases, and `vars()` is empty on
    the slotted form. A silently-empty iteration here would make
    `--deterministic-scene` a no-op that still logged success, which is worse
    than not having the flag -- the point of the flag is that the caller can
    trust the scene is fixed.
    """
    events = getattr(env_cfg, "events", None)
    if events is None:
        return []
    out = []
    for name in dir(events):
        if name.startswith("_"):
            continue
        term = getattr(events, name, None)
        if getattr(term, "params", None) is not None:
            out.append((name, term))
    return out


def zero_reset_randomization(env_cfg):
    """Pin every range on every reset event term. Returns what changed.

    Keyed on the SHAPE of the parameter rather than on a list of known names.
    The nominal `-Eval` variant is NOT un-randomized -- it already jitters light
    exposure, mat yaw, ego focal length and external camera pose -- and the
    `-DR-Eval` variant adds robot colour and sky light on top. A name allowlist
    would silently miss whichever term the workshop adds next, and the caller
    would believe the scene was fixed when it was not.

    Two shapes are recognised:
      * a dict of axis -> (lo, hi): `pose_range`, `rot_range`, `pos_range`;
      * a bare (lo, hi) pair: `exposure_range`, `focal_length_range`.

    Both collapse to `(lo, lo)` rather than to `(0, 0)`. `focal_length_range`
    (12.0, 15.0) is an ABSOLUTE focal length, so zeroing it would produce a
    degenerate camera, while `pose_range` (-0.04, 0.04) is an offset whose
    identity is 0 -- and there is no way to tell which a given term is from its
    name. Taking the low end makes the draw a constant without needing to know:
    the value is arbitrary but FIXED, which is the only property this promises.
    """
    changed = []
    for name, term in event_terms(env_cfg):
        if getattr(term, "mode", None) not in (None, "reset"):
            continue
        for key, val in list(term.params.items()):
            if not key.endswith("_range"):
                continue
            if isinstance(val, dict):
                term.params[key] = {k: (v[0], v[0]) for k, v in val.items()}
                changed.append("%s.%s" % (name, key))
            elif isinstance(val, (tuple, list)) and len(val) == 2:
                term.params[key] = (val[0], val[0])
                changed.append("%s.%s" % (name, key))
        if "rack_placement_prob" in term.params:
            # Explicit rather than relying on the 0.33 default: a vial pre-placed
            # in a bore changes which bore is free, which is exactly the kind of
            # run-to-run variation this flag exists to remove.
            term.params["rack_placement_prob"] = 0.0
            changed.append("%s.rack_placement_prob" % name)
    return changed


# --- observation slicing ------------------------------------------------------

def env_visual_slice(visual_obs, i):
    """A 1-env VIEW of the visual observation dict, positioned at env i.

    `LeRobotSO101Interface.sim_obs_to_policy_processor` indexes its visual
    argument with a hardcoded `[0]`, so handing it a slice that STARTS at i is
    what makes that workshop method usable per env without forking it. Slicing
    (rather than indexing) keeps the leading dimension, so the `[0]` inside the
    method selects env i. A slice is a view, not a copy; the method clones what
    it needs.
    """
    return {k: v[i:i + 1] for k, v in visual_obs.items()}
