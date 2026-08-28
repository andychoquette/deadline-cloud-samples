#!/usr/bin/env python3
"""Reduce the parallel Evaluate step's per-shard JSON into one eval_summary.json.

This is the "gather" half of the scatter/gather that step `4 - Evaluate` opens
up: N independent shards each write `eval/shard_NN.json`, and this collapses
them into a single success rate plus per-shard and per-condition breakdowns, and
prints a table so the demo can just show this task's log.

Deliberately NOT run inside the Isaac Sim container, and deliberately
stdlib-only:

  * The container is ~9 GB. A worker that has never pulled it pays ~4 minutes
    before running anything. Reading a handful of small JSON files does not need
    Omniverse Kit, CUDA, LeRobot, or torch, so paying an image pull to do it
    would make the cheapest step in the job one of the slowest.
  * Stdlib-only (no numpy, no pandas) means it runs on the worker's own
    `python3` with no conda queue environment and no pip install, which is what
    lets the step declare no GPU and no software requirements at all.

Python 3.9 compatible on purpose: that is what Amazon Linux 2023 ships as
`python3`, and service-managed fleet workers are AL2023.

Exit codes:
  0  aggregated (a low or zero success rate is a RESULT, not a failure)
  2  one or more expected shards are missing
  3  no eval directory at all
  4  a shard file exists but is not readable/parseable JSON
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

SHARD_RE = re.compile(r"^shard_(\d+)\.json$")


def log(msg):
    print(f"[aggregate] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate per-shard Isaac Sim policy-evaluation results."
    )
    p.add_argument("--output-dir", required=True,
                   help="The job's shared OutputDir. Shards are read from "
                        "<output-dir>/eval/ and the summary is written to "
                        "<output-dir>/eval_summary.json.")
    p.add_argument("--expected-shards", type=int, required=True,
                   help="How many shards the Evaluate step was configured to run. "
                        "Shards 1..N must all be present; a missing one fails the "
                        "task rather than silently reporting a success rate over a "
                        "subset, which would look like a valid result.")
    p.add_argument("--summary-name", default="eval_summary.json")
    return p.parse_args()


def load_shards(eval_dir, expected):
    """Return (shards_by_index, stale_files). Exits on unreadable JSON."""
    found = {}
    stale = []
    for name in sorted(os.listdir(eval_dir)):
        m = SHARD_RE.match(name)
        if not m:
            continue
        idx = int(m.group(1))
        path = os.path.join(eval_dir, name)
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            log(f"ERROR: cannot read {path}: {exc!r}")
            sys.exit(4)
        # A shard index outside 1..N is left over from an earlier, wider run.
        # OutputDir is dataFlow: INOUT, so yesterday's 16-shard output is
        # re-uploaded as today's input. Counting it would silently mix two runs
        # into one success rate -- exactly the failure mode the Datagen step
        # guards against by clearing its dataset directory. Evaluate cannot
        # clear the directory (concurrent tasks would delete each other's
        # results), so the reduce step filters instead.
        if idx < 1 or idx > expected:
            stale.append(name)
            continue
        found[idx] = data
    return found, stale


def fmt_row(cells, widths):
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths)).rstrip()


def table(header, rows):
    widths = [len(h) for h in header]
    for r in rows:
        widths = [max(w, len(str(c))) for w, c in zip(widths, r)]
    out = [fmt_row(header, widths), fmt_row(["-" * w for w in widths], widths)]
    out.extend(fmt_row(r, widths) for r in rows)
    return out


def pct(successes, total):
    return 100.0 * successes / total if total else 0.0


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.output_dir)
    eval_dir = os.path.join(out_dir, "eval")
    if not os.path.isdir(eval_dir):
        log(f"ERROR: no {eval_dir}. The Evaluate step must run first, and because "
            "OutputDir is dataFlow: INOUT its outputs must have been synced back.")
        return 3

    shards, stale = load_shards(eval_dir, args.expected_shards)
    for name in stale:
        log(f"WARNING: ignoring {name} -- shard index outside 1..{args.expected_shards}, "
            "so it is left over from an earlier run with a different shard count.")

    missing = [i for i in range(1, args.expected_shards + 1) if i not in shards]
    if missing:
        log(f"ERROR: {len(missing)} of {args.expected_shards} shards missing: "
            f"{', '.join('shard_%02d.json' % i for i in missing)}")
        log("Refusing to report a success rate over a subset of the shards. Check "
            "the corresponding Evaluate tasks in the job's task list.")
        return 2

    # --- reduce --------------------------------------------------------------
    per_shard = []
    by_condition = {}
    by_num_envs = {}
    all_episodes = 0
    all_successes = 0
    videos = []
    video_problems = {}
    total_env_steps = 0
    total_rollout_s = 0.0

    for idx in sorted(shards):
        data = shards[idx]
        eps = data.get("episodes") or []
        succ = sum(1 for e in eps if e.get("success"))
        cond = data.get("condition") or "unspecified"
        all_episodes += len(eps)
        all_successes += succ

        agg = by_condition.setdefault(cond, {"shards": 0, "episodes": 0, "successes": 0})
        agg["shards"] += 1
        agg["episodes"] += len(eps)
        agg["successes"] += succ

        # num_envs is part of the experimental condition, not just a performance
        # setting: Isaac Lab's anti-aliasing mode switches on the COMBINED tiled
        # resolution, which is a function of the env count (PR #1246), and its
        # renderer is documented as stochastic (issue #3505). For a vision policy
        # the image IS the observation, so averaging shards run at different
        # num_envs averages over two different observation distributions.
        # OutputDir is dataFlow: INOUT, so a shard file left over from an earlier
        # run at a different num_envs -- with an index still inside 1..N, so the
        # index filter above cannot see it -- is a realistic way to get here.
        # Reported and split rather than fatal, so the data is never lost, but
        # never averaged SILENTLY either.
        # A missing key means a shard written before num_envs existed, i.e. 1.
        nenv = data.get("num_envs", 1)
        nagg = by_num_envs.setdefault(nenv, {"shards": [], "episodes": 0,
                                            "successes": 0})
        nagg["shards"].append(idx)
        nagg["episodes"] += len(eps)
        nagg["successes"] += succ
        total_env_steps += data.get("env_steps") or 0
        total_rollout_s += data.get("rollout_seconds") or 0.0

        # Video PRESENCE is decided by the filesystem, deliberately: it is the
        # more robust source of truth than a field the shard wrote about itself,
        # and it is what made this step immune when every shard recorded
        # "video": null. video_status below only explains WHY, never whether.
        mp4 = os.path.join("eval", "shard_%02d.mp4" % idx)
        has_mp4 = os.path.isfile(os.path.join(out_dir, mp4))
        if has_mp4:
            videos.append(mp4)
        vstatus = data.get("video_status") or ("written" if has_mp4 else "unknown")
        if not has_mp4 and vstatus not in ("disabled",):
            video_problems.setdefault(vstatus, []).append(idx)

        per_shard.append({
            "shard": idx,
            "condition": cond,
            "task": data.get("task"),
            "seed": data.get("seed"),
            "num_envs": nenv,
            "episodes": len(eps),
            "successes": succ,
            "success_rate": (succ / len(eps)) if eps else 0.0,
            "video_status": vstatus,
            "video_error": data.get("video_error"),
            "mean_steps": (sum(e.get("steps", 0) for e in eps) / len(eps)) if eps else 0.0,
            "video": mp4 if has_mp4 else None,
            # Rollout-only throughput, so the in-process contribution to any
            # speedup can be read off this file without scraping worker logs and
            # without task durations that also contain the image pull and Kit boot.
            "rollout_seconds": data.get("rollout_seconds"),
            "env_steps": data.get("env_steps"),
            "env_steps_per_second": data.get("env_steps_per_second"),
            "deterministic_scene": data.get("deterministic_scene"),
            "video_env": data.get("video_env"),
            "video_episodes": data.get("video_episodes"),
        })

    overall = (all_successes / all_episodes) if all_episodes else 0.0

    # --- human-readable table ------------------------------------------------
    # Printed to the task log on purpose: the demo shows this log rather than
    # downloading and opening a JSON file.
    lines = []
    lines.append("")
    lines.append("=== Policy evaluation: %d shards, %d episodes ==="
                 % (len(per_shard), all_episodes))
    lines.append("")
    lines.extend(table(
        ["shard", "condition", "seed", "envs", "episodes", "successes", "success%",
         "mean steps", "env steps/s"],
        [[r["shard"], r["condition"], r["seed"], r["num_envs"], r["episodes"],
          r["successes"], "%.1f" % (100.0 * r["success_rate"]),
          "%.0f" % r["mean_steps"],
          "-" if r["env_steps_per_second"] is None
          else "%.1f" % r["env_steps_per_second"]]
         for r in per_shard],
    ))
    lines.append("")
    lines.extend(table(
        ["condition", "shards", "episodes", "successes", "success%"],
        [[c, v["shards"], v["episodes"], v["successes"],
          "%.1f" % pct(v["successes"], v["episodes"])]
         for c, v in sorted(by_condition.items())],
    ))
    lines.append("")
    if len(by_num_envs) > 1:
        # Never print a single averaged number without the split when the shards
        # were not run under the same parallelism. See the num_envs comment above.
        lines.append("!! SHARDS DISAGREE ON num_envs -- the aggregate below "
                     "averages over TWO OR MORE observation distributions.")
        lines.append("!! Isaac Lab's anti-aliasing mode is a function of the env "
                     "count, so these are not the same measurement.")
        lines.append("")
    lines.extend(table(
        ["num_envs", "shards", "episodes", "successes", "success%"],
        [[n, ",".join(str(i) for i in v["shards"]), v["episodes"], v["successes"],
          "%.1f" % pct(v["successes"], v["episodes"])]
         for n, v in sorted(by_num_envs.items())],
    ))
    lines.append("")
    lines.append("OVERALL SUCCESS RATE: %d/%d = %.1f%%"
                 % (all_successes, all_episodes, 100.0 * overall))
    if total_rollout_s > 0:
        # Sum of per-shard rollout time, NOT wall clock -- this step cannot see
        # wall clock. Quoted as aggregate throughput so the in-process and the
        # fleet contributions stay separable: env steps/s per shard is the
        # in-process number, and (this total / the step's wall clock) is where the
        # fleet contribution shows up.
        lines.append("ROLLOUT ONLY: %d env-steps over %.0f s of summed shard "
                     "rollout = %.1f env-steps/s per shard (mean)"
                     % (total_env_steps, total_rollout_s,
                        total_env_steps / total_rollout_s))
    lines.append("")
    for ln in lines:
        print(ln, flush=True)

    summary = {
        "expected_shards": args.expected_shards,
        "shards_aggregated": len(per_shard),
        "stale_files_ignored": stale,
        "episodes": all_episodes,
        "successes": all_successes,
        "success_rate": overall,
        "by_condition": {
            c: {
                "shards": v["shards"],
                "episodes": v["episodes"],
                "successes": v["successes"],
                "success_rate": (v["successes"] / v["episodes"]) if v["episodes"] else 0.0,
            }
            for c, v in sorted(by_condition.items())
        },
        # Present whether or not the shards agree, so a consumer never has to
        # infer it: num_envs is null exactly when they disagree, and the split is
        # always in by_num_envs.
        "num_envs": (list(by_num_envs)[0] if len(by_num_envs) == 1 else None),
        "num_envs_consistent": len(by_num_envs) <= 1,
        "by_num_envs": {
            str(n): {
                "shards": v["shards"],
                "episodes": v["episodes"],
                "successes": v["successes"],
                "success_rate": (v["successes"] / v["episodes"]) if v["episodes"] else 0.0,
            }
            for n, v in sorted(by_num_envs.items())
        },
        "env_steps": total_env_steps,
        "rollout_seconds_summed": round(total_rollout_s, 2),
        "by_shard": per_shard,
        "videos": videos,
        # Explains any MISSING video, so a broken encoder is reported here rather
        # than left to be inferred from a short "videos" list.
        "video_problems": video_problems,
        "table": lines,
    }
    path = os.path.join(out_dir, args.summary_name)
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"wrote {path}")
    if len(by_num_envs) > 1:
        # Grep-able, prefixed, and separate from the table -- a warning that only
        # exists inside a pretty-printed block is a warning that gets scrolled
        # past. Deliberately NOT fatal: the per-num_envs split above is a valid
        # result and failing here would throw it away, which is the same mistake
        # the video policy exists to avoid.
        log("WARNING: shards were run at DIFFERENT num_envs "
            f"({', '.join('%s:[%s]' % (n, ','.join(str(i) for i in v['shards'])) for n, v in sorted(by_num_envs.items()))}).")
        log("  The OVERALL success rate above mixes observation distributions and "
            "should NOT be quoted. Use by_num_envs instead.")
        log("  Most likely cause: OutputDir is dataFlow: INOUT, so eval/shard_NN.json "
            "from an earlier run at a different EvalNumEnvs was re-uploaded as input. "
            "Use a fresh OutputDir per experiment.")
    missing_video = len(per_shard) - len(videos)
    if missing_video:
        # Do NOT assert a cause here. A missing MP4 is expected under
        # EvalVideo=false, but it is ALSO what a broken video encoder looks like:
        # Mp4Writer is deliberately non-fatal, so an ffmpeg that cannot encode
        # logs once and the shard still succeeds with its metrics intact and no
        # video. Observed for real -- an LGPL ffmpeg build has no libx264, so
        # 8/8 shards wrote JSON + PNG and 0/8 wrote an MP4. Claiming
        # "expected if EvalVideo=false" would have actively hidden that.
        log(f"note: {missing_video} of {len(per_shard)} shard(s) produced no MP4.")
        for status, shard_ids in sorted(video_problems.items()):
            log(f"  !! video_status={status} on shard(s) "
                f"{', '.join(str(i) for i in shard_ids)}")
            detail = next((s["video_error"] for s in per_shard
                           if s["shard"] == shard_ids[0] and s["video_error"]), None)
            if detail:
                log(f"     {detail}")
        if video_problems:
            log("  These shards' SUCCESS DATA IS STILL VALID -- a video failure "
                "does not invalidate the episodes. Fix the container's ffmpeg.")
    return 0


if __name__ == "__main__":
    # No Isaac Sim here, so none of Kit's non-daemon-thread hang applies and no
    # watchdog is needed. The guard still exists so an unexpected exception
    # produces a traceback AND a deterministic non-zero exit rather than
    # whatever the interpreter would do.
    try:
        sys.exit(main() or 0)
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        sys.exit(1)
