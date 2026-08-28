# The two lines the demo pastes into Window -> Script Editor.
#
# Kept as a file for one reason: it lets the in-session rollout be tested without
# driving the GUI, with
#
#     bash workbench_gui.sh --run-script /job_scripts/workbench_run_one.py
#
# which execs THIS file inside the live session from a Kit update callback -- the
# same context, and the same plain exec(), that the Script Editor's Run button
# uses. So what is verified is what a human would paste, not a separate code path.
#
# The flag is --run-script and NOT --exec on purpose: --exec is one of Kit's own
# CLI flags, so Kit would run this file a second time during app startup. That
# cost one debugging cycle; see workbench_session.py.
import workbench_rollout as wr

wr.run()
