#!/usr/bin/env python3
"""Persistent Isaac Sim GUI session: submitter panel + in-session rollout, one window.

Launched by ``workbench_gui.sh`` inside the workshop container. It does three
things and then gets out of the way:

  1. Starts Kit through Isaac Lab's ``AppLauncher`` with ``enable_cameras=True``
     and ``headless=False``.
  2. Enables ``isaacsim.deadline.submitter`` (and whatever else was asked for) in
     that same Kit session, and makes ``/job_scripts`` importable.
  3. Idles: ``while simulation_app.is_running(): simulation_app.update()``.

It runs NO rollout. The rollout is ``workbench_rollout.py``, which the user
imports from **Window -> Script Editor** in the window this opens.

Why the split, and why the camera flag is here and not there
-----------------------------------------------------------
``enable_cameras`` is a LAUNCH-TIME setting. The policy's observation is two
480x640 cameras rendered by ``TiledCamera``, and a Script Editor cannot turn
cameras on after the fact -- by then the render products do not exist and
``TiledCamera`` silently has nothing to give. ``render_rollout.py`` sets
``ARGS.enable_cameras = True`` immediately before ``AppLauncher`` for exactly
this reason. So the workbench must own the launch, and the Script Editor can only
own what happens afterwards.

What this file must never do
----------------------------
No second ``AppLauncher``, no ``simulation_app.close()``, no ``os._exit()`` in
anything the Script Editor can reach. ``render_rollout.py`` ends that way because
it is a batch job whose process exists only to produce one summary; here the app
IS the deliverable, and closing it kills the submitter panel the demo is about.
(Note also that ``simulation_app.close()`` never returns -- Kit's fast-shutdown
path terminates the process -- which is why guards placed after it in
render_rollout.py were dead code.)

The idle loop is a bare ``simulation_app.update()`` on purpose: that is the same
pump Isaac Lab's own standalone GUI scripts use, and it is what keeps the
viewport, the menus and the submitter's per-frame update subscription alive while
nothing else is happening.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# /job_scripts, so `import workbench_rollout` works from the Script Editor.
# sys.path is process-global, and the Script Editor executes in this process, so
# doing it here is what makes the user's two-line paste work with no path setup.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def parse_args():
    p = argparse.ArgumentParser(
        description="Persistent Isaac Sim workbench session (no rollout).")
    p.add_argument("--checkpoint", default="/checkpoint",
                   help="LeRobot checkpoint dir, remembered as the default for "
                        "workbench_rollout. Bind-mounted by workbench_gui.sh.")
    p.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "/outputs"))
    p.add_argument("--ext-folder", action="append", default=[],
                   help="Kit extension SEARCH path -- the parent of the extension "
                        "directory, not the extension itself. Repeatable.")
    p.add_argument("--enable", action="append", default=[],
                   help="Extension to enable after startup. Repeatable.")
    p.add_argument("--pypath", action="append", default=[],
                   help="Directory appended to sys.path ONLY if a module Kit needs "
                        "is missing from Kit's python. Used for PyYAML; see below.")
    p.add_argument("--task", default="Lerobot-So101-Teleop-Vials-To-Rack-Eval",
                   help="Remembered as workbench_rollout's default task.")
    p.add_argument("--show-window", action="append", default=["Script Editor"],
                   help="Kit window to open at startup, BY TITLE, via "
                        "omni.ui.Workspace.show_window. Defaults to the Script "
                        "Editor so the rollout is two lines away instead of four "
                        "menu clicks. Repeatable; pass 'Submit to AWS Deadline "
                        "Cloud' to open the submitter panel too.")
    # NOT --exec. `--exec` is one of KIT's OWN command line flags, and Kit parses
    # sys.argv itself: naming this option --exec makes Kit run the file too,
    # during SimulationApp._prepare_ui()'s app.update(), which is BEFORE the
    # extensions are enabled. Symptom: one --exec of a file containing a single
    # wr.run() produced TWO episodes. Diagnosed from a stack trace whose outer
    # frames were AppLauncher -> SimulationApp.__init__ -> _prepare_ui ->
    # app.update() -> the exec'd file. Any new flag added here must be checked
    # against Kit's set for the same reason; --ext-folder and --enable below are
    # also Kit flags, but there the collision is harmless (Kit does the same
    # thing, idempotently, and the runtime calls in this file then no-op).
    p.add_argument("--run-script", dest="exec_file", default="",
                   help="Python file to exec INSIDE this session once the app is "
                        "up, from a Kit update callback -- the same context the "
                        "Script Editor's Run uses. This is how the in-session "
                        "rollout is tested without driving the GUI: "
                        "`--run-script workbench_run_one.py`, where that file is "
                        "the two lines a user would paste. Not needed for "
                        "interactive use.")
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


ARGS = parse_args()

# Both of these are launch-time and neither can be fixed later:
#   * enable_cameras -- see the module docstring.
#   * headless -- the workshop image sets HEADLESS=1 in its own ENV and
#     AppLauncher._resolve_headless_settings() only lets the flag RAISE headless,
#     so this assignment is necessary but NOT sufficient. workbench_gui.sh must
#     also pass `-e HEADLESS=0`; without it this line is silently ignored and you
#     get a windowless session. That asymmetry cost a day in run_gui_local.sh.
ARGS.enable_cameras = True
ARGS.headless = False

app_launcher = AppLauncher(ARGS)
simulation_app = app_launcher.app

# --- everything below needs Kit running --------------------------------------
import carb  # noqa: E402
import omni.ext  # noqa: E402  (ExtensionPathType, for the add_path fallback)
import omni.kit.app  # noqa: E402


def log(msg):
    print(f"[workbench] {msg}", flush=True)
    try:
        # Also through carb, which timestamps and writes promptly. Plain stdout
        # from Kit's python and Kit's own logger reach the tee'd log through
        # different buffers, so their relative order in that file is not
        # reliable -- and reading a startup log where one stream lags the other
        # is how a single event gets mistaken for two.
        carb.log_warn(f"[workbench] {msg}")
    except Exception:  # noqa: BLE001
        pass


def ensure_yaml(pypaths):
    """Make `import yaml` work inside Kit, or say clearly that it does not.

    MEASURED on isaacsim-so101-workshop:2.3.2, and the two measurements disagree,
    which is the whole reason this function exists:

      * the image's BARE interpreter cannot import yaml
        (`python3 -c "import yaml"` -> ModuleNotFoundError);
      * the RUNNING KIT APP can -- this logs "available in Kit's python".

    Those are different environments, and the submitter's extension.toml bets on
    the second ("PyYAML ships with Kit's Python"). That bet is correct here, so
    the staged copy is a no-op on this image. It stays because the bet is not
    correct everywhere, bundle_filter.py needs yaml to read template.yaml, and the
    failure mode without it is a panel that opens and then refuses to filter --
    i.e. a demo that dies on stage rather than in a log.

    APPENDED, not prepended, and only when the import actually fails: if a future
    image does ship PyYAML, Kit's own copy keeps winning and this is a no-op.
    """
    try:
        import yaml  # noqa: F401
        log("yaml: available in Kit's python")
        return True
    except ImportError:
        pass
    for path in pypaths:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)
    try:
        import yaml  # noqa: F401
        log(f"yaml {yaml.__version__}: NOT in Kit's python, using the staged copy "
            f"({os.path.dirname(yaml.__file__)})")
        return True
    except ImportError as exc:
        log(f"ERROR: no PyYAML available ({exc}). The submitter panel will open "
            "but cannot read template.yaml.")
        return False


def add_ext_folders(folders):
    """Register extension SEARCH paths with the running extension manager.

    The documented way to do this from a command line is Kit's ``--ext-folder``
    flag, which is what the extension's README tells a human to use. There is no
    equivalent on Isaac Lab's AppLauncher, and ``--/app/exts/folders`` is
    explicitly undocumented, so this uses the extension manager's own
    ``add_path`` at runtime -- the same API the Extensions window's "add a search
    path" button drives.

    The binding's signature has varied across Kit releases (some builds require
    an explicit ExtensionPathType), so both forms are tried and the outcome is
    logged either way rather than assumed.
    """
    mgr = omni.kit.app.get_app().get_extension_manager()
    for folder in folders:
        if not os.path.isdir(folder):
            log(f"ERROR: --ext-folder {folder} is not a directory")
            continue
        # `import omni.ext` is at module scope, NOT here: a function-local import
        # of a dotted package rebinds the name `omni` as a local for the WHOLE
        # function, so the `omni.kit.app` line above raises UnboundLocalError
        # before this branch can ever run. Observed, and it killed the session.
        ok = None
        try:
            ok = mgr.add_path(folder)
        except TypeError:
            try:
                ok = mgr.add_path(folder, omni.ext.ExtensionPathType.COLLECTION_USER)
            except Exception as exc:  # noqa: BLE001
                log(f"ERROR: add_path({folder}) failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            log(f"ERROR: add_path({folder}) failed: {exc}")
        log(f"ext-folder {folder} -> add_path returned {ok!r}")
    # Some Kit builds want an explicit rescan after a new search path; the one in
    # Isaac Sim 5.1.0 has no refresh_extensions() at all, and add_path alone is
    # enough there (verified: the submitter enables immediately afterwards). So
    # this is conditional rather than a try/except that logs a scary warning on
    # every single startup.
    if hasattr(mgr, "refresh_extensions"):
        try:
            mgr.refresh_extensions()
        except Exception as exc:  # noqa: BLE001
            log(f"WARNING: refresh_extensions() failed: {exc}")


def enable_extensions(names):
    """Enable each extension and REPORT the result, per extension.

    ``set_extension_enabled_immediate`` returns None on some builds, so success
    is verified by asking the manager whether the extension is enabled
    afterwards, not by trusting the return value. A submitter that failed to load
    must be visible here, in the log, and not discovered by a missing menu item
    during a demo.
    """
    mgr = omni.kit.app.get_app().get_extension_manager()
    results = {}
    for name in names:
        try:
            mgr.set_extension_enabled_immediate(name, True)
        except Exception as exc:  # noqa: BLE001
            log(f"ERROR: enabling {name} raised: {exc}")
        enabled = False
        try:
            enabled = bool(mgr.is_extension_enabled(name))
        except Exception:  # noqa: BLE001
            pass
        results[name] = enabled
        log(f"extension {name}: enabled={enabled}")
        if not enabled:
            try:
                info = mgr.get_extension_dict(name)
                log(f"  registry entry: {'present' if info else 'MISSING'}")
            except Exception:  # noqa: BLE001
                log("  registry entry: MISSING")
    return results


def report_menu_registration():
    """Evidence that Tools -> Submit to AWS Deadline Cloud exists in THIS session.

    A menu is drawn, so it cannot be asserted from a log line alone -- but the
    action behind it can. The extension registers exactly one action through
    ``make_menu_item_description`` and one menu item under "Tools", so an action
    registered for isaacsim.deadline.submitter is a direct, checkable proxy for
    the menu item existing, and it is queryable without a screenshot.
    """
    try:
        import omni.kit.actions.core
        registry = omni.kit.actions.core.get_action_registry()
        actions = registry.get_all_actions_for_extension(
            "isaacsim.deadline.submitter")
        log(f"submitter actions registered: {len(actions)}")
        for act in actions:
            log(f"  action id={getattr(act, 'id', '?')!r} "
                f"display={getattr(act, 'display_name', '?')!r}")
    except Exception as exc:  # noqa: BLE001
        log(f"could not read the action registry: {exc}")
    # No menu-tree introspection here on purpose. omni.kit.menu.utils has no
    # documented, stable way to enumerate a group's items -- get_merged_menus()
    # exists on this build but returns nothing that contains our label, so a
    # "NONE FOUND" line from it is a false negative that reads like a failure.
    # The registered action above IS the evidence: extension.py obtains its menu
    # item and its action from the SAME make_menu_item_description() call, so an
    # action registered for this extension means the menu item was created.
    # Confirmed visually on 2026-08-28: Tools -> Submit to AWS Deadline Cloud is
    # drawn, below Isaac's own laid-out entries, exactly as the extension README
    # predicts for items outside its MenuLayout.
    log("Tools -> Submit to AWS Deadline Cloud should now exist "
        "(1 registered action == 1 menu item)")


def show_windows(titles):
    """Open Kit windows by title, so the demo does not start with a menu hunt.

    ``omni.ui.Workspace.show_window(title, True)`` is the documented, version-
    stable way to do this and it drives exactly the same visibility path a menu
    click would: for the submitter that means its
    ``set_visibility_changed_fn`` -> ``build_ui()`` runs, so this also PROVES the
    panel constructs in this session rather than only that its menu item exists.
    """
    import omni.ui as ui
    for title in titles:
        try:
            ok = ui.Workspace.show_window(title, True)
            win = ui.Workspace.get_window(title)
            log(f"window {title!r}: show_window={ok!r} "
                f"exists={win is not None} "
                f"visible={getattr(win, 'visible', None)!r}")
        except Exception as exc:  # noqa: BLE001
            log(f"ERROR: could not show window {title!r}: {exc}")


def run_exec_file(path):
    """exec() a file inside this session, once, from a Kit update callback.

    Deliberately the same shape as the Script Editor's Run button: a plain
    ``exec(source, globals_dict)`` on the main thread while the app is updating.
    That is what makes a --run-script test evidence about the Script Editor path and
    not merely about "some code ran in the process".

    Scheduled one frame LATER rather than called here, so the app is already
    pumping frames when it starts -- the same condition the user's paste happens
    under.
    """
    if not path:
        return
    if not os.path.isfile(path):
        log(f"ERROR: --run-script {path} does not exist")
        return
    state = {"ran": False, "sub": None}

    def on_update(_event):
        if state["ran"]:
            return
        state["ran"] = True
        log(f"--run-script {path}: running")
        try:
            with open(path) as handle:
                source = handle.read()
            exec(compile(source, path, "exec"), {"__name__": "__main__"})  # noqa: S102
            log(f"--run-script {path}: returned")
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            log(f"--run-script {path}: raised (see traceback above)")
        finally:
            state["sub"] = None

    state["sub"] = (
        omni.kit.app.get_app()
        .get_update_event_stream()
        .create_subscription_to_pop(on_update, name="workbench_session.exec")
    )
    # Held on the module so the subscription is not garbage collected before it
    # fires.
    globals()["_EXEC_STATE"] = state


def banner(checkpoint, output_dir, task):
    lines = [
        "=" * 78,
        "WORKBENCH READY -- one Isaac Sim session, two jobs.",
        "",
        "  Submit to the farm:   Tools -> Submit to AWS Deadline Cloud",
        "",
        "  Run one attempt here: Window -> Script Editor, then paste:",
        "",
        "      import workbench_rollout as wr",
        "      wr.run()",
        "",
        "  First call builds the env and loads the policy (~1-2 min); later calls",
        "  reuse both. wr.start() is the same episode driven from Kit's update",
        "  loop instead of blocking the Script Editor.",
        "",
        f"  checkpoint: {checkpoint}",
        f"  output dir: {output_dir}",
        f"  task:       {task}",
        "=" * 78,
    ]
    for line in lines:
        print(f"[workbench] {line}", flush=True)


def main():
    # Every step of the setup is individually survivable. A workbench whose
    # session dies because one helper raised is worse than one with a missing
    # panel: the window IS the demo, and a traceback here used to take Kit down
    # with it. Whatever fails, the log says so and the idle loop still runs.
    try:
        ensure_yaml(ARGS.pypath)
        add_ext_folders(ARGS.ext_folder)
        enable_extensions(ARGS.enable)
        # A few app updates so every enabled extension has actually started and
        # registered its menus before anything is reported about them.
        for _ in range(5):
            simulation_app.update()
        report_menu_registration()
        show_windows(ARGS.show_window)
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        log("ERROR: setup failed (see traceback). The session stays up so the "
            "window is still usable -- fix and relaunch.")

    # Defaults for workbench_rollout, so the user's paste needs no arguments.
    # Environment rather than a module import: importing workbench_rollout here
    # would pull torch and Isaac Lab task registration into startup, which is
    # slower and gives the user's first Script Editor call nothing to do.
    os.environ.setdefault("WORKBENCH_CHECKPOINT", ARGS.checkpoint)
    os.environ.setdefault("WORKBENCH_OUTPUT_DIR", ARGS.output_dir)
    os.environ.setdefault("WORKBENCH_TASK", ARGS.task)

    try:
        version = str(carb.settings.get_settings().get("/app/version") or "")
    except Exception:  # noqa: BLE001
        version = ""
    log(f"Isaac Sim / Kit app version: {version or 'unknown'}")
    banner(ARGS.checkpoint, ARGS.output_dir, ARGS.task)
    run_exec_file(ARGS.exec_file)

    # The session. Nothing here closes the app: it ends when the user closes the
    # window (is_running() goes False) or the container is stopped.
    while simulation_app.is_running():
        simulation_app.update()
    log("window closed -- session ending")
    return 0


if __name__ == "__main__":
    # Deliberately NOT render_rollout.py's `finally: simulation_app.close();
    # os._exit()` shape. That exists so a batch task can never wedge a worker; a
    # workbench must not exit just because a helper raised, or a typo in the
    # Script Editor would take the submitter panel down with it.
    sys.exit(main())
