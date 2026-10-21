"""Small CLI. No Tk import unless the GUI subcommand is selected."""

import argparse
import json
import sys
import time

from .config import ConfigError, load, sample_plan, validate
from .network import inspect_source
from .protocol import channels, parse_config
from .runtime import Runtime
from .test_config_input import load_test_config_frame


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pmu-fm", description="Online PMU to local Chronos-2 / TimesFM / Timestral bridge"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    gui = sub.add_parser("gui", help="Open the small desktop window")
    gui.add_argument("config", nargs="?")
    check = sub.add_parser("check", help="Check configuration without connecting by default")
    check.add_argument("config")
    check.add_argument(
        "--online", action="store_true", help="Explicitly connect and request CFG-2 (never START/STOP)"
    )
    check.add_argument(
        "--schema-only",
        action="store_true",
        help="Check settings without requiring weights or channel selection",
    )
    run = sub.add_parser("run", help="Run online until Ctrl+C")
    run.add_argument("config")
    for command in (check, run):
        command.add_argument(
            "--test-config-frame", metavar="HEX_FILE",
            help="TEST ONLY: import one captured CFG-2 hex file; skip initial online CFG-2 request",
        )
    args = parser.parse_args(argv)
    if args.command == "check" and args.online and args.test_config_frame:
        parser.error("--online and --test-config-frame are mutually exclusive")
    try:
        if args.command == "gui":
            try:
                from .gui import launch

                launch(args.config)
            except ImportError as exc:
                raise ConfigError(
                    "Tk is unavailable. Install python3-tk on Linux or Tcl/Tk from the Windows Python installer"
                ) from exc
            return 0
        cfg = load(args.config)
        imported = (
            load_test_config_frame(args.test_config_frame, cfg["source"]["id_code"])
            if args.test_config_frame else None
        )
        if args.command == "check":
            notes = validate(cfg, require_weights=not args.schema_only, require_channels=not args.schema_only)
            print("Static configuration check passed (model execution not tested).")
            for note in notes:
                print(note)
            if args.online or imported is not None:
                selected = {}

                def record(kind, _message, **details):
                    if kind == "command_version_selected":
                        selected.update(details)

                source = parse_config(imported) if imported else inspect_source(cfg["source"], emit=record)
                print(
                    json.dumps(
                        {
                            "configuration": source.to_dict(),
                            "command_version": selected.get("command_version"),
                            "configuration_origin": "test_import" if imported else "online",
                            "channels": channels(source),
                            "sample_plan": sample_plan(cfg, source.interval, source.time_base),
                        },
                        indent=2,
                        default=str,
                    )
                )
            return 0
        runtime = Runtime(cfg, _test_config_frame=imported)
        runtime.start()
        print(f"Run {runtime.run_id}; metadata/results: {runtime.run_directory}; Ctrl+C to stop.", flush=True)
        try:
            while runtime.snapshot()["running"]:
                time.sleep(2)
                print(json.dumps(runtime.snapshot(), default=str), flush=True)
        except KeyboardInterrupt:
            print("Stopping and flushing local resources...", flush=True)
        finally:
            runtime.stop()
        return 1 if runtime.state["phase"] == "failed" else 0
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        print(f"pmu-fm: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
