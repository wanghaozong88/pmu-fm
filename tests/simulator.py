"""Run only as `python -m tests.simulator`; never installed as product input."""

import argparse
import time

from tests.helpers import Simulator


def main():
    parser = argparse.ArgumentParser(description="Test-only fresh online PMU/PDC source on loopback")
    parser.add_argument("--rate", type=int, default=20)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--drop", default="", help="Comma-separated frame indexes to omit")
    parser.add_argument("--duplicate", default="", help="Comma-separated frame indexes to duplicate")
    parser.add_argument("--reorder", action="store_true")
    args = parser.parse_args()

    def indexes(text):
        return [int(x) for x in text.split(",") if x]

    if not 1 <= args.rate <= 120:
        parser.error("--rate must be 1..120")
    with Simulator(
        multi=not args.single,
        rate=args.rate,
        drop=indexes(args.drop),
        duplicate=indexes(args.duplicate),
        reorder=args.reorder,
    ) as source:
        print(
            f"TEST ONLY: 127.0.0.1:{source.port}, stream ID 1; PMUs 10"
            + (" and 20" if not args.single else ""),
            flush=True,
        )
        print("Fresh synthetic online data. Ctrl+C stops this test source.", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
