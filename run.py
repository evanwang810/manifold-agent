"""Entrypoint. One tick per invocation, then exit.

    python run.py              # respects dry_run in config.toml
    python run.py --live       # force real orders regardless of config
    python run.py --dry-run    # force dry run regardless of config
    python run.py --compress   # rewrite the memory summary and print it
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from agent.config import load_config
from agent.runner import Runner, utc_stamp

log = logging.getLogger("manifold-agent")


async def amain(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.live:
        cfg.manifold.dry_run = False
    if args.dry_run:
        cfg.manifold.dry_run = True

    runner = Runner(cfg)
    try:
        if args.compress:
            await runner.memory.compress()
            print(runner.memory.state.get("summary") or "(empty)")
            return 0

        log.info("tick %s  mode=%s", utc_stamp(),
                 "DRY RUN" if cfg.manifold.dry_run else "LIVE")
        report = await runner.tick()
        print("\n" + report.render())
        return 0
    finally:
        await runner.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Manifold Markets trading agent, one tick")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.toml"))
    parser.add_argument("--live", action="store_true", help="force real orders")
    parser.add_argument("--dry-run", action="store_true", help="force dry run")
    parser.add_argument("--compress", action="store_true", help="rewrite memory summary")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)-20s %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        return asyncio.run(amain(args))
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
