"""Entrypoint. One tick per invocation, then exit.

    python run.py              # respects dry_run in config.toml
    python run.py --check      # diagnose both API keys and exit
    python run.py --live       # force real orders regardless of config
    python run.py --dry-run    # force dry run regardless of config
    python run.py --compress   # rewrite the memory summary and print it
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from agent.config import load_config
from agent.runner import Runner, utc_stamp

log = logging.getLogger("manifold-agent")


async def check(cfg) -> int:
    """Answer the only question that matters when nothing works: which side is broken?

    Prints the Manifold identity, the models the LLM key can actually reach, and the
    result of one plain call and one grounded call.
    """
    from agent.llm import GeminiClient, build_llm
    from agent.manifold import ManifoldClient

    ok = True
    print(f"provider     {cfg.llm.provider}\nmodel        {cfg.llm.model}\n")

    client = ManifoldClient(cfg.manifold)
    try:
        me = await client.me()
        print(f"manifold     OK, @{me.get('username')} "
              f"balance M${float(me.get('balance') or 0):,.0f}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"manifold     FAILED  {exc}")
    finally:
        await client.aclose()

    llm = build_llm(cfg.llm)
    try:
        if isinstance(llm, GeminiClient):
            try:
                models = await llm.list_models()
                here = cfg.llm.model in models
                print(f"models       {len(models)} reachable, "
                      f"{cfg.llm.model} is {'present' if here else 'NOT in the list'}")
                if not here:
                    ok = False
                    flash = [m for m in models if "flash" in m][:8]
                    print(f"             try one of: {', '.join(flash) or 'none found'}")
            except Exception as exc:  # noqa: BLE001
                ok = False
                print(f"models       FAILED  {exc}")

        try:
            r = await llm.generate("Reply with the single word: ok", attempts=1)
            print(f"generate     OK, returned {r.text.strip()[:40]!r}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"generate     FAILED  {exc}")

        if llm.supports_search:
            try:
                r = await llm.generate(
                    "What is today's date according to search?", grounded=True, attempts=1
                )
                print(f"search       OK, {len(r.citations)} citation(s)")
            except Exception as exc:  # noqa: BLE001
                ok = False
                print(f"search       FAILED  {exc}")
                print("             set use_search = false in config.toml to run without it")
        else:
            print("search       disabled")
    finally:
        await llm.aclose()

    print("\n" + ("all good" if ok else "something above is broken"))
    return 0 if ok else 1


async def amain(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.live:
        cfg.manifold.dry_run = False
    if args.dry_run:
        cfg.manifold.dry_run = True

    if args.check:
        return await check(cfg)

    runner = Runner(cfg)
    try:
        if args.compress:
            await runner.memory.compress()
            print(runner.memory.state.get("summary") or "(empty)")
            return 0

        log.info("tick %s  mode=%s", utc_stamp(),
                 "DRY RUN" if cfg.manifold.dry_run else "LIVE")
        report = await runner.tick()
        rendered = report.render()
        print("\n" + rendered)

        # In CI this shows up as the run's summary page, which is the closest thing
        # to a dashboard now that there is no server.
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            mode = "dry run" if cfg.manifold.dry_run else "live"
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(f"### tick {utc_stamp()} ({mode})\n\n```\n{rendered}\n```\n")
        return 0
    finally:
        await runner.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Manifold Markets trading agent, one tick")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.toml"))
    parser.add_argument("--live", action="store_true", help="force real orders")
    parser.add_argument("--dry-run", action="store_true", help="force dry run")
    parser.add_argument("--compress", action="store_true", help="rewrite memory summary")
    parser.add_argument("--check", action="store_true", help="diagnose keys, models, search")
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
