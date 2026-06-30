#!/usr/bin/env python3
"""Wait for CAVE's materialize /query endpoint to recover, then run a command.

The CAVE materialization ``/query`` POST endpoint can go down server-side: GETs
to the host keep working (and return ``405`` on the query URL in <1s), but POSTs
to ``/query`` accept the connection and never send a response body. Every spatial
synapse query routes through that one POST, so the whole pipeline stalls.

This watcher probes the endpoint cheaply on a fixed interval, each probe bounded
by a hard wall-clock deadline (so it never hangs), and reports the moment the
endpoint answers again. Optionally it then launches a command — typically the
real run — so a pipeline finishes unattended as soon as CAVE is back.

Examples
--------
    # Just watch and report when CAVE recovers:
    python scripts/wait_for_cave.py --token $CAVE_TOKEN

    # Watch, then run the full pipeline (with synapse caching) on recovery:
    python scripts/wait_for_cave.py --token $CAVE_TOKEN \\
        --synapse-cache-dir ./syn_cache \\
        --run "python scripts/v117_pcfg.py --token $CAVE_TOKEN \\
               --n-boxes 1 --side-um 20 --use-learned \\
               --synapse-cache-dir ./syn_cache"
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

# Allow running directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
for _noisy in ("caveclient", "urllib3", "CAVEclient"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)
log = logging.getLogger("wait_for_cave")

# Same validated, densely-proofread center as the rest of the pipeline.
PROBE_CENTER_NM = (733_592, 513_592, 595_640)


def parse_args() -> argparse.Namespace:
    import os

    p = argparse.ArgumentParser(
        description="Poll CAVE /query until it recovers, then optionally run a command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--token", default=os.environ.get("CAVE_TOKEN"),
                   help="CAVE auth token (or set CAVE_TOKEN)")
    p.add_argument("--version", type=int, default=117,
                   help="Materialization version to probe")
    p.add_argument("--probe-side-um", type=float, default=4.0,
                   help="Side length of the tiny probe box (kept small = cheap)")
    p.add_argument("--probe-timeout", type=float, default=30.0,
                   help="Per-probe wall-clock deadline in seconds")
    p.add_argument("--interval", type=float, default=120.0,
                   help="Seconds between probes while the endpoint is down")
    p.add_argument("--max-wait", type=float, default=3600.0,
                   help="Give up after this many seconds (0 = wait forever)")
    p.add_argument("--synapse-cache-dir", default=None,
                   help="If set, the probe writes its result here (warms the cache)")
    p.add_argument("--run", default=None,
                   help="Shell command to execute once the endpoint recovers")
    return p.parse_args()


def probe_once(token: str | None, version: int, side_um: float,
               timeout_s: float, cache_dir: str | None) -> tuple[bool, str]:
    """One bounded probe of the /query endpoint via a tiny synapse fetch.

    Returns (is_up, detail). A TimeoutError (our watchdog) means the endpoint is
    still stalled; any other exception is reported but also treated as "down".
    """
    from neuronauts.fetch import fetch_synapses, make_cube_bbox_nm

    bbox = make_cube_bbox_nm(PROBE_CENTER_NM, side_um=side_um)
    t0 = time.time()
    try:
        syn = fetch_synapses(
            bbox, version=version, token=token,
            timeout_s=timeout_s, max_retries=1,
            cache_dir=cache_dir,
        )
    except TimeoutError:
        return False, f"stalled (no response in {timeout_s:.0f}s)"
    except Exception as exc:  # network reset, auth, etc. — report and keep waiting
        return False, f"{type(exc).__name__}: {str(exc)[:120]}"
    dt = time.time() - t0
    return True, f"responded in {dt:.1f}s ({syn.n_synapses} synapses in {side_um:.0f}µm probe box)"


def main() -> int:
    args = parse_args()
    if not args.token:
        log.error("No token. Pass --token or set CAVE_TOKEN.")
        return 2

    deadline = None if args.max_wait <= 0 else time.time() + args.max_wait
    log.info(
        "Watching CAVE v%d /query (probe every %.0fs, %.0fs deadline/probe, "
        "give up after %s)...",
        args.version, args.interval, args.probe_timeout,
        "never" if deadline is None else f"{args.max_wait:.0f}s",
    )

    attempt = 0
    while True:
        attempt += 1
        is_up, detail = probe_once(
            args.token, args.version, args.probe_side_um,
            args.probe_timeout, args.synapse_cache_dir,
        )
        if is_up:
            log.info("✅ CAVE /query RECOVERED — %s", detail)
            break
        if deadline is not None and time.time() >= deadline:
            log.error("Gave up after %.0fs (%d probes). Last: %s",
                      args.max_wait, attempt, detail)
            return 1
        log.info("attempt %d: still down — %s; next probe in %.0fs",
                 attempt, detail, args.interval)
        # Sleep, but don't overshoot the overall deadline.
        nap = args.interval
        if deadline is not None:
            nap = min(nap, max(1.0, deadline - time.time()))
        time.sleep(nap)

    if not args.run:
        log.info("No --run command; exiting on recovery.")
        return 0

    log.info("Launching run command:\n  %s", args.run)
    completed = subprocess.run(args.run, shell=True)
    log.info("Run command exited with code %d", completed.returncode)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
