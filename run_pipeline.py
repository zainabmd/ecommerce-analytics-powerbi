"""
One-command pipeline runner.

    python run_pipeline.py                 # generate -> (load to Postgres) -> segment -> EDA report -> Power BI
    python run_pipeline.py --customers 20000 --skip-eda

The Postgres load step runs only when DATABASE_URL is set; otherwise the
pipeline works entirely from Parquet. The last step exports the Power BI star schema
(powerbi/data/*.parquet); open powerbi/Ecommerce Analytics.pbip and click Refresh to load it.
"""
from __future__ import annotations

import argparse
import sys
import time

from src import eda_report, export_powerbi, generate_data, load_to_postgres, rfm_segmentation
from src.config import DATABASE_URL, get_logger

log = get_logger("pipeline")


def main() -> int:
    p = argparse.ArgumentParser(description="Run the full e-commerce analytics pipeline.")
    p.add_argument("--customers", type=int, default=None, help="override number of simulated customers")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--k", type=int, default=None, help="force K-Means k")
    p.add_argument("--skip-generate", action="store_true", help="reuse existing Parquet extracts")
    p.add_argument("--skip-eda", action="store_true")
    p.add_argument("--skip-powerbi", action="store_true", help="skip the Power BI data export")
    args = p.parse_args()

    t0 = time.perf_counter()
    steps: list[tuple[str, callable]] = []
    if not args.skip_generate:
        gen_args = []
        if args.customers:
            gen_args += ["--customers", str(args.customers)]
        if args.seed is not None:
            gen_args += ["--seed", str(args.seed)]
        steps.append(("generate", lambda: generate_data.main(gen_args)))
    if DATABASE_URL:
        steps.append(("load_to_postgres", lambda: load_to_postgres.main([])))
    else:
        log.info("DATABASE_URL not set – skipping PostgreSQL load (Parquet mode)")
    steps.append(("segment", lambda: rfm_segmentation.main(["--k", str(args.k)] if args.k else [])))
    if not args.skip_eda:
        steps.append(("eda_report", eda_report.main))
    if not args.skip_powerbi:
        steps.append(("powerbi_export", lambda: 0 if export_powerbi.run() else 1))

    for name, fn in steps:
        log.info("▶ %s", name)
        t = time.perf_counter()
        rc = fn()
        if rc != 0:
            log.error("Step %s failed with exit code %s – aborting", name, rc)
            return rc
        log.info("✔ %s (%.1fs)", name, time.perf_counter() - t)
    log.info("Pipeline complete in %.1fs. Open powerbi/Ecommerce Analytics.pbip in Power BI Desktop "
             "(Home > Refresh on first open).", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
