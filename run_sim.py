"""Run every scenario x replicate and save raw rows + a summary table.

Usage: python3 run_sim.py --reps 200 --workers 1
Each replicate's seed is (scenario letter, replicate number), so any row can be rerun alone.
Replicates already in results/replicates.csv are skipped (safe to stop and restart).
"""
import argparse
import os
import time
from multiprocessing import Pool

import pandas as pd

from ntsim.core import run_replicate

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "results", "replicates.csv")


ORDER = "minp"


def job(args):
    return run_replicate(*args, order=ORDER)


def summarize(df):
    g = df.groupby(["scenario", "rule"])
    out = pd.DataFrame({
        "reps": g.size(),
        "clean_drop_rate": g.clean_drop_rate.mean(),
        "any_clean_dropped": g.clean_dropped.apply(lambda x: (x > 0).mean()),
        "planted_catch_rate": g.planted_catch_rate.mean(),
        "calib_pass_rate": g.calib_pass.mean(),
        "pass_when_truth_bad": g.apply(lambda d: d.loc[~d.truth_ok, "calib_pass"].mean()
                                       if (~d.truth_ok).any() else float("nan")),
        "mean_false_disc": g.false_disc.mean(),
        "mean_fdp": g.fdp.mean(),
        "mean_power": g.power.mean(),
    }).reset_index()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--order", default="minp", choices=["minp", "count"],
                    help="how R1/R2 pick the worst guide (count = the first run, results/v1_count_order)")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    global ORDER, RAW
    ORDER = a.order
    RAW = os.path.join(a.out, "replicates.csv")
    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    done = set()
    if os.path.exists(RAW):
        old = pd.read_csv(RAW)
        done = set(zip(old.scenario, old.rep))
    todo = [(s, r) for s in "abcd" for r in range(a.reps) if (s, r) not in done]
    print(f"{len(todo)} replicates to run", flush=True)
    t0 = time.time()
    pool = Pool(a.workers) if a.workers > 1 else None
    mapper = pool.map if pool else (lambda f, xs: list(map(f, xs)))
    for i in range(0, len(todo), a.chunk):
        rows = [row for res in mapper(job, todo[i:i + a.chunk]) for row in res]
        pd.DataFrame(rows).to_csv(RAW, mode="a", header=not os.path.exists(RAW), index=False)
        print(f"{i + len(todo[i:i + a.chunk])}/{len(todo)} done, {time.time() - t0:.0f}s", flush=True)
    if pool:
        pool.close()
    df = pd.read_csv(RAW)
    summ = summarize(df)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    diag = pd.crosstab(df[df.rule == "R0_keep_all"].scenario, df[df.rule == "R0_keep_all"].verdict)
    diag.to_csv(os.path.join(a.out, "diagnostic_verdicts.csv"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(summ.round(3).to_string())
    print(diag)


if __name__ == "__main__":
    main()
