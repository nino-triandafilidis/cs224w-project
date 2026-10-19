"""Recompute every JTH statistic quoted in the proposal.

Usage: python analysis/jth_stats.py [data_dir]   (default: data/)
Data: JTH, Zenodo record 21390581 (CC BY-NC 4.0). Download with
    for f in candidates.csv jobs.csv history.csv; do
      curl -L -o data/$f "https://zenodo.org/records/21390581/files/$f?download=1"; done
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
TRAIN_END, VAL_YEAR = 2021, 2022  # train <= 2021, val = 2022, test >= 2023

INTERVIEW_STAGES = {"1st Interview", "2nd Interview", "3rd Interview", "4th Interview",
                    "Offer Received", "Offer Accepted"}
STAGE_DATES = ["spontaneous_application_date", "shortlist_date", "qualification_date",
               "resume_sent_to_company_date", "1st_interview_date", "2nd_interview_date",
               "3rd_interview_date", "4th_interview_date", "job_offer_proposed_date",
               "job_offer_accepted_date"]


def load():
    h = pd.read_csv(DATA / "history.csv")
    jobs = pd.read_csv(DATA / "jobs.csv", usecols=["job_id", "create_date"])
    cands = pd.read_csv(DATA / "candidates.csv", usecols=["candidate_id"])
    for col in STAGE_DATES + ["end_of_process_date"]:
        h[col] = pd.to_datetime(h[col], errors="coerce")
    jobs["job_date"] = pd.to_datetime(jobs.create_date, errors="coerce")
    h = h.merge(jobs[["job_id", "job_date"]], on="job_id", how="left")
    h["first_event"] = h[STAGE_DATES].min(axis=1)
    h["interviewed"] = h.last_stage_reached.isin(INTERVIEW_STAGES)
    h["hired"] = h.last_stage_reached.eq("Offer Accepted")
    h["resume_sent"] = h.resume_sent_to_company_date.notna()
    yr = h.job_date.dt.year
    h["split"] = np.select([yr <= TRAIN_END, yr == VAL_YEAR, yr > VAL_YEAR],
                           ["train", "val", "test"], "none")
    return h, jobs, cands


def main():
    h, jobs, cands = load()
    print("== Size")
    print(f"candidates {len(cands):,}  jobs {len(jobs):,}  pairs {len(h):,}")
    print(f"pairs per job median {h.groupby('job_id').size().median():.0f}")
    once = (h.groupby("candidate_id").size() == 1).mean()
    print(f"candidates appearing in exactly one pair {once:.1%}")

    print("\n== Labels")
    print(f"interviewed {h.interviewed.mean():.1%} ({h.interviewed.sum():,})  "
          f"hired {h.hired.mean():.1%} ({h.hired.sum():,})")
    print(f"interviewed pairs whose resume was sent first "
          f"{h.loc[h.interviewed, 'resume_sent'].mean():.1%}")

    print("\n== Split by job opening year")
    g = h.groupby("split").agg(jobs=("job_id", "nunique"), pairs=("job_id", "size"),
                               interviewed=("interviewed", "sum"), hired=("hired", "sum"))
    print(g.loc[["train", "val", "test"]])
    t25 = h[(h.split == "test") & (h.job_date.dt.year == 2025)]
    print(f"test interviewed from 2025 jobs: {t25.interviewed.sum()}")
    tr = h[h.split == "train"]
    for year in (VAL_YEAR, VAL_YEAR + 1):
        print(f"leakage cut: train pairs with first event >= {year} "
              f"{(tr.first_event.dt.year >= year).mean():.1%}")
    # graph seen at test time: pairs of pre-2023 jobs whose first event is before 2023
    seen = set(h[h.split.isin(["train", "val"]) & (h.first_event.dt.year <= VAL_YEAR)].candidate_id)
    te = h[h.split == "test"]
    cold = ~te.candidate_id.isin(seen)
    print(f"test pairs whose candidate has no earlier pair in the graph {cold.mean():.1%} "
          f"(interviewed only {cold[te.interviewed].mean():.1%})")

    print("\n== Time to first interview / truncation")
    d = (h["1st_interview_date"] - h.job_date).dt.days.dropna()
    d = d[d >= 0]
    print(f"days job opening -> 1st interview: median {d.median():.0f}, 90th pct {d.quantile(.9):.0f}")
    print(f"last dated event {h[STAGE_DATES + ['end_of_process_date']].max().max().date()}")
    q = h.groupby(h.job_date.dt.to_period("Q")).interviewed.mean()
    print("interview rate by job-opening quarter (2022-2025):")
    print(q[q.index >= pd.Period("2022Q1")].map("{:.1%}".format).to_string())

    print("\n== Recruiter benchmark (test jobs)")
    te = h[h.split == "test"]
    per = te.groupby("job_id").agg(n=("job_id", "size"), sent=("resume_sent", "sum"),
                                   iv=("interviewed", "sum"))
    # usable: ranking is informative (>=1 interviewed and >=1 not) and the recruiter sent >=1 resume
    usable = per[(per.iv > 0) & (per.n > per.iv) & (per.sent > 0)]
    tu = te[te.job_id.isin(usable.index)]
    rec_hit = tu.loc[tu.resume_sent, "interviewed"].mean()
    # random order picking the same number of candidates per job as the recruiter
    rand_hit = (usable.sent * usable.iv / usable.n).sum() / usable.sent.sum()
    print(f"usable test jobs: {len(usable)}  pairs per usable job {usable.n.mean():.1f}")
    print(f"resumes sent per usable job: {usable.sent.mean():.1f}")
    print(f"interview rate of sent resumes {rec_hit:.0%} vs random picks {rand_hit:.0%}")


if __name__ == "__main__":
    main()
