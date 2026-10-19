"""Quick signal check: does a colleague/firm signal add lift over an adviser's own features?

Usage: .venv/bin/python analysis/iapd_signal_check.py [T] [T_next] [T_prev]
  defaults: T=2018-03-26 (snapshot), label window ends at the next snapshot 2019-03-25,
  history window starts at T_prev=2017-03-26.
Needs data/iapd/parsed/<T>/ and data/iapd/spells.csv.gz, moves.csv.gz (iapd_careers.py).

Population: advisers registered in snapshot T. Label: a personal (non-mass) firm move in
(T, T_next]. Excluded: exits, mass movers, and advisers absent from snapshot T_next without
a recorded move or exit (unobserved).
Baseline 2: gradient boosting on the adviser's own features.
Baseline 3: baseline 2 + hand-built colleague and firm features.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit

T = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2018-03-26")
T_NEXT = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2019-03-25")
T_PREV = pd.Timestamp(sys.argv[3] if len(sys.argv) > 3 else "2017-03-26")
D = Path("data/iapd")
P = D / "parsed" / str(T.date())

ce = pd.read_csv(P / "crnt_emp.csv.gz", dtype=str, parse_dates=["first_reg_dt"])
ind = pd.read_csv(P / "indvl.csv.gz", dtype={"indvl_pk": str})
br = pd.read_csv(P / "branch.csv.gz", dtype=str)
ex = pd.read_csv(P / "exam.csv.gz", dtype=str)
sp = pd.read_csv(D / "spells.csv.gz", dtype=str, parse_dates=["begin_dt", "end_dt"])
mv = pd.read_csv(D / "moves.csv.gz", dtype={"indvl_pk": str, "from_org": str, "to_org": str},
                 parse_dates=["move_dt", "end_dt", "begin_dt"])
nxt = pd.read_csv(D / "parsed" / str(T_NEXT.date()) / "indvl.csv.gz", dtype={"indvl_pk": str})

# ---- label
pop = ind.indvl_pk
win = (mv.move_dt > T) & (mv.move_dt <= T_NEXT)
movers = set(mv.loc[win & ~mv.mass, "indvl_pk"])
mass = set(mv.loc[win & mv.mass, "indvl_pk"]) - movers
open_at_T = sp[(sp.begin_dt <= T) & (sp.end_dt.isna() | (sp.end_dt > T))]
last_end = open_at_T.groupby("indvl_pk").end_dt.max()
still_open = open_at_T.groupby("indvl_pk").end_dt.apply(lambda x: x.isna().any())
exits = set(last_end[(last_end <= T_NEXT) & ~still_open].index) - movers - mass
unobserved = set(pop) - set(nxt.indvl_pk) - movers - mass - exits
keep = pop[~pop.isin(mass | exits | unobserved)]
y = keep.isin(movers).astype(int).values

# ---- own features
own = pd.DataFrame({"indvl_pk": keep})
cur = ce.groupby("indvl_pk").agg(n_cur_firms=("org_pk", "nunique"), cur_start=("first_reg_dt", "min"),
                                 org_pk=("org_pk", "first"))
own = own.merge(cur, on="indvl_pk", how="left")
own["tenure_days"] = (T - own.cur_start).dt.days
past = sp[sp.begin_dt <= T]
career = past.groupby("indvl_pk").agg(career_start=("begin_dt", "min"), n_firms=("org_pk", "nunique"))
own = own.merge(career, on="indvl_pk", how="left")
own["career_days"] = (T - own.career_start).dt.days
own["n_past_moves"] = own.indvl_pk.map(mv[mv.move_dt <= T].groupby("indvl_pk").size()).fillna(0)
exams = ex.assign(exam_cd=ex.exam_cd.str.upper()).groupby("indvl_pk").exam_cd.agg(set)
own["n_exams"] = own.indvl_pk.map(exams.map(len)).fillna(0)
for code in ["S7", "S66", "S65", "S63", "S24"]:
    own[f"exam_{code}"] = own.indvl_pk.map(exams.map(lambda s, c=code: c in s)).fillna(False).astype(int)
own = own.merge(ind[["indvl_pk", "n_drp", "has_termination", "has_cust_comp"]], on="indvl_pk", how="left")
OWN = ["n_cur_firms", "tenure_days", "career_days", "n_firms", "n_past_moves", "n_exams",
       "exam_S7", "exam_S66", "exam_S65", "exam_S63", "exam_S24", "n_drp", "has_termination", "has_cust_comp"]

# ---- colleague and firm features (only events before T)
# branch = (firm, 5-digit zip) of current employment; colleagues = same branch at T
b = br[br.postl_cd.fillna("").str.len() == 5].drop_duplicates(["indvl_pk", "org_pk", "postl_cd"])
b["branch"] = b.org_pk + "|" + b.postl_cd
branch_size = b.groupby("branch").indvl_pk.nunique()
# people who left each branch in the past 12 months: seen in snapshot T_PREV-ish? use moves out of firm
# with origin branch unknown; approximate a branch's recent leavers by advisers whose ended spell at
# that firm had the same zip in the T_PREV snapshot.
prev_dir = max((p for p in (D / "parsed").iterdir() if pd.Timestamp(p.name) <= T_PREV), key=lambda p: p.name)
bp = pd.read_csv(prev_dir / "branch.csv.gz", dtype=str)
bp = bp[bp.postl_cd.fillna("").str.len() == 5].drop_duplicates(["indvl_pk", "org_pk", "postl_cd"])
bp["branch"] = bp.org_pk + "|" + bp.postl_cd
hist = (mv.move_dt > pd.Timestamp(prev_dir.name)) & (mv.move_dt <= T)
left = mv[hist & ~mv.mass][["indvl_pk", "from_org"]].rename(columns={"from_org": "org_pk"})
left_b = left.merge(bp[["indvl_pk", "org_pk", "branch"]], on=["indvl_pk", "org_pk"])
branch_leavers = left_b.groupby("branch").indvl_pk.nunique()
prev_branch_size = bp.groupby("branch").indvl_pk.nunique()
bf = b[b.indvl_pk.isin(keep)].copy()
bf["branch_size"] = bf.branch.map(branch_size)
bf["branch_leave_share"] = (bf.branch.map(branch_leavers).fillna(0) / bf.branch.map(prev_branch_size)).fillna(0)
bf["branch_leavers"] = bf.branch.map(branch_leavers).fillna(0)
colleague = bf.groupby("indvl_pk").agg(branch_size=("branch_size", "max"),
                                       branch_leave_share=("branch_leave_share", "max"),
                                       branch_leavers=("branch_leavers", "max"))
# firm headcount change and firm outflow over the past year
firm_now = ce.groupby("org_pk").indvl_pk.nunique()
ce_prev = pd.read_csv(prev_dir / "crnt_emp.csv.gz", dtype=str, usecols=["indvl_pk", "org_pk"])
firm_prev = ce_prev.groupby("org_pk").indvl_pk.nunique()
firm_out = left.groupby("org_pk").indvl_pk.nunique()
firm = pd.DataFrame({"firm_size": firm_now})
firm["firm_growth"] = (firm_now / firm_prev.reindex(firm_now.index)).replace([np.inf], np.nan)
firm["firm_outflow_share"] = (firm_out.reindex(firm_now.index).fillna(0) / firm_prev.reindex(firm_now.index))
own = own.merge(colleague, on="indvl_pk", how="left").merge(firm, left_on="org_pk", right_index=True, how="left")
NET = OWN + ["branch_size", "branch_leave_share", "branch_leavers", "firm_size", "firm_growth", "firm_outflow_share"]


def precision_at(y_true, score, frac):
    k = max(int(len(score) * frac), 1)
    return y_true[np.argsort(-score)[:k]].mean()


# split by firm so colleagues never sit in both train and test
split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
tr, te = next(split.split(own, y, groups=own.org_pk.fillna("na")))
print(f"T={T.date()} label window to {T_NEXT.date()}, history from {prev_dir.name}")
print(f"population {len(pop):,}; kept {len(keep):,} (excluded: mass {len(mass):,}, exits {len(exits):,}, "
      f"unobserved {len(unobserved):,}); movers {y.sum():,}; base rate {y.mean():.2%}")
tenure_rank = -own.tenure_days.fillna(own.tenure_days.median()).values  # baseline 1: shortest tenure first? test both
for name, cols in [("B2 own features", OWN), ("B3 own + colleague/firm", NET)]:
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    clf.fit(own.iloc[tr][cols], y[tr])
    s = clf.predict_proba(own.iloc[te][cols])[:, 1]
    print(f"{name:28s} PR-AUC {average_precision_score(y[te], s):.3f}  "
          f"P@1% {precision_at(y[te], s, .01):.3f}  P@5% {precision_at(y[te], s, .05):.3f}  "
          f"(base rate {y[te].mean():.3f})")
for name, sc in [("B1 shortest tenure first", tenure_rank), ("B1 longest tenure first", -tenure_rank)]:
    s = sc[te]
    print(f"{name:28s} PR-AUC {average_precision_score(y[te], s):.3f}  P@1% {precision_at(y[te], s, .01):.3f}")
