"""Rebuild adviser careers from parsed IAPD snapshots and check feasibility.

Usage: python analysis/iapd_careers.py [parsed_dir]   (default data/iapd/parsed)
Writes data/iapd/spells.csv.gz and data/iapd/moves.csv.gz, prints go/no-go stats.

Definitions
- spell: one adviser registered with one firm (org_pk) from begin to end (end empty = open).
  Current employments give open spells (start = earliest current registration); previous
  registrations give closed spells. Spells are unioned over all snapshots.
- move: a spell at firm A ends and a spell at a different firm B starts within GAP_DAYS after
  (or up to OVERLAP_DAYS before) A's end.
- mass move: a firm pair (A -> B) with >= MASS_N advisers whose moves fall within MASS_WINDOW
  days of each other; treated as a merger / rebrand / acquisition, not a personal choice.
- exit: an adviser's last spell ends and no later spell starts within GAP_DAYS.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PARSED = Path(sys.argv[1] if len(sys.argv) > 1 else "data/iapd/parsed")
OUTDIR = PARSED.parent
GAP_DAYS, OVERLAP_DAYS = 90, 30
MASS_N, MASS_WINDOW = 20, 30
HORIZON = pd.DateOffset(months=12)


def load_snapshots():
    snaps = sorted(p for p in PARSED.iterdir() if (p / "_done").exists())
    return [(pd.Timestamp(p.name), p) for p in snaps]


def build_spells(snaps):
    rows, present = [], []
    for date, p in snaps:
        ce = pd.read_csv(p / "crnt_emp.csv.gz", dtype=str)
        pr = pd.read_csv(p / "prev_reg.csv.gz", dtype=str)
        ce = ce.rename(columns={"first_reg_dt": "begin_dt"}).assign(end_dt=np.nan)
        rows.append(ce[["indvl_pk", "org_pk", "org_nm", "begin_dt", "end_dt"]].assign(seen=date))
        rows.append(pr[["indvl_pk", "org_pk", "org_nm", "begin_dt", "end_dt"]].assign(seen=date))
        ind = pd.read_csv(p / "indvl.csv.gz", dtype={"indvl_pk": str}, usecols=["indvl_pk"])
        present.append(ind.assign(snap=date))
    s = pd.concat(rows, ignore_index=True)
    s["begin_dt"] = pd.to_datetime(s.begin_dt, errors="coerce")
    s["end_dt"] = pd.to_datetime(s.end_dt, errors="coerce")
    s = s.dropna(subset=["begin_dt", "org_pk"])
    # one spell per (adviser, firm, begin): keep the latest view (closed beats open)
    s = s.sort_values(["seen", "end_dt"], na_position="first")
    s = s.groupby(["indvl_pk", "org_pk", "begin_dt"], as_index=False).agg(
        org_nm=("org_nm", "last"), end_dt=("end_dt", "last"), last_seen=("seen", "max"))
    return s, pd.concat(present, ignore_index=True)


def find_moves(s):
    s = s.sort_values(["indvl_pk", "begin_dt"])
    a = s[s.end_dt.notna()][["indvl_pk", "org_pk", "end_dt"]].rename(columns={"org_pk": "from_org"})
    b = s[["indvl_pk", "org_pk", "begin_dt"]].rename(columns={"org_pk": "to_org"})
    m = a.merge(b, on="indvl_pk")
    gap = (m.begin_dt - m.end_dt).dt.days
    m = m[(m.from_org != m.to_org) & (gap <= GAP_DAYS) & (gap >= -OVERLAP_DAYS)]
    # the first new firm after each ended spell
    m = m.sort_values("begin_dt").groupby(["indvl_pk", "from_org", "end_dt"], as_index=False).first()
    m["move_dt"] = m.begin_dt
    # mass moves: many advisers, same firm pair, within a short window
    m = m.sort_values("move_dt")
    m["mass"] = False
    for (fa, fb), g in m.groupby(["from_org", "to_org"]):
        if len(g) < MASS_N:
            continue
        t = g.move_dt.values.astype("datetime64[D]").astype(int)
        lo = np.searchsorted(t, t - MASS_WINDOW, "left")
        hi = np.searchsorted(t, t + MASS_WINDOW, "right")
        m.loc[g.index[(hi - lo) >= MASS_N], "mass"] = True
    return m


def cutoff_stats(s, m, present, snaps):
    snap_dates = [d for d, _ in snaps]
    last_obs = max(snap_dates)
    personal = m[~m.mass]
    seen = present.groupby("indvl_pk").snap.agg(["min", "max"])
    out = []
    for T in pd.date_range("2016-06-30", last_obs - HORIZON, freq="6MS") + pd.offsets.MonthEnd(0):
        if T + HORIZON > last_obs:
            break
        nxt = min(d for d in snap_dates if d >= T + HORIZON)
        active = s[(s.begin_dt <= T) & (s.end_dt.isna() | (s.end_dt > T))]
        # an open spell is only trusted if the adviser still appears in a snapshot on or after T
        ok = seen.index[seen["max"] >= T]
        active = active[active.indvl_pk.isin(ok) | active.end_dt.notna()]
        pop = active.indvl_pk.unique()
        win = lambda d: (d > T) & (d <= T + HORIZON)
        movers = set(personal.loc[win(personal.move_dt) & personal.indvl_pk.isin(pop), "indvl_pk"])
        mass_movers = set(m.loc[m.mass & win(m.move_dt) & m.indvl_pk.isin(pop), "indvl_pk"]) - movers
        # exits: active at T, every spell open at T ends in the window, and no move
        ended = active.groupby("indvl_pk").end_dt.max()
        has_open = active.groupby("indvl_pk").end_dt.apply(lambda x: x.isna().any())
        exits = set(ended[win(ended) & ~has_open].index) - movers - mass_movers
        # unobserved: no move or exit recorded, and absent from the first snapshot after T+12m
        # (they vanished before we could see the end of the window; a move may be hidden)
        later = set(seen.index[seen["max"] >= nxt])
        unobserved = set(pop) - movers - mass_movers - exits - later
        prv = max((d for d in snap_dates if d <= T), default=None)
        n = len(pop)
        labelled = n - len(exits) - len(mass_movers) - len(unobserved)
        out.append(dict(cutoff=T.date(), active=n, movers=len(movers),
                        base_rate=len(movers) / max(labelled, 1),
                        mass_movers=len(mass_movers), exits=len(exits),
                        unobserved=len(unobserved), unobs_share=len(unobserved) / n,
                        snap_before=prv.date() if prv is not None else None,
                        snap_after=nxt.date() if nxt is not None else None))
    return pd.DataFrame(out)


def branch_coverage(snaps):
    rows = []
    for date, p in snaps:
        ce = pd.read_csv(p / "crnt_emp.csv.gz", dtype=str, usecols=["indvl_pk", "org_pk"])
        br = pd.read_csv(p / "branch.csv.gz", dtype=str)
        has_zip = br[br.postl_cd.fillna("").str.len() == 5][["indvl_pk", "org_pk"]].drop_duplicates()
        has_city = br[br.city.fillna("") != ""][["indvl_pk", "org_pk"]].drop_duplicates()
        k = ce.drop_duplicates()
        rows.append(dict(snapshot=date.date(), current_employments=len(k),
                         with_branch_city=len(k.merge(has_city)) / len(k),
                         with_branch_zip=len(k.merge(has_zip)) / len(k)))
    return pd.DataFrame(rows)


def main():
    snaps = load_snapshots()
    print("snapshots:", ", ".join(str(d.date()) for d, _ in snaps))
    s, present = build_spells(snaps)
    m = find_moves(s)
    s.to_csv(OUTDIR / "spells.csv.gz", index=False)
    m.to_csv(OUTDIR / "moves.csv.gz", index=False)
    print(f"advisers ever seen {s.indvl_pk.nunique():,}; spells {len(s):,}; firms {s.org_pk.nunique():,}")
    print(f"moves {len(m):,}; flagged mass moves {m.mass.mean():.1%}")
    pd.set_option("display.width", 200)
    print("\n== Per cutoff (12-month window; base rate excludes exits and mass movers)")
    print(cutoff_stats(s, m, present, snaps).to_string(index=False))
    print("\n== Branch coverage of current employments")
    print(branch_coverage(snaps).round(3).to_string(index=False))
    # how many advisers vanish between consecutive snapshots (exit-date blur)
    print("\n== Advisers present in one snapshot and absent from the next")
    ps = present.groupby("snap").indvl_pk.apply(set)
    d = list(ps.index)
    for a, b in zip(d, d[1:]):
        gone = len(ps[a] - ps[b])
        print(f"{a.date()} -> {b.date()} ({(b - a).days} days): {gone:,} gone ({gone / len(ps[a]):.1%})")


if __name__ == "__main__":
    main()
