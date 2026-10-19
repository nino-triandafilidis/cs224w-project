# IAPD adviser careers (derived)

Built from the SEC IAPD individual-adviser feed: the current SEC copy (2026-10-17) and 13 Internet Archive copies (2016-06-03 to 2025-07-17). Raw files are public; rebuild with `analysis/iapd_parse.py` then `analysis/iapd_careers.py`.

[`SOURCES.csv`](SOURCES.csv) lists the 14 raw files: snapshot date, exact download URL, size in bytes and SHA-256.

- `roster.csv.gz`: `indvl_pk, snapshot`. Who was registered on each snapshot date.
- `spells.csv.gz`: one row per adviser–firm registration spell (`indvl_pk, org_pk, begin_dt, org_nm, end_dt, last_seen`); `end_dt` empty = still open when last seen.
- `moves.csv.gz`: firm-to-firm moves (`indvl_pk, from_org, end_dt, to_org, begin_dt, move_dt, mass`); `mass` = part of a 20+ adviser same-firm-pair move within 30 days (merger, rebrand, acquisition).

Per-snapshot branch, exam and disclosure tables are not committed (about 300 MB); regenerate them from the raw feeds.
