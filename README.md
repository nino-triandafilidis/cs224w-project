# CS224W project: predicting adviser job moves with heterogeneous GNNs

Project type 1 (real-world application of GNNs). On a cut-off date, predict which registered US investment adviser representatives move to a different firm within 12 months, using their career record and professional network. Data: the SEC's [Investment Adviser Public Disclosure (IAPD)](https://adviserinfo.sec.gov/) registry, rebuilt from 14 snapshots (2016–2026).

## Layout

- `main.tex`, `neurips_2022.sty`, `references.bib` — proposal (NeurIPS'22 style). Compiles in Overleaf as-is.
- `dataset/iapd/` — derived tables (roster, spells, moves) and the raw-source manifest; see its README.
- `analysis/iapd_parse.py`, `analysis/iapd_careers.py` — rebuild the derived tables from the raw feeds.
- `analysis/iapd_signal_check.py` — preliminary baseline signal check (DECISIONS.md Q32).
- `DECISIONS.md` — project decisions; `TODO.md` — open work.
- `docs/course-instructions.md` — official CS224W project instructions; `docs/proposal-feedback.md` — feedback on the proposal.

## Deadlines

Proposal 2026-10-20, milestone 2026-11-10, final (blog post + Colab) 2026-12-10, all 11:59pm PT.
