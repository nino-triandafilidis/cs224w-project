# CS224W project: cold-start candidate ranking with heterogeneous GNNs

Project type 1 (real-world application of GNNs). Data: [JTH — Job Tracking History](https://zenodo.org/records/21390581) (CC BY-NC 4.0, Romero et al., RecSys '26).

## Layout

- `main.tex`, `neurips_2022.sty`, `references.bib` — proposal (NeurIPS'22 style). Compiles in Overleaf as-is.
- `analysis/jth_stats.py` — recomputes every statistic quoted in the proposal.
- `data/` — JTH CSVs, git-ignored.
- `docs/course-instructions.md` — official CS224W project instructions (Project 1 excerpt, pulled 2026-10-18).

## Get the data

```bash
mkdir -p data && for f in candidates.csv jobs.csv history.csv dataset_card.md; do curl -L -o "data/$f" "https://zenodo.org/records/21390581/files/$f?download=1"; done
```

## Reproduce the numbers / build the PDF

```bash
python analysis/jth_stats.py data
latexmk -pdf main.tex
```

## Deadlines

Proposal 2026-10-20, milestone 2026-11-10, final (blog post + Colab) 2026-12-10, all 11:59pm PT.
