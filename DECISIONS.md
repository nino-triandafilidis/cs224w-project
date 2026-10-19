# Project decisions

Decisions from the scoping Q&A (17–18 Oct 2026) that produced the proposal. Each entry gives the decision, the reason, and the alternatives that were rejected. Some decisions were made in side discussions and are recorded here from their summaries. Numbers come from `analysis/jth_stats.py` unless noted.

## Course constraints (from the CS224W project doc)

- Project type 1, real-world application of GNNs. Final deliverables: a Medium blog post (50 pts), a Colab (25 pts) and the Medium account name (5 pts).
- The dataset must be public. The doc calls this a "strict requirement".
- Novelty (a new or improved method) or a comprehensive analysis (ablations, architecture comparisons) is required. Applying an existing model to a new dataset is not enough.
- Proposal: 2 pages, NeurIPS'22 style, no abstract, plus the group listing.
- Deadlines (all 11:59pm PT): proposal Tue 20 Oct 2026; milestone Tue 10 Nov; final Thu 10 Dec.

## Framing

- **Q1. Industry problems motivate; a public dataset is used for experiments.** Company data is private, so it can only provide motivation.
- **Q2. Novelty route: comprehensive analysis as the core, plus targeted novelty.** Safer than pure novelty on a solo timeline.
- **Q3. Keep the post generic and don't name any company.** The blog post is public.
- **Q4/Q10. Solo, Colab Pro.** The course discourages solo projects; the student has handled this with course staff.
- **Q5. Meeting the criteria beats fit with an industry problem.** If no industry problem had fitted, we would have picked the strongest project from the course's example list.
- **Q6. The problem is cold-start candidate–job link prediction.** It is a core problem for AI recruiting platforms.
  - Rejected: match-outcome edge classification and tier node classification (sparse labels, no public analogue); duplicate-profile detection (a weak fit for a GNN).

## Dataset

- **Q7. JTH (Job Tracking History), with no fallback.** Romero et al., RecSys '26; Zenodo record 21390581; CC BY-NC 4.0; open download, no login.
  - About it: a French tech recruiting agency, 2018–2025; 37,554 candidates, 6,011 jobs, 42,288 pairs.
  - Each pair has dated funnel stages: application, shortlist, qualification, resume sent, interviews 1–4, offer, hire.
  - Attributes are structured only, with no free text and no company id. Dates are randomized for anonymity, with funnel order preserved.
  - The non-commercial licence is fine for coursework and the blog post. A model trained on it could not ship in a product, which doesn't matter here.
  - Rejected: CareerBuilder (Kaggle 2012), whose rules allow use "solely for purposes of the Competition" and require the data to be destroyed afterwards.
  - Rejected: XING (RecSys 2016/17), which is ideal and two-sided but withdrawn, and forbids sharing.
  - Rejected: Zhaopin/Tianchi, which has no timestamps, Chinese text and uncertain availability.
  - Rejected: LinkedIn Job Postings, which has no users or interactions.
  - Rejected: the person–job-fit papers (DPGNN, PJFNN and others), which all used private data.
  - A fallback was dropped because the only candidate, CareerBuilder, had worse licence problems than JTH.

## Task and labels

- **Q11. A "pair" is a candidate and a job that appear together**, either because a recruiter picked the candidate or because the candidate applied. A pair reflects recruiter selection, not candidate interest, and the post must say so.
- **Q17. One question, graded two ways:** for a new job, rank candidates so that those who will be interviewed come first.
  - The main score is within each job's pairs, where the labels are trustworthy.
  - The secondary score covers the whole population. It is noisy, because unpaired candidates are unknown rather than negative.
  - An earlier split into "who to target" vs "who to contact first" was merged. They are the same ranking, and the split came from gaps in the data, not from the business.
- **Interviewed** means the pair reached a company interview or beyond: 7.4% of pairs, the first point where the company says yes. **Picked but not interviewed** means the pair stopped at shortlist, qualification or resume sent. **Random** means no pair for that job. Hires (2.0%) are a subset of interviewed. Don't use the word "shallow"; it was unclear.

## Training target (novelty 1)

- **Q12. A three-level BPR loss: interviewed > picked but not interviewed > random.**
  - The main term ranks interviewed above both other levels.
  - A helper term with weight λ ranks picked-but-not-interviewed above random. It never ranks a non-interviewed pair above an interviewed one, so it doesn't fight the test objective.
  - The experiment, λ = 0 vs λ > 0, answers whether recruiter picks help or hurt. Hires count more than interviews only in NDCG grading.
  - Rejected: a 4-grade funnel-depth scheme and a head predicting the exact stage. Deep stages are too thin (interviews 5.3%, offer or hire 2.1%).
  - Rejected: predicting rejection. There are no reject reasons and no record of who rejected, and 72% of pairs have no end date.

## Split and leakage

- **Q13. Split by job opening year:**
  - Train 2018–2021: 2,144 jobs, 15,048 pairs, 1,648 interviewed.
  - Validation 2022: 720 jobs, 9,919 pairs, 743 interviewed. It sets early stopping and all hyperparameters, including λ.
  - Test 2023–2025: 1,252 jobs, 17,321 pairs, 743 interviewed (70 of them from 2025 jobs).
- **Test jobs have no pairs in the graph.**
- **Leakage cut:** training pairs whose first event falls after the start of the evaluated period are dropped: 4.8% for validation, 0.2% for test. The first-event date is used because dates between stages are unreliable; 44% of multi-stage pairs have identical stage dates, which looks backfilled.
- **No candidate-pool rule:** every candidate is rankable for every test job, with the population standing in for the market.
  - Rejected: restricting to candidates who existed when the job opened. Only 35% of test positives were already in the pool, so the rule would have discarded most positives and measured the weaker channel.
  - Industry context: recruiting firms both reach out to people in the wider market and match candidates already in their pool.
- **Cold start, measured:** every test job is new, and 82% of test pairs involve a candidate with no earlier pair in the graph. 68.5% of candidates appear in only one pair.
- **Sensitivity checks:**
  - Scores with and without jobs opened in the last 97 days of the data. 90% of first interviews come within 97 days of a job opening (median 31); the last event is 2025-05-03.
  - 2023 and 2024–25 reported separately. Quarterly interview rates were 6–9% in 2022, 1.5–4.2% in 2023 and about 5% in 2024. The cause is unknown.

## Graph

- **Q14. Shared attribute values become nodes:** skill, location (département), seniority, contract, industry and job category. They bridge jobs and candidates and are the only paths into new jobs and new candidates.
  - Hand-entered and LLM skills are merged into one normalised vocabulary, with an edge flag recording the source.
  - Numeric and one-sided attributes are node features: salary, years of experience and company-type flags.
  - The rule of thumb: shared categorical values become nodes; numbers, very common flags (which would become hubs) and properties with no counterpart on the other side stay features.
  - "Attributes as nodes vs as features" is an ablation. The gradient-boosting baseline needs the features-only version anyway.
- **Starting vectors:** jobs and candidates start from their features through a learned linear layer, which works for unseen nodes. Each attribute node has its own learned vector.
- Data-cleaning steps are tracked in `TODO.md`.

## Models (novelty 2)

- **Q19. Both main models use the same heterogeneous GraphSAGE encoder** (PyG `to_hetero`) and differ only in how they score a pair:
  - **Two-tower:** run the encoder once and score with a dot product.
  - **ID-GNN** (You et al., AAAI 2021): mark the query job with a learned identity vector, run the encoder, then score the candidates it reaches with an MLP. This needs one forward pass per job. RelBench (Robinson et al. 2024) found ID-GNN averaged 6.74 vs 1.85 test MAP against two-tower on recommendation tasks, but it lost where popularity dominated. It is a hypothesis to test. PyG has no one-line ID-GNN, so we adapt the RelBench example code.
  - **Grid:** encoder (GraphSAGE; HGT as a stretch goal) × scoring setup.
  - Rejected: RGCN, since heterogeneous GraphSAGE with per-relation weights is essentially RGCN.
  - Rejected: LightGCN, which can't embed unseen nodes.

## Evaluation and baselines

- **Main score:** NDCG@K and MRR within each test job's pairs, with gain 2 for a hire and 1 for an interview.
- **Q15. Recruiter benchmark (headline):**
  - On the 410 test jobs with at least one interviewed pair, one non-interviewed pair and one resume sent, the recruiter sent 3.1 resumes per job and 55% reached an interview, against 14% for random picks of the same size. The model picks the same number per job.
  - Caveat: the recruiter had screening calls the model can't see, and 95% of interviewed pairs had their resume sent first, so recruiter picks shape the label. The wording is still to be clarified; see `TODO.md`.
  - A side discussion reported 374 jobs, 54% and 16% under a slightly different filter. The filter used here is the one in the script.
- **Baselines:** gradient boosting on candidate and job attributes; MiniLM text similarity (production-style); shared-skill count. Popularity was dropped.
- **Secondary:** Recall@K over the whole population (noisy), and a top-K check on seniority, contract, location and salary. A manual audit is a stretch goal.
- **Q18. Segments:** results split by candidates with vs without earlier pairs in the training graph.
  - An earlier claim that candidates "found for this job" were hired 2.4× as often was withdrawn. It rested on a create-date proxy.
  - The `application_source` field is filled for only 24% of pairs and is suggestive at most. Internal pool converts 4.6% to interview and 0.5% to hire, LinkedIn 6.1% and 1.2%, and blank rows 8.2%. Footnote only.

## Sensitive attributes

- **Q16. The student chose to use the LLM-inferred sex, nationality and age bucket as model inputs.**
  - Concerns raised before the decision: French labour law (Code du travail L1132-1), the EU AI Act's high-risk category for recruiting AI, and the fact that the values are LLM guesses.
  - The post must say the model could not be deployed as-is. A fairness check by group is a stretch goal.

## Proposal writing

- **Q20. Outline:**
  1. Application and motivation. The student writes this section, informed by the #intel Slack channel on AI in recruiting; it is kept generic.
  2. Dataset, task and metrics.
  3. Graph ML techniques: schema figure, equations, loss.
  4. Experiments.
  5. References.
- **Q21. Workflow:** Claude drafts LaTeX in this repo; the student edits in Overleaf, which syncs through GitHub. Pull before editing, and push only with the student's approval.

## Proposal scope (18 Oct)

- **Q22. The proposal mirrors the course checklist (course doc §1.3; see `docs/course-instructions.md`).** It has a group line; §1 Application domain and dataset (one framing sentence, the dataset and public status, dataset/task/main metric, why this dataset); §2 Graph ML techniques (models with citations, figure and equations, why they fit); and references. The checklist asks for no motivation, experiments, baselines or ablations. The removed content (the recruiter-benchmark details, secondary score, sensitivity checks, segments and the sensitive-attribute note) stays in this file for the milestone and blog post.
- **Q23. §2 ends with a short "Planned evaluation" note** naming the comparisons that carry the novelty: two-tower vs ID-GNN, λ = 0 vs λ > 0, the recruiter benchmark and the baselines. Mentors are assigned from the proposals, so a reviewer should see where the novelty is.
- **Q24. The motivation work moves to the blog post**, which grades motivation at 10 of its 50 points. The #intel channel is a source for it; nothing from it is stored in the repo.
- This supersedes the Q20 outline.


## Re-plan: SEC adviser careers (18 Oct)

The JTH plan above is **dropped**. JTH has no full resumes (no employer or school names, no career histories), and its LLM-extracted fields can't be verified: some were invented when the CV text was empty, and they often contradict the recruiter-entered fields. A model trained on it would not see what real hiring decisions rest on. The decisions above are kept as history. The problem survey is in `docs/problem-space.md`.

- **Q25. Data: SEC IAPD individual-adviser feed.** The current SEC copy (2026-10-17) plus 13 Internet Archive copies (2016-06-03 to 2025-07-17). The May 2017 capture was corrupt, so the 2017-05-04 capture is used instead. These are public records: named employers with firm IDs, registration dates to the day, and branch office addresses. Derived tables are in `dataset/iapd/` (roster, spells, moves); scripts are `analysis/iapd_parse.py` and `analysis/iapd_careers.py`.
  - Fallback if the data fails: ORCID + OpenAlex academic hiring.
- **Q26. Question: who is about to move?** On a cut-off date, predict for each registered adviser whether they switch to a different firm within the next 12 months. This is node classification. Firm quality, destination prediction and academic hiring are set aside.
- **Q27. What a cut-off date means.** The combined dataset is one timeline of every adviser's employment spells. A cut-off is a "pretend it's today" moment: the model sees only what was known on that date (who works where, with whom, past moves) and predicts moves in the following window. Each cut-off gives one labelled set, one row per adviser registered that day.
- **Q28. Label window: 12 months.** It is a trade-off. A shorter window is more useful to a recruiter, but only about 1% move per quarter, so positives are scarce and date noise matters more. A longer window weakens the link between the state at the cut-off and the outcome, and needs snapshots further ahead. 12 months is standard in attrition work and smooths bonus seasons; 6 months is the main alternative.
- **Q29. Exclusions** (from both classes):
  - **Mass moves.** When a firm is bought or rebranded, all its advisers switch firm IDs on one day: 32.5% of recorded moves. Nobody chose to move, nobody would recruit them, and it can't be predicted from adviser data. It also fakes the colleague signal, since everyone's colleagues "leave" at once. Real team breakaways (typically 2–15 people) must be kept. How to detect mergers is open; see `TODO.md`.
  - **Leaving the industry** (retirement, career change). It is a different outcome: a recruiter can't hire these people into another advisory firm, and labelling them either "moved" or "didn't move" is wrong. A third outcome class was rejected as complexity for little gain.
  - **Unobserved advisers**, who vanish before the end of the label window can be seen. A move may be hidden.
- **Q30. Graph: adviser, firm and branch nodes.**
  - Edges: current employment, past employment with dates, and branch membership.
  - Licences are dropped as nodes; at most they become a count per adviser.
  - A branch is approximated as firm plus 5-digit postcode, the best available definition of "who you actually work with". Cleaning is open; see `TODO.md`.
  - **Former colleagues are in the graph** through past-employment edges: you → old firm → former colleague → their new firm. With current jobs only, former colleagues would not be connected, and this multi-hop "ex-colleagues pull you along" pattern is where a GNN could beat hand-built features.
- **Q31. Novelty: one question, answered in steps.**
  - *Question:* does an adviser's professional network (current branch colleagues, former colleagues, firm-level departures) predict who moves next beyond their own career record, and do you need graph ML to capture it?
  - *(1) A new public benchmark:* the first graph-ML formulation of job-move prediction on public regulatory career records, rebuilt from 14 snapshots without survivorship bias and released with code. Prior work on adviser mobility is descriptive economics.
  - *(2) The answer in steps:* B2 (the adviser's own record, gradient boosting) → B3 (plus hand-built network features) → heterogeneous GraphSAGE (the network learned from the graph). B1 (tenure rules) is the floor.
  - *(3) Which relationships matter:* edge ablation removing current-branch, former-colleague and firm edges.
  - Metrics: top-1% hit rate and PR-AUC, always next to the base rate and averaged across test cut-offs.
  - A temporal GNN is a stretch goal. The "fading signal" analysis is dropped.
- **Q32. Preliminary signal check: left out of the proposal.** Cut-off 2018-03-26, 312K labelled advisers, base rate 4.65%. PR-AUC: B1 0.074, B2 0.087, B3 0.094. Top-1% hit rate: B2 13.5%, B3 13.9%. This is one split with no confidence intervals: a small network effect, not yet known to be real.
- **Q33. Publishing.** The processed tables are committed to the repo, and names may be kept. No ethics section in the proposal.
- **Q34. Sequencing.** Most dataset work (cleaning, merger detection, cut-off design, branch tables) waits until after the proposal is submitted on 20 Oct.
