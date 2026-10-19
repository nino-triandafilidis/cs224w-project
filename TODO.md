# TODO

## IAPD dataset (after the proposal is submitted)

- [ ] Cut-off design around snapshot gaps. With no census between 2019-03-25 and 2022-02-10, the population on a date inside the gap can only be rebuilt from people still registered in 2022 (survivors), and moves by people who moved then left are hidden. Decide which cut-offs to use and how to split them into train/val/test.
- [ ] Mass-move detection: replace the 20-advisers-in-30-days threshold with direct evidence of a merger, acquisition or rebrand where possible (e.g. firm-level records), keeping real team breakaways.
- [ ] Branch cleaning: inconsistent city names, about 5% of records missing a postcode, advisers listed at several branches, one postcode grouping many unrelated people at big firms.
- [ ] Add adviser names to the derived tables (the parser does not extract them yet).
- [ ] Commit the per-snapshot branch tables needed for colleague edges (about 90 MB), or a compacted version.

## JTH (dropped plan; kept for reference)

## Proposal (main.tex)

Revisit before submitting (the proposal no longer has a motivation section; see DECISIONS.md Q22):

- [ ] (Blog post, no longer in the proposal) Clarify the recruiter-benchmark caveat. Proposed wording: "Two factors favour the recruiter: they had screening calls with candidates, which the model cannot see, and companies almost only interview candidates whose resume the recruiter sent (95% of interviewed pairs), so a candidate the recruiter did not send rarely had the chance to be interviewed and counts as a miss for the model."
- [ ] Clarify node starting vectors in the Encoder paragraph ("projects node features / learned embedding"). Proposed wording: "Jobs and candidates start from their features (salary, experience, company type, ...), mapped to a vector by a learned linear layer, so a new job gets a starting vector without having been seen in training. Attribute nodes have no features, so each value (e.g. the skill Python) has its own learned vector."

## Data cleaning (Colab)

- [ ] Treat the masking token `_rare_skill_` in LLM skill fields as missing, never as a skill node (it appears in 63% of jobs' `llm_hard_skills`; as a node it would become a hub).
- [ ] Location: JTH's `zipcode` holds only the two-digit département code. Convert candidate values (stored as floats, e.g. `75.0`) to two-digit strings matching the jobs' codes.
- [ ] Treat département `0` as missing, not as a location node.
