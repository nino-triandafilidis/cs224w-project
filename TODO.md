# TODO

## IAPD dataset (after the proposal is submitted)

- [x] Cut-off design: training cut-offs 2016-06 to about 2018-03, validation and test 2022-02 to about 2025-10, none inside the gap for now (DECISIONS.md Q36).
- [ ] Decide after the milestone whether to rebuild cut-offs inside the 2019–2022 gap.
- [ ] Mass-move detection: replace the 20-advisers-in-30-days threshold with direct evidence of a merger, acquisition or rebrand where possible (e.g. firm-level records), keeping real team breakaways.
- [ ] Branch cleaning: inconsistent city names, about 5% of records missing a postcode, advisers listed at several branches, one postcode grouping many unrelated people at big firms.
- [ ] Add adviser names to the derived tables (the parser does not extract them yet).
- [ ] Commit the per-snapshot branch tables needed for colleague edges (about 90 MB), or a compacted version.
