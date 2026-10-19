# Problem space: recruiting problems graph ML could address

Phase 1 of the re-plan (18 Oct 2026), after the JTH-based plan was dropped because its data was too thin (see `DECISIONS.md`). These are problems a startup trying to disrupt recruiting with AI might be solving today, limited to ones that graph ML techniques eligible for the course could plausibly address.

The problems are stated without reference to any dataset. Phase 2 checks which data can serve each one.

## The decision we serve

**For an open role, whom should we put forward?** Recruiters make this call for two kinds of candidate: people they reach out to in the wider market, and people already in their own pool. #1 and #2 attack this decision directly. Each of the other problems improves one part of it.

Hiring teams reveal their judgement at several points. Each one can serve as a signal:

1. **Profile review.** The hiring manager likes or passes on a profile.
2. **Interview.** The company chooses to meet the person.
3. **Offer or hire.**
4. **After the hire.** Tenure and promotion.

Earlier signals are closer to the team's judgement and more plentiful, but rarely public. Later ones are more often observable, but they mix in the candidate's own choice. Below, **"approved"** means any of these. Phase 2 records which signals each dataset provides.

## Fields used for each problem

- **Question:** what the model is given and what it must produce.
- **The pain:** who struggles today, and how.
- **Part of the decision:** which part of the decision above the problem improves.
- **Graph:** what the nodes and edges are.
- **Technique:** the eligible graph ML method.
- **How answers are checked:** every observed fact the predictions could be scored against, strongest first, and what each one cannot tell us.
- **What the model must see:** the entities, identifiers, date precision and scale the model needs as input.
- **Realism check:** whether the model sees what recruiters actually use: named employers, named schools and a dated sequence of roles. The earlier JTH plan failed this test.
- **Shallow vs deep:** the superficial version of the project, and the one sharp question a solo seven-week project can answer properly. The deep version is where the novelty lies.
- **Graph value:** an honest view of how much a graph adds.

## A. Sourcing and matching

### 1. Finding candidates from a role description

- **Question:** Given a description of a role (company, title, level, location, skills), rank people by how likely the hiring team is to approve them.
- **The pain:** Keyword search on job titles misses strong people with unusual titles or adjacent backgrounds, and pulls in weak people whose titles happen to match.
- **Part of the decision:** The decision itself, when the hiring team describes the role.
- **Graph:** People, employers, schools, titles and skills as nodes. Edges are "worked at" and "studied at" (both dated) and "has skill". The role is a new node wired to its company, title, location and skill nodes. The task is to predict role → person links.
- **Technique:** Heterogeneous GNN link prediction, with cold start for the new role.
- **How answers are checked:**
  - The hiring manager's feedback on profiles for that role.
  - Interview and offer decisions for that role.
  - Who was hired into that role, or into that company and title family.
  - Blind spots: the earlier signals cover only people someone already chose to put forward. Hires mix in the candidate's own choice and rarely identify the exact role.
- **What the model must see:** Role descriptions; profiles with named employers, named schools and dated roles; approvals for enough roles to evaluate on.
- **Design choice:** Whether the model may use the company's past decisions. Default yes, since a recruiter always knows the client.
- **Realism check:** Passes if profiles carry named employers, named schools and dated roles.
- **Shallow vs deep:**
  - Shallow: train a heterogeneous GNN and report Recall@K.
  - Deep: Does career structure (the employer and school graph) beat title and text similarity at predicting approvals? Where does it win, for example career changers or little-known employers? Which node types carry the signal (ablation)?
- **Graph value:** High. A career is a path through employers and schools.

### 2. Finding candidates from example profiles

- **Question:** Given a few profiles the hiring team approved for a role, rank everyone else by how likely the team is to approve them too.
- **The pain:** A hiring manager likes two or three profiles and asks for more like them. Recruiters can only guess what "like" means. In practice, searches move from #1 to #2: they start from the description, and once the hiring manager has reviewed a first batch, they are steered by examples.
- **Part of the decision:** The same decision as #1, when the hiring team points at people instead of describing the role.
- **Graph:** Same as #1, except that the role node is wired to the example people instead of to company, title and skill nodes. When both are available, it is wired to both. One model can serve #1 and #2, and the difference lies in the graph's input.
- **Technique:** Link prediction from the role node, or set-based retrieval over GNN embeddings.
- **How answers are checked:** The same signals as #1. The examples and the targets come from the same signal, split by time: earlier approvals are the examples, later ones the targets. Blind spot: the team's reasons are never recorded, only its decisions.
- **What the model must see:** As #1, plus roles with enough approvals to split into examples and targets.
- **Realism check:** As #1.
- **Shallow vs deep:**
  - Shallow: average the example embeddings and take the nearest neighbours.
  - Deep: Do a few approved profiles predict the team's later approvals better than the role description? What do they carry that the description misses, such as employer and school prestige or career path? Does combining both help, and how many examples are enough?
- **Graph value:** Medium–high.

### 3. Who is about to move?

- **Question:** Given a person's career and their colleagues' careers up to a date, predict whether they will leave their employer within the next N months.
- **The pain:** Most outreach reaches people who aren't open to a move. Recruiters rely on tenure rules of thumb or "open to work" flags.
- **Part of the decision:** Is this person reachable now?
- **Graph:** A dated person–employer graph, with colleague links from overlapping stints at the same employer. "Turnover contagion", where colleagues' departures predict your own, is documented in the literature.
- **Technique:** Temporal GNN, node classification.
- **How answers are checked:**
  - An observed departure within N months. Blind spot: a profile without an end date may just be out of date.
  - An employer's own attrition records. Blind spot: covers one employer only.
  - Signs of openness: replies to outreach, "open to work" flags, applications sent. Blind spot: openness is not the same as moving.
- **What the model must see:** Start and end dates at month level or finer; several people per employer, so colleague signals aren't empty.
- **Realism check:** Passes with named employers and dated roles.
- **Shallow vs deep:**
  - Shallow: a GNN classifier on top of tenure features.
  - Deep: Do colleagues' departures predict yours beyond tenure, employer size and industry trends? Compare a temporal GNN with a survival-model baseline, and ablate the colleague edges.
- **Graph value:** High. Colleague departures are a signal that tabular features don't capture.

### 4. Where will they go next?

- **Question:** Given a person's career up to a date, rank employers by how likely the person is to join them next.
- **The pain:** Recruiters need to know which openings to pitch a candidate, and which companies to pitch to which talent.
- **Part of the decision:** Which companies does this person fit? This is #1 seen from the candidate's side: the same person–company link, queried from the other end.
- **Graph:** A temporal person → company graph, plus company → company talent-flow edges.
- **Technique:** Temporal link prediction (TGN, GraphMixer-style models).
- **How answers are checked:**
  - The next employer actually joined. Blind spot: it mixes the person's choice with the employer's.
  - Companies the person applied to or replied to. Blind spot: shows interest, not a match.
- **What the model must see:** Dated stints with named employers; enough repeat destinations that the targets aren't mostly one-off companies.
- **Realism check:** Passes with named employers and dated roles.
- **Shallow vs deep:**
  - Shallow: TGN on move edges, reporting MRR.
  - Deep: What predicts where someone goes beyond popularity: talent flows, the prestige gap, the school network? Compare temporal GNN vs static GNN vs a popularity baseline, with results split by moves to rarely seen companies.
- **Graph value:** High.

## B. Reading careers

### 5. How good is a stint at company X? (employer quality)

- **Question:** Given many people's moves between employers, score each employer's quality, including employers that are poorly known.
- **The pain:** Recruiters weigh employer prestige heavily but judge it by gut feel. They over-reward famous brands and can't judge unknown startups or foreign firms.
- **Part of the decision:** How much is this employer history worth?
- **Graph:** A company → company graph of talent flows, built from many people's careers rather than one candidate's. Each company is a node. A directed edge from company A to company B counts the people who worked at A and later moved to B; A is the employer they left and B is the employer that hired them. For a single candidate, A is a past employer and B is the company now looking to hire. Prestige is revealed by where talent chooses to go: a company that many people leave other firms to join, and that few people leave for weaker firms, ranks high.
- **Technique:** Graph-based ranking (PageRank- or SpringRank-style) vs a GNN that infers prestige for little-known companies from their neighbours (semi-supervised node regression).
- **How answers are checked:** There is no direct truth.
  - Agreement with published rankings for a subset of companies.
  - Whether the score predicts held-out moves (#4).
  - Whether hiring teams actually reward it: does it predict approvals in #1 and #2 where those are observed? This is the strongest test.
- **What the model must see:** Named employers with consistent identities across people; dated moves; enough moves per company to rank it.
- **Realism check:** Passes. This is exactly the signal recruiters use.
- **Shallow vs deep:**
  - Shallow: run PageRank and show a top-20 list.
  - Deep: Do talent flows rank employers better than size or brand? Can a GNN place little-known companies from their neighbours? Does the score improve next-employer prediction (#4) or approval prediction (#1, #2)?
- **Graph value:** Very high. Prestige essentially is a graph property.

### 6. School and program quality from where graduates go

- **Question:** Given where graduates of each school and program go to work, score each school's and program's quality, including ones that rankings miss.
- **The pain:** Same as #5, for schools, especially non-US schools and specific programs.
- **Part of the decision:** How much is this education worth?
- **Graph:** A school → employer placement graph, combined with the employer graph of #5.
- **Technique:** As in #5.
- **How answers are checked:** There is no direct truth.
  - Agreement with published school rankings for a subset.
  - Whether the score predicts graduates' later careers.
  - Whether hiring teams reward it in #1 and #2 where approvals are observed.
- **What the model must see:** Education records with named schools (and ideally programs), graduation dates and the employers that follow.
- **Realism check:** Passes if schools are named.
- **Shallow vs deep:**
  - Shallow: run PageRank on the placement graph.
  - Deep: Where a placement-based score disagrees with published rankings, which one better predicts graduates' later careers? Can it rank programs and non-US schools that rankings miss?
- **Graph value:** High.

## C. Skills

### 7. What can this person actually do? (skill inference)

- **Question:** Given a person's roles, employers and some listed skills, predict the other skills they have.
- **The pain:** Profiles list a few keywords. Real skills follow from roles, employers and colleagues, so matches are missed.
- **Part of the decision:** What does this person bring?
- **Graph:** Person–skill–occupation–company, plus a skill taxonomy as a knowledge graph.
- **Technique:** Link prediction or knowledge-graph completion.
- **How answers are checked:**
  - Listed skills, held out. Blind spot: a skill that isn't listed isn't necessarily absent.
  - Verified skills, such as assessments or certifications.
  - Skills required by roles the person was approved for.
- **What the model must see:** Skill lists linked to the person's roles and employers; a skill taxonomy.
- **Realism check:** Partial. Recruiters skim skills, but trust employers and titles more.
- **Shallow vs deep:**
  - Shallow: link prediction on person–skill edges, reporting AUC.
  - Deep: Do employer and colleague neighbourhoods recover held-out skills beyond what an LLM infers from the job title alone? Which relations carry the signal?
- **Graph value:** Medium–high.

### 8. Turning job postings into skills and occupations

- **Question:** Given a job posting, predict the occupation and the skills it requires.
- **The pain:** Postings are messy. Matching needs to know what the job really requires.
- **Part of the decision:** What does this role need?
- **Graph:** Posting–skill–occupation knowledge graph.
- **Technique:** Knowledge-graph completion or edge classification.
- **How answers are checked:**
  - Occupations and skills coded by people against a taxonomy.
  - Codes assigned by job boards or statistics agencies. Blind spot: coding quality varies.
- **What the model must see:** Posting text; a skill and occupation taxonomy.
- **Realism check:** Passes on the job side; says nothing about candidates.
- **Shallow vs deep:**
  - Shallow: classify postings into occupations.
  - Deep: Does a taxonomy graph plus a posting–skill graph beat a text or LLM classifier on rare occupations? Hard to win.
- **Graph value:** Medium. LLMs do much of this already.

## Problem × data match

Phase 2 research, 18 Oct 2026. Facts were checked against official pages or with small API and SPARQL queries. Where a figure comes from a small sample, the table says so.

### Finding

No public dataset records a hiring team's own decisions (profile feedback or interviews) about candidates with rich profiles. Every industry dataset with real recruiter decisions is private, and the public ones either record no decisions or carry synthetic or LLM-generated labels. So every option below is checked against hires (who joined where) or what happened after the hire, and treats everyone not hired as unlabelled rather than rejected.

No public source covers industry careers with both named employers and named schools at scale. The usable sources each cover one labour market.

### Data sources

| Source | What it holds | Realism | Access and licence | Main risks |
|---|---|---|---|---|
| **SEC IAPD adviser feed** (US investment adviser representatives) | About 442K advisers. Past and current firms with firm IDs, start and end dates to the day, monthly employment history, dated licensing exams, designations, branch addresses, and a flag for being fired by the employer. In a sample of 6,900 records, 69% had 2+ distinct firms. | Partial: named employers with IDs and dated roles; **no schools**, no titles beyond registration type | One daily bulk download (169 MB), no login. sec.gov content can be redistributed; IAPD's own terms still need checking | Privacy is not a concern: SEC publishes these records for anyone to look up. Each day's feed holds current registrants only, so people who left are missing; fixed by the archived snapshots below. Finance only |
| **ORCID Public Data File** (+ OpenAlex, ROR) | About 30M records, 14M with an affiliation. Employment and education with organisation, department, role title and dates. From 400 sampled records, an estimated 1–2M people have 2+ dated employers. OpenAlex adds topics per author (10.9M authors have an ORCID iD). | Pass: named employers **and** schools, dated roles | CC0. 46 GB download, or a free BigQuery copy | Mostly researchers. Self-reported, edited after the fact. About half of employer entries lack an organisation ID. Role titles are free text. 1–2 weeks of data work |
| **US DOL PERM filings**, old form, FY2008 to mid-2023 | One row per worker an employer sponsored: named employer, job title, occupation code, required skills, offered wage, the worker's named school, major and graduation year, decision date | Pass for one hire; no career histories | Public domain, yearly files | No person ID across filings. Foreign workers only. Fill rate of the skills field not yet checked |
| **CMS Doctors & Clinicians** | 3.6M rows; 965K doctors name a medical school and graduation year, plus a group practice with an ID. Moves come from joining yearly archives (2014–2021) on each doctor's ID | Pass for school and employer; no titles | US public data | Residency, between school and employer, is missing. Medicine only. Archive access not yet checked |
| **Wikidata** | 797K people with dated employers; 554K with a dated employer and a school; only about 5K with 2+ dated company employers | Partial: dominated by universities and notable people | CC0 | Too thin for industry careers. Best used as a source of IDs |
| **DOL prevailing-wage records** | About 242K determinations a year: job title, employer, requirements, the employer's suggested occupation code, and the code assigned by agency staff | Pass on the job side only | Public domain | No duties text |
| **Analogues** (OpenReview review decisions, Wikipedia admin elections) | Real like/pass decisions about people | Fail or partial | Varies; OpenReview's terms restrict reuse | Not recruiting. Heavily used in prior work |

### Links to the sources

- **SEC IAPD adviser feed**
  - Download page: https://adviserinfo.sec.gov/compilation
  - Current file list: https://reports.adviserinfo.sec.gov/reports/CompilationReports/CompilationReports.manifest.json
  - Archived copies, 2016–2019: https://web.archive.org/cdx/search/cdx?url=adviserinfo.sec.gov/IAPD/Content/BulkFeed/&matchType=prefix
  - Archived copies, 2022 onward: https://web.archive.org/cdx/search/cdx?url=reports.adviserinfo.sec.gov/reports/CompilationReports/&matchType=prefix
  - SEC reuse statement: https://www.sec.gov/privacy
  - Egan, Matvos & Seru adviser panel (on request, non-commercial): https://eganmatvoserru.stanford.edu/index.php/download-adviser-level-data/
- **ORCID Public Data File**
  - Annual files: https://info.orcid.org/documentation/features/annual-data-files/
  - BigQuery copy: https://info.orcid.org/orcid-public-data-file-on-google-big-query/
  - Use policy: https://info.orcid.org/public-data-file-use-policy/
  - Record format: https://info.orcid.org/documentation/integration-guide/orcid-record/
- **OpenAlex**
  - Licence (CC0): https://openalex.s3.amazonaws.com/LICENSE.txt
  - Author records: https://help.openalex.org/api-entities/authors/author-object
- **US DOL PERM and prevailing-wage disclosure files**
  - Download page: https://www.dol.gov/agencies/eta/foreign-labor/performance
  - PERM old-form layout: https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/PERM_Record_Layout_FY2022_Q4.pdf
  - Prevailing-wage layout: https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/FY26Q3/PW_Record_Layout_FY2026_Q3.pdf
- **CMS Doctors & Clinicians** (National Downloadable File): https://data.cms.gov/provider-data/dataset/mj5m-pzi6
- **Wikidata dumps**: https://www.wikidata.org/wiki/Wikidata:Database_download
- **Analogues**
  - ICLR on OpenReview (berenslab dataset): https://github.com/berenslab/iclr-dataset
  - OpenReview terms: https://openreview.net/legal/terms
  - Wikipedia admin elections (SNAP wiki-RfA): https://snap.stanford.edu/data/wiki-RfA.html
- **Taxonomies**
  - ESCO: https://esco.ec.europa.eu/en/use-esco/download
  - O*NET: https://www.onetcenter.org/database.html
- **Prior work cited above**
  - Dies, Liu & Eliassi-Rad, faculty placement from co-authorship networks: https://arxiv.org/abs/2507.14696
  - JobXMLC (job→skill GNN): https://aclanthology.org/2023.findings-eacl.163

**IAPD survivorship check (18 Oct 2026).** Each day's feed lists current registrants only: every record in samples from the 2026 and 2019 files has a current employer with an active status. The Internet Archive holds full copies of the individual feed from 2016 (Jun, Oct, Nov, Dec), 2017 (Feb, May), 2018 (Mar, Sep), 2019 (Mar), 2022 (Feb, Nov), 2024 (Sep) and 2025 (Jul), and SEC serves the current one. A sample of the March 2019 copy has the same format. Each snapshot is a census of who was registered on that date. So:

- Leaving the industry is observable as disappearing between two snapshots, known to within the gap between them; the gap from 2019 to 2022 is about three years.
- Firm switches keep day-level dates.
- The past can be rebuilt without survivorship bias.

Caveats: someone who drops the adviser registration but keeps a broker registration also disappears, and the copies depend on the Internet Archive staying available.

Ruled out: JobHop, KARRIEREWEGE and the survey panels (no employer names); Djinni (no decisions); Hugging Face resume–job "fit" sets (synthetic labels); Kaggle resume sets (scraped); US federal workforce records (no stable person ID since 2016); Crunchbase, FINRA BrokerCheck and sports-reference data (terms forbid publishing); CSRankings (no-derivatives licence); faculty-hiring network data (aggregate counts only; useful to validate a ranking).

### Match table

| # | Problem | Best data | Checked against | Realism | Can it reach the novelty bar? | Verdict |
|---|---|---|---|---|---|---|
| 1 | Candidates from a role description | ORCID: an assistant-professor start at institution X is a hire by X. IAPD: joining firm X, but the "role description" is thin (firm, branch, registration type) | Hires; tenure or promotion afterwards | ORCID pass; IAPD partial | Yes. The closest prior work (Dies et al., EPJ Data Science 2026) predicts only a top-k prestige tier for CS faculty and does not publish its data | **Strong with ORCID**, for the academic labour market |
| 2 | Candidates from example profiles | ORCID: a department's earlier hires are the examples, its later hires the targets | Hires | Pass | Yes, beyond Dies et al. | **Strong with ORCID**; pairs with #1 |
| 3 | Who is about to move | IAPD: colleagues are defined exactly (same firm and branch), exits dated to the day | Observed firm switch | Partial (no schools) | Yes. Economics has descriptive work on co-worker contagion among advisers; no graph-ML work found | **Strong with IAPD** |
| 4 | Where they go next | IAPD (next firm, with an ID); ORCID as an alternative | Observed next employer | IAPD partial; ORCID pass | Yes for IAPD; no graph-ML work found | **Strong with IAPD** |
| 5 | Employer quality | IAPD firm-to-firm flows, with firm size and assets from the firm feed. ORCID for institutions | No direct truth: firm traits, predictive validity on #4 | IAPD partial; ORCID pass | Yes for IAPD firms. Weaker for universities, where prestige from hiring flows is well studied (SpringRank and later work) | **Strong with IAPD**; crowded with ORCID |
| 6 | School quality from placements | PERM (named school → named industry employer, with offered wage); ORCID; CMS (medical school → practice) | Published rankings for a subset; does the score predict wage or later career? | Pass | Likely for PERM; no graph work found on these filings | **Promising**, but no career histories |
| 7 | Skill inference | ORCID employment + OpenAlex topics | Held-out keywords; topics from later papers | Pass (researchers) | Yes; no prior work found using ORCID employment this way | Good, but less central to recruiting |
| 8 | Postings to codes | DOL prevailing-wage records (agency-assigned codes) | Agency coding | Pass on the job side | Hard: a GNN beat on job→skill already exists (JobXMLC, 2023), and LLMs code titles well | **Weak** |

### Recommendation

1. **Financial-adviser careers (IAPD) for #3, #4 and #5, as one project.** One temporal heterogeneous graph of advisers, firms, branches and exams. The sharp question: do colleagues' departures and talent-flow firm quality predict who moves and where they go, beyond tenure, firm size and popularity? It is a real industry labour market, with resolved employer IDs and exact dates, a single small download, and no prior graph-ML work. Its weaknesses are no schools and one industry. Privacy is not a concern, since SEC publishes these records for anyone to look up. Day-one gate: confirm IAPD's terms of use.
2. **Academic hiring (ORCID + OpenAlex) for #1 and #2.** Rank candidates for a department from its role description or from its earlier hires. It is the only option that passes the realism check in full (named employers and named schools) and that attacks the matching decision directly. Its weaknesses are that it covers the academic labour market only, has noisy self-reported records, needs about 1–2 weeks of data work, and has close prior work on faculty placement.

Choose 1 for feasibility and a clean industry story; choose 2 if named schools and the matching decision itself are essential. PERM is a possible add-on to either, for #6 school quality from industry placements.
