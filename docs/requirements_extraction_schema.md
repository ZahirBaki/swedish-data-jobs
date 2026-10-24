# Step 4 requirements extraction — version 1.0

Proceed with available accepted, source-validated relevant AI classifications.
No complete classification of all 5,023 ads, classification retry, publication-audit
gate or paid API is required. Accepted means technically accepted and source
grounded, not independently human-verified relevance. Retain classification model,
prompt, source run, policy, review flags and all original text. Irrelevant jobs are
excluded; accepted uncertain judgments and technical failures remain set aside.
Later failed attempts do not erase earlier valid judgments. Later accepted semantic
judgments supersede earlier ones for selection; every version remains in history.

The inventory uses canonical completed Step 3D combined, Step 3E recovered,
Step 3F recovered and the completed three-case Step 3F 1.1 reliability run. It does
not count archived copies as new jobs. Relevant, uncertain and irrelevant decisions
are source-validated with their original schema. No AI-assisted review labels are
treated as truth. The independent 200-job human reserve stays untouched.

## Outputs

| File | Unit and purpose |
| --- | --- |
| classification_registry.csv | All 5,023 original IDs/texts, preserved rules plus selected AI fields and Step 4 disposition |
| classification_history.json | Canonical pilot decision versions and source/validation provenance; original attempt logs remain untouched |
| pilot_input.csv | Up to 50 existing accepted relevant ads, fixed-seed category-stratified development sample |
| jobs_requirements.csv / .jsonl | One row per selected job, full original description/classifications plus extracted fields |
| requirements_evidence.csv | One row per matched fact/occurrence, including qualification, negated, ambiguous and mention-only evidence |
| job_skills.csv | Programming languages, cloud platforms and tools; one job may have many skills and occurrences |
| job_language_requirements.csv | English/Swedish evidence with modality and conflict/alternative context |
| job_education_requirements.csv | Degree levels and required/preferred/alternative/mentioned modalities |
| job_experience_requirements.csv | Explicit years, bounds, modality and context; independent requirements never collapsed into one inferred threshold |
| job_seniority.csv | Explicit own-role seniority, including exact title or description support |
| job_employment_location.csv | Explicit employment/work arrangement and original employer/date/region/municipality metadata |
| requirements_summary.json | Counts, distinct-job denominators, category/scope/region tables and missing-data limits |
| manual_review_requirements.csv | Blank human validity/modality/value/notes/verification/reviewer fields per extracted record |
| manual_review_jobs.csv | Blank per-ad missing-requirement review; output-level precision cannot establish recall |
| ambiguous_requirements_queue.csv | Optional inspection queue, not a blocking completion requirement or automatic API work list |
| dry_run_report.json / run_summary.json | Selection, coverage and deterministic $0 API cost |
| optional_llm_plan.json | Optional planning-only forecast; no executable LLM backend |
| manifest.json / preservation_manifest.json | Input/code hashes and verification that pre-existing data files were unchanged |

CSV encoding is UTF-8 with BOM for Windows spreadsheet compatibility. JSON/JSONL
are UTF-8. Lists/dictionaries in CSV cells are JSON, not comma-delimited guesses.
Source text is never normalized or translated for matching. Open original source
columns as text in spreadsheets; authoritative originals remain unchanged.

## Job-level fields

- `skills_required`, `skills_preferred`, `skills_alternative_required`,
  `skills_mentioned_only`: canonical skill lists. Full occurrence details stay in
  normalized tables. Missing matches produce empty lists, not proof of no skills.
- `swedish_requirement`, `english_requirement`: `required`, `preferred`,
  `explicitly_not_required` or `not_specified`. Contradictory and either/or language
  evidence stays in the evidence table, sets a review flag and abstains from a
  single unqualified language claim. Application language, employer location and
  team language alone are not applicant qualifications.
- `education_levels`: one or more of `bachelor`, `master`, `phd`,
  `degree_unspecified`, or `not_specified`. `education_level` is that single level,
  `multiple_levels`, or `not_specified`; `education_modalities` preserves required,
  preferred and alternative requirements. Bachelor-or-master is not silently
  converted into a mandatory master's degree. A thesis or supervising PhD students
  does not establish a degree requirement.
- `experience_requirements`: records with `stated_years`, `minimum_years`,
  `maximum_years`, `bound_type`, modality and requirement ID. `3–5` supplies bounds;
  `5+` / `at least five` supplies a minimum; bare `5 years` is a stated duration,
  not an invented exact/minimum/maximum interpretation. Study duration, age,
  graduation dates and nonnumeric experience do not become professional years.
- `seniority`: `junior`, `mid_level`, `senior`, `lead` or `unspecified`. Match only
  explicit own-role title/body wording, never infer from years or colleagues.
  Multiple explicit title labels use lead/senior/mid/junior precedence; all matches
  remain auditable. Seniority is not a qualification modality.
- `employment_type`, `working_time`, `work_arrangement`: lists of explicit values
  or `not_specified`. Hybrid cloud architecture is not hybrid work. Original
  employer/municipality/region/date/vacancy count remain unchanged source metadata.
- `requirements_data_usable`: at least one non-mention qualification fact among
  skills, language, degree or numeric experience, including explicit negation.
  Title seniority or location alone does not make qualification data usable.
- `requirements_review_required`: broad conservative inspection flag, not an
  instruction to fix/classify/retry every ad or a probability of correctness.

## Evidence schema

Each normalized record has `requirement_id`, `advertisement_id`, `kind`, `value`,
`modality`, `source_field`, `match_start`, `match_end`, `matched_text`,
`evidence_start`, `evidence_end`, `evidence_text`, `rule_id`, `extractor_version`
and `extraction_review_required`. Optional fields include qualification-heading
offsets/text, alternative group, numeric experience bounds and metadata value.

Offsets are Python Unicode-character offsets [start,end) into the ORIGINAL
`description_text`, `job_title` or explicitly identified metadata field. Both
matched entity and supporting full clause are exact slices. Qualification-heading
rules retain the separate heading slice, so required/preferred inference is
inspectable even when its cue is on another line. Whitespace, line breaks,
Swedish diacritics, Unicode separators and punctuation are preserved.

Modality is `required`, `preferred`, `alternative_required`,
`explicitly_not_required`, `ambiguous` or `mentioned_only`; seniority/employment use
`explicit` and original metadata uses `provided_metadata`. Each occurrence is kept:
conflicting evidence is not overwritten. Alternative items are not individually
mandatory. Mention-only and ambiguous skills do not enter requested-skill totals.

## Rules, review and limits

The versioned catalog covers common programming languages/cloud/tool names, not
every technology or paraphrase. Explicit requirement/preference/negation cues and
qualification headings establish modality. Glued headings from collected HTML are
recognized by boundaries without modifying text. Duties/tool usage, training,
company descriptions and optional or mixed clauses are not silently made mandatory.
Context/negation scope is rule-based and can miss or misinterpret complex prose;
review exact excerpts before publication. Missing extraction is `not_specified`,
not evidence that a job does not require the qualification.

Summary functions count distinct jobs, not repeated mentions. They expose selected
jobs, usable qualification jobs, skills-specified jobs and field-specific denominators.
Per-category tables include category denominators; regional tables expose selected
and usable jobs per region. Alternative levels/skills can overlap and must not be
summed as disjoint categories. Research, governance, adjacent and core subsets stay
separate. All findings concern the existing AI-pilot subset, not total Swedish
labor demand, prevalence, independent unique vacancies or validated model accuracy.

The evaluation notebook shows evidence and blank manual-review files. Precision
of extracted records can be computed only after explicit human validity judgments
and named verification. Recall needs a complete independently enumerated gold set,
not just a review of system outputs. No extraction accuracy is currently claimed.

Deterministic extraction uses no external model, credentials or network and costs
$0 in API usage. Optional `--plan-llm` only forecasts a bounded subset of ambiguous
ads using configured prices and draft token allowances. It has no API execution
path; it does not bypass the existing SSL unknown-outcome hold. A later optional
LLM extractor requires separately authorized provider/schema/budget/checkpoint work.
Many not_specified fields are genuinely absent and should never be invented by an
LLM. Completion of this queue is not a prerequisite for analysis or a future dashboard.
