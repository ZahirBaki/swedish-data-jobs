# Step 3E policy and prompt change record

Original: `classification_policy.md` (preserved verbatim).
New: `classification_policy_step3e_v1.md`.
New prompt: `prompts/step3e-v1.txt`; identifier `step3e-1.0`.
Existing Step 3D code, prompts, source snapshots, caches and predictions remain intact.

The combined 30-ad pilot and `data/reviews/pilot_classification_assistant_reviewed.csv`
were inspected before these changes. The latter contains 10 relevant and 20
irrelevant assistant suggestions and zero human-verified rows. Its suggestions
motivated boundary clarification only; they are not labels used for evaluation,
sampling or prompts for individual ads.

| Previous ambiguity | Step 3E clarification |
| --- | --- |
| Backend/fullstack + SQL/database/cloud/AI mention | Require substantial qualifying duties; ORM and optional AI tools alone are insufficient |
| HR + recruiting AI/KPIs/dashboards | Distinguish ordinary HR tooling from dedicated substantial people analytics |
| Analysis product use + optional ML | Reporting/insights do not automatically imply Data Scientist |
| Generic engineering title | Allow genuine pipeline/model/production-AI duties to establish relevance |
| Academic AI/ML and internships | Preserve research scope and independent position attributes; do not exclude via relevance |
| Multiple advertised occupations | Flag candidates, retain unresolved primary category when necessary and require review |
| Implicit title-first priority | Actual advertised responsibilities take precedence |

Category spellings and seven-field schema are unchanged. Source-grounded evidence,
safe whitespace-only normalization, review/confidence safeguards and budget/resume
mechanisms are retained. Step 3E has a separate capped-200 runner; Step 3D's capped-30
runner is unchanged. New prompt/policy/request identities prevent reuse of older
cached predictions as if produced under the new policy.
