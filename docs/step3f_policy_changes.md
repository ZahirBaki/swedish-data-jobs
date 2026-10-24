# Step 3F policy-development review

The new policy is a final proposal for human review, not a validated ground truth.
Original policies, prompts, datasets and Step 3D/3E engines remain unchanged.

The 200-advertisement assistant review proposes 78 relevant, 95 irrelevant and
27 uncertain labels. Among the 159 technically accepted original predictions,
there are 30 relevance and 27 primary-category disagreements, affecting 34 distinct
advertisements. The other 41 are technical failures, not valid competing judgments.
No assistant labels are treated as human verified; no quality metrics are computed.

| Recurring weakness | Proposed clarification | Illustrative advertisement IDs |
| --- | --- | --- |
| Domain science treated as AI/ML or data science | Scientific computing, simulation, imaging and a doctorate do not establish substantive AI/ML work; inspect methods and responsibilities | 31582037, 31309910, 31458261, 30952680, 31009156 |
| Software/infrastructure treated as data engineering | SQL, Neo4j, Kubernetes, cloud and ORM alone do not establish analytical platform or pipeline duties | 31174356, 30940238 |
| Hybrid AI/software titles with unclear scope | Require substantial production AI ownership; retain uncertainty where the work or dominant role is vague | 30985069, 31396059, 31337093, 31530734 |
| AI leadership excluded for lack of coding | Accountable Data/AI delivery, governance or programme leadership can qualify without coding; generic adoption is insufficient | 31035548, 30959030, 31136710 |
| Operational reporting treated as a dedicated occupation | Ordinary KPIs and incidental reporting are insufficient; establish dedicated analytics ownership | 31383504 |
| Academic status conflated with occupational category | Track doctoral/postdoctoral/internship/thesis attributes independently; academic AI research requires substantive AI/ML methodology | 31518941, 31590836 |
| Statistics and geodata boundaries | Dedicated statistical methods may qualify separately; GIS/cartography and domain data use need a duties-based assessment | 30963916, 31064402, 31252307, 30967846 |

These examples identify questions, not adjudicated corrections. In particular,
an assistant suggestion that any doctoral role is AI research is not adopted.
Governance is separately selectable; generic legal/compliance and quality work
needs substantial accountable Data/AI governance duties. Named canonical categories
are preserved; scope and opportunity attributes provide separate reporting filters.

## Technical change

The earlier run rejected 36 responses for unmatched quotations and five for
non-relevant predictions containing secondary categories. Step 3F returns source
span IDs rather than retyped quotations. Every character of the original title
and description is supplied in ordered 240-character spans, without normalization
or truncation. The validator resolves IDs to exact source strings and Unicode
character offsets. Unknown IDs, fabricated quotations, contradictory scopes and
invalid category abstentions remain failures. A source-grounded span still needs
human inspection for logical support; this change does not demonstrate accuracy.

Valid relevance outcomes are Relevant, Irrelevant and Uncertain. Rejected or
incomplete responses have null relevance/category and the distinct outcome
Technical failure / requires retry. Planned, unexecuted rows have no final outcome.

## Version and reuse rules

Policy `step3f-v1`, prompt `step3f-1.0`, validation/schema `indexed-spans-1`.
Prior labels are retained as historical columns, never supplied as model truth.
Reuse requires exact content, model, request settings, endpoint, policy, prompt and
schema identity. The prepared manifest also freezes code hashes for safe resume.
Old caches are retained but cannot be reused under this changed request identity.

New requests require explicit `--execute`. Automatic API retries are disabled.
Terminal failures remain preserved on resume; unknown outcomes block repayment.
Before any paid call the runner checks all source/cache/checkpoint identities,
probes write/replace permissions, and durably saves the checkpoint. Windows file
replacement retries are local filesystem operations, not additional API requests.
Reservations are enforced per request; provider-reported token costs and unreported
budget holds remain separate. The application budget cannot guarantee a provider
invoice ceiling if reported usage exceeds its conservative reservation; it records
the overrun and stops immediately. Provider prompt-cache discounts are not assumed.

## Next quality gate

Human review of the proposed boundaries and a small new span-schema pilot should
precede a full paid run. The 200 reviewed ads influenced policy development and
must not be presented as an independent validation holdout. Use a fresh representative
human-reviewed sample to assess relevance and category quality before publication.
Never infer complete Swedish market demand from this candidate dataset.
