"""Build fictional dashboard examples locally, with the production extractor.

No original advertisement, employer, identifier, model output or credential is
copied. This is a deterministic UI fixture, not a miniature research sample.
"""

import json
from pathlib import Path

import pandas as pd

from src.extract_requirements import extract_job

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "examples" / "demo_dataset"
EXAMPLES = [
    ("Data Engineer", "core_industry", "Senior Data Engineer", "Stockholm",
     "Responsibilities\nYou will build data pipelines for analytics.\nRequirements\nYou have experience with Python and SQL.\nEnglish is required.\nPreferred\nExperience with Azure and Databricks is preferred."),
    ("Data Analyst", "core_industry", "Data Analyst", "Västra Götaland",
     "Arbetsuppgifter\nDu analyserar verksamhetsdata och skapar rapporter.\nKvalifikationer\nDu har erfarenhet av SQL och Power BI.\nSvenska är ett krav.\nMeriterande\nKandidatexamen är meriterande."),
    ("ML Engineer", "core_industry", "ML Engineer", "Skåne",
     "Responsibilities\nYou will develop and deploy machine learning models.\nRequirements\nYou have experience with Python and PyTorch.\nYou have at least 3 years of professional experience.\nA master's degree is required.\nWe use Docker in our team."),
    ("AI / ML Research", "academic_ai_ml_research", "AI Researcher", "Uppsala",
     "Responsibilities\nYou will research machine learning methods.\nRequirements\nA PhD degree is required.\nEnglish is required.\nPreferred\nExperience with Python is preferred."),
    ("Data Governance / Data Quality", "governance", "Data Governance Analyst", "Stockholm",
     "Responsibilities\nYou will define data quality controls and governance processes.\nRequirements\nEnglish is required.\nPreferred\nExperience with SQL is preferred."),
    ("Data Scientist", "opportunities", "Data Science Internship", "Skåne",
     "Responsibilities\nYou will develop predictive models with guidance.\nRequirements\nYou have experience with Python.\nEnglish is required.\nOur team uses AWS."),
]


def build(destination=DESTINATION):
    """Rebuild only the explicitly designated fictional fixture directory."""
    destination = Path(destination)
    jobs, evidence = [], []
    for index, (category, scope, title, region, description) in enumerate(EXAMPLES, 1):
        identifier = f"demo-{index:03d}"
        quote = description.splitlines()[1]
        start = description.index(quote)
        opportunities = ["internship"] if scope == "opportunities" else []
        row = dict(
            advertisement_id=identifier, job_title=title, employer=f"Fictional Example {index}",
            publication_date="2026-06-01", region=region, municipality="Fictional municipality",
            number_of_vacancies="1", description_text=description, original_advertisement_url="",
            ai_status="synthetic_fixture", ai_relevance="relevant", ai_primary_category=category,
            ai_secondary_categories=[], ai_scope=scope, ai_opportunity_types=opportunities,
            ai_multi_role=False, ai_review_required=False, ai_confidence="not_applicable",
            ai_reasoning="Handwritten fictional example; no AI classification was performed.",
            ai_evidence=[quote], ai_evidence_spans=[dict(source_field="description_text", start=start,
                                                      end=start + len(quote), text=quote)],
            ai_model="synthetic_fixture_no_model", ai_prompt_version="synthetic_fixture",
            classification_source_valid=True, classification_human_verified=False,
            classification_selected_run="examples/demo_dataset (fictional fixture)",
            reporting_scope=scope, analysis_scope=scope, primary_category_label=category,
            analysis_opportunity_types=opportunities, opportunity_group="restricted_or_student_opportunity" if opportunities else "ordinary_employment",
            multi_role_group="not_flagged_multi_role", strict_audit_approved=index == 1,
            v3_audit_status="synthetic_approved_example" if index == 1 else "synthetic_not_approved_example",
            v3_audit_reasons=["Fictional UI state; no real audit was performed."],
            dashboard_inclusion_method="synthetic_fixture", dashboard_inclusion_policy_version="demo-only",
            classification_provisional=True,
        )
        enriched, facts = extract_job(row)
        jobs.append(enriched)
        evidence.extend(facts)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "jobs_requirements.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in jobs), encoding="utf-8"
    )
    pd.DataFrame(evidence).to_csv(destination / "requirements_evidence.csv", index=False, encoding="utf-8-sig")
    summary = dict(total_collected=len(jobs), total_successfully_classified=0, expanded_total=len(jobs),
                   strict_audit_approved=1, scopes=pd.Series([row["analysis_scope"] for row in jobs]).value_counts().to_dict())
    (destination / "step7_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([dict(advertisement_id=row["advertisement_id"], step7_status="synthetic_fixture") for row in jobs]).to_csv(
        destination / "corpus_coverage.csv", index=False, encoding="utf-8"
    )
    (destination / "demo_manifest.json").write_text(json.dumps(dict(
        synthetic=True, real_advertisements=0, paid_requests=0, generated_by="python -m tools.build_demo",
        warning="Fictional UI fixtures. No real classifications, research evidence or market statistics."
    ), indent=2) + "\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    print(build())
