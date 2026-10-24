# Fictional dashboard demo

`demo_dataset/` contains six handwritten examples with invented employers and
IDs beginning with `demo-`. No original advertisement, contact details, private
path, API response or provider receipt is copied from the research collection.

The production deterministic extractor generated the requirement tables, with
exact source offsets and the same qualification modalities used by the dashboard.
Classification and audit states are **synthetic UI fixtures**, not real model
outputs or audited research evidence. They are labelled in `demo_manifest.json`
and in the dashboard. Demo counts must never be added to research totals.

Run `python -m tools.build_demo` to regenerate only these fixtures, then set
`JOBS_DASHBOARD_DATA=examples/demo_dataset` before launching Streamlit. A fresh
clone automatically uses this demo when no completed local research publication
is available. Existing full local publications retain priority.
