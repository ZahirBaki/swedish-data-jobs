"""Extract all available accepted relevant advertisements without API calls."""
import argparse
import json
from pathlib import Path

from src.dashboard_data import load_dataset, summary
from src.run_step4 import run
from src.step4_jobs import load_existing
from src.validate_classification import ROOT, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    registry, _, _ = load_existing()
    available = int(registry.step4_disposition.eq('eligible_relevant').sum())
    plan = run(args.output, size=available, dry_run=args.dry_run)
    if not args.dry_run:
        jobs, facts = load_dataset(args.output)
        if len(jobs) != available:
            raise ValueError('Not all eligible advertisements were extracted')
        report = summary(jobs, facts)
        report['extraction_summary'] = plan
        report['code_sha256'] = {name: sha256(ROOT / 'src' / name)
                                  for name in ['run_step5.py', 'dashboard_data.py']}
        (args.output / 'step5_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        table = __import__('pandas').DataFrame(report['coverage'])
        table.to_csv(args.output / 'requirement_coverage.csv', index=False, encoding='utf-8-sig')
        text = '# Step 5 — available-data extraction\n\n'
        text += f'{len(jobs)} unique accepted relevant advertisements; deterministic extraction; $0 API cost.\n\n'
        text += 'Collection: ' + report['collection_period'] + '.\n\n'
        text += table.to_string(index=False) + '\n\n' + report['limitations'] + '\n'
        (args.output / 'report.md').write_text(text, encoding='utf-8')
        print(json.dumps(report, ensure_ascii=True, indent=2))
    else:
        print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
