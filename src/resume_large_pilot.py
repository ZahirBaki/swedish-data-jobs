"""Resume a recovered Step 3E run; filesystem retries do not retry API calls."""
import argparse
from pathlib import Path

from src.llm_classifier import Config
from src.recover_large_pilot import inspect,writable_probe,resilient_checkpoint_writes
from src.run_large_pilot import run
from src.validate_classification import ROOT


def resume(run_dir,budget_usd,execute=False,cache_dir=None,session=None,sleep=None):
    run_dir=Path(run_dir)
    if not (run_dir/'recovery_manifest.json').is_file():raise ValueError('Use a recovered run directory')
    report,_,_,plan=inspect(run_dir,budget_usd,cache_dir)
    if report['pending_markers'] or report['unfinished_ledger_attempts']:
        raise ValueError('Unfinished requests need inspection; no automatic retry is allowed')
    writable_probe(run_dir);writable_probe(cache_dir or ROOT/'data/processed/.llm_cache')
    config=Config(**plan['identity']['config'])
    with resilient_checkpoint_writes():
        return run(plan['prepared'],config,budget_usd,execute=execute,resume=run_dir,
            max_ads=plan['identity']['max_ads'],cache_dir=cache_dir,session=session,sleep=sleep)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--budget-usd',required=True,type=float)
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--execute',action='store_true');modes.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();_,summary=resume(args.run,args.budget_usd,execute=args.execute)
    print('Status:',summary['status'],'Accounted cost:',summary['accounted_cost_usd'])
    return 0 if summary['status'] in {'completed','dry_run_only'} else 2


if __name__=='__main__':raise SystemExit(main())
