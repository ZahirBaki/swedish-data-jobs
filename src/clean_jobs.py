"""Non-destructive cleaning of the Step 2 CSV; original columns remain intact."""
import json
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {'advertisement_id', 'job_title', 'description_text', 'publication_date'}


def normalize_text(value):
    if value is None or pd.isna(value):
        return ''
    text = unicodedata.normalize('NFKC', str(value))
    text = ''.join(' ' if unicodedata.category(c) in {'Zs', 'Zl', 'Zp', 'Cf'} else c for c in text)
    return re.sub(r'\s+', ' ', text).strip().casefold()


def latest_collection(root=ROOT / 'data' / 'collections'):
    completed = []
    for path in Path(root).glob('*/summary.json'):
        summary = json.loads(path.read_text(encoding='utf-8'))
        if summary.get('status') == 'queries_exhausted' and (path.parent / 'jobs.csv').is_file():
            completed.append((summary['finished_at'], path.parent))
    if not completed:
        raise FileNotFoundError('No completed collection with jobs.csv found')
    return max(completed)[1]


def load_jobs(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    if missing := REQUIRED - set(frame.columns):
        raise ValueError(f'Missing required CSV columns: {sorted(missing)}')
    return frame


def publication_day(value):
    try:
        stamp = datetime.fromisoformat(str(value).strip().replace('Z', '+00:00'))
        if stamp.tzinfo is not None:
            stamp = stamp.astimezone(ZoneInfo('Europe/Stockholm'))
        return stamp.date()
    except (ValueError, TypeError, OverflowError):
        return None


def clean_jobs(frame, start=None, end=None):
    """Retain invalid/out-of-window rows with explicit statuses for audit.

    Only repeated nonempty exact IDs are removed (first CSV occurrence wins).
    Missing IDs stay separate. Inclusive bounds use Swedish calendar days.
    """
    start = date.fromisoformat(start) if isinstance(start, str) else start
    end = date.fromisoformat(end) if isinstance(end, str) else end
    if start and end and start > end:
        raise ValueError('Start must not exceed end')
    out = frame.copy(deep=True).fillna('')
    duplicate = out.advertisement_id.ne('') & out.advertisement_id.duplicated()
    out = out.loc[~duplicate].copy()
    for source, target in [('job_title', 'job_title_normalized'), ('description_text', 'description_normalized')]:
        out[target] = out[source].map(normalize_text)
    out['missing_title'] = out.job_title_normalized.eq('')
    out['missing_description'] = out.description_normalized.eq('')
    out['missing_id'] = out.advertisement_id.map(normalize_text).eq('')
    days = out.publication_date.map(publication_day)
    out['publication_day'] = days.map(lambda d: d.isoformat() if d else '')
    out['date_status'] = days.map(lambda d: 'invalid_or_missing' if d is None else
                                  'outside_window' if (start and d < start) or (end and d > end) else 'valid')
    out['in_analysis_window'] = out.date_status.eq('valid')
    return out.reset_index(drop=True)
