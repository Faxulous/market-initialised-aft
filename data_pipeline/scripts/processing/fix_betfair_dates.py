#!/usr/bin/env python3
"""
Snap Betfair CSV filenames to FBref schedule dates.

Betfair market dates can differ from FBref kickoff dates by 1-2 days (UTC market
creation, reschedules). The CSV filename is used as fbref_match_id downstream, so
rename any CSV whose team pair matches a schedule fixture within +/-3 days.

Usage:
    python scripts/processing/fix_betfair_dates.py --league premier-league --season 25-26
"""

import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd

BASE_PATH = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description='Snap Betfair CSV dates to FBref schedule')
    ap.add_argument('--league', required=True)
    ap.add_argument('--season', required=True)
    args = ap.parse_args()

    sched = pd.read_parquet(BASE_PATH / 'data' / args.league / args.season / 'fbref_schedule.parquet')
    by_pair = {}  # "Home-Away" -> [date, ...]
    for m in sched['match_id']:
        d, pair = m.split(' ', 1)
        by_pair.setdefault(pair, []).append(datetime.strptime(d, '%Y-%m-%d'))

    csv_dir = BASE_PATH / 'OUTPUT_LEAGUES' / args.league / args.season
    renamed = 0
    for f in sorted(csv_dir.glob('*.csv')):
        d, pair = f.stem.split(' ', 1)
        fd = datetime.strptime(d, '%Y-%m-%d')
        cands = [s for s in by_pair.get(pair, []) if 0 < abs((s - fd).days) <= 3]
        if not cands or fd in by_pair.get(pair, []):  # exact match already, or no near fixture
            continue
        target = min(cands, key=lambda s: abs(s - fd))
        new = f.with_name(f"{target.strftime('%Y-%m-%d')} {pair}.csv")
        if new.exists():
            raise FileExistsError(f"refusing to overwrite {new.name} with {f.name}")
        f.rename(new)
        renamed += 1
        print(f"{f.name} -> {new.name}")
    print(f"\nRenamed {renamed} files")


if __name__ == '__main__':
    main()
