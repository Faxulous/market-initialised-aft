#!/usr/bin/env python3
"""
Parse Betfair OVER_UNDER_* markets into kick-off O/U lines per match.

Reads BASIC/*.bz2 files, keeps OVER_UNDER_05..45 markets, takes the LAST traded
price for each Over/Under runner BEFORE the market goes in-play (= kick-off odds),
and writes one row per match:

    data/{league}/{season}/over_under.parquet
    columns: match_id, over_0.5, under_0.5, ..., over_4.5, under_4.5

match_id = "{date} {home}-{away}" (FBref format, same as betfair_odds CSVs).

Usage:
    python scripts/processing/parse_betfair_ou.py --league premier-league --season 25-26
"""

import bz2, json, argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_betfair_by_league_mp_v2 import BASE_PATH, load_config, classify_match, get_season_from_date
from aggregate_betfair_odds import align_to_schedule


def process_file(file_path: str, config: dict):
    """Extract (league, season, match_id, threshold, {runner: (pt, ltp)}) from one OU .bz2 file."""
    try:
        with bz2.open(file_path, 'rt', encoding='utf-8') as f:
            first = f.readline()
            if not first:
                return None
            md = json.loads(first).get('mc', [{}])[0].get('marketDefinition')
            if not md or not md.get('marketType', '').startswith('OVER_UNDER_'):
                return None
            thresh = int(md['marketType'].rsplit('_', 1)[1]) / 10  # OVER_UNDER_25 -> 2.5

            dt = datetime.fromisoformat(md['marketTime'].replace('Z', '+00:00'))
            season = get_season_from_date(dt)
            cls = classify_match(md.get('eventName', ''), season, config)
            if not cls:
                return None
            league, home, away = cls
            match_id = f"{dt.strftime('%Y-%m-%d')} {home}-{away}"

            runners = {r['id']: r.get('name', '') for r in md.get('runners', [])}
            last = {}  # runner name -> (pt, ltp), latest pre-in-play trade
            f.seek(0)
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pt = rec.get('pt')
                for mc in rec.get('mc', []):
                    if mc.get('marketDefinition', {}).get('inPlay'):
                        return (league, season, match_id, thresh, last)  # kick-off reached
                    for rc in mc.get('rc', []):
                        if rc.get('ltp') is not None:
                            last[runners.get(rc['id'], '')] = (pt, rc['ltp'])
            return (league, season, match_id, thresh, last)  # market never went in-play
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description='Parse Betfair O/U kick-off odds')
    ap.add_argument('--league', required=True)
    ap.add_argument('--season', required=True, help='YY-YY, e.g. 25-26')
    ap.add_argument('--workers', type=int, default=cpu_count())
    args = ap.parse_args()

    config = load_config()
    files = [str(p) for p in (BASE_PATH / 'BASIC').rglob('*.bz2')]
    print(f"Scanning {len(files)} .bz2 files with {args.workers} workers...")

    # best[(match_id, thresh, side)] = (pt, ltp): latest pre-KO trade wins across duplicate markets
    best = {}
    with Pool(args.workers) as pool:
        for res in pool.imap_unordered(partial(process_file, config=config), files, chunksize=200):
            if res is None:
                continue
            league, season, match_id, thresh, last = res
            if league != args.league or season != args.season:
                continue
            for name, (pt, ltp) in last.items():
                side = 'over' if name.lower().startswith('over') else 'under'
                key = (match_id, thresh, side)
                if key not in best or (pt or 0) > (best[key][0] or 0):
                    best[key] = (pt, ltp)

    if not best:
        raise SystemExit(f"No O/U markets found for {args.league} {args.season}")

    rows = {}
    for (match_id, thresh, side), (_, ltp) in best.items():
        rows.setdefault(match_id, {})[f'{side}_{thresh}'] = ltp
    df = pd.DataFrame([{'match_id': m, **v} for m, v in sorted(rows.items())])
    cols = ['match_id'] + [f'{s}_{t}' for t in (0.5, 1.5, 2.5, 3.5, 4.5) for s in ('over', 'under')]
    df = df.reindex(columns=[c for c in cols if c in df.columns])

    # league fixtures only (cup ties / friendlies dropped; rescheduled kick-offs renamed to the schedule id,
    # an exact-id row winning over a renamed one)
    sched = set(pd.read_parquet(BASE_PATH / 'data' / args.league / args.season / 'fbref_schedule.parquet')['match_id'])
    df['aligned'] = align_to_schedule(df['match_id'], sched)
    df['exact'] = df['aligned'] == df['match_id']
    print(f"Dropped {df['aligned'].isna().sum()} non-league markets; renamed {(df['aligned'] != df['match_id']).sum() - df['aligned'].isna().sum()}")
    df = df.dropna(subset=['aligned']).sort_values('exact', ascending=False).drop_duplicates('aligned')
    df = df.assign(match_id=df['aligned']).drop(columns=['aligned', 'exact']).sort_values('match_id')
    out = BASE_PATH / 'data' / args.league / args.season / 'over_under.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Saved {len(df)} matches x {df.shape[1]-1} price columns -> {out}")


if __name__ == '__main__':
    main()
