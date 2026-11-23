"""
Aggregate Betfair betting odds CSV files into single parquet per season.

This script consolidates individual match odds CSV files into a single parquet file:
- Searches for all Betfair CSV files for a league/season
- Calculates match_minute from timestamps
- Aggregates odds by minute
- Saves to data/{league}/{season}/betfair_odds.parquet

Output schema:
    fbref_match_id, minute, odds_home, odds_away, odds_draw, is_real_update

Where:
    - is_real_update: True if ANY odds update in that minute was real (not forward-filled)

Usage:
    python aggregate_betfair_odds.py --league premier-league --season 24-25
    python aggregate_betfair_odds.py --all  # Process all leagues/seasons
"""

import pandas as pd
import json
from pathlib import Path
import sys
import argparse
import numpy as np

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

LEAGUE_MAPPING = {
    'premier-league': 'ENG-Premier League',
    'bundesliga': 'GER-Bundesliga',
    'la-liga': 'ESP-La Liga',
    'ligue-1': 'FRA-Ligue 1',
    'serie-a': 'ITA-Serie A'
}

# Base paths (repo root = two levels above scripts/processing/)
BASE_PATH = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_PATH / 'data'
BETFAIR_PATH = BASE_PATH / 'OUTPUT_LEAGUES'  # per-match CSVs written by parse_betfair_by_league_mp_v2.py

def load_team_mapping(league: str = 'premier-league'):
    """Load team name mapping from JSON config."""
    config_path = BASE_PATH / 'config' / 'team_name_mapping.json'

    if not config_path.exists():
        print(f"[WARNING] Team mapping file not found: {config_path}")
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)

    if league in mapping_data:
        return mapping_data[league]
    elif league.replace('-', '_') in mapping_data:
        return mapping_data[league.replace('-', '_')]
    else:
        return {}

def find_all_odds_files(league: str, season: str, team_mapping: dict = None) -> list:
    """Find all Betfair odds CSV files for a league/season."""

    # New unified structure: data/betfair/{league}/{season}/*.csv
    odds_dir = BETFAIR_PATH / league / season

    if not odds_dir.exists():
        print(f"   [WARNING] Betfair odds directory not found: {odds_dir}")
        return []

    csv_files = list(odds_dir.glob('*.csv'))
    print(f"   Found {len(csv_files)} CSV files in {odds_dir}")
    # keep league fixtures only: cup ties / friendlies between the same clubs would otherwise be
    # keyed onto the league match downstream; rescheduled kick-offs are renamed to the schedule id
    sched = set(pd.read_parquet(DATA_PATH / league / season / 'fbref_schedule.parquet')['match_id'])
    ids = pd.Series([f.stem for f in csv_files], index=range(len(csv_files)))
    aligned = align_to_schedule(ids, sched)
    out = []
    for f, a in zip(csv_files, aligned):
        if pd.isna(a):
            print(f"   [SKIP] not a league fixture: {f.stem}")
        elif a != f.stem and a in set(ids):   # exact file exists too: the renamed one is a relisted market
            print(f"   [SKIP] duplicate of {a}: {f.stem}")
        else:
            if a != f.stem:
                print(f"   [RENAME] {f.stem} -> {a}")
            out.append((f, a))
    if len({a for _, a in out}) != len(out):
        raise ValueError("two odds files align to one fixture")
    return out

def half_durations(league: str, season: str) -> dict:
    """Map fbref_match_id -> (1H duration, 2H duration) in minutes, from WhoScored expanded minutes."""
    d = DATA_PATH / league / season
    ws = pd.read_parquet(d / 'whoscored_events.parquet', columns=['fbref_match_id', 'period', 'expanded_minute'])
    ws = ws[ws['period'].isin(['FirstHalf', 'SecondHalf'])]
    g = ws.groupby(['fbref_match_id', 'period'])['expanded_minute'].max().unstack()
    # 2H expanded_minute is cumulative (includes 1H stoppage), so 2H duration = exp2H - exp1H
    return {m: (int(r['FirstHalf']) + 1, int(r['SecondHalf']) - int(r['FirstHalf'])) for m, r in g.iterrows()}


HT_BREAK = 15            # median halftime break in wall-clock minutes (25-26 PL: 12-17 for 80% of feeds)
HT_BREAK_OK = (10, 20)   # implied break outside this range => feed truncated, anchor 2H forward instead


def align_to_schedule(ids: pd.Series, sched_ids: set, tol_days: int = 7) -> pd.Series:
    """Map Betfair match ids ('{date} {home}-{away}') onto FBref schedule ids: exact match, else the
    unique league fixture with the same team pair within tol_days (rescheduled kick-off), else NaN
    (cup ties, friendlies). Callers drop the NaNs and report them."""
    key = lambda m: (m.split(' ', 1)[1], pd.Timestamp(m[:10]))
    by_pair = {}
    for m in sched_ids:
        t, d = key(m)
        by_pair.setdefault(t, []).append((d, m))

    def one(m):
        if m in sched_ids:
            return m
        t, d = key(m)
        near = [x for dd, x in by_pair.get(t, []) if abs((dd - d).days) <= tol_days]
        if len(near) > 1:
            raise ValueError(f"{m}: {len(near)} league fixtures within {tol_days} days")
        return near[0] if near else np.nan
    return ids.map(one)


def add_game_clock(df: pd.DataFrame, dur: tuple) -> pd.DataFrame:
    """Add match_minute (game clock, -1 at halftime) and period columns.

    Anchors: 1H = [kickoff, kickoff + dur1H]; 2H = [last row - dur2H, last row] when the feed runs to
    full time (implied halftime break within HT_BREAK_OK), else forward from 1H end + HT_BREAK
    (truncated feed). Matches the hand-repaired 24-25 betfair_odds.parquet semantics.
    """
    if not dur:  # no WhoScored durations (should not happen once ids are schedule-aligned)
        df['match_minute'], df['period'] = df['minute'].astype(float), 'unknown'
        df.loc[df['minute'] < 0, 'period'] = 'PreMatch'
        return df
    dur1, dur2 = dur
    start2 = df['minute'].max() - dur2
    if not HT_BREAK_OK[0] <= start2 - dur1 <= HT_BREAK_OK[1]:
        start2 = dur1 + HT_BREAK
    # Offsets calibrated so Betfair goal-reaction latency vs WhoScored matches the 24-25 file
    # (eval/goals/goal_latency.py: median 0, mean -0.3 in both halves). 1H mm = wall - 2; 2H mm = 43 + wall - start2.
    wall = df['minute']
    df['period'] = np.select(
        [wall <= 0, wall <= dur1, wall < start2], ['PreMatch', 'FirstHalf', 'halftime'], 'SecondHalf')
    df['match_minute'] = np.select(
        [wall <= 0, wall <= dur1, wall < start2],
        [wall.astype(float), (wall - 2).astype(float), -1.0], (43 + wall - start2).astype(float))
    return df


def process_odds_file(odds_file: Path, fbref_match_id: str) -> pd.DataFrame:
    """Load and process a single betting odds CSV file."""

    try:
        df = pd.read_csv(odds_file)
    except Exception as e:
        print(f"      [ERROR] Failed to read {odds_file.name}: {e}")
        return pd.DataFrame()

    if 'minute' not in df.columns or 'in_play' not in df.columns:
        print(f"      [SKIP] Missing required columns in {odds_file.name}")
        return pd.DataFrame()

    minute_dt = pd.to_datetime(df['minute'])
    real_ts = pd.to_datetime(df['timestamp'])  # actual update time; NaT on forward-filled rows

    # Find match start (first in-play row)
    in_play_rows = df[df['in_play'] == True]
    if len(in_play_rows) == 0:
        print(f"      [SKIP] No in-play data in {odds_file.name}")
        return pd.DataFrame()

    # Wall-clock minutes from kickoff (first in-play minute); pre-match rows are negative
    kickoff_time = minute_dt[in_play_rows.index[0]]
    wall = ((minute_dt - kickoff_time).dt.total_seconds() / 60).astype(int)

    # Team odds columns: parser writes home, draw, away in file-specific order; recover by name
    meta_cols = ['minute', 'timestamp', 'market_status', 'in_play', 'is_real_update', 'match_event']
    odds_cols = [col for col in df.columns if col not in meta_cols]
    if len(odds_cols) != 3:
        print(f"      [SKIP] Expected 3 odds columns in {odds_file.name}, got {odds_cols}")
        return pd.DataFrame()
    home_name = fbref_match_id.split(' ', 1)[1].rsplit('-', 1)[0]
    draw_col = next(c for c in odds_cols if c.lower() == 'the draw')
    team_cols = [c for c in odds_cols if c != draw_col]
    # Betfair runner names -> FBref names via config mapping (exact, no fuzzy matching)
    with open(BASE_PATH / 'config' / 'betfair_team_mapping.json', encoding='utf-8') as f:
        bf_map = {k: v for league_map in json.load(f).values() if isinstance(league_map, dict) for k, v in league_map.items()}
    mapped = {c: bf_map.get(c, c) for c in team_cols}
    home_col = next((c for c, fb in mapped.items() if fb == home_name), None)
    if home_col is None:
        raise ValueError(f"{odds_file.name}: no runner maps to home '{home_name}' (runners: {team_cols})")
    away_col = next(c for c in team_cols if c != home_col)

    out = pd.DataFrame({
        'fbref_match_id': fbref_match_id, 'minute': wall,
        'odds_home': df[home_col], 'odds_away': df[away_col], 'odds_draw': df[draw_col],
        'is_real_update': df['is_real_update'], 'timestamp': real_ts.fillna(minute_dt),
    })
    inv = out[['odds_home', 'odds_draw', 'odds_away']].rdiv(1)
    normed = inv.div(inv.sum(axis=1), axis=0)
    out['prob_home'], out['prob_draw'], out['prob_away'] = normed['odds_home'], normed['odds_draw'], normed['odds_away']
    return out

def process_league_season(league: str, season: str):
    """Process all betting odds files for a league-season combination."""
    print(f"\n{'='*80}")
    print(f"{league.upper()} {season} - AGGREGATING BETFAIR ODDS")
    print(f"{'='*80}")

    # Load team name mapping
    team_mapping = load_team_mapping(league)
    if team_mapping:
        print(f"[OK] Loaded {len(team_mapping)} team name mappings")

    # Find all odds files
    print(f"\n[STEP 1] Finding betting odds CSV files...")
    odds_files = find_all_odds_files(league, season, team_mapping)

    if not odds_files:
        print(f"[ERROR] No betting odds files found for {league} {season}")
        print(f"   Searched in: {BETFAIR_PATH / league / season}")
        return

    print(f"[OK] Found {len(odds_files)} unique odds files")

    # Process each file
    print(f"\n[STEP 2] Processing {len(odds_files)} odds files...")

    all_odds = []
    processed_count = 0
    error_count = 0

    for idx, (odds_file, mid) in enumerate(odds_files):
        try:
            print(f"\n   [{idx+1}/{len(odds_files)}] {odds_file.name}")

            odds_df = process_odds_file(odds_file, mid)

            if len(odds_df) > 0:
                all_odds.append(odds_df)
                processed_count += 1
                print(f"      [OK] {len(odds_df)} minute-level odds records")

        except Exception as e:
            print(f"      [ERROR] Failed: {e}")
            error_count += 1
            continue

    # Combine all odds
    if len(all_odds) > 0:
        print(f"\n[STEP 3] Combining all odds, deriving game clock from WhoScored half durations...")
        durs = half_durations(league, season)
        no_ws = [d['fbref_match_id'].iloc[0] for d in all_odds if d['fbref_match_id'].iloc[0] not in durs]
        # League fixtures without WhoScored events (not yet played / feed missing) get period='unknown'
        all_odds = [add_game_clock(d, durs.get(d['fbref_match_id'].iloc[0])) for d in all_odds]
        print(f"   Game clock derived; {len(no_ws)} matches without WhoScored durations marked 'unknown': {no_ws}")
        combined = pd.concat(all_odds, ignore_index=True)
        combined = combined[['fbref_match_id', 'minute', 'match_minute', 'period', 'odds_home', 'odds_away',
                             'odds_draw', 'is_real_update', 'timestamp', 'prob_home', 'prob_draw', 'prob_away']]

        # Save to new structure: data/{league}/{season}/betfair_odds.parquet
        output_dir = DATA_PATH / league / season
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / 'betfair_odds.parquet'
        combined.to_parquet(output_path, index=False)

        # Print summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Matches processed: {processed_count}/{len(odds_files)}")
        print(f"Matches failed: {error_count}")
        print(f"Total minute-level odds: {len(combined):,}")
        print(f"Unique matches: {combined['fbref_match_id'].nunique()}")

        # Show data quality if is_real_update exists
        if 'is_real_update' in combined.columns:
            real_pct = combined['is_real_update'].sum() / len(combined) * 100
            print(f"Real updates: {combined['is_real_update'].sum():,} ({real_pct:.1f}%)")
            print(f"Forward-filled: {(~combined['is_real_update']).sum():,} ({100-real_pct:.1f}%)")

        print(f"\nOutput: {output_path}")
        print(f"Size: {combined.shape}")
        print(f"Columns: {list(combined.columns)}")

    else:
        print("[WARNING] No odds were successfully processed")

def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate Betfair odds CSV files into parquet')
    parser.add_argument('--league', choices=list(LEAGUE_MAPPING.keys()),
                       help='League name')
    parser.add_argument('--season', help='Season (YY-YY format)')
    parser.add_argument('--all', action='store_true',
                       help='Process all leagues and seasons')
    return parser.parse_args()

def main():
    args = parse_args()

    if args.all:
        leagues = list(LEAGUE_MAPPING.keys())
        seasons = ['17-18', '18-19', '19-20', '20-21', '21-22', '22-23', '23-24', '24-25']

        print(f"Processing {len(leagues)} leagues × {len(seasons)} seasons")

        for league in leagues:
            for season in seasons:
                try:
                    process_league_season(league, season)
                except Exception as e:
                    print(f"[ERROR] Failed {league} {season}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    elif args.league and args.season:
        process_league_season(args.league, args.season)

    else:
        print("ERROR: Must specify either --league and --season, or --all")
        sys.exit(1)

    print(f"\n{'='*80}")
    print("DONE! Aggregated odds saved to data/{league}/{season}/betfair_odds.parquet")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
