#!/usr/bin/env python3
"""
Parse Betfair historical odds data using multiprocessing (optimized v2).

Major optimizations:
1. Simplified duplicate tracking using dict instead of list comprehension
2. Batch file writes instead of individual writes
3. Removed unnecessary tracking data structures
4. Process multiple months in parallel

Expected speedup: 10-20x depending on core count and I/O.

Usage:
    python parse_betfair_by_league_mp_v2.py --year 2019
    python parse_betfair_by_league_mp_v2.py --year 2019 --month 01
    python parse_betfair_by_league_mp_v2.py --all
"""

import os
import bz2
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
import argparse

# Suppress FutureWarning about fillna downcasting
warnings.filterwarnings('ignore', category=FutureWarning, message='.*Downcasting object dtype.*')

# ============================================================================
# BETFAIR TICK SIZES - Valid odds values
# ============================================================================

BETFAIR_TICKS = [
    1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1,
    1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2,
    1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.28, 1.29, 1.3,
    1.31, 1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.38, 1.39, 1.4,
    1.41, 1.42, 1.43, 1.44, 1.45, 1.46, 1.47, 1.48, 1.49, 1.5,
    1.51, 1.52, 1.53, 1.54, 1.55, 1.56, 1.57, 1.58, 1.59, 1.6,
    1.61, 1.62, 1.63, 1.64, 1.65, 1.66, 1.67, 1.68, 1.69, 1.7,
    1.71, 1.72, 1.73, 1.74, 1.75, 1.76, 1.77, 1.78, 1.79, 1.8,
    1.81, 1.82, 1.83, 1.84, 1.85, 1.86, 1.87, 1.88, 1.89, 1.9,
    1.91, 1.92, 1.93, 1.94, 1.95, 1.96, 1.97, 1.98, 1.99, 2.0,
    2.02, 2.04, 2.06, 2.08, 2.1, 2.12, 2.14, 2.16, 2.18, 2.2,
    2.22, 2.24, 2.26, 2.28, 2.3, 2.32, 2.34, 2.36, 2.38, 2.4,
    2.42, 2.44, 2.46, 2.48, 2.5, 2.52, 2.54, 2.56, 2.58, 2.6,
    2.62, 2.64, 2.66, 2.68, 2.7, 2.72, 2.74, 2.76, 2.78, 2.8,
    2.82, 2.84, 2.86, 2.88, 2.9, 2.92, 2.94, 2.96, 2.98, 3.0,
    3.05, 3.1, 3.15, 3.2, 3.25, 3.3, 3.35, 3.4, 3.45, 3.5,
    3.55, 3.6, 3.65, 3.7, 3.75, 3.8, 3.85, 3.9, 3.95, 4.0,
    4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0,
    5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0,
    6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0,
    8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0,
    10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0,
    15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0,
    21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
    32.0, 34.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0,
    55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0,
    110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0,
    210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0,
    310.0, 320.0, 330.0, 340.0, 350.0, 360.0, 370.0, 380.0, 390.0, 400.0,
    410.0, 420.0, 430.0, 440.0, 450.0, 460.0, 470.0, 480.0, 490.0, 500.0,
    510.0, 520.0, 530.0, 540.0, 550.0, 560.0, 570.0, 580.0, 590.0, 600.0,
    610.0, 620.0, 630.0, 640.0, 650.0, 660.0, 670.0, 680.0, 690.0, 700.0,
    710.0, 720.0, 730.0, 740.0, 750.0, 760.0, 770.0, 780.0, 790.0, 800.0,
    810.0, 820.0, 830.0, 840.0, 850.0, 860.0, 870.0, 880.0, 890.0, 900.0,
    910.0, 920.0, 930.0, 940.0, 950.0, 960.0, 970.0, 980.0, 990.0, 1000.0
]

# ============================================================================
# CONFIGURATION - Loaded once and passed to workers
# ============================================================================

BASE_PATH = Path(__file__).resolve().parents[2]  # repo root

def load_config():
    """Load all configuration data once."""
    config = {}

    # Load team rosters
    roster_file = BASE_PATH / "config/team_rosters_all_seasons.json"
    if roster_file.exists():
        with open(roster_file, "r", encoding="utf-8") as f:
            config['rosters'] = json.load(f)
    else:
        config['rosters'] = {}

    # Load team name mappings
    mapping_file = BASE_PATH / "config/betfair_team_mapping.json"
    if mapping_file.exists():
        with open(mapping_file, "r", encoding="utf-8") as f:
            config['mappings'] = json.load(f)
    else:
        config['mappings'] = {}

    # League configuration
    config['leagues'] = {
        'premier-league': {
            'display_name': 'Premier League'
        },
        'bundesliga': {
            'display_name': 'Bundesliga'
        },
        'la-liga': {
            'display_name': 'La Liga'
        },
        'serie-a': {
            'display_name': 'Serie A'
        },
        'ligue-1': {
            'display_name': 'Ligue 1'
        }
    }

    return config

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_season_from_date(match_date: datetime) -> str:
    """Determine season from match date (e.g., 2018-01-15 -> 17-18)."""
    year = match_date.year
    month = match_date.month

    if month >= 8:  # Aug-Dec = start of season
        return f"{str(year)[-2:]}-{str(year + 1)[-2:]}"
    else:  # Jan-May = end of season
        return f"{str(year - 1)[-2:]}-{str(year)[-2:]}"

def normalize_team_name(name: str, league: str, mappings: dict) -> str:
    """Normalize team name using config mappings."""
    name = name.strip()

    # Try league-specific mapping first
    if league and league in mappings:
        if name in mappings[league]:
            return mappings[league][name]

    # Try all league mappings
    for league_key in mappings:
        if isinstance(mappings[league_key], dict):
            if name in mappings[league_key]:
                return mappings[league_key][name]

    return name

def should_skip_match(event_name: str) -> Tuple[bool, str]:
    """Check if match should be skipped."""
    if not event_name:
        return True, "No event name"

    # Filter women's and youth matches
    if '(W)' in event_name or 'Women' in event_name:
        return True, "Women's match"

    if any(youth in event_name for youth in ['U19', 'U20', 'U21', 'U23', '(Res)', 'Reserve']):
        return True, "Youth/reserve match"

    # Check format
    if ' v ' not in event_name:
        return True, "Invalid format"

    return False, ""

def round_to_betfair_tick(value):
    """Round a value to the nearest valid Betfair tick."""
    if pd.isna(value):
        return value
    closest_tick = min(BETFAIR_TICKS, key=lambda x: abs(x - value))
    return closest_tick

def fill_missing_odds(df, team_columns):
    """Fill missing odds using smart interpolation, round to Betfair ticks."""
    df = df.copy()
    last_n = min(5, len(df))
    end_threshold = len(df) - last_n

    for col in team_columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue

        missing_idx = df[df[col].isnull()].index

        for idx in missing_idx:
            before_vals = df.loc[:idx-1, col].dropna()
            after_vals = df.loc[idx+1:, col].dropna()

            # Don't fill if at end with no future values
            if len(after_vals) == 0 and idx >= end_threshold:
                continue

            is_end_of_match = idx >= end_threshold

            if is_end_of_match and len(after_vals) > 0:
                recent_vals = df.loc[max(0, idx-5):, col].dropna()
                if len(recent_vals) > 0:
                    if recent_vals.min() <= 1.05:
                        other_cols = [c for c in team_columns if c != col]
                        other_vals = df.loc[max(0, idx-5):, other_cols].values.flatten()
                        other_vals = other_vals[~pd.isna(other_vals)]
                        if len(other_vals) > 0 and other_vals.min() > 50:
                            df.loc[idx, col] = recent_vals.iloc[-1]
                            continue
                    if recent_vals.max() >= 100:
                        df.loc[idx, col] = recent_vals.iloc[-1]
                        continue

            if len(before_vals) > 0 and len(after_vals) > 0:
                interpolated = (before_vals.iloc[-1] + after_vals.iloc[0]) / 2
                df.loc[idx, col] = round_to_betfair_tick(interpolated)
            elif len(before_vals) > 0 and idx < end_threshold:
                df.loc[idx, col] = before_vals.iloc[-1]
            elif len(after_vals) > 0:
                df.loc[idx, col] = after_vals.iloc[0]

    return df

def classify_match(event_name: str, season: str, config: dict) -> Optional[Tuple[str, str, str]]:
    """Classify match by league using season-specific rosters."""
    # Pre-filter
    should_skip, reason = should_skip_match(event_name)
    if should_skip:
        return None

    # Parse teams
    parts = event_name.split(' v ')
    if len(parts) != 2:
        return None

    home_team = normalize_team_name(parts[0], None, config['mappings'])
    away_team = normalize_team_name(parts[1], None, config['mappings'])

    # Check each league using season-specific rosters
    rosters = config['rosters']
    for league_key in config['leagues'].keys():
        if league_key in rosters and season in rosters[league_key]:
            roster_teams = rosters[league_key][season]
            normalized_roster = {normalize_team_name(t, league_key, config['mappings']): t for t in roster_teams}

            if home_team in normalized_roster and away_team in normalized_roster:
                return (league_key, normalized_roster[home_team], normalized_roster[away_team])

    return None

# ============================================================================
# FILE PROCESSING - Worker function
# ============================================================================

def process_single_file(file_path: str, config: dict, output_base: str) -> dict:
    """
    Process a single .bz2 file, write output if successful, and return results.
    This function is called by worker processes.

    Returns dict with:
        - status: 'match_odds', 'not_match_odds', 'skip_women_youth', 'unclassified', 'error'
        - match_key: tuple for deduplication (if successful)
        - num_rows: row count (if successful)
        - filepath: output filepath (if successful)
        - error: error message if error
    """
    result = {
        'status': None,
        'match_key': None,
        'num_rows': None,
        'filepath': None,
        'unclassified': None
    }

    try:
        # Quick check: read just first line to determine if Match Odds
        with bz2.open(file_path, 'rt', encoding='utf-8') as f:
            first_line = f.readline()
            if not first_line:
                result['status'] = 'not_match_odds'
                return result

            d = json.loads(first_line)
            if 'mc' not in d or not d['mc']:
                result['status'] = 'not_match_odds'
                return result

            mc = d['mc'][0]
            if 'marketDefinition' not in mc:
                result['status'] = 'not_match_odds'
                return result

            md = mc['marketDefinition']

            # Early exit if not Match Odds
            if md.get('marketType') != 'MATCH_ODDS':
                result['status'] = 'not_match_odds'
                return result

            event_name = md.get('eventName', '')
            market_time = md.get('marketTime', '')

            # Determine season
            match_datetime = datetime.fromisoformat(market_time.replace('Z', '+00:00'))
            match_season = get_season_from_date(match_datetime)

            # Classify match
            classification = classify_match(event_name, match_season, config)

            if not classification:
                # Check if it's women's/youth
                should_skip, reason = should_skip_match(event_name)
                if should_skip and ('Women' in reason or 'Youth' in reason):
                    result['status'] = 'skip_women_youth'
                else:
                    result['status'] = 'unclassified'
                    result['unclassified'] = {
                        'event_name': event_name,
                        'season': match_season,
                        'date': match_datetime.strftime('%Y-%m-%d')
                    }
                return result

            league_key, home_team, away_team = classification
            result['status'] = 'match_odds'

            # Now extract full odds data
            runners_map = {}
            for runner in md.get('runners', []):
                runners_map[runner.get('id')] = runner.get('name', '')

            odds_data = []
            current_in_play = False
            current_status = None

            # Reset and read all lines
            f.seek(0)
            for line in f:
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                    pt = record.get('pt')
                    if not pt:
                        continue

                    timestamp = datetime.fromtimestamp(pt / 1000)

                    for mc_record in record.get('mc', []):
                        if 'marketDefinition' in mc_record:
                            md_record = mc_record['marketDefinition']
                            current_status = md_record.get('status')
                            current_in_play = md_record.get('inPlay', current_in_play)

                        if 'rc' in mc_record:
                            odds_record = {
                                'timestamp': timestamp,
                                'market_status': current_status,
                                'in_play': current_in_play
                            }

                            for rc in mc_record['rc']:
                                runner_id = rc.get('id')
                                ltp = rc.get('ltp')
                                runner_name = runners_map.get(runner_id, f'Runner_{runner_id}')
                                odds_record[runner_name] = ltp

                            if len(odds_record) > 3:
                                odds_data.append(odds_record)

                except json.JSONDecodeError:
                    continue

            if len(odds_data) == 0:
                result['status'] = 'no_odds_data'
                return result

            # SOPHISTICATED DATA PROCESSING (from original script)
            # Convert to DataFrame
            df = pd.DataFrame(odds_data)

            # Group by minute (take last update per minute)
            df['minute'] = df['timestamp'].dt.floor('min')
            df = df.groupby('minute').last().reset_index()

            # Create complete minute-by-minute time series (only for match period)
            if len(df) > 0:
                # Find the actual match start time (first in_play=True)
                match_start_rows = df[df['in_play'] == True]

                if len(match_start_rows) > 0:
                    # Get the minute value of first in_play=True
                    match_start_minute = match_start_rows.iloc[0]['minute']

                    # Start time series from 1 minute before match start
                    series_start = match_start_minute - pd.Timedelta(minutes=1)

                    # FILTER: Keep only in_play=True rows (actual match data)
                    df = df[df['in_play'] == True].copy()

                    # Mark original rows as real data (before reindexing)
                    df['is_real_update'] = True

                    # Set minute as index AFTER marking
                    df = df.set_index('minute')

                    # Create complete range from series_start (1 min before match) to end of match
                    full_range = pd.date_range(
                        start=series_start,  # Start 1 minute before match
                        end=df.index.max(),  # End at last data point
                        freq='1min'
                    )

                    # Reindex to include all minutes
                    team_cols = [c for c in df.columns if c not in ['timestamp', 'market_status', 'in_play', 'is_real_update']]
                    df = df.reindex(full_range)
                    df.index.name = 'minute'

                    # Mark filled rows (rows that didn't exist before reindex)
                    df['is_real_update'] = df['is_real_update'].fillna(False).infer_objects(copy=False)

                    # Forward fill odds (carry forward last known odds)
                    for col in team_cols:
                        df[col] = df[col].ffill()

                    # Forward fill in_play status (set True for all rows since we're only keeping match data)
                    df['in_play'] = df['in_play'].fillna(True).infer_objects(copy=False).ffill()

                    # Reset index
                    df = df.reset_index()

                    # Add match_event column to mark start and end
                    df['match_event'] = ''
                    df.loc[1, 'match_event'] = 'start'  # Row 1 is first in-play minute (row 0 is pre-match)
                    df.loc[len(df)-1, 'match_event'] = 'end'  # Last row
                else:
                    # No in_play data, skip this match
                    result['status'] = 'no_odds_data'
                    return result

            # Fill any remaining missing odds using sophisticated interpolation
            team_cols = [c for c in df.columns if c not in ['minute', 'timestamp', 'market_status', 'in_play', 'match_event', 'is_real_update']]
            df = fill_missing_odds(df, team_cols)

            # WRITE FILE IMMEDIATELY (in parallel, within worker)
            match_date = match_datetime.strftime('%Y-%m-%d')

            # Create output directory
            output_path = Path(output_base)
            league_dir = output_path / league_key / match_season
            league_dir.mkdir(parents=True, exist_ok=True)

            # Create filename
            filename = f"{match_date} {home_team}-{away_team}.csv"
            filepath = league_dir / filename

            # Write CSV (happens in parallel!)
            df.to_csv(filepath, index=False)

            # Return metadata only (not DataFrame - can't pickle efficiently)
            match_key = (league_key, match_season, match_date, home_team, away_team)
            result['match_key'] = match_key
            result['num_rows'] = len(df)
            result['filepath'] = str(filepath)

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)

    return result

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_directory_parallel(input_dir: str, output_base: str = "OUTPUT_LEAGUES", num_workers: int = None):
    """Process directory using multiprocessing."""

    input_path = Path(input_dir)
    output_path = Path(output_base)
    output_path.mkdir(exist_ok=True)

    if not input_path.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        return

    print("=" * 80)
    print("BETFAIR DATA PARSER - MULTIPROCESSING V2 (OPTIMIZED)")
    print("=" * 80)
    print(f"\nInput:  {input_dir}")
    print(f"Output: {output_base}/")
    print(f"Workers: {num_workers or cpu_count()} parallel processes")
    print()

    # Load configuration once
    print("Loading configuration...")
    config = load_config()
    print(f"Loaded rosters for {len(config['rosters'])} leagues")
    print(f"Loaded mappings for {len(config['mappings'])} leagues")
    print()

    # Find all .bz2 files
    bz2_files = list(input_path.glob('*.bz2'))
    print(f"Found {len(bz2_files)} .bz2 files to process\n")

    if len(bz2_files) == 0:
        print("No .bz2 files found. Exiting.")
        return

    # Process files in parallel
    stats = {
        'total_files': 0,
        'match_odds_markets': 0,
        'skipped_not_match_odds': 0,
        'skipped_women_youth': 0,
        'skipped_unclassified': 0,
        'processed_by_league': {league: 0 for league in config['leagues'].keys()},
        'errors': 0
    }

    # Track unique matches and handle duplicates
    match_tracker = {}  # (league, season, date, home, away) -> (num_rows, filepath)
    unclassified_matches = []
    processed_matches = []

    # Create pool with progress tracking
    print(f"Processing {len(bz2_files)} files with {num_workers or cpu_count()} workers...")
    print("(Files are being written in parallel as they're processed)")
    print()

    worker_func = partial(process_single_file, config=config, output_base=output_base)

    if num_workers is None:
        num_workers = cpu_count()

    with Pool(processes=num_workers) as pool:
        # Process in chunks for better progress reporting
        chunk_size = 500
        for i in range(0, len(bz2_files), chunk_size):
            chunk = bz2_files[i:i+chunk_size]
            results = pool.map(worker_func, [str(f) for f in chunk])

            # Process results
            for result in results:
                stats['total_files'] += 1

                if result['status'] == 'match_odds':
                    stats['match_odds_markets'] += 1
                    match_key = result['match_key']
                    num_rows = result['num_rows']
                    filepath = Path(result['filepath'])

                    # Check for duplicate - keep file with more rows
                    is_new_match = match_key not in match_tracker

                    if not is_new_match:
                        prev_rows, prev_filepath = match_tracker[match_key]
                        if num_rows <= prev_rows:
                            # New file has fewer rows, delete it
                            if filepath.exists():
                                filepath.unlink()
                            continue
                        else:
                            # New file has more rows, delete old one
                            if prev_filepath.exists():
                                prev_filepath.unlink()

                    # Track this match
                    match_tracker[match_key] = (num_rows, filepath)

                    # Only update stats and summary for NEW matches (not replacements)
                    if is_new_match:
                        league_key, season, match_date, home_team, away_team = match_key
                        stats['processed_by_league'][league_key] += 1

                        processed_matches.append({
                            'league': config['leagues'][league_key]['display_name'],
                            'season': season,
                            'date': match_date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'filename': filepath.name
                        })

                elif result['status'] == 'not_match_odds':
                    stats['skipped_not_match_odds'] += 1
                elif result['status'] == 'skip_women_youth':
                    stats['skipped_women_youth'] += 1
                elif result['status'] == 'unclassified':
                    stats['skipped_unclassified'] += 1
                    if result['unclassified']:
                        unclassified_matches.append(result['unclassified'])
                elif result['status'] == 'error':
                    stats['errors'] += 1

            # Progress update
            print(f"Processed {min(i + chunk_size, len(bz2_files))}/{len(bz2_files)} files... ({len(match_tracker)} unique matches written)", end='\r')

    print()  # Clear progress line
    print()

    # Save unclassified matches report
    if unclassified_matches:
        unclassified_path = output_path / 'unclassified_matches.csv'
        unclassified_df = pd.DataFrame(unclassified_matches)
        unclassified_df.to_csv(unclassified_path, index=False)
        print(f"[INFO] Saved {len(unclassified_matches)} unclassified matches to: {unclassified_path}")
        print()

    # Print summary
    print("=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)

    print(f"\nFiles Processed: {stats['total_files']}")
    print(f"  Match Odds markets: {stats['match_odds_markets']}")
    print(f"  Skipped (not Match Odds): {stats['skipped_not_match_odds']}")
    print(f"  Skipped (women's/youth): {stats['skipped_women_youth']}")
    print(f"  Skipped (unclassified): {stats['skipped_unclassified']}")
    print(f"  Errors: {stats['errors']}")

    print(f"\nMatches Processed by League:")
    for league_key, count in stats['processed_by_league'].items():
        print(f"  {config['leagues'][league_key]['display_name']}: {count} matches")

    total_matches = sum(stats['processed_by_league'].values())
    print(f"\nTotal Matches Saved: {total_matches}")

    print(f"\nOutput saved to: {output_base}/")

    # Save processing summary
    if processed_matches:
        summary_df = pd.DataFrame(processed_matches)
        summary_path = output_path / 'processing_summary.csv'
        summary_df.to_csv(summary_path, index=False)

    print("=" * 80)

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Parse Betfair data in parallel (optimized v2)')
    parser.add_argument('--year', type=str, help='Year to process (e.g., 2019)')
    parser.add_argument('--month', type=str, help='Month to process (01-12)')
    parser.add_argument('--all', action='store_true', help='Process all available data')
    parser.add_argument('--workers', type=int, help='Number of worker processes (default: CPU count)')

    args = parser.parse_args()

    output_base = str(BASE_PATH / "OUTPUT_LEAGUES")

    if args.all:
        # Process all BASIC directories
        basic_path = BASE_PATH / "BASIC"
        if not basic_path.exists():
            print("ERROR: BASIC directory not found")
            return 1

        for year_dir in sorted(basic_path.iterdir()):
            if not year_dir.is_dir():
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                print(f"\nProcessing {year_dir.name}/{month_dir.name}...")
                process_directory_parallel(str(month_dir), output_base, args.workers)

    elif args.year:
        if args.month:
            # Process specific month
            input_dir = str(BASE_PATH / f"BASIC/{args.year}/{args.month}")
            if not Path(input_dir).exists():
                print(f"ERROR: Directory not found: {input_dir}")
                return 1
            process_directory_parallel(input_dir, output_base, args.workers)
        else:
            # Process all months in year
            year_path = BASE_PATH / f"BASIC/{args.year}"
            if not year_path.exists():
                print(f"ERROR: Year directory not found: {year_path}")
                return 1

            # Find all month directories
            month_dirs = sorted([d for d in year_path.iterdir() if d.is_dir()])
            if not month_dirs:
                print(f"ERROR: No month directories found in {year_path}")
                return 1

            print(f"Found {len(month_dirs)} directories to process:")
            for month_dir in month_dirs:
                num_files = len(list(month_dir.glob('*.bz2')))
                print(f"  {args.year}/{month_dir.name}: {num_files:,} files")
            print()

            for i, month_dir in enumerate(month_dirs, 1):
                print("=" * 80)
                print(f"[{i}/{len(month_dirs)}] Processing {args.year}/{month_dir.name}")
                print("=" * 80)
                print()
                process_directory_parallel(str(month_dir), output_base, args.workers)
                print()

    else:
        # Default: process BASIC/2018/01
        input_dir = str(BASE_PATH / "BASIC/2018/01")
        if not Path(input_dir).exists():
            print(f"ERROR: Default directory not found: {input_dir}")
            print("Make sure BASIC/2018/01 exists with downloaded .bz2 files")
            print("\nTo process all data, use: python parse_betfair_by_league_mp_v2.py --all")
            return 1
        process_directory_parallel(input_dir, output_base, args.workers)

    return 0

if __name__ == '__main__':
    exit(main())
