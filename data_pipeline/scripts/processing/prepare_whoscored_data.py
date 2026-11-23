"""
Prepare WhoScored events for analysis.

This script processes WhoScored event data to:
1. Extract nested qualifiers into flat boolean columns
2. Add match linking keys (fbref_match_id, ws_match_id)
3. Structure data for easy querying
4. Include ALL events (shots, passes, tackles, fouls, etc.)

Output: data/{league}/{season}/whoscored_events.parquet
        - ALL WhoScored events including shots
        - Flat, queryable structure
        - Linked to FBRef data via fbref_match_id
        - NO betting odds (kept separate in betfair_odds.parquet)

Why include shots?
        - WhoScored has second-level precision timing
        - Additional qualifiers and context
        - Can cross-reference with FBRef xG data

Usage:
    python prepare_whoscored_data.py --league premier-league --season 24-25
    python prepare_whoscored_data.py --all  # Process all leagues/seasons
"""

import pandas as pd
import json
from pathlib import Path
import sys
import argparse
from typing import Dict
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

def load_whoscored_fbref_alignment(league: str, season: str) -> pd.DataFrame:
    """Load existing WhoScored-FBRef alignment table."""
    # New unified structure
    alignment_path = DATA_PATH / league / season / 'match_alignment.parquet'

    # Fallback to old location for backwards compatibility
    if not alignment_path.exists():
        alignment_path = DATA_PATH / 'alignment' / league / season / 'match_alignment.parquet'

    if not alignment_path.exists():
        print(f"[WARNING] No alignment table found at {alignment_path}")
        print(f"[INFO] Run 'python link_matches.py --league {league} --season {season}' first")
        return None

    return pd.read_parquet(alignment_path)

def extract_qualifiers_to_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Extract nested qualifiers array into flat boolean/string columns."""
    if 'qualifiers' not in df.columns or len(df) == 0:
        return df

    df = df.copy()

    # Initialize new columns
    df['is_big_chance'] = False
    df['is_assisted'] = False
    df['is_intentional_assist'] = False
    df['is_from_corner'] = False
    df['is_first_touch'] = False
    df['is_volley'] = False
    df['is_own_goal'] = False
    df['is_penalty'] = False
    df['is_fast_break'] = False
    df['is_through_ball'] = False
    df['is_cross'] = False
    df['is_long_ball'] = False
    df['is_chipped'] = False
    df['is_headed_pass'] = False
    df['is_high_press'] = False
    df['is_counter_attack'] = False

    df['pass_body_part'] = None
    df['pass_height'] = None
    df['pass_technique'] = None

    # Qualifier type mappings
    BOOLEAN_QUALIFIERS = {
        'BigChance': 'is_big_chance',
        'Assisted': 'is_assisted',
        'IntentionalAssist': 'is_intentional_assist',
        'FromCorner': 'is_from_corner',
        'FirstTouch': 'is_first_touch',
        'Volley': 'is_volley',
        'OwnGoal': 'is_own_goal',
        'Penalty': 'is_penalty',
        'FastBreak': 'is_fast_break',
        'ThroughBall': 'is_through_ball',
        'Cross': 'is_cross',
        'LongBall': 'is_long_ball',
        'Chipped': 'is_chipped',
        'HeadPass': 'is_headed_pass',
        'HighPress': 'is_high_press',
        'CounterAttack': 'is_counter_attack'
    }

    BODY_PARTS = {'LeftFoot', 'RightFoot', 'Head', 'OtherBodyPart'}

    # Process each row
    for idx, row in df.iterrows():
        qualifiers = row['qualifiers']

        if qualifiers is None or (isinstance(qualifiers, float) and pd.isna(qualifiers)):
            continue

        if isinstance(qualifiers, np.ndarray):
            qualifiers = qualifiers.tolist()

        if not isinstance(qualifiers, list):
            continue

        for qual in qualifiers:
            if not isinstance(qual, dict) or 'type' not in qual:
                continue

            qual_type = qual['type']
            if isinstance(qual_type, dict) and 'displayName' in qual_type:
                qual_name = qual_type['displayName']
            else:
                continue

            # Boolean qualifiers
            if qual_name in BOOLEAN_QUALIFIERS:
                df.at[idx, BOOLEAN_QUALIFIERS[qual_name]] = True

            # Body part
            elif qual_name in BODY_PARTS:
                df.at[idx, 'pass_body_part'] = qual_name

    return df

def restructure_events(df: pd.DataFrame) -> pd.DataFrame:
    """Restructure events dataframe to be more intuitive."""
    if len(df) == 0:
        return df

    df = df.copy()

    # Extract qualifiers first
    df = extract_qualifiers_to_columns(df)

    # Rename columns for clarity
    rename_map = {
        'game_id': 'ws_match_id',
        'type': 'event_type',
        'outcome_type': 'event_outcome',
        'x': 'start_x',
        'y': 'start_y',
        'player': 'player_name',
        'team': 'team_name',
    }

    df = df.rename(columns=rename_map)

    # Drop qualifiers (already extracted)
    if 'qualifiers' in df.columns:
        df = df.drop(columns=['qualifiers'])

    # Organize columns
    column_order = [
        # Match identifiers
        'fbref_match_id',
        'ws_match_id',

        # Timing
        'match_minute',
        'expanded_minute',
        'minute',
        'second',
        'period',

        # Event type
        'event_type',
        'event_outcome',

        # Team and player
        'team_id',
        'team_name',
        'player_id',
        'player_name',

        # Location
        'start_x',
        'start_y',
        'end_x',
        'end_y',

        # Event flags
        'is_touch',
        'is_shot',
        'is_goal',

        # Pass details
        'is_cross',
        'is_through_ball',
        'is_long_ball',
        'is_chipped',
        'is_headed_pass',
        'pass_body_part',
        'pass_height',
        'pass_technique',

        # Context flags
        'is_from_corner',
        'is_first_touch',
        'is_penalty',
        'is_own_goal',
        'is_fast_break',
        'is_counter_attack',
        'is_high_press',
        'is_assisted',
        'is_intentional_assist',
        'is_volley',
        'is_big_chance',

        # Cards
        'card_type',

        # Relations
        'related_event_id',
        'related_player_id',
    ]

    # Add any remaining columns
    remaining_cols = [col for col in df.columns if col not in column_order]
    column_order.extend(remaining_cols)

    # Reorder columns
    final_columns = [col for col in column_order if col in df.columns]
    df = df[final_columns]

    return df

def process_single_match(fbref_match_id: str, ws_game_id: int, league: str, season: str) -> pd.DataFrame:
    """Process a single match: load ALL WhoScored events (including shots)."""

    # Load WhoScored events
    ws_events_path = DATA_PATH / 'whoscored' / league / season / 'events_by_match' / f'match_{ws_game_id}_events.parquet'
    if not ws_events_path.exists():
        print(f"      [SKIP] No WhoScored events found for game_id {ws_game_id}")
        return pd.DataFrame()

    ws_events = pd.read_parquet(ws_events_path)

    # Keep ALL events including shots (shots have second-level precision and qualifiers)
    # FBRef shots.parquet has xG data; WhoScored has timing/context - both useful!

    # Add match identifiers
    ws_events['fbref_match_id'] = fbref_match_id
    ws_events['ws_game_id'] = ws_game_id

    # Calculate match minute
    if 'expanded_minute' in ws_events.columns:
        ws_events['match_minute'] = ws_events['expanded_minute'].fillna(0).astype(int)
    else:
        ws_events['match_minute'] = 0

    # Drop ws_game_id (redundant with ws_match_id from game_id)
    if 'ws_game_id' in ws_events.columns:
        ws_events = ws_events.drop(columns=['ws_game_id'])

    return ws_events

def process_league_season(league: str, season: str):
    """Process all matches for a league-season combination."""
    print(f"\n{'='*80}")
    print(f"{league.upper()} {season}")
    print(f"{'='*80}")

    # Load alignment table
    print("\n[STEP 1] Loading alignment tables...")
    ws_fb_alignment = load_whoscored_fbref_alignment(league, season)
    if ws_fb_alignment is None or len(ws_fb_alignment) == 0:
        print("[ERROR] No alignment data found. Run link_matches.py first.")
        return

    print(f"[OK] Loaded {len(ws_fb_alignment)} match alignments")

    # Process each match
    print(f"\n[STEP 2] Processing {len(ws_fb_alignment)} matches...")

    all_events = []
    processed_count = 0
    error_count = 0

    for idx, row in ws_fb_alignment.iterrows():
        try:
            print(f"\n   [{idx+1}/{len(ws_fb_alignment)}] {row['fbref_match_id']}")

            events = process_single_match(
                fbref_match_id=row['fbref_match_id'],
                ws_game_id=row['ws_game_id'],
                league=league,
                season=season
            )

            if len(events) > 0:
                all_events.append(events)
                processed_count += 1
                print(f"      [OK] {len(events)} events")

        except Exception as e:
            print(f"      [ERROR] Failed: {e}")
            error_count += 1
            continue

    # Combine all events
    if len(all_events) > 0:
        print(f"\n[STEP 3] Combining all events...")
        combined = pd.concat(all_events, ignore_index=True)

        # Restructure to flat format
        print(f"\n[STEP 4] Restructuring to flat format...")
        combined = restructure_events(combined)

        # Save to new structure: data/{league}/{season}/whoscored_events.parquet
        output_dir = DATA_PATH / league / season
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / 'whoscored_events.parquet'
        combined.to_parquet(output_path, index=False)

        # Print summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Matches processed: {processed_count}/{len(ws_fb_alignment)}")
        print(f"Total events: {len(combined):,}")
        print(f"  Shot events: {(combined['event_type'] == 'Shot').sum():,}")
        print(f"  Pass events: {(combined['event_type'] == 'Pass').sum():,}")
        print(f"  Tackle events: {(combined['event_type'] == 'Tackle').sum():,}")
        print(f"\nOutput: {output_path}")
        print(f"Size: {combined.shape}")
        print(f"Columns: {len(combined.columns)}")

        # Show column list
        print(f"\nKey columns for GNN:")
        print(f"  Match linking: fbref_match_id, ws_match_id")
        print(f"  Timing: match_minute, minute, second")
        print(f"  Event: event_type, event_outcome")
        print(f"  Players: player_id, player_name, team_id, team_name")
        print(f"  Location: start_x, start_y, end_x, end_y")

    else:
        print("[WARNING] No events were successfully processed")

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare WhoScored events (including shots) for analysis')
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
    print("DONE! All WhoScored events (including shots) saved to:")
    print("  data/{league}/{season}/whoscored_events.parquet")
    print("\nCross-reference with:")
    print("  data/{league}/{season}/fbref_shots.parquet (xG data)")
    print("  data/{league}/{season}/betfair_odds.parquet (betting odds)")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
