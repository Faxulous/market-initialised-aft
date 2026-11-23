"""
Update match alignment table between WhoScored and FBref.
Simplified version that works with the new per-league-season structure.

Usage:
    python update_match_alignment_v2.py --league premier-league --season 24-25
    python update_match_alignment_v2.py --all  # Process all leagues/seasons
"""

import pandas as pd
import json
from pathlib import Path
import sys
import argparse

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

BASE_PATH = Path(__file__).resolve().parents[2]  # repo root

def load_team_mapping():
    """Load team name mapping from JSON config."""
    config_path = BASE_PATH / 'config' / 'team_name_mapping.json'

    if not config_path.exists():
        print(f"[ERROR] Team mapping file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)

    return mapping_data['premier_league']

def normalize_team_name(team_name, mapping):
    """Normalize team name using mapping dictionary."""
    if team_name in mapping:
        return mapping[team_name]

    for ws_name, fb_name in mapping.items():
        if ws_name.lower() == team_name.lower():
            return fb_name

    return None  # Unmapped

def process_league_season(league_dir, season, team_mapping):
    """Create alignment table for one league-season."""
    print(f"\n{'='*80}")
    print(f"{league_dir.upper()} {season}")
    print(f"{'='*80}")

    # Note: season is already in YY-YY format (e.g., "24-25")
    # No conversion needed for WhoScored paths

    # Load WhoScored schedule
    ws_schedule_path = BASE_PATH / 'data' / 'whoscored' / league_dir / season / 'schedule.parquet'
    if not ws_schedule_path.exists():
        print(f"[SKIP] No WhoScored data found at: {ws_schedule_path}")
        return

    ws_schedule = pd.read_parquet(ws_schedule_path)
    print(f"[OK] Loaded {len(ws_schedule)} matches from WhoScored")

    # Load FBref match IDs (from download_fbref_data.py outputs, format: "YYYY-MM-DD Home-Away")
    fbref_data_path = BASE_PATH / 'data' / league_dir / season
    if not fbref_data_path.exists():
        print(f"[SKIP] No FBref data found at: {fbref_data_path}")
        return

    fbref_match_ids = set()
    for parquet_file in ['fbref_shots.parquet', 'fbref_schedule.parquet']:
        file_path = fbref_data_path / parquet_file
        if file_path.exists():
            df = pd.read_parquet(file_path)
            id_col = 'match_id' if 'match_id' in df.columns else 'game'
            fbref_match_ids.update(df[id_col].unique())

    print(f"[OK] Found {len(fbref_match_ids)} unique FBref matches")

    # Create alignment
    alignment_records = []
    unmapped_teams = set()
    matched_count = 0

    # soccerdata>=1.9 WhoScored schedules use long names where the event feed uses short ones;
    # downstream code matches events.team_name == alignment.ws_home_team, so use event names
    WS_SCHEDULE_TO_EVENT = {'Manchester United': 'Man Utd', 'Manchester City': 'Man City'}
    for idx, row in ws_schedule.iterrows():
        ws_game_id = row['game_id']
        ws_date = row['date'].strftime('%Y-%m-%d')
        ws_home = WS_SCHEDULE_TO_EVENT.get(row['home_team'], row['home_team'])
        ws_away = WS_SCHEDULE_TO_EVENT.get(row['away_team'], row['away_team'])

        # Normalize team names
        fb_home = normalize_team_name(ws_home, team_mapping)
        fb_away = normalize_team_name(ws_away, team_mapping)

        if fb_home is None:
            unmapped_teams.add(ws_home)
            fb_home = ws_home
        if fb_away is None:
            unmapped_teams.add(ws_away)
            fb_away = ws_away

        # Construct FBref match_id
        fbref_match_id = f"{ws_date} {fb_home}-{fb_away}"
        exists_in_fbref = fbref_match_id in fbref_match_ids

        if exists_in_fbref:
            matched_count += 1

        # Column names/order must match earlier seasons' match_alignment.parquet exactly
        alignment_records.append({
            'ws_game_id': ws_game_id,
            'fbref_match_id': fbref_match_id,
            'understat_game_id': None,       # understat not collected from 25-26 onwards
            'date': ws_date,
            'ws_home_team': ws_home,
            'ws_away_team': ws_away,
            'fbref_home_team': fb_home,
            'fbref_away_team': fb_away,
            'fbref_matched': exists_in_fbref,
            'understat_matched': False,
            'home_score': row['home_score'],
            'away_score': row['away_score']
        })

    alignment_df = pd.DataFrame(alignment_records)
    alignment_df['understat_game_id'] = alignment_df['understat_game_id'].astype('Int64')  # nullable int, matches old int64 key

    # Report unmapped teams
    if unmapped_teams:
        print(f"\n[WARNING] {len(unmapped_teams)} unmapped teams: {sorted(unmapped_teams)}")

    # Save alignment table to new unified structure
    output_dir = BASE_PATH / 'data' / league_dir / season
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / 'match_alignment.csv'
    parquet_path = output_dir / 'match_alignment.parquet'

    alignment_df.to_csv(csv_path, index=False)
    alignment_df.to_parquet(parquet_path, index=False)

    print(f"\n[OK] Saved alignment: {parquet_path}")
    print(f"     Matched: {matched_count}/{len(alignment_df)}")

def parse_args():
    parser = argparse.ArgumentParser(description='Create match alignment between WhoScored and FBref')
    parser.add_argument('--league', choices=list(LEAGUE_MAPPING.keys()),
                       help='League name')
    parser.add_argument('--season', help='Season (YY-YY format)')
    parser.add_argument('--all', action='store_true',
                       help='Process all leagues and seasons')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load team mapping
    team_mapping = load_team_mapping()

    if args.all:
        # Process all leagues and seasons
        leagues = list(LEAGUE_MAPPING.keys())
        seasons = ['17-18', '18-19', '19-20', '20-21', '21-22', '22-23', '23-24', '24-25']

        print(f"Processing {len(leagues)} leagues × {len(seasons)} seasons = {len(leagues)*len(seasons)} combinations")

        for league in leagues:
            for season in seasons:
                try:
                    process_league_season(league, season, team_mapping)
                except Exception as e:
                    print(f"[ERROR] Failed {league} {season}: {e}")
                    continue

    elif args.league and args.season:
        # Process single league-season
        process_league_season(args.league, args.season, team_mapping)

    else:
        print("ERROR: Must specify either --league and --season, or --all")
        sys.exit(1)

    print(f"\n{'='*80}")
    print("DONE!")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
