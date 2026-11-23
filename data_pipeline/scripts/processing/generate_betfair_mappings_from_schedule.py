"""
Generate Betfair team name mappings by matching unclassified matches against WhoScored schedule data.

This script:
1. Loads WhoScored schedule.parquet files for all leagues
2. Reads unclassified Betfair matches
3. Matches them based on date (exact match) and fuzzy team name matching
4. Generates team name mappings for Betfair -> FBRef
"""

import pandas as pd
import json
from pathlib import Path
from difflib import SequenceMatcher
from datetime import datetime

# League mappings
LEAGUES = {
    'premier-league': 'Premier League',
    'bundesliga': 'Bundesliga',
    'la-liga': 'La Liga',
    'serie-a': 'Serie A',
    'ligue-1': 'Ligue 1'
}

def normalize_team_name(name):
    """Basic normalization - remove common variations."""
    name = name.strip()
    # Remove common suffixes
    for suffix in [' FC', ' CF', ' SC', ' United', ' City', ' Town']:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    return name

def fuzzy_match_score(name1, name2):
    """Calculate fuzzy match score between two team names."""
    # Normalize first
    n1 = normalize_team_name(name1).lower()
    n2 = normalize_team_name(name2).lower()

    return SequenceMatcher(None, n1, n2).ratio()

def load_whoscored_schedules():
    """Load all WhoScored schedule.parquet files."""
    schedules = {}

    for league_key in LEAGUES.keys():
        league_schedules = []

        # Find all schedule.parquet files for this league
        league_dir = Path(f"data/whoscored/{league_key}")
        if not league_dir.exists():
            print(f"Warning: {league_dir} does not exist, skipping {LEAGUES[league_key]}")
            continue

        for season_dir in sorted(league_dir.glob("*")):
            if not season_dir.is_dir():
                continue

            schedule_file = season_dir / "schedule.parquet"
            if schedule_file.exists():
                df = pd.read_parquet(schedule_file)
                # Add season column
                df['season'] = season_dir.name
                league_schedules.append(df)
                print(f"Loaded {len(df)} matches from {league_key}/{season_dir.name}")

        if league_schedules:
            schedules[league_key] = pd.concat(league_schedules, ignore_index=True)
            print(f"Total {LEAGUES[league_key]}: {len(schedules[league_key])} matches")

    return schedules

def parse_betfair_match_name(event_name):
    """Parse Betfair event name into home and away teams."""
    if ' v ' in event_name:
        parts = event_name.split(' v ')
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    return None, None

def match_betfair_to_schedule(betfair_match, schedules):
    """
    Try to match a Betfair match to a WhoScored schedule entry.

    Returns: (league_key, home_team, away_team, confidence) or None
    """
    event_name = betfair_match['event_name']
    match_date_str = betfair_match['date']  # YYYY-MM-DD format
    season = betfair_match['season']

    # Parse Betfair team names
    betfair_home, betfair_away = parse_betfair_match_name(event_name)
    if not betfair_home or not betfair_away:
        return None

    best_match = None
    best_score = 0.0

    # Try each league
    for league_key, schedule_df in schedules.items():
        # Convert season format: '18-19' -> '1819'
        season_normalized = season.replace('-', '')

        # Filter by season and date
        schedule_df['date_only'] = pd.to_datetime(schedule_df['date']).dt.date
        match_date = datetime.strptime(match_date_str, '%Y-%m-%d').date()

        matches_on_date = schedule_df[
            (schedule_df['season'] == season_normalized) &
            (schedule_df['date_only'] == match_date)
        ]

        if len(matches_on_date) == 0:
            continue

        # For each potential match, calculate similarity score
        for _, row in matches_on_date.iterrows():
            ws_home = row['home_team']
            ws_away = row['away_team']

            # Calculate match scores
            home_score = fuzzy_match_score(betfair_home, ws_home)
            away_score = fuzzy_match_score(betfair_away, ws_away)

            # Average score
            avg_score = (home_score + away_score) / 2.0

            # Must have decent match on both teams
            if home_score >= 0.6 and away_score >= 0.6 and avg_score > best_score:
                best_score = avg_score
                best_match = {
                    'league_key': league_key,
                    'home': ws_home,
                    'away': ws_away,
                    'betfair_home': betfair_home,
                    'betfair_away': betfair_away,
                    'confidence': avg_score
                }

    return best_match

def generate_mappings():
    """Generate Betfair team name mappings from unclassified matches."""

    print("=" * 80)
    print("GENERATING BETFAIR TEAM NAME MAPPINGS FROM SCHEDULE DATA")
    print("=" * 80)
    print()

    # Load WhoScored schedules
    print("Loading WhoScored schedule data...")
    schedules = load_whoscored_schedules()
    print()

    if not schedules:
        print("ERROR: No schedule data found!")
        return

    # Load unclassified matches
    unclassified_file = Path("OUTPUT_LEAGUES/unclassified_matches.csv")
    if not unclassified_file.exists():
        print(f"ERROR: {unclassified_file} not found!")
        print("Run the parser first to generate unclassified matches.")
        return

    print(f"Loading unclassified Betfair matches from {unclassified_file}...")
    unclassified_df = pd.read_csv(unclassified_file)
    print(f"Found {len(unclassified_df)} unclassified matches")
    print()

    # Match each unclassified match
    mappings_by_league = {league: {} for league in LEAGUES.keys()}
    matched_count = 0
    high_confidence_count = 0

    print("Matching Betfair matches to WhoScored schedule...")
    for idx, row in unclassified_df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(unclassified_df)} matches...", end='\r')

        match_result = match_betfair_to_schedule(row, schedules)

        if match_result:
            matched_count += 1
            league_key = match_result['league_key']

            # Add mappings for both teams
            mappings_by_league[league_key][match_result['betfair_home']] = match_result['home']
            mappings_by_league[league_key][match_result['betfair_away']] = match_result['away']

            if match_result['confidence'] >= 0.85:
                high_confidence_count += 1

    print()
    print(f"Matched {matched_count}/{len(unclassified_df)} matches ({high_confidence_count} high confidence)")
    print()

    # Load existing Betfair mappings
    config_file = Path("config/betfair_team_mapping.json")
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            existing_mappings = json.load(f)
    else:
        existing_mappings = {}

    # Merge new mappings with existing ones
    print("Merging with existing mappings...")
    for league_key, new_mappings in mappings_by_league.items():
        if league_key not in existing_mappings:
            existing_mappings[league_key] = {}

        new_count = 0
        for betfair_name, fbref_name in new_mappings.items():
            if betfair_name not in existing_mappings[league_key]:
                existing_mappings[league_key][betfair_name] = fbref_name
                new_count += 1

        if new_count > 0:
            print(f"  {LEAGUES[league_key]}: Added {new_count} new mappings")

    # Save updated mappings
    print()
    print(f"Saving updated mappings to {config_file}...")

    # Ensure description and format fields are present
    if 'description' not in existing_mappings:
        existing_mappings['description'] = "Team name normalization mapping from Betfair to FBRef"
    if 'format' not in existing_mappings:
        existing_mappings['format'] = "betfair_name -> fbref_name (based on FBRef 'game' column format)"

    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(existing_mappings, f, indent=2, ensure_ascii=False, sort_keys=True)

    print("Done!")
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for league_key in LEAGUES.keys():
        count = len(existing_mappings.get(league_key, {}))
        print(f"{LEAGUES[league_key]:20s}: {count} total mappings")
    print("=" * 80)

if __name__ == '__main__':
    generate_mappings()
