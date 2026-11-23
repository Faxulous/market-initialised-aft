"""
Download WhoScored data for a single league-season combination.
Designed to be called by batch script for multiple leagues/seasons.

Usage:
    python download_whoscored_league_season.py --league premier-league --season 24-25
    python download_whoscored_league_season.py --league bundesliga --season 23-24
"""

import soccerdata as sd
import pandas as pd
from pathlib import Path
import sys
import time
import argparse

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# League name mapping (soccerdata format -> directory format)
LEAGUE_MAPPING = {
    'premier-league': 'ENG-Premier League',
    'bundesliga': 'GER-Bundesliga',
    'la-liga': 'ESP-La Liga',
    'ligue-1': 'FRA-Ligue 1',
    'serie-a': 'ITA-Serie A'
}

def parse_args():
    parser = argparse.ArgumentParser(description='Download WhoScored data for one league-season')
    parser.add_argument('--league', required=True,
                       choices=list(LEAGUE_MAPPING.keys()),
                       help='League name (e.g., premier-league, bundesliga)')
    parser.add_argument('--season', required=True,
                       help='Season in format YY-YY (e.g., 24-25, 23-24)')
    parser.add_argument('--headless', action='store_true',
                       help='Run Chrome in headless mode (may not work with newer Selenium)')
    return parser.parse_args()

def main():
    args = parse_args()

    league_dir = args.league
    league_ws = LEAGUE_MAPPING[args.league]
    season = args.season
    season_ws = '20' + season  # Convert 24-25 -> 2024-25

    print("=" * 80)
    print(f"WhoScored Download: {league_ws} {season}")
    print("=" * 80)

    # Setup output directory (repo data/whoscored/, where prepare_whoscored_data.py reads from)
    output_dir = Path(__file__).resolve().parents[2] / 'data' / 'whoscored' / league_dir / season
    output_dir.mkdir(parents=True, exist_ok=True)
    events_dir = output_dir / 'events_by_match'
    events_dir.mkdir(exist_ok=True)

    # Initialize WhoScored
    print(f"\n1. Initializing WhoScored for {league_ws} {season}...")
    try:
        ws = sd.WhoScored(
            leagues=league_ws,
            seasons=season_ws,
            headless=args.headless,
            no_cache=False,
            no_store=False
        )
        print("   [OK] WhoScored initialized successfully")
    except Exception as e:
        print(f"   [ERROR] Failed to initialize WhoScored: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Get match schedule
    print("\n2. Fetching match schedule...")
    try:
        time.sleep(3)
        schedule = ws.read_schedule()
        print(f"   [OK] Found {len(schedule)} total matches")

        # Save schedule
        schedule_path = output_dir / 'schedule.parquet'
        schedule.to_parquet(schedule_path, compression='snappy')
        print(f"   [OK] Saved schedule to {schedule_path}")

        # Filter for completed matches
        completed_matches = schedule[schedule['status'] == 6]
        print(f"   [OK] {len(completed_matches)} matches completed")

        if len(completed_matches) == 0:
            print("   [WARNING] No completed matches found - nothing to download")
            sys.exit(0)

    except Exception as e:
        print(f"   [ERROR] Failed to fetch schedule: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Get missing players data
    print("\n3. Fetching missing players data...")
    try:
        missing_players = ws.read_missing_players()

        if missing_players is not None and len(missing_players) > 0:
            missing_players_path = output_dir / 'missing_players.parquet'
            missing_players.to_parquet(missing_players_path, compression='snappy')
            print(f"   [OK] Saved {len(missing_players)} missing player records")
        else:
            print("   [INFO] No missing players data available")

    except Exception as e:
        print(f"   [WARNING] Failed to fetch missing players: {e}")
        # Continue anyway - this is optional data

    # Download events for all completed matches
    print(f"\n4. Downloading events for {len(completed_matches)} completed matches...")
    print("   Progress will be saved incrementally.")
    print("   Already downloaded matches will be skipped.\n")

    all_events = []
    successful_downloads = 0
    failed_downloads = 0
    skipped_matches = 0

    for idx, (match_idx, match_info) in enumerate(completed_matches.iterrows(), 1):
        match_id = match_info['game_id']
        home_team = match_info['home_team']
        away_team = match_info['away_team']
        match_date = match_info['date']

        # Check if already downloaded
        match_file = events_dir / f"match_{match_id}_events.parquet"
        if match_file.exists():
            print(f"   [{idx}/{len(completed_matches)}] SKIP: {home_team} vs {away_team} (already downloaded)")
            skipped_matches += 1
            # Load existing data for combined file
            try:
                existing_events = pd.read_parquet(match_file)
                all_events.append(existing_events)
            except Exception as e:
                print(f"      [WARNING] Could not load existing file: {e}")
            continue

        print(f"   [{idx}/{len(completed_matches)}] Downloading: {home_team} vs {away_team} ({match_date.date()})")

        try:
            # Download events for this match
            events = ws.read_events(match_id=[match_id], output_fmt='events')

            if len(events) == 0:
                print(f"      [WARNING] No events found for match {match_id}")
                failed_downloads += 1
                continue

            # Save individual match file
            events.to_parquet(match_file, compression='snappy', index=False)

            # Add to combined data
            all_events.append(events)

            successful_downloads += 1
            print(f"      [OK] Downloaded {len(events)} events")

            # Small delay to be respectful
            time.sleep(2)

        except Exception as e:
            print(f"      [ERROR] Failed to download match {match_id}: {e}")
            failed_downloads += 1
            continue

    # Combine all events
    print("\n5. Creating combined events file...")
    if len(all_events) > 0:
        combined_events = pd.concat(all_events, ignore_index=True)

        # Save combined file
        combined_path = output_dir / 'all_events.parquet'
        combined_events.to_parquet(combined_path, compression='snappy', index=False)

        print(f"   [OK] Saved combined events: {combined_path}")
        print(f"      - Total events: {len(combined_events):,}")
        print(f"      - File size: {combined_path.stat().st_size / 1024 / 1024:.1f} MB")

        # Statistics
        print("\n6. Data Statistics:")
        print(f"   - Matches with events: {len(all_events)}")
        print(f"   - Total events: {len(combined_events):,}")

        # Count by event type
        top_event_types = combined_events['type'].value_counts().head(5)
        print(f"   - Top event types:")
        for event_type, count in top_event_types.items():
            print(f"      {event_type}: {count:,} ({count/len(combined_events)*100:.1f}%)")
    else:
        print("   [WARNING] No events were downloaded!")

    # Summary
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"League: {league_ws}")
    print(f"Season: {season}")
    print(f"Total matches in schedule: {len(completed_matches)}")
    print(f"Successfully downloaded: {successful_downloads}")
    print(f"Skipped (already downloaded): {skipped_matches}")
    print(f"Failed: {failed_downloads}")
    print(f"\nData saved to: {output_dir}")
    print("=" * 80)

if __name__ == '__main__':
    main()
