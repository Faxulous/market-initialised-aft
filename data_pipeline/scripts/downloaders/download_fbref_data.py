"""
Download comprehensive FBRef data for a league/season.

This script fetches all available FBRef data including:
- Match schedule (dates, teams, scores, attendance)
- Shot events (with SCA data)
- Team statistics
- Player statistics
- Match lineups and formations

Output structure:
    data/{league}/{season}/
        ├── fbref_shots.parquet           # Shot-level data with SCA
        ├── fbref_schedule.parquet         # Match schedule
        ├── fbref_team_stats.parquet       # Team-level statistics
        └── fbref_lineups.parquet          # Lineups and formations (if available)

Usage:
    python download_fbref_data.py --league premier-league --season 24-25
    python download_fbref_data.py --all  # Download all leagues/seasons
"""

import soccerdata as sd
import pandas as pd
import cloudscraper
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

# Base paths (repo root = two levels above scripts/downloaders/)
BASE_PATH = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_PATH / 'data'

def patch_soccerdata():
    """Patch soccerdata with cloudscraper to avoid rate limiting."""
    original_init = sd._common.BaseRequestsReader._init_session

    def new_init_session(self):
        session = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
        )
        session.headers.update({
            'sec-ch-ua': '"Not A Brand";v="99", "Chromium";v="138", "Google Chrome";v="138"'
        })
        session.proxies.update(self.proxy())
        return session

    sd._common.BaseRequestsReader._init_session = new_init_session
    print("[INFO] Patched soccerdata with cloudscraper")

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-level column names."""
    df.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
    return df

def download_schedule(fbref, output_dir: Path):
    """Download match schedule."""
    print("\n[1/4] Downloading match schedule...")
    try:
        schedule = fbref.read_schedule()
        schedule_df = flatten_columns(schedule.reset_index())

        # Rename 'game' to 'match_id' for consistency
        if 'game' in schedule_df.columns:
            schedule_df.rename(columns={'game': 'match_id'}, inplace=True)

        output_path = output_dir / 'fbref_schedule.parquet'
        schedule_df.to_parquet(output_path, compression='snappy', index=False)

        print(f"   [OK] Saved {len(schedule_df)} matches to {output_path.name}")
        print(f"        Columns: {list(schedule_df.columns[:8])}...")
        return True
    except Exception as e:
        print(f"   [ERROR] Failed to download schedule: {e}")
        return False

def download_shots(fbref, output_dir: Path):
    """Download shot events with SCA data."""
    print("\n[2/4] Downloading shot events with SCA data...")
    print("        (This may take 5-10 minutes for a full season)")
    try:
        shots = fbref.read_shot_events()
        shots_df = flatten_columns(shots.reset_index())

        # Rename 'game' to 'match_id' for consistency
        if 'game' in shots_df.columns:
            shots_df.rename(columns={'game': 'match_id'}, inplace=True)

        output_path = output_dir / 'fbref_shots.parquet'
        shots_df.to_parquet(output_path, compression='snappy', index=False)

        # Show SCA coverage
        sca_cols = [col for col in shots_df.columns if 'SCA' in col]
        if sca_cols:
            sca1_coverage = shots_df['SCA 1 player'].notna().sum() / len(shots_df) * 100 if 'SCA 1 player' in shots_df.columns else 0
            print(f"   [OK] Saved {len(shots_df)} shots to {output_path.name}")
            print(f"        SCA coverage: {sca1_coverage:.1f}%")
        else:
            print(f"   [OK] Saved {len(shots_df)} shots to {output_path.name}")

        return True
    except Exception as e:
        print(f"   [ERROR] Failed to download shots: {e}")
        return False

def download_team_stats(fbref, output_dir: Path):
    """Download team-level statistics."""
    print("\n[3/4] Downloading team statistics...")
    try:
        # Try different stat types
        stats_types = ['schedule', 'shooting', 'passing', 'defense']

        for stat_type in stats_types:
            try:
                stats = fbref.read_team_season_stats(stat_type=stat_type)
                stats_df = flatten_columns(stats.reset_index())

                output_path = output_dir / f'fbref_team_{stat_type}.parquet'
                stats_df.to_parquet(output_path, compression='snappy', index=False)
                print(f"   [OK] Saved team {stat_type} stats: {stats_df.shape}")
            except:
                continue

        return True
    except Exception as e:
        print(f"   [SKIP] Team stats not available: {e}")
        return False

def download_lineups(fbref, output_dir: Path):
    """Download match lineups and formations."""
    print("\n[4/4] Downloading lineups and formations...")
    try:
        lineups = fbref.read_lineup()
        lineups_df = flatten_columns(lineups.reset_index())

        # Rename 'game' to 'match_id' for consistency
        if 'game' in lineups_df.columns:
            lineups_df.rename(columns={'game': 'match_id'}, inplace=True)

        output_path = output_dir / 'fbref_lineups.parquet'
        lineups_df.to_parquet(output_path, compression='snappy', index=False)

        print(f"   [OK] Saved {len(lineups_df)} lineup records to {output_path.name}")
        return True
    except Exception as e:
        print(f"   [SKIP] Lineups not available: {e}")
        return False

def download_league_season(league: str, season: str):
    """Download all FBRef data for a league-season."""
    print(f"\n{'='*80}")
    print(f"DOWNLOADING FBREF DATA: {league.upper()} {season}")
    print(f"{'='*80}")

    # Create output directory
    output_dir = DATA_PATH / league / season
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Patch soccerdata
    patch_soccerdata()

    # Initialize FBref reader
    print(f"\nInitializing FBref reader for {LEAGUE_MAPPING[league]} {season}...")
    fbref = sd.FBref(leagues=LEAGUE_MAPPING[league], seasons=season)

    # Download each data type
    results = {
        'schedule': download_schedule(fbref, output_dir),
        'shots': download_shots(fbref, output_dir),
        'team_stats': download_team_stats(fbref, output_dir),
        'lineups': download_lineups(fbref, output_dir)
    }

    # Print summary
    print(f"\n{'='*80}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*80}")
    for data_type, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {data_type}")

    print(f"\nFiles saved to: {output_dir}")
    for file in sorted(output_dir.glob('fbref_*.parquet')):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name} ({size_mb:.1f} MB)")

def parse_args():
    parser = argparse.ArgumentParser(description='Download comprehensive FBRef data')
    parser.add_argument('--league', choices=list(LEAGUE_MAPPING.keys()),
                       help='League name')
    parser.add_argument('--season', help='Season (YY-YY format)')
    parser.add_argument('--all', action='store_true',
                       help='Download all leagues and seasons')
    return parser.parse_args()

def main():
    args = parse_args()

    if args.all:
        leagues = list(LEAGUE_MAPPING.keys())
        seasons = ['17-18', '18-19', '19-20', '20-21', '21-22', '22-23', '23-24', '24-25']

        print(f"Downloading {len(leagues)} leagues × {len(seasons)} seasons")

        for league in leagues:
            for season in seasons:
                try:
                    download_league_season(league, season)
                except Exception as e:
                    print(f"\n[ERROR] Failed {league} {season}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    elif args.league and args.season:
        download_league_season(args.league, args.season)

    else:
        print("ERROR: Must specify either --league and --season, or --all")
        sys.exit(1)

    print(f"\n{'='*80}")
    print("DONE! FBRef data saved to data/{league}/{season}/fbref_*.parquet")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
