"""
Download BASIC soccer data from Betfair.

Downloads raw .bz2 files month by month and saves them to BASIC/ directory.
Requires BETFAIR_LOGIN_DIR env var (see config.py get_trading_client).

Usage:
    python scripts/downloaders/download_betfair_soccer.py --country GB --start-year 2025 --start-month 8 --end-year 2026 --end-month 5
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for config import
from config import get_trading_client
from betfairlightweight.exceptions import InvalidResponse

# Settings
OUTPUT_BASE = str(Path(__file__).resolve().parents[2] / "BASIC")
SPORT = "Soccer"
# MATCH_ODDS for 1X2; OVER_UNDER_* for the over_under.parquet kick-off lines (0.5-4.5)
MARKET_TYPES = ["MATCH_ODDS", "OVER_UNDER_05", "OVER_UNDER_15", "OVER_UNDER_25",
                "OVER_UNDER_35", "OVER_UNDER_45"]
COUNTRY_CODE = "GB"
START_DATE = datetime(2025, 8, 1)   # 25-26 season
END_DATE = datetime(2026, 5, 31)

# Download settings
DOWNLOAD_TIMEOUT = 60  # Increased from default 16 to 60 seconds
MAX_RETRIES = 3  # Number of retry attempts for failed downloads
RETRY_DELAY = 2  # Seconds to wait between retries


def get_month_ranges(start: datetime, end: datetime) -> List[tuple]:
    """Generate list of (start_date, end_date) tuples for each month"""
    ranges = []
    current = start.replace(day=1)

    while current <= end:
        # First day of month
        month_start = current

        # Last day of month
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
        month_end = next_month - timedelta(days=1)

        # Don't go past END_DATE
        if month_end > end:
            month_end = end

        ranges.append((month_start, month_end))

        # Move to next month
        current = next_month

    return ranges


def download_month_data(trading, year: int, month: int, month_start: datetime, month_end: datetime, country_code: str) -> Dict:
    """Download all BASIC soccer files for a given month"""

    print(f"\n{'='*60}")
    print(f"  {month_start.strftime('%B %Y')}")
    print(f"{'='*60}")

    # Create output directory
    output_dir = os.path.join(OUTPUT_BASE, str(year), f"{month:02d}")
    os.makedirs(output_dir, exist_ok=True)

    # Get file list for this month
    print(f"Querying files from {month_start.date()} to {month_end.date()}...")

    try:
        # Note: The betfairlightweight library has string type hints but the Betfair API
        # actually expects integers for dates. Pass as integers directly.
        files = trading.historic.get_file_list(
            sport=SPORT,
            plan='Basic Plan',
            from_day=month_start.day,
            from_month=month_start.month,
            from_year=month_start.year,
            to_day=month_end.day,
            to_month=month_end.month,
            to_year=month_end.year,
            event_id=None,
            event_name=None,
            market_types_collection=MARKET_TYPES,
            countries_collection=[country_code],
            file_type_collection=None  # Get all file types
        )
    except InvalidResponse as e:
        # This HTML error page usually means authentication failed or session expired
        print(f"Error getting file list - Authentication/Session issue: {e}")
        print("  Tip: Try re-authenticating or check if your session has expired")
        return {"total": 0, "downloaded": 0, "skipped": 0, "failed": 0}
    except Exception as e:
        print(f"Error getting file list: {e}")
        return {"total": 0, "downloaded": 0, "skipped": 0, "failed": 0}

    if not files:
        print("  No files found for this month")
        return {"total": 0, "downloaded": 0, "skipped": 0, "failed": 0}

    # Note: get_file_list() returns a list of file path strings, not dictionaries
    # Filter for BASIC files only
    basic_files = [f for f in files if 'BASIC' in f]

    print(f"  Found {len(files)} total files, {len(basic_files)} BASIC files")

    stats = {"total": len(basic_files), "downloaded": 0, "skipped": 0, "failed": 0}

    # Download each file
    for i, file_path in enumerate(basic_files, 1):
        # file_path is a string like: /xds_nfs/edp_processed/BASIC/2025/Oct/1/34773095/1.248230404.bz2

        # Create local filename
        filename = os.path.basename(file_path)
        local_path = os.path.join(output_dir, filename)

        # Skip if already downloaded
        if os.path.exists(local_path):
            print(f"  [{i}/{len(basic_files)}] Skipped (exists): {filename}")
            stats["skipped"] += 1
            continue

        # Download file with retry logic
        download_success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if attempt == 1:
                    print(f"  [{i}/{len(basic_files)}] Downloading: {filename}", end=" ... ")
                else:
                    print(f"Retry {attempt}/{MAX_RETRIES}", end=" ... ")

                trading.historic.download_file(
                    file_path=file_path,
                    store_directory=output_dir
                )

                # Check if downloaded
                if os.path.exists(local_path):
                    size_mb = os.path.getsize(local_path) / 1024 / 1024
                    print(f"OK ({size_mb:.1f} MB)")
                    stats["downloaded"] += 1
                    download_success = True
                    break
                else:
                    print("FAILED (file not found after download)")

                # Small delay to avoid rate limiting
                time.sleep(0.5)

            except Exception as e:
                error_msg = str(e)
                # Check if it's a timeout error
                if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
                    if attempt < MAX_RETRIES:
                        print(f"TIMEOUT (attempt {attempt}/{MAX_RETRIES})", end=" ... ")
                        time.sleep(RETRY_DELAY)  # Wait before retry
                    else:
                        print(f"ERROR: {e}")
                else:
                    # Non-timeout errors don't retry
                    print(f"ERROR: {e}")
                    break

        if not download_success:
            stats["failed"] += 1

    return stats


def main():
    """Main download function"""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download Betfair BASIC soccer data')
    parser.add_argument('--country', type=str, default=COUNTRY_CODE,
                       help='Country code (GB=England, ES=Spain, DE=Germany, IT=Italy, FR=France)')
    parser.add_argument('--start-year', type=int, default=START_DATE.year)
    parser.add_argument('--end-year', type=int, default=END_DATE.year)
    parser.add_argument('--start-month', type=int, default=START_DATE.month)
    parser.add_argument('--end-month', type=int, default=END_DATE.month)
    args = parser.parse_args()

    # Override settings with arguments
    country_code = args.country.upper()
    start_date = datetime(args.start_year, args.start_month, 1)
    
    # Handle end date correctly (last day of month)
    if args.end_month == 12:
        end_date = datetime(args.end_year, 12, 31)
    else:
        # First day of next month - 1 day
        next_month = datetime(args.end_year, args.end_month + 1, 1)
        end_date = next_month - timedelta(days=1)

    # Country name mapping
    country_names = {
        'GB': 'England (Premier League)',
        'ES': 'Spain (La Liga)',
        'DE': 'Germany (Bundesliga)',
        'IT': 'Italy (Serie A)',
        'FR': 'France (Ligue 1)'
    }
    country_display = country_names.get(country_code, country_code)

    print("="*60)
    print("  Betfair BASIC Soccer Data Download")
    print(f"  {start_date.year} - {end_date.year}")
    print("="*60)
    print(f"Output directory: {OUTPUT_BASE}/")
    print(f"Sport: {SPORT}")
    print(f"Market types: {', '.join(MARKET_TYPES)}")
    print(f"Country: {country_code} ({country_display})")
    print()

    # Authenticate
    print("Authenticating...")
    try:
        trading = get_trading_client(timeout=DOWNLOAD_TIMEOUT)
        # IMPORTANT: Must call login() to set session_token for Historic Data API
        trading.login()
        print("[OK] Authenticated successfully")
        print(f"[OK] Session token set for Historic Data API (timeout: {DOWNLOAD_TIMEOUT}s)\n")
    except Exception as e:
        print(f"[ERROR] Authentication failed: {e}")
        print("\nPlease check your credentials in config.py")
        return

    # Get month ranges
    month_ranges = get_month_ranges(start_date, end_date)
    print(f"Total months to process: {len(month_ranges)}\n")

    # Download each month
    total_stats = {"total": 0, "downloaded": 0, "skipped": 0, "failed": 0}
    consecutive_auth_failures = 0
    MAX_AUTH_FAILURES = 5

    for i, (month_start, month_end) in enumerate(month_ranges):
        year = month_start.year
        month = month_start.month

        stats = download_month_data(trading, year, month, month_start, month_end, country_code)

        # Check for authentication failures
        if stats["total"] == 0 and stats["downloaded"] == 0:
            consecutive_auth_failures += 1
            if consecutive_auth_failures >= MAX_AUTH_FAILURES:
                print(f"\n{'='*60}")
                print("  WARNING: Multiple consecutive authentication failures")
                print(f"{'='*60}")
                print(f"Failed {consecutive_auth_failures} times in a row.")
                print("This suggests:")
                print("  1. Your session may have expired")
                print("  2. You may not have access to historical data")
                print("  3. The Betfair API may be experiencing issues")
                print("\nAttempting to re-authenticate...")
                try:
                    trading = get_trading_client(timeout=DOWNLOAD_TIMEOUT)
                    trading.login()  # Must call login() to set session_token
                    print("[OK] Re-authenticated successfully\n")
                    consecutive_auth_failures = 0
                except Exception as e:
                    print(f"[ERROR] Re-authentication failed: {e}")
                    print("\nStopping download process.")
                    break
        else:
            consecutive_auth_failures = 0

        # Update totals
        for key in total_stats:
            total_stats[key] += stats[key]

        # Print month summary
        print(f"  Month summary: {stats['downloaded']} downloaded, "
              f"{stats['skipped']} skipped, {stats['failed']} failed")

    # Final summary
    print("\n" + "="*60)
    print("  DOWNLOAD COMPLETE")
    print("="*60)
    print(f"Total files found: {total_stats['total']}")
    print(f"Downloaded: {total_stats['downloaded']}")
    print(f"Skipped (already exist): {total_stats['skipped']}")
    print(f"Failed: {total_stats['failed']}")
    print(f"\nFiles saved to: {os.path.abspath(OUTPUT_BASE)}/")
    print("="*60)


if __name__ == "__main__":
    main()
