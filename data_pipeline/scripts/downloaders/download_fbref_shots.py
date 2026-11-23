#!/usr/bin/env python3
"""
Scrape FBref shots ("shots_all" table) per match into fbref_shots.parquet.

Replaces soccerdata's read_shot_events(), removed in soccerdata 1.9. Output schema
matches earlier seasons exactly (league, season, match_id, match_minute, player, team,
xG, PSxG, outcome, distance, body_part, notes, SCA 1/2 player+event, minute, period).

Reads data/{league}/{season}/fbref_schedule.parquet for match_report URLs; caches one
parquet per match (resume-safe) then concatenates.

Usage:
    python scripts/downloaders/download_fbref_shots.py --league premier-league --season 25-26
"""

import argparse, time
from io import StringIO
from pathlib import Path
import pandas as pd
import cloudscraper

BASE_PATH = Path(__file__).resolve().parents[2]
FBREF = 'https://fbref.com'
DELAY = 7  # seconds between requests (FBref bans above 10 req/min; stay under)
LEAGUE_NAMES = {'premier-league': 'ENG-Premier League', 'bundesliga': 'GER-Bundesliga',
                'la-liga': 'ESP-La Liga', 'ligue-1': 'FRA-Ligue 1', 'serie-a': 'ITA-Serie A'}
# FBref renamed some teams in 25-26; normalise back to the convention of earlier seasons
TEAM_RENAMES = {'Newcastle': 'Newcastle Utd', 'Nottingham': "Nott'ham Forest"}


def parse_shots(html: str, match_id: str, league: str, season: str) -> pd.DataFrame:
    """Extract and normalise the shots_all table from a match report page."""
    html = html.replace('<!--', '').replace('-->', '')  # FBref ships tables inside HTML comments
    tbl = pd.read_html(StringIO(html), attrs={'id': 'shots_all'})[0]
    tbl.columns = [f"{a} {b}".strip() if not a.startswith('Unnamed') else b for a, b in tbl.columns]
    tbl = tbl[tbl['Minute'].astype(str).str.strip().ne('') & tbl['Minute'].notna()]
    tbl = tbl[tbl['Minute'].astype(str) != 'Minute']  # repeated header rows

    df = pd.DataFrame({
        'league': LEAGUE_NAMES[league], 'season': season.replace('-', ''), 'match_id': match_id,
        'match_minute': tbl['Minute'].astype(str), 'player': tbl['Player'],
        'team': tbl['Squad'].replace(TEAM_RENAMES),
        'xG': pd.to_numeric(tbl['xG'], errors='coerce').astype('Float64'),
        'PSxG': pd.to_numeric(tbl['PSxG'], errors='coerce').astype('Float64'),
        'outcome': tbl['Outcome'], 'distance': pd.to_numeric(tbl['Distance'], errors='coerce'),
        'body_part': tbl['Body Part'], 'notes': tbl['Notes'],
        'SCA 1 player': tbl['SCA 1 Player'], 'SCA 1 event': tbl['SCA 1 Event'],
        'SCA 2 player': tbl['SCA 2 Player'], 'SCA 2 event': tbl['SCA 2 Event'],
    })
    # '45+2' -> minute 47, period by base minute (matches repair_shot_minutes, verified on 23-24/24-25)
    parts = df['match_minute'].str.split('+')
    base = pd.to_numeric(parts.str[0], errors='raise')
    extra = pd.to_numeric(parts.str[1], errors='coerce').fillna(0)
    df['minute'] = (base + extra).astype(int)
    df['period'] = (base > 45).astype(int) + 1
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description='Scrape FBref shots tables')
    ap.add_argument('--league', required=True, choices=list(LEAGUE_NAMES))
    ap.add_argument('--season', required=True, help='YY-YY, e.g. 25-26')
    ap.add_argument('--limit', type=int, help='only scrape first N matches (testing)')
    ap.add_argument('--browser', action='store_true', help='fetch via seleniumbase UC Chrome (beats IP bans; run with DISPLAY unset for invisibility)')
    ap.add_argument('--wayback', action='store_true', help='fetch in-season Wayback snapshots (FBref removed shots tables from live pages in 2026)')
    args = ap.parse_args()

    sched = pd.read_parquet(BASE_PATH / 'data' / args.league / args.season / 'fbref_schedule.parquet')
    sched = sched[sched['match_report'].notna() & sched['score'].notna()]  # played matches only
    if args.limit:
        sched = sched.head(args.limit)
    cache = BASE_PATH / 'data' / 'fbref_shots_cache' / args.league / args.season
    cache.mkdir(parents=True, exist_ok=True)

    if args.wayback:
        import requests
        sess = requests.Session()
        sess.headers['User-Agent'] = 'football-inplay research (academic)'

        def cdx_retry(params, tries=5):
            for a in range(tries):
                r = sess.get('http://web.archive.org/cdx/search/cdx', params=params, timeout=60)
                if r.status_code == 200 and r.text.strip().startswith('['):
                    return r.json()
                time.sleep(10 * (a + 1))  # CDX 503s under load; back off
            raise RuntimeError(f"CDX failed after {tries} tries: {r.status_code}")

        def fetch_wayback(row):
            day_after = (pd.Timestamp(row.match_id[:10]) + pd.Timedelta(days=1)).strftime('%Y%m%d')
            rows = cdx_retry({'url': FBREF + row.match_report, 'output': 'json',
                              'from': day_after, 'to': '20260801', 'filter': 'statuscode:200', 'limit': 8})
            for ts in [r[1] for r in rows[1:]]:
                html = sess.get(f'http://web.archive.org/web/{ts}id_/{FBREF}{row.match_report}', timeout=120).text
                if 'shots_all' in html:
                    return html
            raise ValueError(f'no snapshot with shots table ({len(rows)-1} snapshots tried)')
    elif args.browser:
        from seleniumbase import Driver
        driver = Driver(uc=True, headless=False)  # run under Xvfb: DISPLAY=:99 for invisible CF solving
        def fetch(url):
            driver.uc_open_with_reconnect(url, reconnect_time=4)
            if 'Just a moment' in driver.get_title():  # Cloudflare challenge
                driver.uc_gui_click_captcha()
                driver.sleep(4)
            if 'Just a moment' in driver.get_title():
                raise RuntimeError('Cloudflare challenge not passed')
            return driver.page_source
    else:
        # Same profile as download_fbref_data.py's soccerdata patch (plain linux UA gets 403)
        scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False})
        scraper.headers.update({'sec-ch-ua': '"Not A Brand";v="99", "Chromium";v="138", "Google Chrome";v="138"'})
        def fetch(url):
            r = scraper.get(url, timeout=30)
            r.raise_for_status()
            return r.text

    frames, failed = [], []
    for i, row in enumerate(sched.itertuples(), 1):
        f = cache / f"{row.game_id}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
            continue
        try:
            html = fetch_wayback(row) if args.wayback else fetch(FBREF + row.match_report)
            df = parse_shots(html, row.match_id, args.league, args.season)
            if len(df) == 0:
                raise ValueError('empty shots table')
            df.to_parquet(f, index=False)
            frames.append(df)
            print(f"[{i}/{len(sched)}] {row.match_id}: {len(df)} shots")
        except Exception as e:
            failed.append(row.match_id)
            print(f"[{i}/{len(sched)}] FAILED {row.match_id}: {e}")
        time.sleep(DELAY)

    if failed:
        print(f"\n[WARNING] {len(failed)} matches failed: {failed}")
    out = BASE_PATH / 'data' / args.league / args.season / 'fbref_shots.parquet'
    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(out, index=False)
    print(f"\nSaved {len(combined)} shots from {combined['match_id'].nunique()} matches -> {out}")


if __name__ == '__main__':
    main()
