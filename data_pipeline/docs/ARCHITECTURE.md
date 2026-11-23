# ARCHITECTURE

## Overview
Multi-source football analytics system combining WhoScored events, Betfair odds, and FBRef shots.

## Workflow
1. **Download WhoScored**: `scripts/download_whoscored_league_season.py --league "ENG-Premier League" --season "2019-2020"`
2. **Download Betfair**: `download_basic_soccer.py --country GB --start-year 2019 --end-year 2020`
3. **Parse Betfair**: `parse_betfair_by_league.py --all` (auto-generates team mappings)
4. **Download FBRef**: `download_shots.py --league "ENG-Premier League" --season "2019-2020"`
5. **Align Matches**: `scripts/update_match_alignment_v2.py --league premier-league --season 19-20`
6. **Merge All**: `scripts/merge_all_data.py --league premier-league --season 19-20`

**Output**: `data/merged/premier-league/19-20/all_events_enriched.parquet` (events + odds + xG)

## Structure
```
data/
├── whoscored/         # Event-level match data (passes, shots, tackles)
├── fbref/             # Shot details (xG, PSxG, SCA)
├── alignment/         # WhoScored↔FBRef match mapping
└── merged/            # Final enriched data (events + odds + shots)

OUTPUT_LEAGUES/        # Parsed Betfair odds by league/season
BASIC/                 # Raw Betfair data (.bz2 files)

config/
└── team_name_mapping.json  # Auto-updated by parser
```