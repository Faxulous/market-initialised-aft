# Football Data Merger - Usage Guide

## Overview

The merge script combines three data sources at the **event level**:
1. **WhoScored** - Base event data (second-by-second, ~1,600 events per match)
2. **Betting Odds** - Enriched with minute-level odds (odds_home, odds_away, odds_draw)
3. **FBRef Shots** - Enriched with shot details when event is a shot (PSxG, SCA1, etc.)

## Output Format

**One parquet file per league-season** with all events from all matches:
- File: `data/merged/{league}/{season}/all_events_enriched.parquet`
- Format: Event-level data (each row = one event)
- Structure: WhoScored events + odds columns + FBRef shot columns

## Example Output (17-18 Season)

```
Total events: 607,856
Events with odds: 27,511 (4.5%)
Shot events: 9,328
Goal events: 1,018
Unique matches: 380
Dimensions: (607856, 34)
```

## Data Structure

### Base Columns (from WhoScored)
- `game_id`: WhoScored match ID
- `fbref_match_id`: FBRef match ID (format: "YYYY-MM-DD Home-Away")
- `period`: FirstHalf, SecondHalf, etc.
- `minute`, `second`, `expanded_minute`: Event timing
- `match_minute`: Integer minute for odds matching
- `type`: Event type (Pass, Shot, Goal, Foul, etc.)
- `outcome_type`: Successful, Unsuccessful, etc.
- `team`, `team_id`: Team information
- `player`, `player_id`: Player information
- `x`, `y`, `end_x`, `end_y`: Pitch coordinates
- `is_shot`, `is_goal`, `is_touch`: Event flags
- `qualifiers`: Additional event metadata

### Enriched Columns

#### Betting Odds (populated for ~4.5% of events)
- `odds_home`: Home team win odds
- `odds_away`: Away team win odds
- `odds_draw`: Draw odds

**Note**: Odds are populated every minute (when available), so most second-by-second events will have null odds values.

#### FBRef Shot Data (populated only for shot events)
- `PSxG`: Post-shot Expected Goals
- `SCA1`, `SCA2`: Shot-Creating Actions
- `Distance`: Distance from goal
- `Body_Part`: Header, Left Foot, Right Foot
- *(Additional columns when FBRef shots data is available)*

## Usage

### 1. Create Alignment Tables First

```bash
python scripts/update_match_alignment_v2.py --league premier-league --season 17-18
```

### 2. Merge All Data Sources

```bash
# Single season
python scripts/merge_all_data.py --league premier-league --season 17-18

# All leagues and seasons
python scripts/merge_all_data.py --all
```

### 3. Load and Analyze

```python
import pandas as pd

# Load enriched events
df = pd.read_parquet('data/merged/premier-league/17-18/all_events_enriched.parquet')

# Filter events with odds
odds_events = df[df['odds_home'].notna()]

# Filter shot events
shot_events = df[df['is_shot'] == True]

# Analyze a specific match
match_events = df[df['fbref_match_id'] == '2018-01-01 Brighton-Bournemouth']
```

## Data Availability

### By Season (Premier League)

| Season | FBRef Events | WhoScored Events | Betting Odds | FBRef Shots |
|--------|--------------|------------------|--------------|-------------|
| 17-18  | ✓            | ✓                | 17 matches   | ✗           |
| 18-19  | ✓            | ✓                | Pending      | ✗           |
| 19-20  | ✓            | ✓                | Pending      | ✗           |
| 20-21  | ✓            | ✓                | Pending      | ✗           |
| 21-22  | ✓            | ✓                | Pending      | ✗           |
| 22-23  | ✓            | ✓                | Pending      | ✗           |
| 23-24  | ✓            | ✓                | Pending      | ✗           |
| 24-25  | ✓            | ✓                | 0 matches    | ✓           |

**Note**: As you extract more betting odds from `BASIC/` into `OUTPUT_LEAGUES/YYYY-YY/`, they will automatically be included in the merge.

## Use Cases

### 1. Build ML Features

```python
# Create minute-by-minute features
features = df.groupby(['fbref_match_id', 'match_minute']).agg({
    'is_shot': 'sum',
    'is_goal': 'sum',
    'odds_home': 'first',  # Take first odds value per minute
    'odds_away': 'first',
    'odds_draw': 'first'
}).reset_index()

# Add cumulative stats
features['cumulative_shots'] = features.groupby('fbref_match_id')['is_shot'].cumsum()
features['cumulative_goals'] = features.groupby('fbref_match_id')['is_goal'].cumsum()
```

### 2. Analyze Odds Movement

```python
# Events that caused large odds shifts
df['next_odds_home'] = df.groupby('fbref_match_id')['odds_home'].shift(-1)
df['odds_shift'] = df['next_odds_home'] - df['odds_home']

# Find biggest shifts
big_shifts = df[df['odds_shift'].abs() > 0.5]
print(big_shifts[['fbref_match_id', 'type', 'player', 'odds_shift']])
```

### 3. Shot Analysis with Context

```python
# Shots with odds context
shots_with_odds = df[(df['is_shot'] == True) & (df['odds_home'].notna())]

# High-pressure shots (team is behind in odds)
high_pressure = shots_with_odds[shots_with_odds['odds_home'] > 3.0]
```

## File Organization

```
data/
├── whoscored/
│   └── premier-league/
│       └── 17-18/
│           ├── schedule.parquet
│           └── events_by_match/*.parquet
├── fbref/
│   └── premier-league/
│       └── 17-18/
│           ├── events.parquet
│           └── shots.parquet (24-25 only)
├── alignment/
│   └── premier-league/
│       └── 17-18/
│           └── match_alignment.parquet
└── merged/                    ← OUTPUT
    └── premier-league/
        └── 17-18/
            └── all_events_enriched.parquet
```

## Next Steps

1. **Extract more betting odds**: Process more BASIC data into OUTPUT_LEAGUES
2. **Collect FBRef shots**: Add shot-level data for older seasons
3. **Feature engineering**: Create ML features from enriched events
4. **Build models**: Use minute-by-minute data for predictions
