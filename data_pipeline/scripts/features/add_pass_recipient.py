import pandas as pd, glob

if __name__ == "__main__":
    for f in glob.glob("data/**/whoscored_events.parquet", recursive=True):
        print(f"Processing {f}...")
        df = pd.read_parquet(f)
        g = df.groupby(['ws_match_id', 'team_name'])['player_name']
        recip = pd.Series(None, index=df.index, dtype=object)
        for i in range(1, 11): recip = recip.fillna(g.shift(-i)[df.player_name != g.shift(-i)])
        df.loc[(df.event_type == 'Pass') & (df.event_outcome == 'Successful'), 'pass_recipient'] = recip
        df.to_parquet(f)
