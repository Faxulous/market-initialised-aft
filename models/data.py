"""Shared data loading for model training pipelines."""
import os
import pandas as pd


def load_seasons(seasons, root):
    """Load and concatenate ws/align/sched across multiple seasons."""
    all_ws, all_align, all_sched = [], [], []
    for s in seasons:
        ws = pd.read_parquet(os.path.join(root, "data", s, "whoscored_events.parquet"))
        align = pd.read_parquet(os.path.join(root, "data", s, "match_alignment.parquet"))
        sched = pd.read_parquet(os.path.join(root, "data", s, "fbref_schedule.parquet"))
        ws["season"], align["season"], sched["season"] = s, s, s
        all_ws.append(ws); all_align.append(align); all_sched.append(sched)
    return pd.concat(all_ws, ignore_index=True), pd.concat(all_align, ignore_index=True), pd.concat(all_sched, ignore_index=True)
