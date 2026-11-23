"""Betfair price realism: the +2-minute shift and the staleness of the last-traded price.

Rebuilds the Betfair benchmark at BF_SHIFT in {0, 2} on the evaluation points of
eval/predictions.parquet — shift 2 must recover the shipped `betfair` rows exactly — and
measures how old the served price is as seen from the evaluation minute.

Run: python eval/analysis/market_data.py
"""
import os, sys

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from eval.utils import SEASON_CFG, load_betfair_data, spike_mask
from eval.acc import accuracy, logloss, rps
import models.strategies.betfair as bfmod

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEASON, = SEASON_CFG              # the paper evaluates 2024/25 only
SHIFTS = [0, 2]                   # BF_SHIFT values compared (the paper reports 2)
STALE_SET = "eval points: lag from the evaluation minute (+2 shift)"
OUT = os.path.join(ROOT, "eval", "analysis", "out")
NAME = "market_data"

console = Console()
BF_SHIFT_DEFAULT = bfmod.BF_SHIFT  # restored at the end of main(); the loop mutates it


def q(a, name, n_no_age):
    """Quantile + threshold summary for an array of minutes. NaN is an error, not a drop;
    prints with no definable age are passed as an explicit count (`n_no_age`)."""
    a = np.asarray(a, float)
    if np.isnan(a).any():
        raise ValueError(f"{name}: {int(np.isnan(a).sum())} NaN of {len(a)} — no silent drop")
    return {"set": name, "n": len(a), "n_no_age": int(n_no_age),
            "mean": a.mean(), "median": np.median(a),
            "p90": np.percentile(a, 90), "p99": np.percentile(a, 99), "max": a.max(),
            "share_0": (a == 0).mean(), "share_le_1": (a <= 1).mean(), "share_ge_3": (a >= 3).mean(),
            "n_ge_5": int((a >= 5).sum()), "share_ge_5": (a >= 5).mean(),
            "n_ge_10": int((a >= 10).sum()), "share_ge_10": (a >= 10).mean()}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ODDS FRAME WITH STALENESS (mirrors eval.utils.load_betfair_data key construction)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_odds_meta(season):
    """One row per odds_lookup key, carrying the age of the price print. load_betfair_data
    drops is_real_update/timestamp, so the same key construction (broken-match match_minute
    repair, in-play filter, dedup) is repeated here, including its team-pair-only merge.

    The feed polls from minute -30, so every in-play print has an earlier fresh poll and
    therefore a definable age; a key without one is a data fault and raises.
    """
    o = pd.read_parquet(os.path.join(ROOT, "data", season, "betfair_odds.parquet"))
    a = pd.read_parquet(os.path.join(ROOT, "data", season, "match_alignment.parquet"))
    teams = lambda x: str(x).split(" ", 1)[1] if " " in str(x) else str(x)
    o["_t"], a["_t"] = o["fbref_match_id"].apply(teams), a["fbref_match_id"].apply(teams)
    o = o.merge(a[["_t", "ws_game_id"]], on="_t", how="inner").drop(columns="_t")
    o["timestamp"] = pd.to_datetime(o["timestamp"])

    # Price age: minutes since the last poll that returned a fresh, complete price.
    ok = o["is_real_update"] & o[["odds_home", "odds_draw", "odds_away"]].notna().all(axis=1)
    o = o.sort_values(["ws_game_id", "minute"])
    o["age"] = o["minute"] - o["minute"].where(ok).groupby(o["ws_game_id"]).ffill()

    o = o[o[["odds_home", "odds_draw", "odds_away"]].notna().all(axis=1)].copy()
    inv = o[["odds_home", "odds_draw", "odds_away"]].rdiv(1)
    normed = inv.div(inv.sum(axis=1), axis=0)
    o["ph"], o["pdr"], o["pa"] = normed["odds_home"], normed["odds_draw"], normed["odds_away"]

    ip = o["period"] != "PreMatch"   # transient spikes dropped, as in eval.utils.load_betfair_data
    o = o.drop(o.index[ip][spike_mask(o.loc[ip, ["ph", "pdr", "pa"]], o.loc[ip, "fbref_match_id"], o.loc[ip, "minute"]).values])
    o = o.sort_values("timestamp", na_position="first")
    # Half-time rows continue the first-half clock through the break (as in eval.utils)
    fl = o[o["period"] == "FirstHalf"].sort_values("minute").groupby("ws_game_id").last()
    htm = o["period"] == "halftime"
    off = o.loc[htm, "ws_game_id"].map(fl["minute"] - fl["match_minute"])
    if off.isna().any():
        raise ValueError(f"{season}: half-time prices without first-half prices in {off.isna().sum()} rows")
    o.loc[htm, "match_minute"] = o.loc[htm, "minute"] - off
    o = o[((o["match_minute"] >= 0) | (o["match_minute"] < -1)) & o["match_minute"].notna()]

    # Repair matches whose match_minute is wall-clock (no halftime gap) — as in eval.utils
    for wid, grp in o.groupby("ws_game_id"):
        fh, sh = grp[grp["period"] == "FirstHalf"], grp[grp["period"] == "SecondHalf"]
        if len(fh) == 0 or len(sh) == 0 or sh["minute"].min() - fh["minute"].max() > 2:
            continue
        ss = sh.sort_values("minute")
        d = pd.to_datetime(ss["timestamp"]).diff().dt.total_seconds().fillna(0)
        big = d[d > 300].index
        first2h = int(ss.loc[big[0], "minute"]) if len(big) else int(fh["minute"].max()) + 18
        m = (o["ws_game_id"] == wid) & o["period"].isin(["SecondHalf", "unknown"])
        o.loc[m, "match_minute"] = o.loc[m, "minute"] - (first2h - 45)

    cols = ["ws_game_id", "match_minute", "ph", "pdr", "pa", "age"]
    fh = o[o["period"].isin(["FirstHalf", "halftime"]) | ((o["period"] == "unknown") & (o["match_minute"] < 45))]
    fh = fh.drop_duplicates(subset=["ws_game_id", "match_minute"], keep="last")[cols].assign(per=1)
    sh = o[o["period"].isin(["SecondHalf", "unknown"]) & (o["match_minute"] >= 45)]
    sh = sh.drop_duplicates(subset=["ws_game_id", "match_minute"], keep="last")[cols].assign(per=2)
    meta = pd.concat([fh, sh], ignore_index=True)
    if meta["age"].isna().any():
        raise ValueError(f"{season}: {int(meta['age'].isna().sum())} lookup keys have no price age")
    meta["mid"] = meta["ws_game_id"].astype(int)
    meta["mm"] = meta["match_minute"].astype(int)
    return meta


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BENCHMARK REBUILD AT ARBITRARY SHIFT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class _Dummy:
    """Stand-in for TeamState in eval/test.py: no `period` attribute. The pipeline passes
    `period=` explicitly to predict_probs (so 1H-stoppage rows are priced off 1H odds)."""


def classify(mid, minute, per, om, shift, lut):
    """Which BetfairStrategy fallback branch produced the price at odds_minute `om`."""
    off = om - minute
    if om == -1:   # minute 0 at 0-0: last price before kick-off (eval/utils.py), identical at every shift
        return "pre_kickoff"
    if off == shift:
        return "exact"
    if 0 <= off < shift:
        return "locf_back"
    if off > shift:
        return "forward_search"
    return "back_search" if (mid, om, per) in lut else "prev_half"


def run_shift(shift, pts, lut, age_lut, no_age):
    """Rebuild the Betfair benchmark at BF_SHIFT=shift on the given evaluation points.

    `age` = age of the print at the odds minute served. `lag` = how old that price is
    measured from the *evaluation* minute: max(0, eval_minute - odds_minute) + age.
    Keys in `no_age` (pre-kick-off prints, whose age is not recorded) get NaN age/lag and
    are counted explicitly downstream; any other missing key raises.
    """
    bfmod.BF_SHIFT = shift  # runtime override; models/strategies/betfair.py untouched
    st = bfmod.BetfairStrategy(odds_lookup=lut)
    dummy, rows = _Dummy(), []
    for mid, minute, per in pts:
        r = st.predict_probs(dummy, dummy, 0, 1.0, None, 1, match_id=mid, minute=minute, period=per)
        if r is None:
            raise ValueError(f"shift {shift}: no price for evaluation point ({mid}, {minute}, {per})")
        om = int(r["odds_minute"])
        br = classify(mid, minute, per, om, shift, lut)
        k = (mid, om, 1 if br == "prev_half" else per)
        if k not in age_lut and k not in no_age:
            raise KeyError(f"no price age for served odds key {k}")
        age = age_lut[k] if k in age_lut else np.nan
        p = np.maximum([r["home"], r["draw"], r["away"]], 1e-3)  # floor + renormalise
        p = np.round(p / p.sum(), 4)                             # + round, as in eval/test.py
        rows.append({"pred_home": p[0], "pred_draw": p[1], "pred_away": p[2],
                     "age": age, "lag": max(0, minute - om) + age})
    return pd.DataFrame(rows)


def metrics(sub):
    ph, pdr, pa, act = (sub["pred_home"].values, sub["pred_draw"].values,
                        sub["pred_away"].values, sub["actual"].values)
    return {"n": len(sub), "accuracy": accuracy(ph, pdr, pa, act),
            "rps": rps(ph, pdr, pa, act).mean(), "logloss": logloss(ph, pdr, pa, act).mean()}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    os.makedirs(OUT, exist_ok=True)
    meta = load_odds_meta(SEASON)
    lut, _, _ = load_betfair_data(SEASON, ROOT)
    plut = {(r.mid, r.mm, r.per): (r.ph, r.pdr, r.pa) for r in meta.itertuples()}
    # Validate the mirrored key construction against the production lookup. Minute -1 keys are
    # pre-kick-off prices from load_prematch_1x2, not part of the in-play feed mirrored here.
    bad = sum(1 for k, v in lut.items() if k[1] != -1 and (k not in plut or abs(plut[k][0] - v["home"]) > 1e-12))
    if bad:
        raise ValueError(f"{SEASON}: {bad}/{len(lut)} odds_lookup keys not reproduced by load_odds_meta")
    console.print(f"[green]{SEASON} key check OK[/]: {len(lut)} odds_lookup entries reproduced")
    age_lut = {(r.mid, r.mm, r.per): r.age for r in meta.itertuples()}
    no_age = {k for k in lut if k[1] == -1}  # pre-kick-off print: age not recorded

    preds = pd.read_parquet(os.path.join(ROOT, "eval", "predictions.parquet"))
    bf0 = preds[preds["model"] == "betfair"].copy()
    if set(bf0["season"]) != {SEASON}:
        raise ValueError(f"predictions seasons {set(bf0['season'])} != {{{SEASON}}}")
    pts = list(zip(bf0["match_id"].astype(int), bf0["eval_minute"].astype(int), bf0["period"].astype(int)))

    # run_shift preserves bf0's row order, so `actual` attaches positionally.
    # (A merge on (match_id, eval_minute) would duplicate the 1H-stoppage rows.)
    shifted = {}
    for s in SHIFTS:
        d = run_shift(s, pts, lut, age_lut, no_age).reset_index(drop=True)
        shifted[s] = d.assign(actual=bf0["actual"].reset_index(drop=True))
    dev = float(np.abs(shifted[2]["pred_home"].values - bf0["pred_home"].values).max())
    if dev > 1e-4:
        raise ValueError(f"shift 2 does not reproduce the betfair column: max deviation {dev:.4f}")

    # ── STALENESS: how old the price in use is at the minute the bettor would act ──
    lag = shifted[2]["lag"]
    stale = pd.DataFrame([q(lag.dropna(), STALE_SET, lag.isna().sum())])
    stale.to_csv(os.path.join(OUT, f"{NAME}_staleness.csv"), index=False)

    # ── BENCHMARK DELAY SENSITIVITY: identical evaluation points at each shift ──
    shift_tbl = pd.DataFrame([{"model": "betfair", "shift": s, "convention": "all rows",
                               **metrics(shifted[s])} for s in SHIFTS])
    shift_tbl.to_csv(os.path.join(OUT, f"{NAME}_shift_metrics.csv"), index=False)
    bfmod.BF_SHIFT = BF_SHIFT_DEFAULT  # leave the imported module as we found it

    def show(title, df, fmt="{:.4f}"):
        t = Table(title=title, header_style="bold cyan")
        for c in df.columns:
            t.add_column(str(c), justify="right", overflow="fold")
        for _, r in df.iterrows():
            t.add_row(*[fmt.format(v) if isinstance(v, float) else str(v) for v in r])
        console.print(t)

    console.print(f"shift 2 reproduces the shipped betfair column (max abs deviation {dev:.1e})")
    show("Price staleness (min) — lag from the evaluation minute",
         stale[["set", "n", "n_no_age", "mean", "median", "p90", "p99", "max", "share_le_1",
                "n_ge_5", "n_ge_10"]])
    show("Betfair benchmark by shift", shift_tbl)


if __name__ == "__main__":
    main()
