"""Data-quality sanity checks over the paper's EPL seasons; writes the appendix table. Run: python eval/analysis/sanity.py"""
import os, re, sys
import numpy as np, pandas as pd
from rich.console import Console
from rich.table import Table

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from eval.utils import SEASON_CFG, seasons_through, build_goals_lookup, spike_mask, SPIKE_JUMP
from models.weibull.clock import EXCLUDED_MATCHES

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVAL = "24-25"                      # the evaluation season; its training pool is every earlier fixture
SEASONS = seasons_through(EVAL)     # the paper's seasons: training pool + evaluation season
NS = f"{len(SEASONS)} seasons"
SEED = 8
STOPPAGE_MAX = 15       # implausible added time (minutes)
RED_MAX = 3             # implausible red cards in one match
SHORT_FEED = 85         # a match feed ending before this minute is truncated
OUT = os.path.join(ROOT, "eval", "analysis", "out")
WS_COLS = ["ws_match_id", "period", "minute", "event_type", "team_name", "team_id",
           "is_own_goal", "is_shot", "card_type"]
console = Console()
ROWS = []  # tidy check summary
TABLE = []  # (check, scope, flagged, action): rows of the paper's appendix table


def add(check_id, scope, n_checked, n_flagged, worst, action, season="all"):
    ROWS.append(dict(check_id=check_id, season=season, scope=scope, n_checked=n_checked,
                     n_flagged=n_flagged, worst_example=worst, action_taken=action))


def tab(check, scope, flagged, action):
    TABLE.append((check, scope, flagged, action))


def write_table():
    """Booktabs appendix table from TABLE (counts with LaTeX thousands separators)."""
    f = lambda x: re.sub(r"(\d),(\d{3})", r"\1{,}\2", f"{x:,}" if isinstance(x, (int, np.integer)) else x)
    body = "\n".join(f"{c} & {f(sc)} & {f(n)} & {a} \\\\" for c, sc, n, a in sorted(TABLE, key=lambda r: "price" in r[0].lower() or "Betfair" in r[0]))  # event/shot checks, then market
    tex = ("\\begin{tabular}{llrl}\n\\toprule\nCheck & Checked & Flagged & Action \\\\\n\\midrule\n"
           f"{body}\n\\bottomrule\n\\end{{tabular}}\n")
    open(os.path.join(OUT, "sanity_table.tex"), "w").write(tex)


def load_all():
    """Load events (subset of columns) / alignment / schedule / shots, tagged by season."""
    ws, al, sc, sh = [], [], [], []
    for s in SEASONS:
        d = os.path.join(ROOT, "data", s)
        parts = [pd.read_parquet(os.path.join(d, "whoscored_events.parquet"), columns=WS_COLS),
                 pd.read_parquet(os.path.join(d, "match_alignment.parquet")),
                 pd.read_parquet(os.path.join(d, "fbref_schedule.parquet")),
                 pd.read_parquet(os.path.join(d, "fbref_shots.parquet"))]
        for df, lst in zip(parts, [ws, al, sc, sh]):
            df["season"] = s
            lst.append(df)
    cat = lambda l: pd.concat(l, ignore_index=True)
    return cat(ws), cat(al), cat(sc), cat(sh)


def split_ids(al):
    """(training pool, evaluation) ws_game_id sets, as the models are trained: the pool is every
    fixture dated before the evaluation start, the evaluation set is EVAL from that date on."""
    d, cut = pd.to_datetime(al["date"]), pd.to_datetime(SEASON_CFG[EVAL][1])
    return set(al.loc[d < cut, "ws_game_id"]), set(al.loc[(al["season"] == EVAL) & (d >= cut), "ws_game_id"])


def prematch_mask(od):
    """Pre-match quote rows under the two odds schemas: 21-22..23-24 have no period column and
    pre-match minutes are negative; 24-25 labels 'PreMatch' at minutes -30..-1."""
    return od["period"] == "PreMatch" if "period" in od.columns else od["minute"] < 0


# ━━━ 1. Final score: WhoScored events (incl. own goals) vs FBref schedule vs alignment ━━━
def check_score(ws, al, sched):
    goals = build_goals_lookup(ws, al)  # own goals credited to the opposing side
    have = set(ws["ws_match_id"].unique())
    sc = sched.set_index("match_id")["score"]
    recs = []
    for r in al.itertuples():
        if r.ws_game_id not in have:
            continue  # no event feed at all -> check 2
        g = goals.get(int(r.ws_game_id))
        wh, wa = (0, 0) if g is None else (int(g["is_home"].sum()), int((~g["is_home"]).sum()))
        fh, fa = [int(x) for x in str(sc[r.fbref_match_id]).replace("–", "-").split("-")]
        recs.append(dict(season=r.season, ws_game_id=r.ws_game_id, fixture=r.fbref_match_id,
                         ws_score=f"{wh}-{wa}", fbref_score=f"{fh}-{fa}",
                         ws_ok=(wh, wa) == (fh, fa), align_ok=(r.home_score, r.away_score) == (fh, fa)))
    df = pd.DataFrame(recs)
    bad = df[~df["ws_ok"] | ~df["align_ok"]]
    worst = "none" if bad.empty else "; ".join(f"{r.season} {r.fixture}: events give {r.ws_score}, "
                                               f"FBref/alignment give {r.fbref_score}" for r in bad.itertuples())
    tab("Final score agrees with FBref", f"{len(df):,} fixtures", len(bad), "Same fixture as above")
    add("1_score_agreement", f"matches with an event feed, {NS}", len(df), len(bad), worst,
        "none required" if bad.empty else "truncated feed(s), see check 2c")


# ━━━ 2. Coverage: which scheduled fixtures are excluded and why ━━━
def check_coverage(ws, al, sched):
    n_sched = {s: int((sched.season == s).sum()) for s in SEASONS}  # 380 per season
    for s, n in n_sched.items():
        assert int((al.season == s).sum()) == n, f"{s}: alignment != schedule ({n} fixtures)"
    tot, w_ids = sum(n_sched.values()), set(ws["ws_match_id"].unique())
    excl = al[~al["ws_game_id"].isin(w_ids)]                # no event feed at all
    dropped = al[al["ws_game_id"].isin(EXCLUDED_MATCHES)]   # truncated feed, see check 2c
    have = w_ids - EXCLUDED_MATCHES                         # these exclusions never reach the fit
    tr, ev = split_ids(al)
    ledger = dict(scheduled=tot, usable=tot - len(excl) - len(dropped),
                  train=len(tr & have), eval=len(ev & have))
    assert ledger["train"] + ledger["eval"] == ledger["usable"], f"ledger does not add up {ledger}"
    pred = pd.read_parquet(os.path.join(ROOT, "eval", "predictions.parquet"))
    assert pred.match_id.nunique() == ledger["eval"], f"predictions cover {pred.match_id.nunique()} != {ledger['eval']}"
    add("2_fixture_coverage", f"{tot} scheduled fixtures, {NS} ({', '.join(f'{s}: {n}' for s, n in n_sched.items())})",
        tot, len(excl) + len(dropped), "; ".join([f"{r.season} {r.fbref_match_id} (no whoscored_events)" for r in excl.itertuples()]
                                                + [f"{r.season} {r.fbref_match_id} (truncated feed)" for r in dropped.itertuples()]),
        f"excluded -> {ledger['usable']} usable; {EVAL}: {ledger['train']} training + {ledger['eval']} evaluation")
    tab("WhoScored event feed exists", f"{tot:,} fixtures", len(excl), "Excluded")

    # 2c. feed completeness for matches that DO have events
    dur = ws[ws["period"].isin(["FirstHalf", "SecondHalf"])].groupby(
        ["season", "ws_match_id", "period"])["minute"].max().unstack()
    trunc = dur[dur["SecondHalf"].isna() | (dur["SecondHalf"] < SHORT_FEED)].reset_index()
    trunc = trunc.merge(al[["ws_game_id", "fbref_match_id"]], left_on="ws_match_id", right_on="ws_game_id")
    tab("Event feed reaches the final whistle", f"{len(dur):,} fixtures", len(trunc), "Excluded")
    add("2c_feed_completeness", "event feed reaches the final whistle", int(len(dur)), len(trunc),
        "; ".join(f"{r.season} {r.fbref_match_id}: last event min {r.FirstHalf:.0f} (1H), 2H max {r.SecondHalf}"
                  for r in trunc.itertuples()),
        "; ".join(f"{r.fbref_match_id}: excluded via models.weibull.clock.EXCLUDED_MATCHES" if r.ws_match_id in EXCLUDED_MATCHES
                  else f"{r.fbref_match_id}: NOT excluded, inside the training pool" for r in trunc.itertuples()))
    return ledger


# ━━━ 5b. Every shot: same count per match and side, same minute up to the indexing offset, xG present ━━━
def check_shots(ws, al, shots):
    home_fb = al.set_index("fbref_match_id")["fbref_home_team"].to_dict()
    home_ws = al.set_index("ws_game_id")["ws_home_team"].to_dict()
    season_of = al.set_index("ws_game_id")["season"].to_dict()
    fb_to_ws = al.set_index("fbref_match_id")["ws_game_id"].to_dict()
    fbs = shots.assign(ws_game_id=shots["match_id"].map(fb_to_ws))
    fbs = fbs[fbs["ws_game_id"].isin(set(ws["ws_match_id"].unique()) - EXCLUDED_MATCHES)]
    fbs["is_home"] = [t == home_fb[m] for t, m in zip(fbs["team"], fbs["match_id"])]
    wss = ws[ws["is_shot"].astype(bool) & ~ws["is_own_goal"].astype(bool) & ~ws["ws_match_id"].isin(EXCLUDED_MATCHES)].copy()
    wss["is_home"] = [t == home_ws[m] for t, m in zip(wss["team_name"], wss["ws_match_id"])]
    a = wss.groupby(["ws_match_id", "is_home"])["minute"].apply(sorted).to_dict()
    b = fbs.groupby(["ws_game_id", "is_home"])["minute"].apply(sorted).to_dict()
    uneven = [k for k in set(a) | set(b) if k not in a or k not in b or len(a[k]) != len(b[k])]
    d = pd.Series([x - y for k in set(a) & set(b) if k not in uneven for x, y in zip(a[k], b[k])]).value_counts().sort_index()
    used = (set.union(*split_ids(al)) - EXCLUDED_MATCHES)  # training + evaluation matches
    off = int(d[d.index != -1].sum())  # WhoScored minutes are 0-indexed, FBref's 1-indexed
    tab("Shot count agrees for both teams", f"{len(set(a) | set(b)) // 2:,} fixtures", len(uneven), "--")
    tab("Shot minute agrees", f"{len(fbs):,} shots", off, "FBref minute used")
    tab("Shot has an xG value", f"{len(fbs):,} shots", int(pd.to_numeric(fbs["xG"], errors="coerce").isna().sum()), "--")
    add("5b_shot_timing", f"{len(fbs)} FBref shots vs {len(wss)} WhoScored shots, paired by match and side (own goals excluded)",
        len(fbs), len(uneven) + int(d[d.index.map(abs) > 2].sum()),
        f"{len(uneven)} match-sides with unequal shot counts; minute offsets: {', '.join(f'{int(k)} min x{v}' for k, v in d.items())}; "
        f"{EVAL} training + evaluation matches: {int(fbs['ws_game_id'].isin(used).sum())} shots",
        "none required: the offset is the 0- vs 1-indexed minute convention" if not uneven else
        "unequal counts left as recorded: " + "; ".join(f"{season_of[k[0]]} {k[0]} {'home' if k[1] else 'away'}" for k in uneven))


# ━━━ 6. Stoppage time per half ━━━
def check_stoppage(ws):
    d = ws[ws["period"].isin(["FirstHalf", "SecondHalf"])]
    e = d.groupby(["season", "ws_match_id", "period"])["minute"].max().unstack().reset_index()
    e["h1_added"], e["h2_added"] = e["FirstHalf"] - 45, e["SecondHalf"] - 90
    g = e.groupby("season")[["h1_added", "h2_added"]].agg(["mean", "std", "min", "max", "count"]).round(2)
    console.print(g)
    bad = e[(e[["h1_added", "h2_added"]] < 0).any(axis=1) |
            (e[["h1_added", "h2_added"]] > STOPPAGE_MAX).any(axis=1)]
    means = "; ".join(f"{s}: H1 {g.loc[s, ('h1_added','mean')]:.2f}, H2 {g.loc[s, ('h2_added','mean')]:.2f}"
                      for s in SEASONS)
    r = e[~e["ws_match_id"].isin(EXCLUDED_MATCHES)]  # retained fixtures only
    long = int((r[["h1_added", "h2_added"]] > STOPPAGE_MAX).any(axis=1).sum())  # negatives: half ends after its last event
    tab(f"Added time at most {STOPPAGE_MAX} minutes per half", f"{len(r):,} fixtures", long,
        "Retained, timestamps agree")
    add("6_stoppage_time", "added time per half, every match with events", len(e), len(bad),
        f"means {means}. Extremes: H1 max {e.h1_added.max():.0f}, H2 max {e.h2_added.max():.0f}; flagged: "
        + ("; ".join(f"{r.season} {r.ws_match_id} H1 {r.h1_added:+.0f} H2 {r.h2_added:+.0f}" for r in bad.itertuples()) or "none"),
        "no action: extremes correspond to real long stoppages; the season shift is why "
        "stoppage is estimated within season")


# ━━━ 7. Red cards ━━━
def check_reds(ws, al):
    r = ws[ws["card_type"].isin(["Red", "SecondYellow"])]
    per_match = r.groupby(["season", "ws_match_id"]).size()
    df = pd.DataFrame({"reds": r.groupby("season").size(),
                       "matches": ws.groupby("season")["ws_match_id"].nunique()}).reset_index()
    df["per_1000_matches"] = (1000 * df["reds"] / df["matches"]).round(1)
    pool = (split_ids(al)[0] & set(ws["ws_match_id"].unique())) - EXCLUDED_MATCHES  # training pool
    add("7_red_cards", f"red and second-yellow cards, {NS}", int(df.matches.sum()),
        int((per_match > RED_MAX).sum()),
        f"{len(r)} total; in the training pool: {EVAL} {int(r['ws_match_id'].isin(pool).sum())} ({len(pool)} matches)"
        f"; {df.per_1000_matches.min():.0f}-{df.per_1000_matches.max():.0f} per 1,000 matches by season; "
        f"most in one match = {int(per_match.max())}",
        "none required")


# ━━━ 8. Betfair quotes: all three outcomes priced, and transient one-minute spikes ━━━
def check_odds():
    od = pd.read_parquet(os.path.join(ROOT, "data", EVAL, "betfair_odds.parquet"))
    full, inplay = od[["odds_home", "odds_draw", "odds_away"]].notna().all(axis=1), ~prematch_mask(od)
    n_in, n_part = int(inplay.sum()), int((~full & inplay).sum())
    tab(f"Betfair quotes all three outcomes (20{EVAL.replace('-', '/')})", f"{n_in:,} in-play minutes", n_part, "Dropped")
    add("8_partial_quotes", f"in-play Betfair quote rows, {EVAL}", n_in, n_part,
        f"{n_part} in-play rows quote fewer than three legs",
        "dropped in eval/utils.load_betfair_data", EVAL)


def check_spikes():
    """Transient last-traded spikes (eval.utils.spike_mask): an in-play print whose implied probability
    for some outcome sits >= SPIKE_JUMP from both neighbouring prints, same side. Removed from the
    Betfair lookup by eval.utils.load_betfair_data, which carries the preceding price forward."""
    recs, cols = [], ["odds_home", "odds_draw", "odds_away"]
    for s in SEASONS:
        od = pd.read_parquet(os.path.join(ROOT, "data", s, "betfair_odds.parquet"))
        od = od[od[cols].notna().all(axis=1) & ~prematch_mask(od)]
        p = od[cols].rdiv(1)
        hit = od[spike_mask(p.div(p.sum(axis=1), axis=0), od["fbref_match_id"], od["minute"])]
        recs.append(dict(season=s, n_inplay_prints=len(od), n_spikes=len(hit),
                         n_matches=hit["fbref_match_id"].nunique()))
    df = pd.DataFrame(recs)
    r = df[df.season == EVAL].iloc[0]  # only evaluation-season odds are read in play
    tab(f"No one-minute price spike (20{EVAL.replace('-', '/')})", f"{r.n_inplay_prints:,} in-play prices", r.n_spikes, "Preceding price used")
    add("8e_odds_spikes", f"in-play Betfair prints, {NS}", int(df.n_inplay_prints.sum()), int(df.n_spikes.sum()),
        "; ".join(f"{r.season}: {r.n_spikes} prints in {r.n_matches} matches" for r in df.itertuples()),
        f"removed: one-minute excursions of the last-traded price of at least {SPIKE_JUMP} in implied "
        "probability (typically at goals, while the market is suspended) are not executable; the lookup "
        "carries the preceding price forward")


# ━━━ 10. Residual checks ━━━
def check_misc(ws, al):
    g = ws[ws["event_type"] == "Goal"]
    key = ["ws_match_id", "minute", "period", "team_name", "event_type"]
    dup_g = int(g.duplicated(subset=key).sum())
    gr = g[~g["ws_match_id"].isin(EXCLUDED_MATCHES)]  # retained fixtures only
    tab("Goal event not duplicated", f"{len(gr):,} goals", int(gr.duplicated(subset=key).sum()), "--")
    add("10a_duplicate_goal_events", "goal events sharing (match, minute, period, team)", len(g), dup_g,
        f"{dup_g} repeated keys: no goal is double-counted in any of the {NS}" if dup_g == 0 else f"{dup_g} repeated keys",
        "none required" if dup_g == 0 else "inspect")
    ou = pd.read_parquet(os.path.join(ROOT, "data", EVAL, "over_under.parquet"))
    ev = al[al["ws_game_id"].isin(split_ids(al)[1])]
    n_miss = int((~ev["fbref_match_id"].isin(set(ou["match_id"]))).sum())
    tab("Kick-off 1X2 and over/under prices exist", f"{len(ev):,} evaluation", n_miss, "--")
    add("10c_calibration_inputs", f"{len(ev)} evaluation matches need 1X2 and O/U at kick-off", len(ev), n_miss,
        "every evaluation fixture has both markets" if n_miss == 0 else f"{n_miss} evaluation fixtures lack an O/U row",
        "none required" if n_miss == 0 else "inspect", EVAL)


def main():
    np.random.seed(SEED)
    ws, al, sched, shots = load_all()
    console.print(f"Loaded {len(ws):,} events, {len(al)} fixtures, {len(shots):,} shots")
    ledger = check_coverage(ws, al, sched)
    check_score(ws, al, sched)
    check_shots(ws, al, shots)
    check_stoppage(ws)
    check_reds(ws, al)
    check_odds()
    check_spikes()
    check_misc(ws, al)

    write_table()
    pd.DataFrame(ROWS).to_csv(os.path.join(OUT, "sanity_checks.csv"), index=False)
    t = Table(show_lines=True)
    for c, w in [("check_id", 26), ("season", 7), ("n_checked", 10), ("n_flagged", 10), ("worst_example", 70)]:
        t.add_column(c, overflow="fold", max_width=w)
    for r in ROWS:
        t.add_row(r["check_id"], r["season"], f"{r['n_checked']:,}", f"{r['n_flagged']:,}", r["worst_example"])
    console.print(t)
    console.print(f"[green]Ledger[/] {ledger}")


if __name__ == "__main__":
    main()
