"""Match-clustered uncertainty for the predictive-performance differences vs Betfair.

Non-parametric cluster bootstrap over the evaluation matches: pooled accuracy / RPS / log-loss
per model, clustered and naive i.i.d. SEs with their design effects, and the paired difference
against Betfair with a match-clustered 95% interval. Every model, metric and difference shares
one multinomial weight matrix, so all comparisons are paired on the same B resamples.

Run: python eval/analysis/uncertainty.py
"""
import os, sys
import numpy as np, pandas as pd
from rich.console import Console
from rich.table import Table

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from eval.acc import rps, logloss, accuracy          # identical metric definitions
from eval.utils import SEASON_CFG

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEED, B = 8, 10000
PRED = os.path.join(ROOT, "eval", "predictions.parquet")
CSV = os.path.join(ROOT, "eval", "analysis", "out", "uncertainty_bootstrap.csv")
REF = "betfair"                                   # paired comparator
MODELS = ["weibull_no_cov", "weibull", "weibull_calibd", "zou", "zou_calibd",
          "maia", "maia_calibd", "betfair"]
KEYS = ["match_id", "period", "eval_minute"]
LBL = {"correct": "Accuracy", "rps": "RPS", "logloss": "Log-loss"}
console = Console()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA -> ALIGNED PER-POINT LOSS MATRICES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load():
    """Return (base grid, {metric: (n_pts, n_models) loss matrix})."""
    df = pd.read_parquet(PRED)
    missing = set(MODELS) - set(df["model"].unique())
    if missing:
        raise ValueError(f"models absent from {PRED}: {sorted(missing)}")
    if set(df["season"].unique()) != set(SEASON_CFG):
        raise ValueError(f"seasons in {PRED} {sorted(df['season'].unique())} != SEASON_CFG {list(SEASON_CFG)}")
    g = {m: df[df["model"] == m].sort_values(KEYS).reset_index(drop=True) for m in MODELS}
    base = g[MODELS[0]][KEYS + ["actual"]]
    for m in MODELS:                               # all models must share an identical grid
        if not g[m][KEYS + ["actual"]].equals(base):
            raise ValueError(f"prediction grid for {m} differs from {MODELS[0]}")

    act = base["actual"].values
    cols = {"correct": [], "rps": [], "logloss": []}
    for m in MODELS:
        ph, pd_, pa = g[m]["pred_home"].values, g[m]["pred_draw"].values, g[m]["pred_away"].values
        pred = np.argmax(np.stack([ph, pd_, pa], 1), 1)
        cols["correct"].append((pred == act).astype(float))
        cols["rps"].append(rps(ph, pd_, pa, act))
        cols["logloss"].append(logloss(ph, pd_, pa, act))
        if abs(cols["correct"][-1].mean() - accuracy(ph, pd_, pa, act)) > 1e-12:
            raise ValueError(f"accuracy indicator disagrees with eval.acc.accuracy for {m}")
    return base, {k: np.column_stack(v) for k, v in cols.items()}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLUSTER BOOTSTRAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def boot_weights(n_match):
    """(B, n_match) multinomial counts == resampling n_match matches with replacement."""
    rng = np.random.default_rng(SEED)
    return rng.multinomial(n_match, np.full(n_match, 1.0 / n_match), size=B).astype(float)


def p_two_sided(d_boot):
    """CI-inversion two-sided bootstrap p-value with (B+1) continuity correction."""
    lo = (1 + (d_boot <= 0).sum()) / (B + 1.0)
    hi = (1 + (d_boot >= 0).sum()) / (B + 1.0)
    return min(1.0, 2 * min(lo, hi))


def analyse(P, midx, W, subset, ref_i):
    """Pooled metrics, clustered/naive SEs and paired deltas vs REF for every model."""
    nm = W.shape[1]
    C = np.bincount(midx, minlength=nm).astype(float)        # points per match (model-invariant)
    if C.min() == 0:
        raise ValueError(f"subset '{subset}' has matches with zero points: "
                         f"{int((C == 0).sum())} of {nm}")
    denom = W @ C
    rows = []
    for metric, Y in P.items():
        A = np.stack([np.bincount(midx, weights=Y[:, j], minlength=nm) for j in range(Y.shape[1])], 1)
        est = A.sum(0) / C.sum()                             # pooled point estimate per model
        bt = (W @ A) / denom[:, None]                        # (B, n_models) pooled per resample
        d_bt = bt - bt[:, [ref_i]]                           # paired: same resample both arms
        d_pt = est - est[ref_i]
        d_pp = Y - Y[:, [ref_i]]                             # per-point paired difference
        for j, m in enumerate(MODELS):
            lo, hi = np.percentile(bt[:, j], [2.5, 97.5])
            dlo, dhi = np.percentile(d_bt[:, j], [2.5, 97.5])
            se_c, se_i = bt[:, j].std(ddof=1), Y[:, j].std(ddof=1) / np.sqrt(len(Y))
            dse_c = d_bt[:, j].std(ddof=1)
            dse_i = d_pp[:, j].std(ddof=1) / np.sqrt(len(Y))
            deff_l = (se_c / se_i) ** 2                       # design effect on the level
            # delta design effect undefined where the paired difference is identically zero
            # (calibrated models share Betfair's argmax class at minute 0) -> NaN, never a default
            deff_d = (dse_c / dse_i) ** 2 if dse_i > 0 else np.nan
            rows.append(dict(subset=subset, model=m, metric=metric, n_points=len(Y),
                             n_matches=nm, est=est[j], se_clust=se_c, ci_lo=lo, ci_hi=hi,
                             se_iid=se_i, deff_level=deff_l, se_ratio_level=np.sqrt(deff_l),
                             n_eff_level=len(Y) / deff_l,
                             delta_vs_bf=d_pt[j], d_se_clust=dse_c, d_ci_lo=dlo, d_ci_hi=dhi,
                             p_boot=np.nan if j == ref_i else p_two_sided(d_bt[:, j]),
                             d_se_iid=np.nan if j == ref_i else dse_i,
                             deff_delta=np.nan if j == ref_i else deff_d,
                             se_ratio_delta=np.nan if j == ref_i else np.sqrt(deff_d),
                             n_eff_delta=np.nan if j == ref_i else len(Y) / deff_d))
    return pd.DataFrame(rows)


def main():
    season, = SEASON_CFG
    base, P = load()
    midx = pd.factorize(base["match_id"], sort=True)[0]      # 0..nm-1 match index per row
    nm, ref_i = midx.max() + 1, MODELS.index(REF)
    subset = f"all ({len(base):,})"                          # label derived, never hardcoded
    console.print(f"[bold]{season}[/] cluster bootstrap B={B}, SEED={SEED}, {nm} matches, "
                  f"{len(base)} points/model, {len(MODELS)} models")
    res = analyse(P, midx, boot_weights(nm), subset, ref_i).assign(season=season)
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    res.to_csv(CSV, index=False)

    t = Table(title=f"{season}: Delta vs Betfair, match-clustered 95% CI ({subset} rows)",
              header_style="bold cyan")
    for c in ["Model", "Metric", "Delta", "95% CI", "p", "SE ratio", "n_eff"]:
        t.add_column(c, justify="left" if c == "Model" else "right")
    for m in MODELS:
        if m == REF: continue
        for metric in ["correct", "rps", "logloss"]:
            r = res[(res.model == m) & (res.metric == metric)].iloc[0]
            t.add_row(m, LBL[metric], f"{r.delta_vs_bf:.4f}",
                      f"[{r.d_ci_lo:.4f}, {r.d_ci_hi:.4f}]", f"{r.p_boot:.4f}",
                      f"{r.se_ratio_delta:.2f}", f"{r.n_eff_delta:,.0f}")
    console.print(t)
    console.print(f"[green]Wrote[/] {CSV}")


if __name__ == "__main__":
    main()
