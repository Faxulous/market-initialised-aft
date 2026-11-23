# A market-initialised accelerated failure time model for in-play football forecasting

Code for the paper:

> Clegg, L., Song, Z. and Cartlidge, J. (2026). A market-initialised accelerated
> failure time model for in-play football forecasting. *Journal of Quantitative
> Analysis in Sports*. https://doi.org/10.1515/jqas-2026-0103

The model is a Weibull accelerated failure time model of goal arrival times,
used to forecast home win, draw and away win probabilities at every minute of
English Premier League matches. It is compared with the birth process of Zou et
al. (2020), the Cox process of Maia et al. (2026), three market-implied
benchmarks and Betfair Exchange in-play prices.

Training uses 1,236 matches from 2021/22 to gameweek 10 of 2024/25. Evaluation
uses the 280 matches from gameweek 11 of 2024/25 onwards.


## What is included

No data are included. The event data (WhoScored.com), shot data (FBref.com) and
prices (Betfair Exchange historical data) all have terms of use that restrict
redistribution. This repository therefore holds no season data, no prices and
no per-match or per-minute forecasts. It holds the code, and the aggregate
result tables and figures reported in the paper.

Rebuilding the results needs the data, obtained from the three sources
(Section 5).


## 1. Market initialisation on made-up prices

    python examples/market_initialisation.py

This runs the initialisation of Section 3.7 of the paper for one match, using
invented 1X2 and over/under odds, and needs no data. It prints the prices, the
model's kick-off probabilities before and after initialisation, and the shift
in each team's log-expected scoring time.


## 2. Reproduce the paper's tables and figures

These read the forecasts in `eval/predictions/`, which Section 3 builds from
the data, and run in seconds to a few minutes. Run from the repository root.

| Paper item | Script | Output |
|---|---|---|
| Covariate comparison table, shape and labelling results, stoppage means | `eval/refit.py` | `eval/refit/` |
| Predictive performance table | `eval/gen_tables.py --benchmarks` | LaTeX to stdout |
| Log-loss by minute figure | `eval/acc.py` | `eval/acc/pdf/logloss.pdf`, `eval/acc/summary.csv` |
| Intervals for the gap to Betfair | `eval/analysis/uncertainty.py` | `eval/analysis/out/uncertainty_bootstrap.csv` |
| Betting table | `eval/bet.py` | `eval/bet/summary.csv` |
| Reliability and betting figures, calibration table, return intervals | `eval/analysis/bet_robust.py` | `eval/analysis/out/` |
| Market-implied benchmarks | `eval/analysis/benchmarks.py` | `eval/analysis/out/benchmark_preds.parquet` |
| Betfair shift and price recency | `eval/analysis/market_data.py`, `eval/analysis/goal_latency.py` | `eval/analysis/out/` |
| Data consistency table | `eval/analysis/sanity.py` | `eval/analysis/out/sanity_table.tex` |
| Market-initialisation shift figure | `eval/shift.py` | `eval/acc/pdf/calibration_shift.pdf`, `eval/acc/shift.csv` |
| Mean against true stoppage time | `eval/test_stoppage.py` | `eval/acc/stoppage.csv` |

The aggregate outputs are included as generated for the paper. The per-match
outputs (the forecast files, `benchmark_preds.parquet`, `shift.csv`,
`bet_robust_match_pnl.csv` and the shift figure) are not, and are written when
the scripts run.

`eval/test_clock.py` holds unit tests for the playing-time clock, likelihood
and simulator.


## 3. Rebuild everything

    eval/run_all.sh                 # team parameters, forecasts, all outputs
    eval/run_all.sh --skip-params   # reuse fitted team parameters

The pipeline fits expanding-window team parameters (one snapshot per matchday),
runs the eight forecasting strategies, initialises the three models from
kick-off prices and regenerates every output above. It runs up to three jobs at
once and needs about 30 GB of memory. Set `PY` to choose the interpreter.

A single strategy:

    python eval/test.py --eval-season 24-25 --strategy weibull_calibd
    python eval/test.py --merge     # rebuild eval/predictions.parquet


## 4. Models

| Strategy | Paper name | Description |
|---|---|---|
| `weibull_no_cov` | Weibull | Weibull AFT with a common shape |
| `weibull` | Weibull, shots | adds red-card difference and shots deviation |
| `weibull_calibd` | Weibull, shots, market-initialised | team strengths from kick-off prices |
| `zou`, `zou_calibd` | Zou | birth process with Bayesian updating, exact recursion |
| `maia`, `maia_calibd` | Maia | Cox process over goals, red cards and stoppage |
| `betfair` | Betfair | in-play 1X2 last-traded prices |

Market initialisation fits each match's scoring parameters to the kick-off
Betfair 1X2 and over/under prices.

    eval/utils.py                 configuration, prediction loading, odds helpers
    eval/test.py                  prediction loop
    eval/run_all.sh               full pipeline
    eval/analysis/                analyses behind the paper's robustness results
    examples/                     market initialisation on made-up prices
    models/weibull/               simulator, clock, fit and market initialisation
    models/weibull/team_params/   expanding-window maximum likelihood
    models/zoubirthprocess/       Zou et al. (2020)
    models/maia/                  Maia et al. (2026)
    models/strategies/            strategy registry
    models/covariates/            stoppage estimates, covariate likelihood


## 5. Data

Season data are not included. Each `data/<season>/` folder holds
`whoscored_events`, `fbref_schedule`, `fbref_shots`, `match_alignment`,
`betfair_odds` and `over_under` (all `.parquet`), joined on match id. Event
data come from WhoScored.com, shots from FBref.com and prices from the Betfair
Exchange historical data service.

`data_pipeline/` downloads, parses and joins these sources into
`data_pipeline/data/premier-league/<season>/`, from which the season folders
above are copied; see `data_pipeline/docs/ARCHITECTURE.md`. Run it from inside
`data_pipeline/`. The Betfair downloader needs your own Betfair account and API
credentials.


## 6. Dependencies

Python 3.10 or later; see `requirements.txt`.


## 7. Licence and citation

The code is released under the PolyForm Noncommercial License 1.0.0 (see
`LICENSE`). It is free to use, change and share for research, teaching and
other noncommercial purposes, provided the copyright notice is kept. Commercial
use needs the author's permission.

If you use this code, please cite the paper:

```bibtex
@article{clegg2026market,
  title   = {A market-initialised accelerated failure time model for in-play football forecasting},
  author  = {Clegg, Lawrence and Song, Zixing and Cartlidge, John},
  journal = {Journal of Quantitative Analysis in Sports},
  year    = {2026},
  doi     = {10.1515/jqas-2026-0103}
}
```

`CITATION.cff` holds the same reference, which GitHub uses for its "Cite this
repository" button.
