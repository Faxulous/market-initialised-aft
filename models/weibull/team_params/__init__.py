"""Team-specific Weibull parameters with pre-computed historical files."""
import math, os
from typing import Dict
import pandas as pd, torch



class TeamParamBaseline:
    """Team params from pre-computed historical files (one per gameweek)."""

    def __init__(self, params_dir: str):
        """
        Args:
            params_dir: Directory containing index.pt and params_*.pt files
        """
        self.params_dir = params_dir
        self.index = None
        self.param_dates = []  # Sorted list of available param dates
        self.params_cache: Dict[str, dict] = {}  # date_str -> params dict

        idx_path = os.path.join(params_dir, "index.pt")
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"No index.pt in {params_dir}. Run: python -m models.weibull.team_params.estimate_params --all")

        self.index = torch.load(idx_path, weights_only=False)
        self.dir = params_dir
        from models.weibull.fit import VERSION
        if self.index["clock_version"] != VERSION:
            raise ValueError("Rebuild Weibull team parameters for the continuous clock")
        self.param_dates = sorted(self.index["dates"])
        print(f"Loaded param index: {len(self.param_dates)} dates ({self.param_dates[0]} to {self.param_dates[-1]})")

    def _get_params_for_date(self, match_date) -> dict:
        """Get params that were valid at match_date (most recent params before that date)."""
        dt_str = str(pd.to_datetime(match_date).date())

        # Find most recent params file strictly before this date
        valid_date = None
        for d in self.param_dates:
            if d < dt_str:
                valid_date = d
            else:
                break

        if valid_date is None:
            raise ValueError(f"No params available before {dt_str}. Earliest: {self.param_dates[0]}")

        # Load from cache or file
        if valid_date not in self.params_cache:
            params_path = os.path.join(self.dir, os.path.basename(self.index["param_files"][valid_date]))  # basename: old indexes stored absolute paths
            self.params_cache[valid_date] = torch.load(params_path, weights_only=False)

        return self.params_cache[valid_date]

    def get_log_et(self, team: int, opp: int, home: bool, date) -> float:
        """Get linear predictor η = μ + home + att + def."""
        p = self._get_params_for_date(date)
        if team not in p["attack"]:
            raise KeyError(f"Team {team} not in params")
        return p["mu"] + home * p["home"] + p["attack"][team] + p["defense"][opp]

    def get_params(self, team_id: int, opp_id: int, is_home: bool, date) -> dict:
        """Get team params dict with log_et for use without graphs."""
        log_et = self.get_log_et(team_id, opp_id, is_home, date)
        return {"baseline_log_et": log_et}
