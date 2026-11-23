"""Betfair implied probability strategy (market benchmark).

Uses a fixed +2 minute shift when looking up Betfair odds, so prices
have had time to absorb any recent events (goals, VAR checks).
LOCF backward from target if exact minute unavailable.
"""
from typing import Dict
from ..weibull import Strategy

BF_SHIFT = 2  # Fixed shift: conservative but consistent across all minutes


class BetfairStrategy(Strategy):
    """Betfair implied probabilities benchmark (looks up pre-cached odds)."""
    name = "betfair"

    def __init__(self, odds_lookup: dict = None):
        self.odds_lookup = odds_lookup or {}
        self.fallback_count = self.fallback_max_offset = self.total_lookups = 0
        self.missing_lookups = []

    def predict_probs(self, home, away, sd: int, rem: float, dev, n_sims: int = 1000,
                      match_id: int = None, minute: int = None, **kw) -> Dict[str, float]:
        if match_id is None or minute is None:
            return None
        self.total_lookups += 1
        mid = int(match_id)
        p = kw.get("period", getattr(home, "period", None))  # explicit kwarg fixes 1H-stoppage rows
        period = int(p.item() if hasattr(p, "item") else p) if p is not None else (1 if minute < 45 else 2)

        # Kick-off: the last price BEFORE kick-off, i.e. the same information the kappa models are
        # initialised from (models/market.py). The shifted in-play price would already see two minutes of play.
        if minute == 0 and period == 1:
            return {**self.odds_lookup[(mid, -1, 1)], "odds_offset": -1, "odds_minute": -1}

        shift = BF_SHIFT
        target = minute + shift

        # Primary: LOCF backward from target to eval minute
        for off in range(shift + 1):
            m = int(target - off)
            result = self.odds_lookup.get((mid, m, period))
            if result:
                result = result.copy()
                result["odds_offset"], result["odds_minute"] = m - minute, m
                if m != target:
                    self.fallback_count += 1
                    self.fallback_max_offset = max(self.fallback_max_offset, abs(m - target))
                return result

        # Fallback: search backward from eval minute
        for off in range(1, minute + 1):
            m = int(minute - off)
            result = self.odds_lookup.get((mid, m, period))
            if result:
                self.fallback_count += 1
                self.fallback_max_offset = max(self.fallback_max_offset, off)
                result = result.copy()
                result["odds_offset"], result["odds_minute"] = -off, m
                return result

        # Period start: search forward up to 5 minutes past target
        period_start = 0 if period == 1 else 45
        if minute <= period_start + 5:
            for off in range(1, 6):
                m = int(target + off)
                result = self.odds_lookup.get((mid, m, period))
                if result:
                    self.fallback_count += 1
                    self.fallback_max_offset = max(self.fallback_max_offset, off)
                    result = result.copy()
                    result["odds_offset"], result["odds_minute"] = m - minute, m
                    return result

        # 2H with no prices: fall back to last 1H price
        if period == 2:
            for m in range(44, -1, -1):
                result = self.odds_lookup.get((mid, m, 1))
                if result:
                    self.fallback_count += 1
                    self.fallback_max_offset = max(self.fallback_max_offset, minute - m)
                    result = result.copy()
                    result["odds_offset"], result["odds_minute"] = m - minute, m
                    return result

        self.missing_lookups.append((mid, minute, period))
        return None
