"""Strategy registry and exports."""
from ..weibull import Strategy
from .betfair import BetfairStrategy
from .extended import (ConditionalCovariateStrategy,
                       SingleGammaPhiStrategy, CalibratedCovariateStrategy)
from .zou_strategy import ZouBayesianStrategy, ZouCalibratedStrategy
from .maia_strategy import MaiaStrategy, MaiaCalibratedStrategy

STRATEGIES = {
    "weibull_no_cov": SingleGammaPhiStrategy,
    "weibull": ConditionalCovariateStrategy,
    "weibull_calibd": CalibratedCovariateStrategy,
    "betfair": BetfairStrategy,
    "zou": ZouBayesianStrategy,
    "zou_calibd": ZouCalibratedStrategy,
    "maia": MaiaStrategy,
    "maia_calibd": MaiaCalibratedStrategy,
}


def get_strategy(name: str, **kw) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name](**kw)


def list_strategies() -> list:
    return list(STRATEGIES.keys())


__all__ = ["Strategy", "get_strategy", "list_strategies", "BetfairStrategy",
           "ConditionalCovariateStrategy", "SingleGammaPhiStrategy", "CalibratedCovariateStrategy"]
