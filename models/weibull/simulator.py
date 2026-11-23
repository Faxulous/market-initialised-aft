"""Strategy interface for match outcome prediction."""
from abc import ABC, abstractmethod
from typing import Dict


class Strategy(ABC):
    """Abstract base for prediction strategies."""

    name: str = "base"

    @abstractmethod
    def predict_probs(self, home, away, sd: int, rem: float, dev, n_sims: int = 1000, **kw) -> Dict[str, float]:
        """Return {"home", "draw", "away"} outcome probabilities."""
