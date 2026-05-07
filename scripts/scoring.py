from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CombinedScorer:
    metric_weight: float
    metric_anchor: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_weight", max(0.0, min(1.0, float(self.metric_weight))))
        object.__setattr__(self, "metric_anchor", max(float(self.metric_anchor), 1e-12))

    def score(self, metric_loss: float, feasibility: float) -> float:
        norm_loss = float(metric_loss) / self.metric_anchor
        return (1.0 - self.metric_weight) * (1.0 - norm_loss) + self.metric_weight * float(feasibility)
