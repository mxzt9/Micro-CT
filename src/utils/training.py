from __future__ import annotations

from dataclasses import dataclass


class MetricTracker:
    """Накопитель средних значений метрик для train/val циклов."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, name: str, value: float, n: int = 1) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + float(value) * n
        self.counts[name] = self.counts.get(name, 0) + n

    def avg(self, name: str, default: float = 0.0) -> float:
        count = self.counts.get(name, 0)
        if count == 0:
            return default
        return self.totals[name] / count

    def as_dict(self) -> dict[str, float]:
        return {name: self.avg(name) for name in self.totals}

    def postfix(self, *names: str) -> dict[str, str]:
        keys = names or tuple(self.totals.keys())
        return {name: f"{self.avg(name):.4f}" for name in keys if name in self.totals}


@dataclass
class EarlyStopping:
    """Ранняя остановка по выбранной валидационной метрике."""

    patience: int = 5
    min_delta: float = 0.0
    mode: str = "min"
    best: float | None = None
    bad_epochs: int = 0
    stopped_epoch: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("mode должен быть 'min' или 'max'")
        if self.patience < 1:
            raise ValueError("patience должен быть положительным")

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float, epoch: int | None = None) -> bool:
        improved = self.is_improvement(value)
        if improved:
            self.best = float(value)
            self.bad_epochs = 0
            self.stopped_epoch = None
            return True

        self.bad_epochs += 1
        if self.should_stop:
            self.stopped_epoch = epoch
        return False
