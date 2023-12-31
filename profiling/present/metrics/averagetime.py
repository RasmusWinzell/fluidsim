from collections.abc import Callable
from typing import Any

from profiling.present.units import MicroSeconds, Unit
from profiling.profilers.stat import Stat

from .number_metric import NumberMetric


class AverageTime(NumberMetric):
    def __init__(
        self,
        name: str = "Average Time",
        unit: Unit = MicroSeconds(),
        format: str = "{:.3f}{}",
        just_func: Callable[[str], str] = None,
    ):
        super().__init__(name, unit, format, just_func)

    def __call__(self, stats: dict[Any, Stat]) -> dict[Any, float]:
        return {
            k: stat.total / stat.count if stat.count else 0 for k, stat in stats.items()
        }
