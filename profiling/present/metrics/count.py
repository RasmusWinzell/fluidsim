from collections.abc import Callable
from typing import Any

from profiling.present.units import Unit
from profiling.profilers.stat import Stat

from .number_metric import NumberMetric


class Count(NumberMetric):
    def __init__(
        self,
        name: str = "Count",
        unit: Unit = Unit(),
        format: str = None,
        just_func: Callable[[str], str] = None,
    ):
        super().__init__(name, unit, format, just_func)

    def __call__(self, stats: dict[Any, Stat]) -> dict[Any, int]:
        return {k: stat.count for k, stat in stats.items()}
