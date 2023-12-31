from collections.abc import Callable
from typing import Any

from profiling.present.units import Seconds, Unit
from profiling.profilers.stat import Stat

from .number_metric import NumberMetric


class TotalTime(NumberMetric):
    def __init__(
        self,
        name: str = "Total Time",
        unit: Unit = Seconds(),
        format: str = None,
        just_func: Callable[[str], str] = None,
    ):
        super().__init__(name, unit, format, just_func)

    def __call__(self, stats: dict[Any, Stat]) -> dict[Any, float | int]:
        return {k: stat.total for k, stat in stats.items()}
