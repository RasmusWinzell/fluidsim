from collections.abc import Callable
from typing import Any

from profiling.present.units import Unit
from profiling.profilers.stat import Stat

from .number_metric import NumberMetric


class LineNumber(NumberMetric):
    def __init__(
        self,
        name: str = "Line Number",
        unit: Unit = None,
        format: str = "{}",
        just_func: Callable[[str], str] = str.rjust,
    ):
        super().__init__(name, unit, format, just_func)

    def __call__(self, stats: dict[Any, Stat]) -> dict[Any, int]:
        stat = [stat for stat in stats.values() if stat.source_start is not None][0]
        return {
            stat.source_start + i: stat.source_start + i
            for i, line in enumerate(stat.all_lines)
        }
