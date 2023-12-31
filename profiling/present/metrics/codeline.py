from collections.abc import Callable
from typing import Any

from profiling.profilers.stat import Stat

from .metric import Metric


class CodeLine(Metric):
    def __init__(
        self,
        name: str = "Code",
        format: str = None,
        just_func: Callable[[str], str] = str.ljust,
    ):
        super().__init__(name, format, just_func)

    def __call__(self, stats: dict[Any, Stat]) -> dict[Any, str]:
        stat = [stat for stat in stats.values() if stat.source_start is not None][0]
        return {stat.source_start + i: line for i, line in enumerate(stat.all_lines)}
