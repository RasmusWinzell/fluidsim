from collections.abc import Callable
from typing import Any

from profiling.present.units import Unit
from profiling.profilers.stat import Stat

from .number_metric import Metric


class Name(Metric):
    def __init__(
        self,
        name: str = "Key",
        format: str = "{}",
        just_func: Callable[[str], str] = str.rjust,
    ):
        super().__init__(name, format, just_func)

    def __call__(self, stats: dict[Any, Stat]) -> dict[Any, int]:
        return {k: v.name for k, v in stats.items()}
