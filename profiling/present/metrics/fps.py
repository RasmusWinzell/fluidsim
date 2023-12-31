from collections.abc import Callable
from typing import Any

from profiling.present.units import FPS, Unit
from profiling.profilers.stat import Stat

from .number_metric import NumberMetric


class FPS(NumberMetric):
    def __init__(
        self,
        name: str = "FPS",
        unit: Unit = FPS(),
        format: str = "{:.1f}{}",
        just_func: Callable[[str], str] = None,
    ):
        super().__init__(name, unit, format, just_func)

    def __call__(self, stats: dict[Any, Stat]) -> dict[Any, float]:
        return {
            k: stat.count / stat.total if stat.count else 0 for k, stat in stats.items()
        }
