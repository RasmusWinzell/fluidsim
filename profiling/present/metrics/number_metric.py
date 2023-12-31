from collections.abc import Callable
from typing import Any

from profiling.present.units import Unit
from profiling.profilers.stat import Stat

from .metric import Metric


class NumberMetric(Metric):
    DEFAULT_FORMAT = "{}{}"
    DEFAULT_UNIT = Unit()

    def __init__(
        self,
        name: str,
        unit: Unit = None,
        format: str = None,
        just_func: Callable[[str], str] = None,
    ):
        super().__init__(name, format, just_func)
        self._unit = unit if unit else self.DEFAULT_UNIT

    @property
    def unit(self) -> Unit:
        return self._unit

    def _format(self, value: Any) -> str:
        return self._format_str.format(value * self.unit.scale, self.unit)

    def __call__(self, stats: dict[Any, Stat]) -> dict[Any, float]:
        raise NotImplementedError
