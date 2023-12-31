from collections.abc import Callable
from typing import Any

from profiling.profilers.stat import Stat


class Metric:
    DEFAULT_FORMAT = "{}"

    def __init__(
        self, name: str, format: str = None, just_func: Callable[[str], str] = None
    ):
        self._name = name
        self._format_str = format if format else self.DEFAULT_FORMAT
        self.just_func = just_func if just_func else str.center

    @property
    def name(self) -> str:
        return self._name

    def format(self, stats: dict[Any, Stat]) -> dict[Any, str]:
        return {k: self._format(v) for k, v in self(stats).items()}

    def _format(self, value: Any) -> str:
        return self._format_str.format(value)

    def __call__(self, stats: dict[Any, Stat]) -> dict[Any, Any]:
        raise NotImplementedError
