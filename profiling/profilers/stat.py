import time
from copy import copy
from typing import Any, Self


class Stat:
    def __init__(
        self, name: Any, parent: Self = None, total: float = 0, count: int = 0
    ):
        self.total = total
        self.count = count
        self.name = name

        self.start_line_number = None
        self.end_line_number = None
        self.source = None
        self.source_start = None

        self._start = 0

        self.parent = parent
        self.substats = {}

    @property
    def all_lines(self) -> list[str] | None:
        if self.source is None:
            return None
        return self.source.splitlines(False)

    @property
    def lines(self) -> list[str] | None:
        if self.source is None:
            return None
        return self.all_lines[
            self.start_line_number
            - self.source_start : self.end_line_number
            - self.source_start
            + 1
        ]

    @property
    def is_leaf(self) -> bool:
        return len(self.substats) == 0

    def get_substat(self, name: Any) -> Self:
        if name not in self.substats:
            stat = Stat(name, parent=self)
            self.substats[name] = stat
        return self.substats[name]

    def set(self, value: float) -> None:
        self.total = value
        self.count = 1

    def add(self, value: float) -> None:
        self.total += value
        self.count += 1

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> None:
        self.add(time.perf_counter() - self._start)

    def __str__(self):
        return f"{self.name}: {self.total} ({self.count})"

    def __copy__(self):
        stat = Stat(self.name)
        stat.__dict__.update(self.__dict__)
        return stat

    def __add__(self, other: Self) -> Self:
        if isinstance(other, Stat):
            new_stat = copy(self)
            new_stat.total += other.total
            new_stat.count += other.count
            return new_stat
        else:
            return self

    def __radd__(self, other: Self) -> Self:
        return self.__add__(other)
