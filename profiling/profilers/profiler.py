import inspect
from collections import defaultdict
from typing import Any, Self

from .stat import Stat


class Profiler:
    SOURCE_VAR_NAME = "__source__"
    SOURCE_START_VAR_NAME = "__source_start__"
    __profilers = defaultdict(lambda: Profiler())

    def __init__(self, name: str = None):
        if name is None:
            name = "default"
        self._name = name
        self._root = Stat(0)
        self._current_stat = self.root

        Profiler.__profilers[name] = self

    @property
    def name(self) -> str:
        return self._name

    @property
    def root(self) -> Stat:
        return self._root

    @property
    def profilers(self) -> list[Self]:
        return [self]

    def start(self, name: Any):
        self._current_stat = self._current_stat.get_substat(name)
        self._current_stat.start()
        if self._current_stat.start_line_number is None:
            self._set_start_caller()

    def stop(self, name: Any):
        if not self._current_stat.name == name:
            raise Exception(
                f"Profiler: stop called on {name}, but current stat is {self._current_stat.name}"
            )
        self._current_stat.stop()
        if self._current_stat.end_line_number is None:
            self._set_stop_caller()
        self._current_stat = self._current_stat.parent

    def _set_start_caller(self):
        frame = self._get_caller(3)
        self._current_stat.start_line_number = frame.lineno + 1

    def _set_stop_caller(self):
        frame = self._get_caller(3)
        self._current_stat.end_line_number = frame.lineno - 1
        try:
            lines = inspect.getsource(frame.frame).splitlines(False)
            source = "\n".join(
                lines[self._current_stat.start_line_number - 1 : frame.lineno - 1]
            )
            self._current_stat.source = source
            self._current_stat.source_start = self._current_stat.start_line_number
        except OSError:
            f_globals = frame.frame.f_globals
            self._current_stat.source = f_globals[self.SOURCE_VAR_NAME]
            source_start = f_globals[self.SOURCE_START_VAR_NAME]
            self._current_stat.source_start = source_start
            self._current_stat.end_line_number += (
                self._current_stat.name - self._current_stat.start_line_number
            )
            self._current_stat.start_line_number = self._current_stat.name

    @staticmethod
    def _get_caller(back_frames: int = 2):
        frames = inspect.getouterframes(inspect.currentframe())
        frame = frames[back_frames]
        return frame

    @staticmethod
    def get(name: str):
        return Profiler.__profilers[name]

    def __copy__(self):
        prof = Profiler(self.name)
        prof._root = self.root
        return prof

    def __bool__(self):
        return any([not p.root.is_leaf for p in self.profilers])

    def __len__(self):
        return len(self.profilers)

    def __iter__(self):
        return iter(self.profilers)
