from profiling.profilers import Profiler


class Presenter:
    PROFILE_SEPARATOR = "-" * 80

    def __init__(self, *profilers: Profiler):
        self.profilers = [
            inner_profiler for profiler in profilers for inner_profiler in profiler
        ]
        self.lines = None

    def present(self, print_in_console: bool = True):
        self.lines = []
        for profiler in self.profilers:
            self._write(self.PROFILE_SEPARATOR)
            if not profiler:
                self._write(f"{profiler.name} - No data")
                continue
            self._write(profiler.name)
            self._present_profiler(profiler)
        self._write(self.PROFILE_SEPARATOR)

        text = "\n".join(self.lines)
        if print_in_console:
            print(text)
        return text

    def _write(self, line: str):
        self.lines.append(line)

    def _present_profiler(self, profiler: Profiler):
        raise NotImplementedError
