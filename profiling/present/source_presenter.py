from profiling.profilers import Profiler
from profiling.profilers.stat import Stat

from .metrics import AverageTime, CodeLine, Count, LineNumber, Metric
from .presenter import Presenter
from .summerize import flatten, format_column, formatted_metrics, update_root


class SourcePresenter(Presenter):
    DEFAULT_METRICS = [AverageTime(), Count(), LineNumber(), CodeLine()]
    DEFAULT_FORMAT = "{} {} {}: {}"

    def __init__(
        self,
        *profilers: Profiler,
        metrics: list[Metric] = None,
        format: str = None,
        profiled_only: bool = False,
        leaves_only: bool = False
    ):
        super().__init__(*profilers)
        self.metrics = metrics or SourcePresenter.DEFAULT_METRICS
        self.profiled_only = profiled_only
        self.leaves_only = leaves_only
        self.format = format or self.DEFAULT_FORMAT

    def set_metrics(self, *metrics: Metric):
        self.metrics = metrics

    def _present_profiler(self, profiler: Profiler):
        update_root(profiler.root)
        stats = flatten(profiler.root)
        metric_values = formatted_metrics(stats, *self.metrics)
        fmv = {
            m.name: format_column(m.name, metric_values[m.name], m.just_func)
            for m in self.metrics
        }
        linenos = self._get_lines(stats)
        self._write(self._format_header(fmv))
        for lineno in linenos:
            self._write(self._format_line(lineno, fmv))

    def _format_header(self, fmv: dict[str, tuple[str, dict[int, str]]]):
        vals = [fmv[metric.name][0] for metric in self.metrics if metric.name in fmv]
        return self.format.format(*vals)

    def _format_line(self, lineno: int, fmv: dict[str, tuple[str, dict[int, str]]]):
        vals = [
            fmv[metric.name][1][lineno]
            if lineno in fmv[metric.name][1]
            else " " * len(fmv[metric.name][0])
            for metric in self.metrics
            if metric.name in fmv
        ]
        return self.format.format(*vals)

    def _get_lines(self, stat_lines: dict[int, Stat]) -> list[int]:
        if self.leaves_only:
            return sorted([line for line, stat in stat_lines.items() if stat.is_leaf])
        elif self.profiled_only:
            return sorted(list(stat_lines.keys()))
        else:
            stat = [
                stat for stat in stat_lines.values() if stat.source_start is not None
            ][0]
            return list(
                range(stat.source_start, stat.source_start + len(stat.all_lines))
            )
