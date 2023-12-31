from collections import defaultdict
from collections.abc import Callable
from typing import Any

from profiling.profilers.stat import Stat

from .metrics import Metric


def metrics(stats: dict[Any, Stat], *metrics: Metric) -> dict[str, dict[Any, Any]]:
    metric_values = {metric.name: metric(stats) for metric in metrics}
    return metric_values


def formatted_metrics(
    stats: dict[Any, Stat], *metrics: Metric
) -> dict[str, dict[Any, str]]:
    metric_values = {metric.name: metric.format(stats) for metric in metrics}
    return metric_values


def format_column(
    header: str, fstats: dict[Any, str], just_fun: Callable[[str], str] = str.ljust
) -> tuple[str, dict[Any, str]]:
    fstat_len = max(len(fstat) for fstat in fstats.values())
    column_width = max(len(header), fstat_len)
    header = just_fun(header, column_width)
    fstats = {k: just_fun(v, column_width) for k, v in fstats.items()}
    return header, fstats


def update_root(root: Stat):
    stats = flatten(root)
    total = sum(stat.total for stat in stats.values()) - root.total
    root.set(total)


def flatten(root: Stat) -> dict[int, Stat]:
    stats = defaultdict(list)

    def _flatten(stat):
        stats[stat.start_line_number].append(stat)
        for substat in stat.substats.values():
            _flatten(substat)

    _flatten(root)

    stats = {k: sum(v) for k, v in stats.items()}
    return stats
