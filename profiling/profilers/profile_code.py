import keyword

from .parenthesis_counter import ParenthesisCounter


class ProfileCode:
    PROFILER_VAR_NAME = "_profiler"

    def __init__(self, code: str, start_line: int = 0):
        self.code = code
        self.start_line = start_line
        self.pc = ParenthesisCounter()
        self.new_lines = []
        self.add_profiling()
        self.profiled_code = "\n".join(self.new_lines)

    def add_profiling(self):
        """Add profiling code to the given code."""
        self.lines = self.code.split("\n")
        self.lines = self.remove_indents(self.lines)

        open_line = None

        for lineno, line in zip(
            range(self.start_line, self.start_line + len(self.lines)), self.lines
        ):
            print(lineno, line)
            if is_empty(line) or is_comment(line) or begins_with_keyword(line):
                self.append_line(line)
                continue

            indent = get_indentation(line)

            if self.pc.correct and open_line is None:
                self.append_start(indent, lineno)
                open_line = lineno

            self.append_line(line)

            if self.pc.correct and open_line is not None:
                self.append_stop(indent, open_line)
                open_line = None

    def append_line(self, line):
        self.pc.add_all(line)
        self.new_lines.append(line)

    def append_start(self, indent, line_number):
        self.new_lines.append(
            f"{indent}{ProfileCode.PROFILER_VAR_NAME}.start({line_number})"
        )

    def append_stop(self, indent, line_number):
        self.new_lines.append(
            f"{indent}{ProfileCode.PROFILER_VAR_NAME}.stop({line_number})"
        )

    @staticmethod
    def remove_indents(lines):
        indents = [get_indentation(line) for line in lines if line.strip()]
        shortest_indent = min(indents, key=len)
        lines = [
            line[len(shortest_indent) :] if len(line) > len(shortest_indent) else line
            for line in lines
        ]
        lines = [line.rstrip() for line in lines]
        return lines

    def __str__(self) -> str:
        return self.profiled_code


def begins_with_keyword(line):
    words = line.strip().split()
    if not words:
        return False
    return line.strip().split()[0] in keyword.kwlist


def is_comment(line):
    line = line.strip()
    return line.startswith("#") or line.startswith('"""') or line.startswith("'''")


def is_empty(line):
    return len(line.strip()) == 0


def get_indentation(line):
    return line.replace(line.lstrip(), "")
