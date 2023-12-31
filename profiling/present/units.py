class Unit:
    def __init__(self, name: str = "", scale: float | int = 1):
        self._name = name
        self._scale = scale

    @property
    def name(self) -> str:
        return self._name

    @property
    def scale(self) -> float | int:
        return self._scale

    def __str__(self) -> str:
        return self.name


class FPS(Unit):
    def __init__(self):
        super().__init__("fps", 1)


class Seconds(Unit):
    def __init__(self):
        super().__init__("s", 1)


class MilliSeconds(Unit):
    def __init__(self):
        super().__init__("ms", 1000)


class MicroSeconds(Unit):
    def __init__(self):
        super().__init__("μs", 1000000)


class Percent(Unit):
    def __init__(self):
        super().__init__("%", 100)
