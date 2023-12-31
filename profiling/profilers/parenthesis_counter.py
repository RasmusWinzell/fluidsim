class ParenthesisType:
    REGULAR = 0
    BRACKET = 1
    BRACE = 2


class ParenthesisCounter:
    def __init__(self):
        self.count = [0] * 3

    @property
    def correct(self):
        return (
            self.count[ParenthesisType.REGULAR] == 0
            and self.count[ParenthesisType.BRACKET] == 0
            and self.count[ParenthesisType.BRACE] == 0
        )

    def add_all(self, line: str):
        for char in line:
            self.add(char)

    def add(self, char: str):
        match char:
            case "(":
                self.count[ParenthesisType.REGULAR] += 1
            case ")":
                self.count[ParenthesisType.REGULAR] -= 1
            case "[":
                self.count[ParenthesisType.BRACKET] += 1
            case "]":
                self.count[ParenthesisType.BRACKET] -= 1
            case "{":
                self.count[ParenthesisType.BRACE] += 1
            case "}":
                self.count[ParenthesisType.BRACE] -= 1
