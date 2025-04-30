# utils/printer.py


class Printer:
    COLORS = {
        "reset": "\033[0m",
        "blue": "\033[94m",
        "yellow": "\033[93m",
        "green": "\033[92m",
        "red": "\033[91m",
        "cyan": "\033[96m",
        "magenta": "\033[95m",
        "bold": "\033[1m",
    }

    def __init__(self, name: str = "LOG"):
        self.name = name.upper()

    def _format(self, color: str, *args) -> str:
        prefix = f"{self.COLORS[color]}[{self.name}]{self.COLORS['reset']}"
        content = " ".join(str(arg) for arg in args)
        return f"{prefix} {self.COLORS[color]}{content}{self.COLORS['reset']}"

    def blue(self, *args):
        print(self._format("blue", *args))

    def yellow(self, *args):
        print(self._format("yellow", *args))

    def info(self, *args):
        print("INFO: ", self._format("cyan", *args))

    def green(self, *args):
        print(self._format("green", *args))

    def red(self, *args):
        print(self._format("red", *args))

    def cyan(self, *args):
        print(self._format("cyan", *args))

    def magenta(self, *args):
        print(self._format("magenta", *args))

    def bold(self, *args):
        print(self._format("bold", *args))
