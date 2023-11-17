from typing import Any


def print_to_file(x: Any, file_name: str = "log.txt"):
    with open(file_name, "w") as f:
        print(x, file=f)
