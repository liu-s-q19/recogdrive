from typing import Sequence


NAVIGATION_COMMANDS = ("turn left", "go straight", "turn right")


def resolve_navigation_command(high_command_one_hot: Sequence[float]) -> str:
    for index, value in enumerate(high_command_one_hot):
        if value == 1 and index < len(NAVIGATION_COMMANDS):
            return NAVIGATION_COMMANDS[index]

    return "unknown"
