from navsim.agents.recogdrive.command_utils import resolve_navigation_command


def test_resolve_navigation_command_handles_three_way_one_hot() -> None:
    assert resolve_navigation_command([0, 1, 0]) == "go straight"


def test_resolve_navigation_command_handles_fourth_bucket_as_unknown() -> None:
    assert resolve_navigation_command([0, 0, 0, 1]) == "unknown"
