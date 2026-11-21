from pathlib import Path, PureWindowsPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_saved_scenario_path(path):
    win_path = PureWindowsPath(str(path))

    if not win_path.is_absolute() or not win_path.drive:
        return Path(path)

    parts = win_path.parts

    if "MAPF_2026_ALPHAProject" not in parts:
        return Path(path)

    idx = parts.index("MAPF_2026_ALPHAProject")
    relative = parts[idx + 1:]

    if not relative or relative[0] != "scenarios":
        return Path(path)

    scenarios_root = (PROJECT_ROOT / "scenarios").resolve()
    local_path = PROJECT_ROOT.joinpath(*relative).resolve()

    if local_path.is_relative_to(scenarios_root) and local_path.is_file():
        return local_path

    return Path(path)
