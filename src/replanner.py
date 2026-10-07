from pathlib import Path
from typing import Any, Sequence

from src.cbs_adapter import (
    CBSAdapter,
    CBSAdapterConfig,
    GridLocation,
    AgentPath,
)


def replan(
    current_positions: Sequence[GridLocation],
    goals: Sequence[GridLocation],
    grid_map: Any,
    solver_root: str | Path | None = None,
    timeout_sec: int = 30,
) -> dict[int, AgentPath]:
    """
    현재 agent 위치를 새로운 start로 사용하여 CBS 재계획을 수행한다.

    Parameters
    ----------
    current_positions:
        현재 시점의 각 agent 위치.
        내부 좌표 표준인 (row, col)을 사용한다.

    goals:
        각 agent의 기존 목표 위치.
        (row, col) 형식이며 agent 순서는 current_positions와 동일해야 한다.

    grid_map:
        2D grid map.
        0 = free, 1 = obstacle.

    solver_root:
        atb033 CBS solver 위치.
        None이면 CBSAdapter 기본 경로를 자동 탐색한다.

    timeout_sec:
        CBS 최대 실행 시간.

    Returns
    -------
    dict[int, AgentPath]
        {
            agent_id: [(row, col), ...]
        }

        반환 path의 index 0은 현재 위치이며,
        이후 index가 새로운 CBS timestep에 해당한다.
    """

    if len(current_positions) != len(goals):
        raise ValueError(
            "current_positions와 goals의 agent 수가 다릅니다: "
            f"{len(current_positions)} != {len(goals)}"
        )

    config = CBSAdapterConfig(
        solver_root=Path(solver_root) if solver_root is not None else None,
        work_dir=Path("outputs/paths/replan"),
        timeout_sec=timeout_sec,
    )

    adapter = CBSAdapter(config)

    replanned_paths = adapter.plan(
        starts=current_positions,
        goals=goals,
        grid=grid_map,
    )

    return replanned_paths