
from typing import Mapping, Sequence

GridLocation = tuple[int, int]


class InvalidPathError(ValueError):
    """잘못된 CBS 경로에 대한 오류."""
    pass


def validate_path_movements(
    paths: Mapping[int, Sequence[GridLocation]]
) -> None:

    if not paths:
        raise InvalidPathError("Path is empty")

    for agent_id, path in paths.items():

        if not path:
            raise InvalidPathError(
                f"Agent {agent_id}: empty path"
            )

        for t in range(len(path) - 1):
            current = path[t]
            next_pos = path[t + 1]

            dr = abs(next_pos[0] - current[0])
            dc = abs(next_pos[1] - current[1])

            distance = dr + dc

            if distance > 1:
                raise InvalidPathError(
                    f"Agent {agent_id}, t={t}->{t+1}: "
                    f"illegal movement {current} -> {next_pos}"
                )




def validate_path_endpoints(
    paths: Mapping[int, Sequence[GridLocation]],
    current_positions: Mapping[int, GridLocation],
    goals: Mapping[int, GridLocation],
) -> None:

    expected_ids = set(current_positions.keys())

    if set(goals.keys()) != expected_ids:
        raise InvalidPathError(
            "Current positions and goals have different Agent IDs"
        )

    if set(paths.keys()) != expected_ids:
        raise InvalidPathError(
            "CBS paths have missing or unexpected Agent IDs"
        )

    for agent_id in sorted(expected_ids):
        path = paths[agent_id]

        if not path:
            raise InvalidPathError(
                f"Agent {agent_id}: empty path"
            )

        if path[0] != current_positions[agent_id]:
            raise InvalidPathError(
                f"Agent {agent_id}: incorrect start position"
            )

        if path[-1] != goals[agent_id]:
            raise InvalidPathError(
                f"Agent {agent_id}: incorrect goal position"
            )




def validate_path_map_and_conflicts(
    paths: Mapping[int, Sequence[GridLocation]],
    grid_map,
) -> None:

    if not paths:
        raise InvalidPathError("Paths are empty")

    # 1. 모든 로봇의 경로 길이 검사
    lengths = [len(path) for path in paths.values()]

    if any(length == 0 for length in lengths):
        raise InvalidPathError("Empty agent path")

    if len(set(lengths)) != 1:
        raise InvalidPathError("Path lengths do not match")

    # 2. 지도 크기 확인
    height = len(grid_map)
    if height == 0:
        raise InvalidPathError("Empty map")

    width = len(grid_map[0])
    if width == 0 or any(len(row) != width for row in grid_map):
        raise InvalidPathError("Invalid map shape")

    agent_ids = sorted(paths.keys())
    total_steps = lengths[0]

    for t in range(total_steps):

        occupied = {}

        for agent_id in agent_ids:
            row, col = paths[agent_id][t]
            pos = (row, col)

            # 3. 지도 범위 검사
            if not (0 <= row < height and 0 <= col < width):
                raise InvalidPathError(
                    f"Agent {agent_id}, t={t}: out of bounds {pos}"
                )

            # 4. 장애물 검사
            if grid_map[row][col] == 1:
                raise InvalidPathError(
                    f"Agent {agent_id}, t={t}: obstacle at {pos}"
                )

            # 5. Vertex Conflict 검사
            if pos in occupied:
                raise InvalidPathError(
                    f"Vertex conflict at t={t}: "
                    f"Agent {occupied[pos]} and {agent_id}"
                )

            occupied[pos] = agent_id

        # 6. Edge Conflict 검사
        if t < total_steps - 1:
            for i, agent_a in enumerate(agent_ids):
                for agent_b in agent_ids[i + 1:]:

                    a_now = tuple(paths[agent_a][t])
                    a_next = tuple(paths[agent_a][t + 1])

                    b_now = tuple(paths[agent_b][t])
                    b_next = tuple(paths[agent_b][t + 1])

                    if a_now == b_next and b_now == a_next:
                        raise InvalidPathError(
                            f"Edge conflict at t={t}->{t+1}: "
                            f"Agent {agent_a} and {agent_b}"
                        )



def validate_path_guard(
    paths: Mapping[int, Sequence[GridLocation]],
    current_positions: Mapping[int, GridLocation],
    goals: Mapping[int, GridLocation],
    grid_map,
) -> None:
    """CBS 경로의 모든 안전 조건을 검사한다."""

    # 1. Agent ID, 시작점, 목표점 검사
    validate_path_endpoints(
        paths, current_positions, goals
    )

    # 2. 점프, 대각선 이동 검사
    validate_path_movements(paths)

    # 3. 지도, 장애물, 충돌, 경로 길이 검사
    validate_path_map_and_conflicts(
        paths, grid_map
    )
