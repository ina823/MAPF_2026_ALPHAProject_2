
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.replan_status import replan_with_status
from src.path_guard import InvalidPathError, validate_path_guard

# 프로젝트 내부 위치 표현: (row, col)
GridPosition = tuple[int, int]


@dataclass(frozen=True)
class CBSRuntimeRequest:
    # 실제 Simulator의 Agent ID 순서
    agent_ids: tuple[int, ...]

    # CBS에 전달할 현재 위치
    current_positions: tuple[GridPosition, ...]

    # CBS에 전달할 목표 위치
    goals: tuple[GridPosition, ...]


def build_cbs_request(
    current_positions: Mapping[int, GridPosition],
    goals: Mapping[int, GridPosition],
) -> CBSRuntimeRequest:

    # 1. 로봇이 하나도 없는 경우 차단
    if not current_positions:
        raise ValueError("No agents provided")

    # 2. 현재 위치와 목표의 Agent ID 일치 여부
    if set(current_positions) != set(goals):
        raise ValueError("Agent IDs do not match")

    # 3. Agent ID 정렬
    agent_ids = tuple(sorted(current_positions))

    # 4. 정렬된 ID 순서로 CBS 입력 구성
    ordered_positions = tuple(
        tuple(current_positions[aid])
        for aid in agent_ids
    )

    ordered_goals = tuple(
        tuple(goals[aid])
        for aid in agent_ids
    )

    # 5. 현재 위치가 서로 겹쳐 있으면 차단
    if len(set(ordered_positions)) != len(ordered_positions):
        raise ValueError("Overlapping current positions")

    # 6. 정리된 입력 반환
    return CBSRuntimeRequest(
        agent_ids=agent_ids,
        current_positions=ordered_positions,
        goals=ordered_goals,
    )


@dataclass(frozen=True)
class CBSRuntimeResult:
    status: str
    paths: dict[int, list[GridPosition]] | None
    latency_sec: float
    error: str | None = None


def run_cbs_runtime(
    current_positions: Mapping[int, GridPosition],
    goals: Mapping[int, GridPosition],
    grid_map: Any,
    timeout_sec: int = 30,
    solver_root: str | Path | None = None,
) -> CBSRuntimeResult:

    # 1. 입력 정리 및 Agent ID 저장
    try:
        request = build_cbs_request(
            current_positions,
            goals,
        )
    except (ValueError, TypeError) as exc:
        return CBSRuntimeResult(
            status="input_error",
            paths=None,
            latency_sec=0.0,
            error=str(exc),
        )

    # 2. 기존 CBS 재계획 함수 호출
    result = replan_with_status(
        current_positions=request.current_positions,
        goals=request.goals,
        grid_map=grid_map,
        solver_root=solver_root,
        timeout_sec=timeout_sec,
    )

    # 3. CBS 실패 시 경로 반환 금지
    if result.status != "success":
        return CBSRuntimeResult(
            status=result.status,
            paths=None,
            latency_sec=result.latency_sec,
            error=result.error,
        )

    if result.paths is None:
        return CBSRuntimeResult(
            status="invalid_path",
            paths=None,
            latency_sec=result.latency_sec,
            error="CBS success without paths",
        )

    # 4. 원래 Agent ID 복원 및 최종 안전 검증
    try:
        expected_indexes = set(range(len(request.agent_ids)))

        if set(result.paths) != expected_indexes:
            raise InvalidPathError("CBS Agent IDs do not match")

        paths_by_agent = {
            aid: [tuple(pos) for pos in result.paths[index]]
            for index, aid in enumerate(request.agent_ids)
        }

        validate_path_guard(
            paths=paths_by_agent,
            current_positions=current_positions,
            goals=goals,
            grid_map=grid_map,
        )

    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return CBSRuntimeResult(
            status="invalid_path",
            paths=None,
            latency_sec=result.latency_sec,
            error=str(exc),
        )

    # 5. 최종 결과 반환
    return CBSRuntimeResult(
        status="success",
        paths=paths_by_agent,
        latency_sec=result.latency_sec,
        error=None,
    )
