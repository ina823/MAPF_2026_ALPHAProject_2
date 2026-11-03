import numpy as np

try:
    # IL 브랜치가 main에 merge되면 이 import가 정상 동작한다.
    from dagger import MAPFSimulator
except ImportError:  # pragma: no cover - IL 브랜치 merge 전에는 항상 이쪽을 탄다.
    class MAPFSimulator:  # type: ignore[no-redef]
        """dagger.py가 아직 없을 때를 위한 임시 대체.
        merge 후에는 위 try 블록의 실제 ABC로 자동 교체된다."""
        pass


class Simulator:
    """
    [ 좌표계 표준 — map_generator / cbs_adapter와 동일하게 통일 ]
      이 클래스의 공개 함수는 항상 내부 표준 (row, col)만 받는다.
      과거 버전은 {'x':.., 'y':..} 딕셔너리를 받았지만,
      wall_channel 인덱싱이 이미 [row, col] 순서였기 때문에
      사실상 이름만 x/y로 잘못 붙어 있었을 뿐이다. 여기서는 이를 바로잡는다.

    [ 역할 ]
      완성된 CBS 경로 전체를 한 번에 검증하는 배치 검증기.
      한 스텝씩 진행하면서 상호작용하는 DAgger용 시뮬레이터는
      아래 MAPFStepSimulator를 참고 (역할이 근본적으로 달라 클래스를 분리함).
    """

    def __init__(self):
        # 행동 정의 약속 (IL spec.py v0.2 표준과 통일: 0=상, 1=하, 2=좌, 3=우, 4=대기)
        # (drow, dcol) 기준: 상=row 감소, 하=row 증가, 좌=col 감소, 우=col 증가
        self.actions = {
            0: (-1, 0),   # Up    : row -1
            1: (1, 0),    # Down  : row +1
            2: (0, -1),   # Left  : col -1
            3: (0, 1),    # Right : col +1
            4: (0, 0),    # Wait
        }

    @staticmethod
    def build_grid_3d(grid_map: np.ndarray) -> np.ndarray:
        """grid_map(2D, row,col) -> grid_3d(3, H, W). 0번 채널만 벽으로 사용.
        (다른 채널은 추후 동적 장애물 등 확장을 위해 남겨둠)"""
        h, w = grid_map.shape
        grid_3d = np.zeros((3, h, w), dtype=np.int64)
        grid_3d[0] = grid_map
        return grid_3d

    def _get_agent_pos_at_t(self, path, t):
        """해당 에이전트의 경로에서 t 시점의 (row, col) 좌표를 안전하게 가져오는 헬퍼 함수"""
        if t < len(path):
            return path[t]
        else:
            return path[-1]

    def validate_and_parse_paths(self, grid_3d, paths):
        """
        단일 맵과 CBS 결과(경로)를 받아 검증 후 정수형 행동 배열을 반환합니다.

        - grid_3d: (3, H, W) Numpy Array (0번 채널: 벽)
        - paths: {agent_id(int): [(row, col), ...], ...}
                 -> map_generator/cbs_adapter와 동일한 내부 표준 (row,col) 포맷
        """
        if not paths:
            return None, False

        wall_channel = grid_3d[0]
        height, width = wall_channel.shape

        agent_ids = sorted(paths.keys())
        num_agents = len(agent_ids)

        # 전체 경로 중 가장 긴 시간(max_t) 계산
        max_t = max(len(paths[aid]) for aid in agent_ids)
        max_time_step = max_t - 1

        # 모방학습용 정수형 행동 배열 초기화 (Shape: 시간, 로봇수) -> 초기값 4(대기)
        # (IL spec v0.2 표준: 0=상, 1=하, 2=좌, 3=우, 4=대기)
        action_array = np.full((max_time_step, num_agents), 4, dtype=np.int64)

        # 시간축(t)을 돌면서 검증 및 액션 추출
        for t in range(max_t):
            current_positions = []

            for idx, agent_id in enumerate(agent_ids):
                path = paths[agent_id]
                agent_name = f"agent{agent_id}"

                # 로봇의 현재(t) 위치 획득 (row, col)
                curr_row, curr_col = self._get_agent_pos_at_t(path, t)

                # 1. 맵 경계 탈출 및 벽 충돌 검증
                if curr_row < 0 or curr_row >= height or curr_col < 0 or curr_col >= width:
                    print(f"❌ [에러] t={t}, {agent_name}이 맵 범위를 벗어났습니다: (row={curr_row}, col={curr_col})")
                    return None, False

                if wall_channel[curr_row, curr_col] == 1:
                    print(f"❌ [에러] t={t}, {agent_name}이 벽(row={curr_row}, col={curr_col})에 충돌했습니다!")
                    return None, False

                # 2. 위치(Vertex) 충돌 검증 (같은 시간, 같은 칸에 두 마리가 있는지)
                if (curr_row, curr_col) in current_positions:
                    print(f"❌ [에러] t={t}, 위치(row={curr_row}, col={curr_col})에서 로봇 간 충돌 발생!")
                    return None, False
                current_positions.append((curr_row, curr_col))

                # 3. 모방학습용 Action 인덱스 계산 (t -> t+1 이동)
                if t < max_time_step and t < len(path) - 1:
                    next_row, next_col = path[t + 1]
                    drow = next_row - curr_row
                    dcol = next_col - curr_col

                    action = 4  # 기본값: 대기 (IL spec v0.2)
                    if drow == -1 and dcol == 0:  action = 0  # 상
                    elif drow == 1 and dcol == 0: action = 1  # 하
                    elif drow == 0 and dcol == -1: action = 2  # 좌
                    elif drow == 0 and dcol == 1:  action = 3  # 우

                    action_array[t, idx] = action

            # 4. 엣지(Edge) 충돌 검증 (t -> t+1 이동 시 서로 크로스(엇갈림) 되는지 체크)
            if t < max_time_step:
                for i in range(num_agents):
                    for j in range(i + 1, num_agents):
                        agent_a_id, agent_b_id = agent_ids[i], agent_ids[j]
                        path_a, path_b = paths[agent_a_id], paths[agent_b_id]

                        a_curr = self._get_agent_pos_at_t(path_a, t)
                        b_curr = self._get_agent_pos_at_t(path_b, t)
                        a_next = self._get_agent_pos_at_t(path_a, t + 1)
                        b_next = self._get_agent_pos_at_t(path_b, t + 1)

                        # A의 현재가 B의 다음 위치고, A의 다음 위치가 B의 현재 위치라면 엇갈림 충돌 발생!
                        if a_curr == b_next and a_next == b_curr:
                            print(f"❌ [에러] t={t} -> t={t+1}, agent{agent_a_id}와 agent{agent_b_id}가 "
                                  f"서로 교차(Edge) 충돌했습니다! {a_curr} ↔ {b_curr}")
                            return None, False

        return action_array, True


# ── DAgger용 step 단위 시뮬레이터 (dagger.py의 MAPFSimulator ABC 구현체) ──
#
# [ 배경 ]
#   위 Simulator는 "완성된 CBS 경로 전체를 한 번에 검증"하는 배치 검증기다
#   (validate_and_parse_paths). DAgger는 그것과 다르게 매 타임스텝마다
#   policy -> action -> 한 칸 이동 -> 그 상태에서 CBS expert라면 뭐라 할지
#   재라벨링, 하는 인터랙티브 롤아웃이 필요해서 별도 클래스로 둔다.
#
# [ 표준(v0.3) 관련 주의사항 ]
#   아래 CH_WALL / CH_OTHER_ROBOT / CH_OTHER_GOAL / ACTION_DELTA는 IL 브랜치의
#   spec.py 값을 그대로 옮겨 적은 것이다. spec.py가 아직 main에 merge되지
#   않아서 지금은 로컬 상수로 둘 수밖에 없다.
#
#   ⚠️ IL 브랜치가 main에 merge된 후에는 아래 상수 블록을 지우고 반드시
#      `from spec import ACTION_DELTA, CH_WALL, CH_OTHER_ROBOT, CH_OTHER_GOAL`
#      로 바꿔야 한다. spec.py가 바뀌면(v0.4 등) 여기서도 자동으로 따라가야
#      하는데, 지금처럼 값을 복사해두면 spec.py만 바뀌고 여기는 안 바뀌는
#      사고가 날 수 있다.

# TODO(IL merge 후 삭제): spec.py에서 import로 교체
_STEP_ACTION_DELTA = {
    0: (-1, 0),   # 상
    1: (1, 0),    # 하
    2: (0, -1),   # 좌
    3: (0, 1),    # 우
    4: (0, 0),    # 대기
}
_CH_WALL = 0          # 벽 / 장애물 / 맵 밖
_CH_OTHER_ROBOT = 1   # 다른 로봇의 현재 위치
_CH_OTHER_GOAL = 2    # 다른 agent의 목표 위치 (내 목표는 미포함)
_FOV_RADIUS = 2       # 5x5 -> 중심에서 반경 2


class MAPFStepSimulator(MAPFSimulator):
    """DAgger 롤아웃용 step 단위 시뮬레이터.

    벽/맵밖/충돌 처리 규칙(우리 팀 컨벤션):
      - 벽이나 맵 밖으로 이동을 시도하면: 그 자리에 그대로 머무른다(제자리 유지).
      - 두 에이전트가 같은 칸으로 이동하려 하면(vertex collision): 관련된
        에이전트 전부 이번 스텝은 이동을 취소하고 이전 위치에 머무른다.
      - 두 에이전트가 서로 자리를 바꾸려 하면(edge collision, swap): 마찬가지로
        둘 다 이번 스텝 이동을 취소한다.
      위 Simulator처럼 완성된 경로를 사후 검증하며 에러를 내는 게 아니라,
      "충돌이 나면 그 자리에 멈추게" 해서 롤아웃이 끝까지 진행되도록 한다
      (DAgger 학습 특성상 매 스텝 계속 진행돼야 하기 때문).
    """

    def __init__(self, cbs_solver_root: str, max_steps: int = 200):
        """
        cbs_solver_root: get_expert_actions에서 CBS 재호출 시 넘길 atb033 솔버 경로.
        max_steps      : 타임아웃 스텝 수 (도달 못 해도 done=True로 끝냄).
        """
        self._cbs_solver_root = cbs_solver_root
        self._max_steps = max_steps

        self._map_grid = None
        self._goals = {}
        self._positions = {}
        self._agent_ids = []
        self._t = 0

    # ── MAPFSimulator ABC 구현 ──────────────────────────────────────────

    def reset(self, map_grid: np.ndarray, starts: dict, goals: dict) -> dict:
        self._map_grid = np.asarray(map_grid)
        self._goals = dict(goals)
        self._positions = dict(starts)
        self._agent_ids = sorted(self._positions.keys())
        self._t = 0
        return self._build_obs()

    def step(self, actions: dict):
        height, width = self._map_grid.shape

        # 1) 각 에이전트의 "의도한" 다음 위치 계산 (벽/맵밖이면 제자리)
        intended = {}
        for agent_id in self._agent_ids:
            row, col = self._positions[agent_id]
            drow, dcol = _STEP_ACTION_DELTA[actions[agent_id]]
            new_row, new_col = row + drow, col + dcol

            out_of_bounds = not (0 <= new_row < height and 0 <= new_col < width)
            hits_wall = (not out_of_bounds) and self._map_grid[new_row, new_col] == 1
            if out_of_bounds or hits_wall:
                intended[agent_id] = (row, col)  # 제자리 유지
            else:
                intended[agent_id] = (new_row, new_col)

        # 2) Vertex / Edge Conflict를 반복 검사하여 연쇄 충돌 방지
        from collections import Counter

        while True:
            to_cancel = set()

            # Vertex Conflict: 같은 칸으로 이동하려는 로봇 탐지
            landing_counts = Counter(intended.values())
            for agent_id in self._agent_ids:
                if landing_counts[intended[agent_id]] > 1:
                    to_cancel.add(agent_id)

            # Edge Conflict: 서로 자리를 교환하려는 로봇 탐지
            for i, agent_a in enumerate(self._agent_ids):
                for agent_b in self._agent_ids[i + 1:]:
                    a_curr = self._positions[agent_a]
                    b_curr = self._positions[agent_b]
                    a_next = intended[agent_a]
                    b_next = intended[agent_b]

                    if (
                        a_next == b_curr
                        and b_next == a_curr
                        and a_next != a_curr
                    ):
                        to_cancel.update((agent_a, agent_b))

            # 더 이상 충돌이 없다면 이동 확정 가능
            if not to_cancel:
                break

            # 충돌한 로봇의 이동을 취소
            changed = False
            for agent_id in to_cancel:
                if intended[agent_id] != self._positions[agent_id]:
                    intended[agent_id] = self._positions[agent_id]
                    changed = True

            # 원래 위치 자체가 중복되는 등 해결 불가능한 상태
            if not changed:
                raise RuntimeError("Unresolvable collision in simulator state")


        self._positions = intended
        self._t += 1

        all_at_goal = all(
            self._positions[aid] == self._goals[aid] for aid in self._agent_ids
        )
        done = all_at_goal or self._t >= self._max_steps

        info = {"t": self._t, "all_at_goal": all_at_goal, "timed_out": self._t >= self._max_steps}
        return self._build_obs(), done, info

    def get_expert_actions(self, obs: dict) -> dict:
        """현재 위치 -> 목표로 CBS를 재호출해서, 각 에이전트의 다음 한 스텝을 추출한다.
        (obs 인자는 인터페이스 규격상 받지만, 내부적으로는 self._positions을
        정답 소스로 사용한다 — 매 스텝 CBS를 다시 도는 방식이라 obs를 다시 파싱할
        필요가 없음)

        NOTE: CBSAdapter.plan()은 {agent_id: (row,col)} dict가 아니라 순서가
        있는 리스트(인덱스 = 내부 agent 번호)를 받는다. 그래서 self._agent_ids
        순서로 리스트를 만들어 넘기고, 반환된 paths(0..n-1 인덱스 기준)를
        다시 실제 agent_id로 매핑해준다.
        """
        from src.cbs_adapter import CBSAdapter, CBSAdapterConfig

        starts_list = [self._positions[aid] for aid in self._agent_ids]
        goals_list = [self._goals[aid] for aid in self._agent_ids]

        adapter = CBSAdapter(CBSAdapterConfig(solver_root=self._cbs_solver_root))
        paths_by_index = adapter.plan(starts_list, goals_list, self._map_grid)

        expert_actions = {}
        for index, agent_id in enumerate(self._agent_ids):
            path = paths_by_index[index]
            cur_row, cur_col = path[0]
            if len(path) < 2:
                expert_actions[agent_id] = 4  # 대기 (이미 목표 또는 더 이상 못 감)
                continue
            next_row, next_col = path[1]
            drow, dcol = next_row - cur_row, next_col - cur_col

            action = 4  # 기본값: 대기
            for candidate_action, (d_row, d_col) in _STEP_ACTION_DELTA.items():
                if (d_row, d_col) == (drow, dcol):
                    action = candidate_action
                    break
            expert_actions[agent_id] = action

        return expert_actions

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────

    def _build_obs(self) -> dict:
        obs = {}
        for agent_id in self._agent_ids:
            obs[agent_id] = {
                "grid": self._extract_local_grid(agent_id),
                "goal_dir": self._goal_dir(agent_id),
            }
        return obs

    def _goal_dir(self, agent_id: int) -> np.ndarray:
        cur_row, cur_col = self._positions[agent_id]
        goal_row, goal_col = self._goals[agent_id]
        return np.array([goal_row - cur_row, goal_col - cur_col], dtype=np.float32)

    def _extract_local_grid(self, agent_id: int) -> np.ndarray:
        """agent_id 중심 5x5x3 로컬 그리드. v0.3 채널 정의:
        ch0=벽/장애물/맵밖, ch1=다른 로봇 현재위치, ch2=다른 agent 목표위치.
        자기 자신의 위치/목표는 표시하지 않음(자기 목표는 goal_dir로 별도 제공).
        각 셀은 최대 하나의 채널만 1이 되도록 wall > other_robot > other_goal
        순서로 우선순위를 둔다.
        """
        height, width = self._map_grid.shape
        cur_row, cur_col = self._positions[agent_id]

        grid = np.zeros((3, 2 * _FOV_RADIUS + 1, 2 * _FOV_RADIUS + 1), dtype=np.float32)

        other_robot_positions = {
            self._positions[aid] for aid in self._agent_ids if aid != agent_id
        }
        other_goal_positions = {
            self._goals[aid] for aid in self._agent_ids if aid != agent_id
        }

        for d_row in range(-_FOV_RADIUS, _FOV_RADIUS + 1):
            for d_col in range(-_FOV_RADIUS, _FOV_RADIUS + 1):
                world_row, world_col = cur_row + d_row, cur_col + d_col
                local_row, local_col = d_row + _FOV_RADIUS, d_col + _FOV_RADIUS

                out_of_bounds = not (0 <= world_row < height and 0 <= world_col < width)
                if out_of_bounds or self._map_grid[world_row, world_col] == 1:
                    grid[_CH_WALL, local_row, local_col] = 1.0
                elif (world_row, world_col) in other_robot_positions:
                    grid[_CH_OTHER_ROBOT, local_row, local_col] = 1.0
                elif (world_row, world_col) in other_goal_positions:
                    grid[_CH_OTHER_GOAL, local_row, local_col] = 1.0
                # 셋 다 아니면 빈 공간 -> 세 채널 모두 0으로 유지

        return grid
