
# CBS Path Guard 및 Recovery 통합 가이드

## 1. 구현 목적

CBS가 반환한 경로를 실제 Simulator에서 실행하기 전에
Path Guard로 검증하고, 성공/실패 상태를 명확하게 반환한다.

## 2. 담당 구현 파일

- src/path_guard.py
  - CBS 경로 안전성 검사

- src/replanner.py
  - CBS 실행 후 Path Guard 호출

- src/replan_status.py
  - CBS 결과 상태 및 실행 시간 반환

- scripts/check_recovery_bridge.py
  - CBS 전체 경로의 Simulator 실행 검증

## 3. CBS Recovery 호출 방법

```python
from src.replan_status import replan_with_status

result = replan_with_status(
    current_positions,
    goals,
    grid_map,
    timeout_sec=30,
)
```

입력:
- current_positions: [(row, col), ...]
- goals: [(row, col), ...]
- grid_map: 0=이동 가능, 1=장애물
- Agent 순서는 current_positions와 goals에서 동일해야 함

반환:
- result.status
- result.paths
- result.latency_sec
- result.error

## 4. 상태 구분

- success: CBS 실행 및 Path Guard 검증 성공
- timeout: CBS 실행 제한 시간 초과
- no_solution: Solver가 해를 찾지 못했다고 보고
- invalid_path: Path Guard 검증 실패
- input_error: 입력값 오류
- solver_error: Solver 실행 또는 파일 처리 오류

success가 아닌 경우 paths=None이며,
해당 경로로 이동 명령을 보내면 안 됨.

timeout은 경로가 존재하지 않는다는 뜻이 아님.

## 5. Path Guard 검증 항목

- Agent ID 일치
- 시작점 및 목표점 일치
- 지도 범위와 장애물 검사
- 한 칸 이동 또는 WAIT만 허용
- Vertex Conflict 검사
- Edge Conflict 검사
- 경로 길이 일치

잘못된 이동을 WAIT으로 자동 변환하지 않음.

## 6. Agent ID 및 Timestep 규칙

Simulator Agent ID를 정렬한 뒤 CBS 입력 리스트 구성.

agent_ids = sorted(starts.keys())

current_positions = [
    starts[aid] for aid in agent_ids
]

goals_list = [
    goals[aid] for aid in agent_ids
]

CBS 반환 인덱스 0, 1, ...는
agent_ids의 순서에 대응함.

Path[0]은 CBS 호출 당시 실제 현재 위치.
Path[1]부터 새로운 이동 명령으로 실행.

Deadlock Trigger가 글로벌 t=5의 sim.step() 직후
발생한 경우 Path[0]은 t=5의 상태이며,
Path[1]은 t=6의 목표 위치임.

## 7. 통합 실행 원칙

1. IL Action을 Simulator에서 실행
2. 실행 이후 Deadlock Monitor 확인
3. Trigger 발생 시 실제 현재 위치 획득
4. replan_with_status() 호출
5. success인 경우에만 CBS 경로 실행
6. 매 timestep에서 실제 위치와 계획 위치 비교
7. Recovery 완료 후 합의된 규칙에 따라 IL 복귀

CBS 실패 시 이동 명령을 차단해야 하며,
실제 안전 정지 정책은 통합 단계에서 별도로 구현 필요.

## 8. 검증 결과

- Path Guard 관련 테스트 17개 PASS
- CBS-Path Guard 연결 테스트 2개 PASS
- CBS 상태 분류 테스트 6개 PASS
- 전체 회귀 테스트 25/25 PASS
- 실제 CBS Solver 호출 성공
- 3×3, 2-Agent 사례에서 전체 CBS 경로 실행 성공
- 최종 목표 도착: True
- 계획 경로와 Simulator 실제 위치 일치

## 9. 아직 미완료인 작업

- Deadlock Monitor의 실제 Trigger 자동 연결
- IL → CBS → IL 전환
- CBS 실패 시 실행 차단 및 안전 정지 통합
- 통합 시 Agent ID 및 timestep 계약 교차 검증
- 대표 Trigger 시나리오에서 자동 Recovery 재검증

독립 Recovery 성공을 전체 Hybrid 통합 완료로
표기하지 않음.
