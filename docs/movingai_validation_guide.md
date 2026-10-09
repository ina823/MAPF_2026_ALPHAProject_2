# MovingAI Benchmark Validation Guide

## 1. 목적

MovingAI `.map` / `.scen` benchmark instance를 Alpha2 내부 표준 `(row, col)` 형식으로 로딩하고, Alpha2의 IL-navhint policy를 `MAPFStepSimulator`에서 실행하여 validation input으로 사용할 수 있는지 확인한다.

이번 검증의 목적은 다음 전체 흐름이 정상적으로 연결되는지 확인하는 것이다.

```text
MovingAI .map / .scen
        ↓
MovingAI Loader
        ↓
Alpha2 (row, col)
        ↓
IL-navhint Policy
        ↓
MAPFStepSimulator
        ↓
Episode Result
        ↓
Common Logging
```

---

## 2. 관련 파일

### MovingAI Loader

```text
src/common/movingai_loader.py
```

역할:

- MovingAI `.map` 파일 로딩
- MovingAI `.scen` 파일 로딩
- 필요한 agent 수 선택
- MovingAI `(x, y)` 좌표를 Alpha2 `(row, col)` 좌표로 변환

### IL-only Benchmark Runner

```text
scripts/run_movingai_il_once.py
```

역할:

- MovingAI instance 로딩
- Alpha2 IL-navhint policy 실행
- `MAPFStepSimulator`에서 episode rollout
- success / timeout 판정
- episode log 저장

### Step Simulator

```text
src/simulator/mapf_step_simulator.py
```

주요 클래스:

```text
MAPFStepSimulator
Simulator
```

- `MAPFStepSimulator`
  - IL policy의 action을 timestep 단위로 실행
  - agent 위치 갱신
  - collision 처리
  - episode 종료 여부 판정

- `Simulator`
  - CBS path 검증
  - vertex / edge conflict 확인
  - CBS path를 Alpha2 action array로 변환

### IL Policy

```text
src/il/policy.py
```

Alpha2의 기존 `NavHintPolicy`를 사용한다.

### IL Model

```text
src/il/model_navhint.py
```

### Checkpoint

```text
checkpoints/cnn_navhint.pt
```

Alpha2에서 사용하는 IL-navhint baseline checkpoint이다.

### Episode Log

```text
outputs/logs/episode_log.csv
```

---

## 3. 좌표 및 Grid 규약

### MovingAI 좌표

MovingAI scenario는 다음 좌표 형식을 사용한다.

```text
(x, y)
```

의미:

```text
x = column
y = row
```

### Alpha2 내부 좌표

Alpha2 내부에서는 다음 형식으로 통일한다.

```text
(row, col)
```

MovingAI loader에서 다음과 같이 변환한다.

```text
MovingAI (x, y)
        ↓
Alpha2 (row, col)
```

예:

```text
MovingAI start = (1, 4)

→ Alpha2 start = (4, 1)
```

### Grid 규약

```text
0 = free
1 = obstacle
```

이 규약은 Alpha2 Simulator와 CBS Adapter에서 동일하게 사용한다.

---

## 4. Validation Instance

검증에 사용한 MovingAI instance:

```text
map:
empty-8-8.map

scenario:
empty-8-8-random-1.scen

agents:
4
```

현재 개발 PC 기준 benchmark 위치:

```text
C:\Users\LG\MAPF_workspace\mapf_bench
```

Map:

```text
C:\Users\LG\MAPF_workspace\mapf_bench\empty-8-8.map
```

Scenario:

```text
C:\Users\LG\MAPF_workspace\mapf_bench\scen-random\empty-8-8-random-1.scen
```

다른 PC에서 재현할 경우 MovingAI benchmark가 저장된 위치에 맞게 `--map-path`, `--scen-path`만 변경한다.

### Loader 검증 결과

```text
grid shape = (8, 8)
walls = 0
```

Starts:

```text
[(4, 1), (0, 1), (6, 1), (6, 4)]
```

Goals:

```text
[(7, 4), (2, 3), (7, 6), (1, 5)]
```

starts / goals가 Alpha2 내부 `(row, col)` 형식으로 정상 변환되는 것을 확인하였다.

---

## 5. IL-only 실행

프로젝트 루트에서 다음 명령으로 실행한다.

```powershell
python scripts\run_movingai_il_once.py `
--map-path C:\Users\LG\MAPF_workspace\mapf_bench\empty-8-8.map `
--scen-path C:\Users\LG\MAPF_workspace\mapf_bench\scen-random\empty-8-8-random-1.scen `
--agents 4 `
--max-steps 64
```

실행 시 Alpha2의 다음 구성요소를 사용한다.

```text
NavHintPolicy
        ↓
cnn_navhint.pt
        ↓
MAPFStepSimulator
```

즉 Alpha1에서 사용했던 `cnn_diverse.pt`나 별도의 `model_cnn.py`를 사용하는 것이 아니라, 현재 Alpha2의 공식 IL-navhint baseline을 사용한다.

---

## 6. Validation 실행 결과

Alpha2에서 실제 실행한 결과:

```text
checkpoint = cnn_navhint.pt
status = success
success = True
steps = 6
timed_out = False
```

Starts:

```text
{
    0: (4, 1),
    1: (0, 1),
    2: (6, 1),
    3: (6, 4)
}
```

Goals:

```text
{
    0: (7, 4),
    1: (2, 3),
    2: (7, 6),
    3: (1, 5)
}
```

Final positions:

```text
{
    0: (7, 4),
    1: (2, 3),
    2: (7, 6),
    3: (1, 5)
}
```

검증 결과:

```text
final_positions == goals = True
```

따라서 4개 agent 모두 6 timestep 이내에 각자의 목표에 도달하였다.

---

## 7. Logging

실행 결과는 다음 파일에 자동 추가된다.

```text
outputs/logs/episode_log.csv
```

현재 필드:

```text
timestamp
mode
instance
map_name
num_agents
status
episode_steps
episode_runtime_ms
episode_timed_out
```

실제 기록 예:

```text
mode = il_only
instance = empty-8-8-random-1
map_name = empty-8-8
num_agents = 4
status = success
episode_steps = 6
episode_runtime_ms = 14.592
episode_timed_out = False
```

단, `episode_runtime_ms`는 실행 환경에 따라 달라질 수 있으므로 단일 실행 시간을 성능 결론으로 사용하지 않는다.

동일 exact instance 반복 실행은 latency variance 확인 용도로만 사용한다.

---

## 8. Episode와 CBS Logging 구분

IL episode와 CBS planning의 시간 및 timeout은 서로 다른 값으로 관리한다.

### IL / Simulator

```text
episode_runtime_ms
episode_timed_out
```

### CBS

```text
cbs_latency_ms
cbs_timed_out
```

두 종류의 latency와 timeout은 의미가 다르므로 하나의 필드로 합치지 않는다.

예:

```text
episode_timed_out
= 전체 IL episode가 max_steps에 도달했는가

cbs_timed_out
= CBS solver가 설정된 planning timeout을 초과했는가
```

---

## 9. Validation / Test 원칙

현재 사용한 다음 instance는 validation 후보로 사용한다.

```text
empty-8-8-random-1.scen
```

최종 evaluation 전에는 validation과 test를 **start-goal instance 단위로 분리**한다.

원칙:

```text
Validation
→ parameter / threshold / interface 확인에 사용

Test
→ 최종 설정이 고정된 뒤에만 실행
```

test instance는 validation/test split이 freeze되기 전에 실행하지 않는다.

또한 deterministic IL에서는 동일한 exact instance를 여러 번 실행했다고 해서 서로 다른 success sample로 세지 않는다.

---

## 10. 현재 검증 상태

현재 Alpha2에서 다음 항목을 확인하였다.

```text
MovingAI .map 로딩 성공
MovingAI .scen 로딩 성공
(x, y) → (row, col) 변환 성공
Alpha2 NavHintPolicy 연결 성공
MAPFStepSimulator rollout 성공
4-agent episode success
final_positions == goals
episode logging 성공
```

따라서 MovingAI benchmark를 Alpha2의 IL-only validation input으로 사용하는 기본 pipeline은 정상 동작하는 상태이다.

아직 완료하지 않은 항목은 최종 validation/test instance 목록 및 split 고정이다.
