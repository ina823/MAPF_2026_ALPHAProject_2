# ROS2/Gazebo 단일 로봇 v1: 독립 Watchdog, Odometry Safety, STOP_CHECK, 장애 주입 (2026-11-26)

> **범위 한정.** 이 문서는 v0(`ros2_gz_single_robot_v0.md`, 덮어쓰지 않음)의 후속 기록이다.
> 단일 데모 로봇(vehicle_blue)의 안전 장치를 시뮬레이터에서 검증하는 단계이며, Ground Truth 검증,
> 최종 성능 평가, 다중 로봇 검증이 아니다. **Watchdog 자체가 강제 종료될 때의 안전성은 확보되지 않았다 (7절 H2).**

- 브랜치: `feature/ina-ros2-gz-bridge` (기준 Commit `882f848`, 이후 변경은 모두 미커밋)
- 결과 표기 구분
  - **[Gazebo·사용자]**: 사용자가 실제 Gazebo에서 직접 실행하고 확인한 결과 (1절). 사용자 보고를 그대로 옮겼다.
  - **[자동화·Claude Code]**: 단위·회귀 테스트, 정적 검사, Gazebo 없는 모의 실행 (5절). **Gazebo를 실행하지 않았다.**
- 2절: 구조 설명, 3~4절: 장애 주입 도구와 STOP_CHECK 시간 제한, 6절: 실행 방법,
  7절: 미해결 후속 과제, 8절: 실험 기록 보존, 9절: 변경 파일

## 1. [Gazebo·사용자] 실제 Gazebo 실험 결과 (최종 코드, 2026-11-26)

- 위치·속도는 모두 DiffDrive Odometry 값이다. **Ground Truth 기반 검증이 아니다.**
- 실행은 항목별 1회(n=1)다.
- 근거 수준: 7번만 자동 기록(JSON, 8절)이 있다. 나머지 7개 항목은 **사용자가 터미널에서 직접 확인한 결과**이며,
  원본 터미널 로그는 보존되지 않았다. 아래 수치는 사용자 보고 값이고, 보고되지 않은 값은 적지 않았다.

| # | 실험 | 결과 | 근거 |
|---|---|---|---|
| 1 | 정상 1 m 직진 (분리 Bridge 환경) | 성공. 위치 오차 0.0063 m (Odometry), STOP_CONFIRMED, Executor 종료 코드 0 | 터미널 확인 |
| 2 | Executor SIGKILL | 종료 코드 137. Watchdog COMMAND_TIMEOUT 0.517 s → FAULT 잠금. STOP_CONFIRMED. Odometry 선속도·각속도 0 | 터미널 확인 |
| 3 | Fault 상태에서 재실행 | `WATCHDOG_FAULT:COMMAND_TIMEOUT`으로 출발 차단, 종료 코드 1 | 터미널 확인 |
| 4 | Watchdog Reset 후 정상 주행 | Reset 성공, READY 복귀, 정상 주행 및 STOP_CONFIRMED | 터미널 확인 |
| 5 | Executor SIGTERM | 종료 코드 143, 정지 요청 10회, Watchdog STOP_CONFIRMED, Odometry 선속도·각속도 0 | 터미널 확인 |
| 6 | Watchdog 미실행 | `WATCHDOG_UNAVAILABLE`로 출발 차단, 종료 코드 1 | 터미널 확인 |
| 7 | 이동 중 Odometry Bridge 장애 주입 (SIGKILL) | 아래 1.1 | **JSON + Executor 로그** |
| 8 | 이동 중 Clock Bridge 장애 주입 | 아래 1.2 | 터미널 확인 |

### 1.1 실험 7: Odometry 전용 Bridge 종료

- 이동 중 Odometry 전용 Bridge만 종료하고 cmd_vel Bridge는 유지했다 (cmd_vel Bridge 프로세스 생존·구독 유지 확인).
- Watchdog ACTIVE 이탈까지 1.299 s, 첫 정지 cmd_vel 관측까지 1.3264 s (주입 시점 기준, injector monotonic).
- Executor가 먼저 감지(`MOVE_ABORTED: ODOM_STAMP_STALE`)해 정지를 요청했다. 종료 코드 1.
- 그래서 Watchdog은 FAULT가 아닌 **WAITING**으로 전환했다 (fault=null, health `ODOM_STAMP_STALE` → `ODOM_RECEIVE_TIMEOUT`).
- STOP_CHECK: PENDING → `NO_FRESH_ODOM` (주입 +5.299 s). `physical_stop: NOT_VERIFIED_ODOMETRY_LOST`.
- 이탈 뒤 0이 아닌 cmd_vel 0건, 0 cmd_vel 109건 (ROS Topic 관측). 사용자가 Gazebo 쪽 cmd_vel에서 정지 명령을 관측했다.
- 종료 뒤 Odometry 수신 0건. injector 판정 8개 항목 모두 통과 (VERDICT PASS, injector 종료 코드 0).
- **Odometry 복구 후 Reset 없이 READY로 돌아가는 정책은 미해결이다 (7절 H1).**

### 1.2 실험 8: Clock 전용 Bridge 종료

- Watchdog ACTIVE, 출력 `[0.3, 0.0]`을 확인한 뒤 Clock Bridge를 종료했다.
- Executor `MOVE_ABORTED: ODOM_STAMP_AHEAD_OF_CLOCK`, 종료 코드 1.
- Watchdog ACTIVE → WAITING → FAULT. CLOCK_TIMEOUT과 TIME_BASE_MISMATCH가 관측됐다. 출력 `[0.0, 0.0]`.
- STOP_CHECK: `UNVERIFIABLE`. **무한 PENDING은 발생하지 않았다** (4절 개선의 실제 확인).
- 종료 뒤 Clock 발행자 0개, Odometry 발행자 1개.
- 수동 Odometry 조회에서 선속도·각속도 0을 확인했다. **독립적인 물리적 정지 검증(Ground Truth 등)은 하지 않았다.**
- TIME_BASE_MISMATCH는 실제 원인(Clock 정지)과 다른 이름의 Fault일 수 있다 (7절 M1).

### 1.3 이전 기록 (v1 STOP_CHECK 개선 이전 코드, 보존용)

v1 작업 전에 사용자가 실행한 결과다. 위 1절과 코드가 다르므로 최종 결과로 쓰지 않는다.

| # | 실험 | 결과 |
|---|---|---|
| 1 | 정상 1 m 이동 | PASS, Odometry 오차 0.0165 m |
| 2 | Executor SIGKILL | 종료 코드 137, COMMAND_TIMEOUT 0.517 s, Watchdog FAULT, STOP_CHECK CONFIRMED |
| 3 | Fault 상태 재출발 | 차단 |
| 4~5 | 명시적 Reset, Reset 이후 정상 이동 | PASS, Odometry 오차 0.0045 m |
| 6 | Executor SIGTERM | 종료 코드 143, 정지 요청 10회, Watchdog READY, fault=null, STOP_CHECK CONFIRMED |
| 7~8 | 정지 중 / 출발 준비 중 Clock 중단 | CLOCK_TIMEOUT, TIME_BASE_MISMATCH 감지 / 출발 차단, 종료 코드 1 |
| 9 | 이동 중 Clock 중단 | ODOM_STAMP_AHEAD_OF_CLOCK, ACTIVE→FAULT, Executor 종료 코드 1. **STOP_CHECK PENDING 유지** (Clock 복구 후 NO_FRESH_ODOM) |
| 10 | Watchdog 미실행 | WATCHDOG_UNAVAILABLE로 출발 차단, 종료 코드 1 |

9번의 PENDING 유지가 4절 STOP_CHECK 시간 제한의 직접 근거다.

## 2. 구조

### 2.1 전체 흐름

```
           (TwistStamped 이동 요청)                (Twist, 유일한 발행자)
Executor ───────────────────────────▶ Watchdog ─────────────────────────▶ cmd_vel Bridge (]) ─▶ Gazebo DiffDrive
   ▲   ◀──────── 상태 (String JSON) ──────┘ ▲                                                     │
   │                                         │                                                     │
   ├────────── Odometry ◀── Odometry Bridge ([) ◀──────────────────────────────────────────────────┤
   └────────── /clock   ◀── Clock Bridge    ([) ◀──────────────────────────────────────────────────┘
               (Executor 와 Watchdog 이 각각 독립적으로 구독·판정)
```

- **Executor** (`gz_executor_node.py`, `safe_motion.py`): 목표까지 제어 명령을 계산하지만 실제 cmd_vel을 발행하지 않는다.
  모든 속도는 Watchdog 이동 요청 Topic으로만 보낸다. 자체 Odometry Safety 감시기와 StopVerifier를 갖는다.
- **Watchdog** (`command_watchdog.py` 순수 로직, `gz_watchdog_node.py` ROS Node): 실제 cmd_vel을 결정하는 유일한 프로세스.
  요청 검증, Command Timeout, Odometry/Clock 감시, Fault 잠금, 정지 확인을 한다.
- 판단 로직은 모두 rclpy 의존 없는 순수 Python 모듈이며 단위 테스트 대상이다. ROS Node는 콜백과 발행만 맡는다.

### 2.2 Watchdog 상태 머신

상태는 저장값이 아니라 매번 다음 우선순위로 계산된다 (`CommandWatchdog.state`).

| 우선순위 | 상태 | 조건 | 출력 |
|---|---|---|---|
| 1 | SHUTDOWN | 종료 처리 중 | 정지 |
| 2 | FAULT | Fault 잠김 (Reset 전까지 유지) | 정지 |
| 3 | ACTIVE | 마지막 수락 명령이 이동 (0이 아님) | 요청 속도 (Command Timeout 이내일 때만) |
| 4 | WAITING | Odometry/Clock 상태가 정상이 아님 (NO_CLOCK, ODOM_RECEIVE_TIMEOUT 등) 또는 Reset 직후 | 정지 |
| 5 | READY | 정상, 이동 명령 없음 | 정지 |

주요 전이

- READY → ACTIVE: 유효한 이동 요청 수락.
- ACTIVE → READY/WAITING: 정지 요청 수락. **이때 상태가 정상이 아니어도 잠그지 않는다 (7절 H1).**
- ACTIVE → FAULT: 매 tick(0.05 s) 판정에서
  - 마지막 유효 요청 이후 `command_timeout_mono`(0.5 s) 초과 → `COMMAND_TIMEOUT`
  - Odometry/Clock 상태 이상 → 그 사유로 잠금 (예: `ODOM_STAMP_STALE`, `CLOCK_TIMEOUT`)
- 모든 상태 → FAULT: Odometry 감시기의 잠금 Fault(2.4), 요청 값 이상(`INVALID_VELOCITY`, `SPEED_LIMIT`, `UNUSED_TWIST_COMPONENT`).
- WAITING ↔ READY: 정지 상태에서 Odometry/Clock 상태 변화에 따라 자동 전환 (잠그지 않음).
- FAULT → WAITING: 명시적 Reset만 (`std_srvs/Trigger`). 이후 Odometry/Clock이 정상이면 READY.
- Reset 거부: 종료 중이거나, 어떤 요청이든(거부된 요청 포함) `command_timeout_mono` 안에 들어온 경우.
- Reset 처리: Odometry 감시기 상태·Fault, StopVerifier, Watchdog Clock, 소유권, seq, 마지막 명령을 모두 지우고 세션 epoch를 올린다.

### 2.3 Executor ↔ Watchdog 통신 프로토콜 (`watchdog_protocol.py`)

표준 메시지만 쓴다 (사용자 정의 msg 빌드 불필요). Topic 이름은 `make_robot_interface(robot_id)`가 만든다.

| 방향 | Topic / Service | 타입 | 내용 |
|---|---|---|---|
| Executor → Watchdog | `/mapf/<id>/cmd_vel_request` | `geometry_msgs/TwistStamped` | 이동 요청 |
| Watchdog → Gazebo | `/model/<id>/cmd_vel` | `geometry_msgs/Twist` | 실제 제어. Watchdog만 발행 |
| Watchdog → Executor | `/mapf/<id>/watchdog_status` | `std_msgs/String` (JSON) | 상태, Fault, 세션 token, 출력, STOP_CHECK 등 |
| 운영자 → Watchdog | `/mapf/<id>/watchdog_reset` | `std_srvs/Trigger` | Fault Reset |

이동 요청 형식

- `header.frame_id = "<session_token>#<sender_id>#<seq>"`
  - session_token = `<boot_id>-<epoch>`. Watchdog 시작마다 새 boot_id, Reset마다 epoch 증가.
  - sender_id: Executor 프로세스마다 새 임의 ID. 한 세션에서는 첫 유효 요청을 보낸 sender만 명령할 수 있다.
    소유 sender가 정지 상태로 Command Timeout 이상 조용하면 소유권이 풀린다.
  - seq: sender별로 엄격히 증가.
- `header.stamp` = Executor가 마지막으로 받은 /clock (sim). 큐에 지연된 요청 차단용.
- `twist.linear.x`, `twist.angular.z`만 사용. 나머지 성분은 0이어야 한다.

Watchdog의 요청 검사 순서: 종료 중 → ID 형식 → 현재 세션 token → Fault 상태 → 소유 sender → seq 증가 →
stamp와 Watchdog /clock 차이 `request_max_age_sim`(0.5 s) 이내 → 값 유한(위반 시 잠금) → 미사용 성분 0(위반 시 잠금) →
속도 상한 0.3 m/s, 0.5 rad/s(위반 시 잠금) → 상태가 정상이 아니면 이동 요청 거부(정지 요청은 수락) → 전달.

Executor 측 판정 (`WatchdogClient.check`, 이동 시작 전과 이동 중 매 주기)

- 상태 메시지 없음 `WATCHDOG_UNAVAILABLE`, 나이 > `status_timeout_mono`(0.5 s) `WATCHDOG_STATUS_TIMEOUT`,
  robot_id 불일치, `WATCHDOG_FAULT:<사유>`, READY/ACTIVE가 아님 `WATCHDOG_NOT_READY:<상태>`,
  이동 시작 때 고정한 token과 다름 `WATCHDOG_SESSION_CHANGED`.
- 이동 중 위 판정이 실패하면 Odometry 이상과 같게 정지 요청 후 Executor 측 감시기를 잠그고 종료한다.

QoS (`qos.py`): 모두 KEEP_LAST depth 1, VOLATILE. Odometry·cmd_vel·이동 요청·상태는 RELIABLE, /clock은 BEST_EFFORT.

### 2.4 Odometry Safety (`odom_safety.py`, `odom_monitor.py`)

시간 기준을 섞지 않는다. `*_mono`는 수신 측 `time.monotonic()`, `*_sim`은 /clock과 `header.stamp`다.
수신 공백은 mono끼리, 생성 시각 나이는 sim끼리만 비교한다.

즉시 잠기는 Fault (Reset 전까지 유지)

| Fault | 조건 |
|---|---|
| `INVALID_CLOCK` | /clock 값이 유한하지 않거나 음수 |
| `SIM_TIME_REGRESSION` | /clock 역행 |
| `INVALID_FRAME` | `frame_id`/`child_frame_id`가 `<id>/odom`, `<id>/chassis`와 다름 |
| `INVALID_STAMP` | stamp가 유한하지 않거나 음수 |
| `TIME_BASE_MISMATCH` | stamp가 마지막 /clock보다 `odom_stamp_max_age_sim` 넘게 앞섬 (예: 벽시계 stamp) |
| `INVALID_POSITION`, `INVALID_QUATERNION`, `INVALID_TWIST` | NaN/Inf, quaternion norm 오차 > 0.05 |
| `POSITION_JUMP` | 연속 두 메시지 거리 > `jump_max_speed × dt + jump_margin` |

중복·역순 stamp는 무시하고 수신 시각을 갱신하지 않는다.

비잠금 상태 판정 (`check()`, 이동 중이면 호출 측이 잠금)

| 사유 | 조건 (데모 1 Hz 값) |
|---|---|
| `NO_CLOCK` / `CLOCK_TIMEOUT` | /clock 없음 / 마지막 수신 후 1.0 s (mono) |
| `SIM_TIME_STALLED` | sim 시간이 1.0 s (mono) 동안 증가하지 않음 |
| `NO_ODOM` / `ODOM_RECEIVE_TIMEOUT` | Odometry 없음 / 마지막 수락 후 2.0 s (mono) |
| `ODOM_STAMP_AHEAD_OF_CLOCK` | stamp가 /clock보다 0.2 s (sim) 넘게 앞섬 |
| `ODOM_STAMP_STALE` | stamp가 /clock보다 1.5 s (sim) 넘게 오래됨 |

Executor 이동 루프 (`safe_motion.run_move`)

- 시작 전 차단은 잠그지 않는다 (아직 움직이지 않았음). 기존 Fault는 계속 차단된다.
- 이동 중 이상, Watchdog 판정 실패, `move_timeout_mono`(60 s) 초과 → 정지 요청 + Executor 측 감시기 잠금 + 종료. 자동 재개 없음.

### 2.5 StopVerifier (`stop_check.py`)

"정지 명령 발행 성공"과 "실제 정지 확인 성공"을 구분한다. Odometry twist로 후자만 판정한다.

- 정지 명령 시각 + `settle_sim`(0.5 s) 이전 stamp의 샘플은 쓰지 않는다. stamp가 엄격히 증가하는 새 샘플만 쓴다.
- 연속 정지 샘플(|v| ≤ 0.01 m/s, |w| ≤ 0.02 rad/s)이 `window_sim`(1.0 s) 이상, `min_samples`(2) 이상이고
  최신 샘플이 /clock 대비 `sample_max_age_sim`(1.5 s) 이내 → `CONFIRMED`.
- `timeout_sim`(4.0 s) 안에 확인 못하면 `NO_FRESH_ODOM` 또는 `NOT_STOPPED`.
- 시작 후 `timeout_mono`(6.0 s) 안에 결론이 없으면 `UNVERIFIABLE` (Clock 정지 대비, 4절).
- 확정 결과는 다음 정지 판정 시작 전까지 바뀌지 않는다.
- Watchdog 측: 출력이 이동 → 정지로 바뀔 때 시작한다. Odometry 무결성 Fault(Frame, stamp, 급변, 시간 역행 등)
  상태에서는 `UNVERIFIABLE`. 명령 계열 Fault(COMMAND_TIMEOUT 등) 뒤에도 Frame이 맞고 stamp가 Watchdog /clock보다
  앞서지 않은 Odometry로 정지 확인을 계속한다.
- Executor 측: `verify_stop()`이 정지 요청을 반복하며 같은 기준으로 판정한다.

### 2.6 종료 처리 (`shutdown.py`)

- SIGINT/SIGTERM은 `ShutdownRequested`(BaseException)로 바뀌어 finally 정리 코드가 반드시 실행된다.
- 정리 중에는 SIGINT/SIGTERM을 무시한다. 정리 내용: 정지 요청(Executor) 또는 정지 cmd_vel(Watchdog)을
  `shutdown_stop_repeats`(10)회, 0.05 s 간격으로 발행한다. 발행 횟수만 보고하며 실제 정지 확인이 아니다.
- 종료 코드: 0 성공, 1 이동 실패, 2 정지 미확인, 3 예외, 130 SIGINT, 143 SIGTERM.
- SIGKILL은 프로세스 안에서 처리할 수 없다. Executor SIGKILL은 Watchdog의 Command Timeout이 대응한다 (1절 2번).
  **Watchdog SIGKILL에는 대응하는 계층이 없다 (7절 H2).**

### 2.7 단방향 Bridge 분리 (`ros2_launch/gz_split_bridge.launch.py`)

`ros_gz_sim_demos/diff_drive.launch.py`는 하나의 `parameter_bridge` 프로세스가 cmd_vel과 Odometry를
양방향(`@`)으로 함께 연결한다. 이 Bridge만 종료하면 cmd_vel도 같이 끊기므로 Odometry 단독 장애를 만들 수 없고,
양방향 cmd_vel Bridge는 Gazebo→ROS 방향으로 같은 Topic을 다시 발행하므로 cmd_vel 발행자가 Watchdog 하나라고 확인할 수 없다.
신규 Launch는 Bridge를 3개 프로세스로 나눈다.

| Bridge | 사양 (방향) | Node 이름 |
|---|---|---|
| cmd_vel | `/model/vehicle_blue/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist` (ROS→Gazebo) | `mapf_cmd_vel_bridge_vehicle_blue` |
| Odometry | `/model/vehicle_blue/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry` (Gazebo→ROS) | `mapf_odom_bridge_vehicle_blue` |
| Clock | `/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock` (Gazebo→ROS) | `mapf_clock_bridge` |

- Gazebo는 `ros_gz_sim/gz_sim.launch.py`에 `-r diff_drive.sdf`를 넘긴다. `gui:=false`이면 `-s`로 서버만 실행한다.
- RViz와 vehicle_green Bridge는 실행하지 않는다. `odom_bridge:=false`이면 Odometry Bridge를 빼고 실행한다.
- 폴더 이름은 ROS `launch` 패키지와 겹치지 않도록 `ros2_launch/`로 했다.
- `scripts/check_gz_cmd_topology.py`: cmd_vel 발행자 = Watchdog 하나, cmd_vel 구독자에 cmd_vel Bridge 존재,
  Odometry 발행자 = Odometry Bridge 하나를 ROS Graph에서 확인한다 (위반 시 종료 코드 1).
  Gazebo 쪽 발행자는 ROS Graph에 보이지 않으므로 `gz topic -i`로 따로 확인한다.

### 2.8 설정값 (`safety_config.py`)

| 항목 | 데모 1 Hz (`demo`, 실험에 사용) | 50 Hz (`mapf50`, 목표 사양·미검증) |
|---|---|---|
| Odometry 수신 Timeout (mono) | 2.0 s | 0.25 s |
| stamp 최대 나이 / 앞섬 허용 (sim) | 1.5 s / 0.2 s | 0.1 s / 0.05 s |
| Clock 수신 Timeout / sim 정지 (mono) | 1.0 s / 1.0 s | 0.5 s / 0.5 s |
| Executor 제어 주기 | 0.05 s | 0.02 s |
| Command Timeout / tick | 0.5 s / 0.05 s | 0.2 s / 0.02 s |
| 상태 메시지 Timeout | 0.5 s | 0.25 s |
| 속도 상한 | 0.3 m/s, 0.5 rad/s | 0.3 m/s, 0.5 rad/s (임시) |
| STOP_CHECK settle / window / timeout_sim / timeout_mono | 0.5 / 1.0 / 4.0 / 6.0 s | 0.5 / 0.2 / 2.0 / 3.0 s |

Executor(`scripts/run_gz_single_robot.py`)는 항상 데모 설정을 쓴다 (7절 L1).

## 3. 장애 주입 자동화: `scripts/run_gz_odom_fault_injection.py`

순수 로직은 `src/ros_bridge/fault_injection.py`, ROS Graph 조회는 `src/ros_bridge/gz_topology.py`에 있다.

1. **대상 프로세스 검증** (`/proc`): `argv[0]`과 `/proc/<pid>/exe`의 basename이 `parameter_bridge`, `--ros-args` 이전 인자가
   Odometry 단방향 사양 하나뿐, `__node:=mapf_odom_bridge_<id>`, 자기 자신·PID 1 아님. 정확히 하나일 때만 진행한다.
   cmd_vel Bridge도 같은 방식으로 찾아 둔다 (종료 뒤 생존 확인용).
2. **ROS Graph 검증**: 2.7 조건 + 이동 요청 Topic 발행자 없음 (다른 Executor 미실행).
3. Watchdog READY 확인 후 Executor를 자식 프로세스로 실행한다 (`--no-spawn-executor`면 수동 실행).
4. **ARM**: Watchdog ACTIVE, fault 없음, 출력이 0이 아닌 상태를 같은 세션에서 연속 2회 관측할 때만 주입한다. 고정 sleep 없음.
   제한 시간(20 s) 안에 조건이 오지 않거나 Executor가 먼저 끝나면 아무것도 종료하지 않고 중단한다 (종료 코드 2).
5. **주입**: `pidfd_open` 후 `/proc`를 다시 읽어 PID·시작 시각·argv·exe가 같을 때만 `pidfd_send_signal` (기본 SIGKILL).
6. **관측·판정** (`InjectionRecord.evaluate`)

| 검사 | 통과 조건 |
|---|---|
| `watchdog_left_active` | 주입 후 `2.0 + 0.05 + 1.0 = 3.05 s` 안에 ACTIVE 이탈 (WAITING 또는 FAULT) |
| `no_motion_command_after_leave` | ACTIVE 이탈 0.2 s 뒤부터 0이 아닌 cmd_vel 0건 |
| `stop_command_observed` | ACTIVE 이탈 뒤 0 cmd_vel 1건 이상 |
| `odometry_isolated` | Bridge 종료 확인, 종료 뒤 Odometry 수신 0건 |
| `cmd_vel_bridge_kept` | cmd_vel Bridge 프로세스 생존, 구독 유지 |
| `stop_check_not_confirmed` | 최종 STOP_CHECK ≠ CONFIRMED |
| `stop_check_settled` | 최종 STOP_CHECK가 확정 상태 |
| `executor_reported_failure` | (Executor를 직접 실행한 경우) 종료 코드 ≠ 0 |

- `physical_stop`은 항상 `NOT_VERIFIED_ODOMETRY_LOST`다. 이 도구는 ROS cmd_vel Topic의 정지 명령만 관측한다.
- `watchdog_left_active`는 WAITING과 FAULT를 구분하지 않는다. 1.1처럼 Fault 없이 통과할 수 있다.
- 기록: `outputs/logs/gz_fault_injection/<시각>_odom_bridge_kill.json`, `<시각>_executor.log` (git 추적 대상 아님, 8절).
- 종료 코드: 0 PASS, 1 FAIL, 2 ABORTED, 3 예외.
- Clock Bridge 장애(1.2)는 이 도구가 아니라 수동 `kill`로 수행했다 (6절).

## 4. STOP_CHECK 시간 제한

- 발견: 실패 판정이 sim 시간으로만 이루어져 /clock이 멈추거나 없으면 PENDING이 무기한 유지될 수 있었다 (1.3 9번).
- 변경: `StopCheckConfig.timeout_mono` (필수, `≥ timeout_sim`). 판정 시작 뒤 이 시간이 지나도록 결론이 없으면
  `UNVERIFIABLE (STOP_CHECK_DEADLINE ...)`로 확정한다. sim 기준 판정이 가능하면 그쪽을 우선한다.
  확정 결과는 Clock 복구 뒤에도 뒤집지 않는다.
- `StopCheck.failed`에 UNVERIFIABLE 포함, `terminal` 속성 추가. Watchdog은 마지막 monotonic 시각(`last_mono`)으로 평가한다.
  Executor `verify_stop`은 monotonic 시각을 전달하고 바깥 루프 상한은 `max(timeout_sim × 1.5, timeout_mono) + 1.0`이다.
- **[Gazebo·사용자] 확인**: 1.2에서 STOP_CHECK가 UNVERIFIABLE로 확정되고 무한 PENDING이 발생하지 않았다.
  "STOP 후 약 6 s"라는 정확한 시각과 Clock 복구 후 유지 여부는 보고된 값이 없어 기록하지 않는다.

## 5. [자동화·Claude Code] 테스트 결과 (2026-11-26 최종 검수)

Gazebo를 실행하지 않았다. unittest 사용 (venv에 pytest 미설치).

### 5.1 테스트

| 실행 | 전체 | PASS | SKIP | FAIL | ERROR |
|---|---|---|---|---|---|
| `py_compile` (`src/ros_bridge/*.py`, `scripts/run_gz_*.py`, `scripts/check_gz_cmd_topology.py`, Launch 파일, `tests/test_ros_bridge_*.py`) | — | 통과 | — | — | — |
| ROS Bridge 테스트 (`test_ros_bridge_*.py`, ROS 환경) | 322 | 322 | 0 | 0 | 0 |
| 전체 회귀, ROS 환경 | 1503 | 1479 | 24 | 0 | 0 |
| 전체 회귀, ROS 환경변수 제거 (`env -i HOME PATH=/usr/bin:/bin` + venv python, sys.path에 ROS 경로 없음 확인) | 1503 | 1441 | 62 | 0 | 0 |
| 전체 회귀, 실제 CBS Solver 1차 (`PYTHONHASHSEED=0`, `ALPHA2_CBS_SOLVER_ROOT=…/cbs_solver_atb033/centralized/cbs`, Solver `89f69e1`) | 1503 | 1489 | 13 | 0 | **1** |
| 전체 회귀, 실제 CBS Solver 2차 (같은 조건) | 1503 | 1490 | 13 | 0 | 0 |
| `git diff --check` | — | 통과 | — | — | — |

- ROS 환경 SKIP 24: 실제 CBS Solver 없음 11, PENDING/UNIMPLEMENTED 결정 10, 저장된 DEV 기록·데이터 없음 3.
- ROS 제거 환경 SKIP 62 = 위 24 + `rclpy not available` 37 (Executor/Watchdog Node 테스트) + 실제 `parameter_bridge` 테스트 1.
- 실제 CBS 환경 SKIP 13: PENDING/UNIMPLEMENTED 10, 저장된 DEV 기록·데이터 없음 3.
- ROS Bridge 테스트 파일별 수: cell_motion 12, command_watchdog 65, fault_injection 46, grid_frame 8,
  gz_executor_node 27, gz_watchdog_node 10, odom_safety 57, safe_motion 53, shutdown 14, stop_check 30.

### 5.2 기존 CBS 환경 의존 Timeout 문제 (미해결, 이번 변경과 무관)

- 1차 ERROR: `tests.test_selective_real_wiring.TestRealSolverPlannerLevel.test_selective_planner_real_l0_timeout_then_real_g`
  — `l0, gl = rec.attempts`에서 `ValueError: not enough values to unpack (expected 2, got 1)`.
- 1차 실행은 ROS 제거 환경 전체 회귀와 **동시에** 실행되어 CPU 부하가 높았다. L0 예산이 60 ms인 시간 의존 테스트다.
- 같은 클래스(5개)를 단독 실행하면 기준 Commit `882f848` 사본과 현재 작업 트리 모두 3/3회 통과했고, 전체 재실행(2차)도 통과했다.
- `src/ros_bridge` 밖의 소스 변경이 없으므로 이번 변경과 무관하다고 판단한다.
- **근본 원인은 조사하지 않았고 해결되지 않았다.** 테스트도 수정하지 않았다. 전체 회귀를 동시에 실행하지 않는 것이 좋다.

### 5.3 모의 실행 (Gazebo 아님, v1 구현 중 수행)

scratchpad의 임시 도구로 실행했고 저장소에는 넣지 않았다. gz-transport Python으로 만든 가짜 시뮬레이터(cmd_vel 단순 적분,
`/clock` 100 Hz, Odometry 1 Hz), 실제 `parameter_bridge` 3개, 실제 Watchdog, injector를 격리된 Domain/Partition에서 실행했다.
**물리 엔진, DiffDrive 플러그인, Gazebo 타이밍을 검증한 것이 아니다.**

| 실행 | 결과 |
|---|---|
| `/clock` 없음 | ABORTED (NO_CLOCK로 READY 안 됨), 아무것도 종료하지 않음 |
| 실제 Executor | PASS. Executor가 먼저 감지(`ODOM_STAMP_STALE`), Watchdog ACTIVE→WAITING +1.449 s, fault=null |
| Odometry를 무시하는 임시 Executor | PASS. Watchdog이 감지: ACTIVE→FAULT `ODOM_STAMP_STALE (during motion)` +0.599 s |

감지 주체(Executor 또는 Watchdog)는 타이밍에 따라 달라진다. Gazebo 실험 7(1.1)은 Executor가 먼저 감지한 경우다.

### 5.4 최종 검수에서 확인한 H1 재현 (Gazebo 아님)

순수 `CommandWatchdog`(데모 설정)만으로 scratchpad 스크립트를 실행했다. 이동 중 Odometry 중단 → Executor 정지 요청이
Watchdog tick보다 먼저 도착 → WAITING (fault=null, STOP_CHECK `NO_FRESH_ODOM`) → Odometry 복구 → **READY** →
Reset 없이 다른 sender의 이동 요청이 **FORWARDED → ACTIVE**.

## 6. 실행 방법 (사용자 실행)

모든 터미널에서 ROS 환경과 프로젝트 venv를 활성화하고 프로젝트 루트에서 실행한다.
v0의 `ros2 launch ros_gz_sim_demos diff_drive.launch.py`가 실행 중이면 먼저 종료한다 (남아 있으면 Topology 검사 실패).

```bash
# 터미널 1: Gazebo + cmd_vel·Clock Bridge
ros2 launch ros2_launch/gz_split_bridge.launch.py odom_bridge:=false
# 터미널 2: Odometry 전용 Bridge (장애 주입 대상)
ros2 run ros_gz_bridge parameter_bridge \
  '/model/vehicle_blue/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry' \
  --ros-args -r __node:=mapf_odom_bridge_vehicle_blue
# 터미널 3: 독립 Watchdog
python3 -m scripts.run_gz_watchdog --robot-id vehicle_blue
# 터미널 4: 구성 확인
python3 -m scripts.check_gz_cmd_topology --robot-id vehicle_blue   # TOPOLOGY_OK 기대
gz topic -i -t /model/vehicle_blue/cmd_vel
```

| 실험 | 명령 (터미널 4) |
|---|---|
| 정상 1 m 이동 | `python3 -m scripts.run_gz_single_robot` |
| 이동 중 Odometry Bridge 종료 | `python3 -m scripts.run_gz_odom_fault_injection --robot-id vehicle_blue` (`--signal TERM` 가능) |
| 이동 중 Clock Bridge 종료 | Executor 실행 → 상태 ACTIVE·출력 `[0.3, 0.0]` 확인 → `pgrep -af '__node:=mapf_clock_bridge'` → `kill <PID>` |

```bash
ros2 topic echo /mapf/vehicle_blue/watchdog_status --once          # 상태 확인
gz topic -e -t /model/vehicle_blue/cmd_vel                          # Gazebo 쪽 정지 명령 관측
ros2 topic echo /model/vehicle_blue/odometry --once --field twist.twist   # 복구 후 수동 정지 확인
ros2 service call /mapf/vehicle_blue/watchdog_reset std_srvs/srv/Trigger  # Executor 종료 후 Reset
```

## 7. 미해결 후속 과제

모두 **미해결**이다. 이번 단계에서는 코드를 수정하지 않았다.

### 7.1 우선순위 요약

| ID | 심각도 | 해결 시점 |
|---|---|---|
| H1 | High | **다중 이동 및 경로 재생 확장 전에 해결** |
| M2 | Medium | **다중 이동 및 경로 재생 확장 전에 해결** |
| H2 | High | **Watchdog 자체 장애 보호 계층 필요. 다중 로봇 확장 전에 Gazebo 환경에서 검증** |
| M1 | Medium | 이후 단계 |
| L1~L4 | Low | 이후 단계 |

### 7.2 항목

**H1. Odometry 장애 후 Fault 잠금이 감지 순서에 의존하고, Reset 없이 READY로 복귀한다**
- 위치: `command_watchdog.py` `on_request`는 상태가 정상이 아니어도 정지 요청을 수락하고 마지막 명령을 정지로 바꾼다.
  `tick`은 이동 중일 때만 상태 이상을 잠근다.
- 결과: Executor의 정지 요청이 Watchdog 판정보다 먼저 도착하면 잠기지 않고 WAITING이 된다 (Gazebo 1.1, 모의 5.3, 재현 5.4).
  Odometry가 복구되면 STOP_CHECK가 실패(`NO_FRESH_ODOM`)인 채로 READY가 되고, 새 Executor 프로세스는 Reset 없이 출발할 수 있다.
- 같은 종류의 Clock 장애는 M1의 부수 효과로 FAULT가 되어 비대칭이다.
- 같은 Executor 프로세스의 자동 재출발은 없다 (Executor 측 감시기 잠금, 재시도 루프 없음, 종료).
  지연된 옛 요청은 세션 token·seq·stamp 검사로 차단된다. **남은 재출발 경로는 새 Executor 프로세스뿐이다.**
- 방향: 이동 중 Odometry/Clock 상태 이상 또는 정지 확인 실패가 있으면 정지 요청을 누가 먼저 보냈든 잠그고 Reset을 요구한다.

**H2. Watchdog 자체 강제 종료(SIGKILL, 충돌) 시 마지막 이동 명령이 유지될 수 있다**
- cmd_vel의 유일한 발행자가 사라진다. 데모 SDF에 command timeout 설정이 없고, `libgz-sim8-diff-drive-system.so`에서
  "timeout" 문자열이 발견되지 않았다 (정적 확인). 따라서 ACTIVE 중 Watchdog이 죽으면 0.3 m/s 이동이 계속될 가능성이 높다.
  **Gazebo에서 확인하지 않았다.**
- Executor는 0.5 s 안에 `WATCHDOG_STATUS_TIMEOUT`으로 중단하지만 cmd_vel 발행 권한이 없어 정지 요청이 전달되지 않는다.
- 현재 복구 수단: Watchdog 재시작(시작 직후와 매 tick에 STOP 발행, 새 boot_id로 기존 Executor는 세션 변경 감지) 또는 수동 `gz topic` 정지 명령.
- Watchdog SIGTERM 종료도 Gazebo에서 시험하지 않았다.
- 필요: Watchdog 자체 장애에 대한 보호 계층 (예: command timeout이 있는 자체 로봇 SDF·플러그인, 감독 프로세스 재시작, 이중화).
  **다중 로봇 확장 전에 Gazebo 환경에서 Watchdog SIGKILL/SIGTERM 실험으로 검증해야 한다.**

**M1. Clock 장애의 Fault 원인이 TIME_BASE_MISMATCH로 기록될 수 있다**
- `odom_monitor.on_odom`은 /clock 신선도를 확인하지 않고 멈춘 마지막 /clock과 stamp를 비교한다. 실제 원인이 Clock 정지여도
  `TIME_BASE_MISMATCH`(무결성 Fault)로 잠길 수 있다 (Gazebo 1.2에서 CLOCK_TIMEOUT과 함께 관측).
- 결과적으로 정지 중 Clock 장애는 FAULT, 정지 중 Odometry 장애는 Fault 없음으로 처리가 비대칭이다.

**M2. Reset이 물리적 정지 여부를 확인하지 않는다**
- STOP_CHECK가 PENDING이거나 실패여도 Reset이 허용된다. Reset 후 STOP_CHECK가 `NOT_STARTED`로 지워져 직전 정지 확인 근거가 남지 않는다.
- 방향: Reset 조건에 정지 확인 결과(또는 운영자 확인)를 포함하고, Reset 응답·로그에 직전 STOP_CHECK를 남긴다.

**L1. 프로세스 간 설정 불일치를 검출하지 않는다**
- `run_gz_watchdog --profile mapf50`이어도 Executor는 항상 데모 설정이다. `validate_against`는 같은 프로세스 안에서만 검사한다.
  상태 메시지에 설정 이름이 없다. 대부분 차단 쪽으로 실패하지만 오탐 Timeout이 날 수 있다.

**L2. 이동 요청 판정에 쓰는 상태가 최대 1 tick(0.05 s) 늦다**
- `on_request`는 마지막 tick에서 계산한 상태를 쓴다. 그 사이 이상이 생기면 이동 요청이 한 번 더 전달될 수 있고 다음 tick에서 잠긴다.

**L3. 감지 지연이 1 Hz Odometry에 묶여 있다**
- Gazebo 1.1에서 ACTIVE 이탈까지 1.299 s (0.3 m/s 기준 약 0.39 m 이동 가능). 설정상 최악은 약 2 s 이상, 약 0.6 m.

**L4. 형식이 잘못된 요청도 "요청 수신 중"으로 집계되어 Reset을 막는다**
- 다른 발행자가 이동 요청 Topic에 계속 발행하면 Reset이 계속 거부된다 (안전 쪽 실패).

### 7.3 기타 남은 위험

1. Odometry 장애 중 물리적 정지는 확인되지 않는다. injector는 ROS cmd_vel Topic만 관측한다.
2. UNVERIFIABLE은 확정 후 유지된다. Clock 복구 뒤 자동 재확인하지 않으므로 운영자가 직접 확인해야 한다.
3. `timeout_mono`(데모 6 s)는 실시간 배율 약 0.67 미만이면 sim 판정보다 먼저 만료된다 (안전 쪽 오탐).
4. injector 검증은 Node 이름 규칙에 의존한다 (다른 이름이면 중단). pidfd는 Linux 5.3 이상 필요.
5. 0이 아닌 cmd_vel 집계의 0.2 s grace는 경험값이다.
6. injector가 띄운 Executor는 별도 세션에서 실행된다. injector가 비정상 종료하면 Executor가 남을 수 있다 (Command Timeout은 적용).
7. 모든 Gazebo 결과는 항목별 n=1이며 Ground Truth 검증이 아니다.

### 7.4 다음 단계 (이번 범위 밖)

- H1, M2 해결 후 다중 셀 이동·경로 재생 연결
- Watchdog 자체 장애 보호 계층과 Gazebo 검증 (H2), 그 후 다중 로봇 확장
- Ground Truth 위치 모듈, 자체 소형 로봇 SDF (50 Hz Odometry, command timeout), 격자 World 생성

## 8. 실험 기록 보존

- 원본 (git 추적 대상 아님, `.gitignore`의 `outputs/logs/*`)
  - `outputs/logs/gz_fault_injection/20261126_111919_odom_bridge_kill.json`
    SHA256 `2e4efc9989395dbd28fbca2c1ac09858f23da3fbeef1876d6c8f396e569a03ee`
  - `outputs/logs/gz_fault_injection/20261126_111919_executor.log`
    SHA256 `6e9933ce7d73555ab736ad9eebd86cc1d6ae65c2429e7e5d1be5a2bd47f9ef19`
- 백업: `/home/ina/mapf_work/manual_backups_20261126/ros2_gz_single_robot_v1_20261126/`
  - 실험 로그 사본, 미커밋 소스 스냅샷(수정 2개 + 미추적 신규 파일), HEAD·브랜치·`git status`·`git diff`, 파일별 SHA256,
    원본·백업 SHA256 대조 결과, 한계를 적은 `README.md`. 상세 목록과 검증 결과는 그 폴더의 `README.md`, `SHA256SUMS.txt`에 있다.
- **한계**
  - 실험 JSON에는 `git.head=882f848…`과 `dirty=true`만 있고 당시 diff 해시가 없다. 스냅샷과 실험 당시 코드가 같다고 **검증할 수 없다.**
  - 정황 근거: 스냅샷의 모든 소스·테스트 파일 수정 시각(mtime)이 실험 7의 주입 시각(2026-11-26 11:19:19 KST) 이전(최종 11:07:14)이다.
    mtime은 내용 해시가 아니므로 증명이 아니다.
  - 실험 1~6, 8은 실행 시각과 터미널 로그가 남아 있지 않아 코드 버전과 연결할 근거가 없다. 사용자 확인 결과로만 기록한다.
    원본 로그가 없는 실험에 대해 로그나 측정값을 새로 만들지 않았다.

## 9. 변경 파일 (기준 Commit `882f848` 대비, 모두 미커밋)

수정 (2)
- `scripts/run_gz_single_robot.py`: Watchdog 경유 실행, 정지 확인, 종료 처리, 종료 코드
- `src/ros_bridge/gz_executor_node.py`: cmd_vel 직접 발행 제거, Watchdog 이동 요청·상태 구독, Odometry Safety, `verify_stop`, `shutdown_stop`

신규 소스 (13)
- `src/ros_bridge/odom_safety.py`: Odometry 검사 순수 함수
- `src/ros_bridge/odom_monitor.py`: Odometry/Clock 상태와 잠금 Fault
- `src/ros_bridge/safety_config.py`: 설정, Robot Interface, 데모·50 Hz 값
- `src/ros_bridge/safe_motion.py`: 안전 검사 이동 루프
- `src/ros_bridge/stop_check.py`: StopVerifier
- `src/ros_bridge/watchdog_protocol.py`: 요청 ID, 상태 JSON, WatchdogClient
- `src/ros_bridge/command_watchdog.py`: Watchdog 순수 로직
- `src/ros_bridge/gz_watchdog_node.py`: Watchdog ROS Node
- `src/ros_bridge/qos.py`: 공용 QoS
- `src/ros_bridge/shutdown.py`: 신호·종료 처리
- `src/ros_bridge/fault_injection.py`: 장애 주입 순수 로직
- `src/ros_bridge/gz_topology.py`: ROS Graph 조회

신규 실행·Launch (4)
- `scripts/run_gz_watchdog.py`, `scripts/check_gz_cmd_topology.py`, `scripts/run_gz_odom_fault_injection.py`
- `ros2_launch/gz_split_bridge.launch.py`

신규 테스트 (8)
- `tests/test_ros_bridge_odom_safety.py`, `tests/test_ros_bridge_safe_motion.py`, `tests/test_ros_bridge_stop_check.py`,
  `tests/test_ros_bridge_command_watchdog.py`, `tests/test_ros_bridge_gz_watchdog_node.py`, `tests/test_ros_bridge_gz_executor_node.py`,
  `tests/test_ros_bridge_shutdown.py`, `tests/test_ros_bridge_fault_injection.py`

신규 문서 (1)
- `docs/ros2_gz_single_robot_v1.md` (이 문서)

Commit 제외: `ros2_launch/__pycache__/`, `outputs/logs/gz_fault_injection/*` (모두 `.gitignore` 대상), `.claude/settings.local.json`.

MAPF 알고리즘, Selective Hybrid 파라미터, Registry, DEV/Validation/Test 데이터, Solver Timeout 테스트는 변경하지 않았다.
