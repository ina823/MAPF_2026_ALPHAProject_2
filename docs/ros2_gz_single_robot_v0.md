# ROS2/Gazebo 단일 로봇 폐루프 제어 프로토타입 v0 (2026-11-25)

> **범위 한정.** 이 문서는 단일 데모 로봇(vehicle_blue)이 Odometry 피드백만으로
> 1셀(1.0 m) 직진해 목표에 도착하는지 확인한 **기초 프로토타입 기록**이다.
> MAPF-Gazebo 통합 검증이나 최종 성능 평가가 아니며, 실행은 1회뿐이다.
> 아래 "알려진 위험"은 해결되지 않았고, 다중 로봇으로 확장하기 전에 처리할 필수 후속 작업이다.

- 브랜치: `feature/ina-ros2-gz-bridge` (분기 기준 Commit `bc4ffd4`)
- 구현: 사용자 직접 구현 (ChatGPT 보조)
- 코드 검수와 테스트: Claude Code (읽기 전용, 2026-11-25)

## 1. 실행 환경

| 항목 | 값 |
|---|---|
| OS | Ubuntu 24.04.5 LTS |
| ROS2 | Jazzy |
| Gazebo | Harmonic 8.15.0 |
| `ROS_AUTOMATIC_DISCOVERY_RANGE` | `LOCALHOST` |
| Python | 3.12.3 (프로젝트 `alpha2` 가상환경) |
| 설치 패키지 (dpkg) | `ros-jazzy-ros-gz-sim-demos 1.0.25`, `ros-jazzy-ros-gz-bridge 1.0.25`, `ros-jazzy-gz-sim-vendor 0.0.13` |

## 2. 실행 명령 (사용자 실행)

```bash
# 1) Gazebo + Bridge (+ RViz, launch 기본값 rviz:=true)
ros2 launch ros_gz_sim_demos diff_drive.launch.py

# 2) Python 제어 — 반드시 프로젝트 루트에서 -m 방식으로 실행
cd <프로젝트 루트>
python3 -m scripts.run_gz_single_robot

# 3) 최종 정지 확인
ros2 topic echo /model/vehicle_blue/odometry --once --field twist.twist
```

**주의:** `scripts/run_gz_single_robot.py`는 다른 스크립트들과 달리 `sys.path`를 보정하지 않는다.
`python3 scripts/run_gz_single_robot.py`처럼 파일 경로로 실행하면 `src` 패키지를 찾지 못한다.
반드시 프로젝트 루트에서 `python3 -m scripts.run_gz_single_robot`으로 실행해야 한다. 이번 v0에서는 코드를 변경하지 않았다.

## 3. World / Bridge 설정 (설치 파일을 읽기 전용으로 확인)

### Launch: `/opt/ros/jazzy/share/ros_gz_sim_demos/launch/diff_drive.launch.py`

- `ros_gz_sim/launch/gz_sim.launch.py`에 `gz_args: '-r diff_drive.sdf'`를 넘겨 실행한다 (`-r`: 시작 즉시 실행).
- Bridge는 `ros_gz_bridge parameter_bridge` 노드이며 인자는 다음과 같다.
  - `/model/vehicle_blue/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist`
  - `/model/vehicle_blue/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry`
  - (vehicle_green에도 같은 2개)
  - 매개변수: `qos_overrides./model/vehicle_blue.subscriber.reliability: reliable` (green도 같음)
- 별도 지정이 없으면 RViz도 실행된다 (`rviz` 기본값 `true`).

### World: `/opt/ros/jazzy/opt/gz_sim_vendor/share/gz/gz-sim8/worlds/diff_drive.sdf`

- `GZ_SIM_RESOURCE_PATH=/opt/ros/jazzy/share` 기준으로, 시스템 안에 `diff_drive.sdf`는 이 파일 하나뿐이다 (`ros_gz_sim_demos/worlds`에는 없음).
- sha256: `6d6a182ed978681656cf8ffa24b62fe102ec14459e7d2d2fa205cfb299a17375`
- physics: `max_step_size 0.001`, `real_time_factor 1.0`
- World 플러그인: physics, user-commands, scene-broadcaster. Pose publisher 플러그인은 선언되어 있지 않다.
- **vehicle_blue**
  - World 생성 위치: `<pose>0 2 0.325 0 -0 0</pose>` (yaw 0)
  - DiffDrive 설정
    - `wheel_separation 1.25`, `wheel_radius 0.3`
    - **`odom_publish_frequency 1` (1 Hz)**
    - 선가속도 ±1, 각가속도 ±2
    - 선속도 ±0.5, 각속도 ±1
  - SDF에는 command timeout 관련 설정이 없다. 명령이 끊겼을 때 플러그인 기본 동작은 확인하지 않았다.
- vehicle_green: 생성 위치 `0 -2 0.325`. DiffDrive에 `odom_publish_frequency`를 지정하지 않았다 (이번 실험에서 사용하지 않음).

### 설정에서 확인되는 해석

- 실험 출력의 시작 위치는 (0.000, -0.000)이고, SDF의 World 생성 위치는 (0, 2)이다. 따라서 **보고된 좌표는 World 좌표가 아니라 DiffDrive Odometry 좌표계 값**이다.
- 제어 피드백은 1 Hz다. 제어 루프는 최대 약 1초 묵은 위치로 속도를 계산한다.

## 4. 구현 구성

| 파일 | 역할 |
|---|---|
| `src/ros_bridge/grid_frame.py` | (row, col)과 world 좌표 변환: `x = x0 + col·s`, `y = y0 − row·s`. 도착 판정 `is_at_target` |
| `src/ros_bridge/cell_motion.py` | Quaternion→yaw 변환, 각도 정규화. 정지→회전→직진 비례 제어 `compute_velocity` |
| `src/ros_bridge/gz_executor_node.py` | `cmd_vel` 발행, `odometry` 구독. Timeout 처리, 종료 시 정지 발행 |
| `scripts/run_gz_single_robot.py` | 시작 Odometry 위치를 셀 (0,0)으로 두고 (0,1)로 1셀 이동한 뒤 최종 위치 판정 |
| `tests/test_ros_bridge_grid_frame.py` | 단위 테스트 8개 |
| `tests/test_ros_bridge_cell_motion.py` | 단위 테스트 12개 |

제어 상수 (하드코딩):

| 항목 | 값 |
|---|---|
| 비례 이득 | `kp_linear 0.8`, `kp_angular 1.5` |
| 최대 속도 | 선속도 0.3 m/s, 각속도 0.5 rad/s |
| 허용 오차 | 위치 0.05 m, 방향 5° |

Timeout (벽시계 `time.monotonic` 기준):

| 항목 | 값 |
|---|---|
| 최초 유효 Odometry 대기 | 5 s |
| Odometry 공백 | 2.5 s |
| 전체 이동 | 60 s |
| 최종 확인 (정지 발행하며 대기) | 3 s |

## 5. 실험 결과 — 사용자 직접 실행 (1회)

| 항목 | 값 |
|---|---|
| 로봇 | vehicle_blue (단일 데모 로봇) |
| 실행 횟수 | 1 |
| 목표 이동 거리 | 1 m (grid (0,0) → (0,1)) |
| 초기 위치 | (0.000, -0.000) |
| 초기 방향 | 약 0° |
| 목표 위치 | (1.000, -0.000) |
| 최종 위치 | (0.988, 0.000) |
| 위치 오차 (Odometry 기준) | 0.0119 m |
| 허용 오차 | 0.05 m |
| 최종 선속도 / 각속도 | 0.0 / 0.0 (`ros2 topic echo ... --field twist.twist`로 확인) |
| 출력 | `TARGET_REACHED`, `PASS: Target reached` |

**이 결과는 Ground Truth 검증이 아니다.**
- 위치와 오차는 모두 제어에 쓴 DiffDrive Odometry 값이다. 같은 값으로 PASS를 판정했으므로 자기 참조 검증이다.
- 바퀴 미끄러짐이나 Odometry 누적 오차는 이 판정에 드러나지 않는다.
- 이번 실험에는 회전이 포함되지 않았다. 초기 방향이 약 0°였고 목표가 +x 방향이었기 때문이다.

## 6. 코드 검수 및 테스트 결과 — Claude Code (2026-11-25, 읽기 전용)

이 절은 사용자 실행 결과와 별개다. Gazebo를 다시 실행하지 않았고, 코드 정적 검수와 단위·회귀 테스트만 수행했다.

| 실행 | 결과 |
|---|---|
| `py_compile` (신규 파일 7개) | 통과 |
| 신규 ROS Bridge 단위 테스트 | **20/20 PASS** |
| 전체 회귀, 기본 환경 | 1201개 중 PASS 1177, SKIP 24, 실패·오류 0 |
| 전체 회귀, ROS 환경변수 제거 (`PYTHONPATH`, `AMENT_PREFIX_PATH`, `ROS_DISTRO`, `LD_LIBRARY_PATH`) | 1201개 중 PASS 1177, SKIP 24, 실패·오류 0 |
| 전체 회귀, 실제 CBS Solver 환경 (`ALPHA2_CBS_SOLVER_ROOT` 지정, `PYTHONHASHSEED=0`) | 1201개 중 PASS 1188, SKIP 13, 실패·오류 0 |
| 기준선 (`bc4ffd4` 상당, 신규 20개 제외) | 1181개 중 PASS 1157, SKIP 24, 실패·오류 0 |

- 신규 회귀 오류는 없다. 기존 추적 파일은 하나도 바뀌지 않았고, 기존 코드는 `src.ros_bridge`를 import하지 않는다.
- ROS가 없는 환경에서도 `grid_frame`, `cell_motion`과 그 테스트는 rclpy 없이 동작한다. rclpy가 필요한 것은 `gz_executor_node.py`와 실행 스크립트뿐이며, 둘 다 unittest 탐색 대상이 아니다.
- 기존의 환경 의존 CBS Timeout 문제는 이번 실행에서 재현되지 않았다. **해결된 것은 아니며**, 이번 변경과는 무관하다.
- 실행에는 unittest를 사용했다 (`alpha2` venv에 pytest 미설치).

## 7. 알려진 위험 (미해결)

1. **오래된 Odometry로 계속 주행한다.** Odometry가 끊겨도 마지막 pose가 남아 있어 2.5초 동안 그 값으로 속도를 계산해 발행한다. 최대 약 0.75 m를 피드백 없이 이동할 수 있다. 정상 상황에서도 1 Hz 피드백 때문에 최대 약 1초 묵은 위치를 쓴다.
2. **신선도를 수신 시각(벽시계)으로 판정한다.** `header.stamp`와 시뮬레이션 시간 기준 정합성은 확인하지 않는다.
3. **프로세스가 강제 종료되면 정지 명령이 나가지 않는다.**
   - `SignalHandlerOptions.NO` 상태라 SIGTERM을 받으면 `finally` 없이 종료된다. SIGKILL이나 충돌은 막을 수 없다.
   - Ctrl+C를 두 번 누르면 정지 발행 루프가 중간에 끊긴다.
   - 독립적인 Watchdog이 없다.
4. **실제 정지 여부를 자동으로 확인하지 않는다.** 이번에는 사용자가 `topic echo`로 수동 확인했다.
5. **위치 피드백의 정상 여부를 검증하지 않는다.** `frame_id`/`child_frame_id` 확인과 위치 점프(리셋 등) 감지가 없다.
6. **Ground Truth로 판정하지 않는다.** 최종 판정도 Odometry로 한다.
7. **제어 방식의 한계.**
   - 직진 중에는 방향을 보정하지 않는다. 1 m 이동 시 옆으로 최대 약 8.7 cm까지 벗어날 수 있다.
   - 목표 앞 5~10 cm 구간에서는 옆으로 조금만 벗어나도 다시 제자리 회전을 한다.
8. **좌표 변환 경계 문제.**
   - `world_to_grid`는 반올림 시 짝수 쪽을 택한다 (0.5→0, 1.5→2).
   - 셀 중심에서 0.49 m 떨어져 있어도 그 셀로 판정한다.
   - row/col/x/y 입력을 검증하지 않는다.
9. **설정이 하드코딩되어 있다.** Topic(`/model/vehicle_blue/...`)과 제어 상수가 코드에 박혀 있고, 허용 오차 0.05가 세 곳에 중복된다.
10. **테스트 공백.** 제어 Node의 Timeout·비정상 입력·정지 처리에 대한 단위 테스트가 없다.

## 8. 미검증 항목

- 자동 회전 후 직진하는 복합 동작
- Gazebo Ground Truth 위치
- 다중 로봇 제어
- 이동 중 물리적 충돌
- 독립적인 Watchdog
- IL Runner 및 Selective Hybrid 실시간 연결
- 반복 실행과 통계 (n=1)

## 9. 후속 작업 — 다중 로봇 확장 전 필수

다중 로봇으로 확장하기 **전에** 아래 항목을 해결하거나 검증해야 한다.

- [ ] 오래된 Odometry 감지 및 즉시 정지
- [ ] `header.stamp` 기반 신선도 검사와 시뮬레이션 시간 기준 정합성
- [ ] 독립적인 Watchdog (명령 타임아웃 시 정지)
- [ ] SIGTERM 종료 처리
- [ ] 실제 정지 여부 자동 확인 (Odometry twist ≈ 0)
- [ ] `frame_id` 및 위치 점프 검증
- [ ] Ground Truth 위치 검증
- [ ] 제어 Node Timeout 및 비정상 입력 단위 테스트
- [ ] 자동 회전 후 직진 복합 동작 검증
- [ ] Topic 및 제어 파라미터 설정화
- [ ] 좌표 변환 경계 및 셀 중심 도착 판정
- [ ] 이후 다중 로봇 Lockstep 연동
