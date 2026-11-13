# Selective Hybrid 실제 연결·E2E 체크포인트 (2026-11-13)

- 브랜치 `feature/ina-selective-real-integration`, 기준 HEAD `f7be39f` (그 위의 미커밋 변경을 이 문서와 함께 정리)
- 이 문서는 **DEV 검증 기록**이다. 모든 시나리오는 개발용 탐색 사례이며 **독립 held-out 성능 평가가 아니다.** 성능 우위를 주장하지 않는다.
- 합의 정책(P1~P5, `docs/selective_hybrid_contract_v0.md` 0장)은 바꾸지 않았다. 아래 수치(예산·timeout)는 모두 **DEV 값**이며 O-1/O-2는 여전히 OPEN_DECISION이다.
- 표기: **[실제]** 실제 atb033 solver·실제 IL·실제 Simulator 실행 결과 / **[Mock]** solver 또는 경로를 대체한 자동 테스트

## 1. 구현 상태 (코드)

| 항목 | 위치 | 상태 |
|---|---|---|
| 실제 Subset CBS 어댑터 | `src/hybrid/subset_cbs_adapter.py` `SubsetCbsPlanner` | 그룹만 CBS, 그룹 밖 Agent = 복사한 grid의 정적 장애물, Agent ID·`(row, col)` 보존, solver 제한 `(timeout_ms − overhead_reserve_ms)/1000` float(올림 없음), 호출마다 새 작업 폴더, 성공 후 전체 Agent 검증 |
| Trigger 예산을 지키는 G | 같은 파일 `BoundedGlobalCbsPlanner` | 같은 경로를 전체 Agent로. `timeout_ms` 없으면 실행 거부(`input_error`) |
| SelectivePlanner opt-in 게이트 | `src/hybrid/selective_planner.py` | 실제 subset planner는 **정확히 `SubsetCbsPlanner` + `allow_real_subset_cbs=True`** 일 때만, 실제 G는 **정확히 `BoundedGlobalCbsPlanner`** 일 때만 허용. 그 외 비-`test_only` planner 거부. 기본 G 모드 경로는 변경 없음 |
| 실제 호출 기록 | 같은 파일 | `real_subset_cbs_called` / `real_global_cbs_called`는 실제 planner 자신의 호출 기록(`called=True`)에서만 True. Mock은 `planner_kind="MOCK"`, `solver_invoked=None` |
| 예산 규칙 | 같은 파일 | 받은 `timeout_ms`를 넘긴(또는 지연을 모르는·NaN·음수) `success`는 실행하지 않고 `timeout`으로 사다리에 전달(`raw_status` 보존, 초과분 차감). 실행 직전 누적 지연 > `total_time_ms`면 `SAFE_STOP:TIME_BUDGET_OVERRUN` |
| 변경하지 않은 것 | `path_guard.py`, `global_hybrid.py`, Simulator, solver, `wfg.py`, `selective_safety.py`, `selective_compose.py`, Monitor, 시나리오 원본 | — |

### 오래된 설명과의 충돌 (이번에 고치지 않음, 기록만)

- `docs/selective_hybrid_contract_v0.md` 상단·0.2·0.3장의 "Selective Runner / 그룹 CBS / 그룹 밖 WAIT 경로 미구현"은 **2026-11-13 기준으로 오래된 설명**이다. 해당 문서 맨 위에 이 문서로 가는 상태 안내만 추가했고 본문(합의·이력)은 그대로 두었다.
- 소스 docstring 중 오래된 문장(수정하지 않음): `src/hybrid/wfg.py`("nothing in the runtime calls it"), `src/hybrid/selective_safety.py`("there is no Selective Runner"), `scripts/run_selective_mock_dev.py`의 `--real` 거부 메시지("no real subset CBS planner exists"; 거부 동작 자체는 여전히 맞음), `docs/warehouse_generator_v3_plan.md` 6행("Selective Runner는 연결되지 않았고"). Selective는 **opt-in DEV 경로로만** 연결되어 있고 기본 런타임(Global)은 이를 사용하지 않는다.

## 2. 검증 결과

> **증거 위치:** 아래 [실제] 결과의 원본 기록(`result.json`, `summary.json`, `predictions.json`, CBS 작업 폴더)은 실행한 PC의 `outputs/selective_real_e2e_dev/`, `outputs/selective_real_e2e_dx_dev/`, `outputs/selective_p3_dev/`에만 있는 **로컬 전용 실행 증거이며 Git에 포함되지 않는다.** 이 커밋에 들어 있는 것은 그 결과를 다시 만드는 DEV 스크립트(6장)와 자동 테스트뿐이다. 다른 환경에서는 6장 명령으로 재실행해 확인해야 한다.

DEV 설정: `TriggerBudget(total 30 000 ms, max_attempts 3, L0 10 000 / L1 10 000 / G 20 000 ms)`, `overhead_reserve_ms=500`, `h_exec=3`, `DevResumeIL`, `DevSingleCbsCall`, max_steps 160. Solver atb033 `89f69e11944b172e11c3bcbbecc57344eaa4be9c`(변경 없음), `PYTHONHASHSEED=0`, Windows 11, 단일 프로세스 순차 실행.

| 시나리오 | 결과 | 증거 |
|---|---|---|
| S1 (corridor 원본 + Agent 2 = (2,8)) | 부분 재계획 **FAIL** / 안전 PASS. WFG {0,1,2} = 전체(`occupy` 0→2, 1→2) → L0/L1 SKIP → G timeout → SAFE_STOP, 이동 0 | [실제] |
| **S1-B** | 실제 Selective E2E **12/12 PASS** (L0 {0,1}, 그룹 밖 2 WAIT, IL_RESUME → SUCCESS) | [실제] |
| S2 (Agent 2가 대피칸 (1,5)) | L0 timeout 9 589.6 ms → G timeout 19 552.6 ms → SAFE_STOP, 이동 0 | [실제] |
| S3 (bottleneck 4) | WFG {0,1},{2,3} → 합집합 = 전체 → SKIP → G 성공 2 399.7 ms → 에피소드 TIMEOUT (Global과 동일) | [실제] |
| **D1** (intersection 7×7 + Agent 2 (0,3) 정지) | 실제 Selective E2E **SUCCESS 12/12** | [실제] |
| **D2** (staggered blocks 9×9 원본) | 실제 Selective E2E **SUCCESS 12/12** (L0 {0,2}, 그룹 밖 1·3) | [실제] |
| D3 (loop w2 7×9 원본) | WFG 4 Agent 연결 → L0/L1 SKIP → G SUCCESS | [실제] |
| P3Q (intersection + Agent 2 (0,3)→(6,3)) | `occupy(2→1)` → WFG 전체 → G timeout → **SAFE_STOP (P3 아님, 분류 C)** | [실제] |
| P3F (DEV 지도 7×11 + Agent 2 전용 통로 이동) | Agent 2가 6회 이동하는 동안 Trigger 지연(t=2 → t=7), Trigger 시 2는 목표 도착 → Selective SUCCESS이나 **P3 조건 미충족 (분류 D)** | [실제] |
| Global 기본 G 모드 회귀 | 5/5 동일 (CORRIDOR, S1, S2, S1-B, S3; latency·작업 폴더만 제외) | [실제] |
| 실패 주입 F1~F5 (S1-B 기반) | 5/5 PASS: 그룹 밖 이동 → EXEC_MISMATCH, 변조 경로 → invalid_path, 실제 L0 timeout → G, 늦은 success 실행 불허, 겹침 → INTEGRITY_ABORT | [실제]+주입 |
| P3 비정상 X1~X5 (P3F 기반, 그룹 밖 Agent는 목표에 정지) | 5/5 PASS: 이동 → EXEC_MISMATCH, 정적 장애물 진입 → `obstacle at (6,10)` 거부, 순간이동 → 거부, 전 단계 실제 timeout → SAFE_STOP, 그룹 목표 점유 → solver 호출 없이 `GOAL_BLOCKED` | [실제]+주입 (X5는 planner 단계, solver 미호출) |
| 회귀 재실행 (S1-B, D1, D2) | 3/3 동일 판정 | [실제] |

**S1-B 지도 변경(원본 대비):** `scenarios/controlled/map_corridor_bay_n2.npy`(5×11)의 셀 **(3,1) 한 칸만 1 → 0**(3행 `11111111111` → `10111111111`). 메모리 사본만 변경, 원본 파일 불변. Agent 2 시작 = 목표 = (3,1). S1 실패를 본 뒤 설계한 DEV 성공 사례다.

**자동 테스트 [Mock]:** `tests/test_selective_real_wiring.py`(20개 중 15개 Mock, 5개 [실제] planner 단계), `tests/test_selective_real_e2e_dev.py`(7, `subprocess.run`만 대체), `tests/test_selective_real_e2e_dx_dev.py`(4, 합성 기록), `tests/test_selective_p3_dev.py`(7, 합성 기록, 실행 전에 작성). Mock 테스트는 판정·차단 로직의 증거이며 실제 계획 성공의 증거가 아니다. Mutation: 새 안전 규칙에 주입한 결함 14/14 검출.

### 측정값 (PC 1대, 표본 작음)

| 측정 | 값 |
|---|---|
| 2-Agent subset 실제 solver 성공 (X2/X3 변조 전 포함) | 286.8–431.8 ms (약 20회) |
| 3-Agent G 성공 (S1-B 계열) | 270–413 ms |
| 4-Agent G 성공 (S3, D3) | 2 399.7 / 2 432.3 ms (S3), 464.7 ms (D3) |
| reserve 0 timeout 초과분 | 57–105 ms (probe), 58.7–87.1 ms (F3/X4) |
| 60 ms 요청 실측 | 118.7–147.1 ms (프로세스 시작 하한) |
| 해 없는 L0 | `no_solution` 대신 timeout까지 실행 (S2: 9.6 s) |

solver는 timeout 시 종료되지만(잔존 프로세스 0) wall-clock 상한은 보장되지 않는다. **실시간 보장이 아니다.**

## 3. P3 구조적 한계 (확인된 사실)

1. v0 Trigger는 `TRIGGER_FULL_REPEAT` = **전체 joint state가 이전 상태와 같을 때만** 발생한다 (`src/common/deadlock_monitor.py` `observe_step`, `state in self._seen`). Trigger 시점에 모든 Agent는 `first_seen` 때와 같은 칸에 있다.
2. 저장된 실제 실행 7건(IL 830 Agent-step)에서 IL은 **목표 밖에서 스스로 WAIT하거나 왕복한 적이 0회**였다. 목표 밖에서 멈춘 Agent는 모두 점유·경합 칸으로 가려다 막힌 경우였고, 이 경우 WFG `occupy`/`contest` 간선이 생겨 L0에 들어간다.
3. 따라서 v0 자연 Trigger에서 그룹 밖 Agent는 사실상 **목표에 정지한 Agent**다(S1-B, D1, D2, P3F 모두 해당). P3F는 무관하게 움직이는 Agent가 교착 감지를 5 step 늦춘다는 것도 보였다.
4. **"이동 중이던 그룹 밖 Agent → WAIT → IL_RESUME → 이동 재개"는 실제 Runner에서 미검증**이다. 이 흐름의 증거는 Mock뿐이다(`tests/test_selective_planner_e2e.py`, 스크립트로 왕복시킨 Agent 7).

## 4. 남은 안전성 위험

- 그룹 밖 WAIT 정책(P3-b)은 **목표에 정지한 Agent에 대해서만** 실제로 검증됨.
- wall-clock 초과(관측 최대 105 ms), Trigger 예산과 에피소드 예산 미연결(O-2): Trigger 하나가 에피소드 남은 예산보다 오래 돌 수 있음.
- 해 없는 하위 문제에 CBS가 배정 시간을 모두 소모(S2: 정지 약 29 s).
- L0 합집합·L1 1단계 규칙의 효용은 사례 수가 적어 판단 불가. PATH_BLOCKED는 런타임에서 계산되지 않음(`group_paths` 미전달).
- CBS 호출이 프로세스 전역 cwd를 바꾸므로 병렬 실행 불가.
- 실물 로봇: CBS 계산 중 정지 정책 없음.

## 5. 다음 개발 계획 — Partial Trigger 후보 [제안, 미구현]

- **FULL 반복 vs 부분 그룹 정체:** FULL은 모든 Agent가 같은 상태로 돌아와야 해서, 무관한 Agent 하나라도 움직이면 발생하지 않는다(P3F). 부분 정체는 일부 Agent만 반복·정지하고 나머지는 진행 중인 상태다.
- **이미 있는 관측 신호:** Monitor는 Trigger와 무관하게 `current_persistent_conflict_streak`/`current_persistent_conflict_key`(같은 충돌 키가 연속되는 길이)와 `current_no_progress_streaks`(Agent별 목표 거리 무개선 연속 길이)를 기록한다. 둘 다 **관측 전용**이며 Trigger를 바꾸지 않는다. 후보 규칙: "충돌 키의 streak ≥ K 이고 그 Agent들의 no_progress streak ≥ K"이면 부분 Trigger. K는 미정.
- **계약 유지:** 부분 Trigger에서도 WFG는 S_{t-1} + intended, CBS 시작은 S_t(P1-c/d). 그룹 밖 Agent는 이동 중일 수 있으므로 P3-c(고정되지 않은 Agent를 장애물로 가정 금지)가 핵심이다. 실행 창 동안 WAIT 강제 + 매 step 전체 Agent expected/actual 비교는 그대로 필수.
- **Global 격리:** 부분 Trigger는 Selective opt-in 경로 전용 별도 설정으로만. 기본 Global(FULL Trigger) 경로·결과는 바이트 단위 동일성을 회귀 테스트로 유지.
- **P3 테스트 설계:** (a) 실제 IL 그룹 + 그룹 밖 Agent만 진행 중인 장면을 부분 Trigger로 잡는 DEV 시나리오, 예측·판정 기준을 실행 전에 고정. (b) 판정은 `scripts/run_selective_p3_dev.py`의 A~E 분류를 그대로 사용(A = 이동 중이던 Agent가 WAIT → IL_RESUME 후 실제 재이동 → 목표 도달). (c) 실패 주입 X1~X5 재실행.
- 필요한 합의: 부분 Trigger 규칙과 K, 그룹 밖 이동 Agent의 WAIT이 안전한 조건(감속·정지 거리 등 실물 정책 포함), O-2(에피소드 예산 연결).

## 6. 재현

```
set PYTHONHASHSEED=0
python -m scripts.run_selective_real_e2e_dev    --solver-root <atb033>/centralized/cbs   # S1, S2, S1-B, S3, G 회귀, F1~F5
python -m scripts.run_selective_real_e2e_dx_dev --solver-root <atb033>/centralized/cbs   # D1~D3 + Global 비교
python -m scripts.run_selective_p3_dev          --solver-root <atb033>/centralized/cbs   # P3Q, P3F, X1~X5
```

세 스크립트 모두 solver 커밋 `89f69e1`과 `PYTHONHASHSEED=0`이 아니면 실행을 거부하고, 결과는 `outputs/` 아래 새 폴더에만 쓴다(기존 폴더 재사용 거부). 결과 폴더(`outputs/selective_real_e2e_dev/`, `outputs/selective_real_e2e_dx_dev/`, `outputs/selective_p3_dev/`)는 저장소에 포함하지 않는다. 실제 solver 테스트는 `ALPHA2_CBS_SOLVER_ROOT`가 atb033 `cbs` 폴더를 가리킬 때만 실행된다.
