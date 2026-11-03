# Global Hybrid Runner v0 (Mock CBS + 실제 CBS Corridor DEV E2E)

## 1. 범위
- IL → TRIGGER → CBS_REQUEST → CBS_EXEC → IL_RESUME / SAFE_STOP 상태기계 1차 구현.
- 상태기계는 Mock planner로 검증했고, 이어서 **실제 CBS Runtime Adapter(`run_cbs_runtime`)와 Path Guard(`validate_path_guard`)를 연결**해 Corridor DEV E2E를 통과했다 (6절).
- 기존 `scripts/run_il.py`, `src/simulator/`, `src/common/`는 변경하지 않음 (import만 사용).
- H_exec, IL 복귀, CBS 재호출, SAFE_STOP 최종 정책은 **CBS 담당(윤지)과 합의 대기** (5절). 6절 E2E의 "전체 경로 실행"은 DEV_ONLY 정책이다.

| 파일 | 역할 |
|---|---|
| `src/hybrid/replan_boundary.py` | CBS 결과 수용 경계 (Path Guard 알고리즘 아님) |
| `src/hybrid/global_hybrid.py` | 상태기계 `run_global_hybrid_episode()` |
| `src/hybrid/cbs_runtime_bridge.py` | 실제 CBS planner(`CbsRuntimePlanner`) + 실제 Path Guard validator(`PathGuardValidator`) |
| `scripts/run_corridor_hybrid_e2e.py` | Corridor DEV E2E 실행 스크립트 (DEV_ONLY 정책 포함) |
| `tests/hybrid_mocks.py` | Mock planner, test-only validator, 정책 stub, 오류 주입 Simulator |
| `tests/test_hybrid_replan_boundary.py` | 경계 단위 테스트 |
| `tests/test_global_hybrid_runner.py` | 상태기계 테스트 |
| `tests/test_cbs_runtime_bridge.py` | Agent ID 계약(`{3, 7}` 포함), 실제 Path Guard, Solver 작업 폴더 격리 |
| `tests/test_corridor_hybrid_e2e.py` | 실제 IL·Simulator·Monitor로 timeout / solver_error / no_solution / invalid_path / 위치 불일치 / 중복 Trigger |

## 2. CBS 계약
```
planner(current_positions, goals, grid) -> ReplanResult(status, paths, latency_sec, error)
status ∈ {success, timeout, no_solution, invalid_path, input_error, solver_error}
```
| `planner.paths_keying` | 입력 | `paths` | 사용처 |
|---|---|---|---|
| `"index"` (기본) | 리스트, 순서 = sorted(agent_id) | `paths[i]` ↔ i번째 agent | Mock planner, `replan_with_status` |
| `"agent_id"` | `{agent_id: (row, col)}` dict | `paths[agent_id]` | `CbsRuntimePlanner` → `run_cbs_runtime` |

## 3. 수용 경계 (`accept_replan_result`)
순서대로 검사하고 하나라도 실패하면 계획 전체를 거절한다 (이동 0회).

| 단계 | 실패 코드 |
|---|---|
| status == success | `CBS_<STATUS>`, `CBS_UNKNOWN_STATUS` |
| paths 형식/개수 (index 모드: 0..n-1, agent_id 모드: `agent_order`와 같은 ID 집합) | `BOUNDARY_PATHS_TYPE`, `BOUNDARY_PATH_COUNT` |
| 셀 형식, 빈 경로 | `BOUNDARY_MALFORMED_PATH`, `BOUNDARY_EMPTY_PATH` |
| Path[0] == S_t | `BOUNDARY_START_MISMATCH` |
| 모든 이동을 `spec.delta_to_action`으로 사전 변환 (점프/대각선 거절) | `BOUNDARY_NON_ADJACENT` |
| 주입 validator (통합 시 CBS 브랜치 `src/path_guard.py`) | `VALIDATOR_REJECT:<사유>`, `VALIDATOR_ERROR` |

- Simulator의 "잘못된 이동 → 제자리" 동작에 의존하지 않는다.
- `test_only = True` validator(예: `AcceptAllValidator`)는 `test_only` planner와만 사용 가능 (`check_validator_pairing`). 실제 planner와 조합하면 `ValueError`.

## 4. 상태 전환
| 현재 | 조건 (우선순위) | 다음 |
|---|---|---|
| IL | overlap / 완료 / max_steps | INTEGRITY_ABORT / SUCCESS / TIMEOUT |
| IL | 새 Trigger (`trigger_timestep == t`, 미처리), CLEAN, `recall_policy` 허용 | TRIGGER |
| TRIGGER | S_t = 현재 Simulator 위치 복사 | CBS_REQUEST |
| CBS_REQUEST | planner 예외 / 경계 거절 | SAFE_STOP (`PLANNER_EXCEPTION` / 경계 코드) |
| CBS_REQUEST | 경계 통과 | CBS_EXEC |
| CBS_EXEC | overlap | INTEGRITY_ABORT |
| CBS_EXEC | 실제 위치 ≠ 계획 위치 | SAFE_STOP (`EXEC_MISMATCH`), 다음 이동 없음 |
| CBS_EXEC | 완료 / max_steps | SUCCESS / TIMEOUT |
| CBS_EXEC | `h_exec` 또는 경로 끝 도달 | IL_RESUME |
| IL_RESUME | `resume_policy.after_exec()` | IL 또는 SAFE_STOP (`RESUME_POLICY_SAFE_STOP`) |
| SAFE_STOP | `safe_stop_handler.on_safe_stop()` 기록 후 종료 | (종료, 추가 이동 0회) |

안전 불변식: 동일 Trigger 재호출 금지(Monitor는 첫 반복 이후에도 `TRIGGER_FULL_REPEAT`를 계속 보고함), overlap 즉시 중단, 거절·불일치·SAFE_STOP 이후 이동 없음.

## 5. 미합의 항목 (필수 주입, 기본값 없음)
| 항목 | 주입 지점 | 테스트 임시값 |
|---|---|---|
| H_exec | `GlobalHybridConfig.h_exec` | 1, 2 |
| IL 복귀 조건 | `resume_policy` | `AlwaysResumeIL`, `AlwaysSafeStopAfterExec` |
| CBS 재호출 | `recall_policy` | `AllowAllRecalls`, `DenyAllRecalls` |
| SAFE_STOP 세부/로그 | `safe_stop_handler` | `RecordSafeStop` |

- v0의 SAFE_STOP은 "추가 이동 없이 종료"만 구현. WAIT 유지 등 다른 방식, CSV 기록 규칙은 합의 후 결정. CSV 스키마는 변경하지 않았다.
- 현재 Monitor는 에피소드당 한 번만 Trigger하므로, 재호출 정책이 허용해도 실제 CBS 호출은 최대 1회. 재무장 방식은 합의 대상.

## 6. 실제 CBS 연결 (Corridor DEV E2E, 2026-10-30)
CBS 브랜치 `origin/feature/global-hybrid-integration@65932d7`에서 파일 단위로만 가져옴:
`src/{cbs_adapter,replanner,replan_status,path_guard,cbs_runtime_adapter}.py`, 관련 테스트 5개
(`test_simulator_ki1_regression.py` 포함 — 우리 Simulator PASS, KI-1 수정 전 Simulator FAIL 확인).

| 항목 | 확인 결과 / 연결 방식 |
|---|---|
| 실제 planner | `src/hybrid/cbs_runtime_bridge.py::CbsRuntimePlanner` → `run_cbs_runtime(dict, dict, grid)` |
| Agent ID 변환 | `run_cbs_runtime` 내부 1회뿐. planner `paths_keying="agent_id"` → Runner는 dict 전달, Boundary는 `paths_keying="agent_id"`로 재매핑 없이 수용. Mock은 기본 `"index"` 유지 |
| 실패 시 | `paths=None`, `latency_sec` 있음, `error` 문자열 → `cbs_calls[i]["error"]`에 기록 |
| Path Guard | 어댑터 내부 2회 + Boundary `PathGuardValidator`(실제 `validate_path_guard`, test_only 아님) 1회 |
| Solver 파일 | `replanner.py`가 cwd 기준 고정 `outputs/paths/replan`에 덮어씀 → bridge가 호출마다 새 `<work_root>/cbs_call_NN`을 cwd로 사용(재사용 금지) |
| 의존성 | PyYAML 6.0.3 (`alpha2_test_env`), atb033 `89f69e1` — `--solver-root`로 명시 (submodule 미사용) |

DEV 실행: `python -m scripts.run_corridor_hybrid_e2e --solver-root <atb033>/centralized/cbs`
→ `outputs/hybrid_e2e/<run_id>/` (result.json, steps CSV, Solver YAML). 실행마다 새 폴더를 만들고, 이미 있으면 중단한다.

### 6.1 Corridor DEV E2E 결과 (`scenario_corridor_bay_n2`, 실제 NavHint IL + 실제 CBS)
| 항목 | 결과 |
|---|---|
| 최초 Trigger | t=5, `TRIGGER_FULL_REPEAT` (두 agent가 (2,5)로 진입하려다 vertex 충돌로 제자리) |
| 실제 S_t | `{0: (2,4), 1: (2,6)}` |
| CBS | `success`, 약 294 ms, 어댑터 Path Guard + Boundary Path Guard 통과 |
| Path[1] | `{0: (2,4), 1: (2,5)}` = 첫 CBS step(t=6) 실제 위치 |
| 실행 | t=6~13 (8 step) 모든 step expected = actual, overlap 0 |
| 종료 | `SUCCESS`, `all_at_goal=True`, steps=13 |

판정 11개 항목 모두 PASS. 범위는 Corridor 시나리오 1개, 1회 실행, CBS 1회 호출이다 (Bottleneck/Intersection 미검증).

### 6.2 DEV_ONLY 정책 (합의된 정책 아님)
| 항목 | 이번 E2E (DEV_ONLY) | 최종 정책 |
|---|---|---|
| H_exec | `max_steps` = CBS 전체 경로 실행 | 윤지와 합의 대기 |
| IL 복귀 | 경로 종료 후 목표 미도달이면 SAFE_STOP (IL 복귀 없음) | 합의 대기 |
| CBS 재호출 | 에피소드당 1회 | 합의 대기 |
| SAFE_STOP | 사유·t·위치 기록 후 추가 이동 0회 | 합의 대기 |

DEV_ONLY 정책은 `scripts/run_corridor_hybrid_e2e.py` 안에만 있고, `src/`에는 기본값이 없다 (`GlobalHybridConfig` 필수 주입 유지).

### 6.3 알려진 한계
- **Solver YAML 고정 경로**: `src/replanner.py`는 cwd 기준 `outputs/paths/replan`에 매 호출 덮어쓴다. bridge는 호출마다 cwd를 새 `<work_root>/cbs_call_NN`으로 바꿔 격리하지만, cwd 변경은 프로세스 전체에 적용되므로 **여러 스레드나 동시 실행에서 CBS를 호출하면 안 된다.** 근본 해결은 CBS 측에 `work_dir` 인자 추가를 요청하는 것이다.
- **CSV**: steps CSV 스키마는 변경하지 않았다. CBS status·latency 전용 열이나 IL/CBS step 구분 열이 없고(`cbs_latency_ms`는 비어 있음), Monitor가 Trigger 이후에도 `TRIGGER_FULL_REPEAT`를 계속 보고한다. CBS 구간은 `result.json`의 `cbs_calls`, `action_sources`, `cbs_exec_trace`로 확인한다.
- **Solver 의존 테스트**: `tests/test_corridor_hybrid_e2e.py`의 위치 불일치·중복 Trigger 테스트는 실제 Solver(`ALPHA2_CBS_SOLVER_ROOT` 또는 `../MAPF_ALPHA2_CBS_CHECK` 복제본)가 없으면 skip된다.
- Path Guard는 어댑터 내부 2회 + Boundary 1회, 총 3회 실행된다 (의도한 중복).

## 7. 세 시나리오 DEV 회귀 (2026-11-03, 실제 NavHint IL + 실제 CBS, `256f54e` 기준)

### 7.1 전체 경로 실행 (`scripts/run_corridor_hybrid_e2e.py --scenario ...`, 코드 변경 없음)
| 항목 | Corridor | Intersection | Bottleneck |
|---|---|---|---|
| 최초 Trigger (KI-1 수정 후 재측정) | t=5 | t=2 | t=2 |
| S_t | `{0:(2,4),1:(2,6)}` | `{0:(3,2),1:(2,3)}` | `{0:(1,2),1:(2,1),2:(1,7),3:(3,7)}` |
| CBS / latency | success / 290 ms | success / 321 ms | success / 3,574 ms |
| CBS 실행 step (= 경로 길이) | 8 | 4 | 13 |
| expected=actual, overlap | 전 step 일치, 0 | 전 step 일치, 0 | 전 step 일치, 0 |
| 종료 | SUCCESS (13 step) | SUCCESS (6 step) | SUCCESS (15 step) |

### 7.2 DEV_ONLY H_exec=3 + IL 복귀 (`scripts/run_hybrid_resume_dev.py`)
DEV_ONLY 정책(합의된 정책 아님): H_exec=3, CBS 창 종료 후 항상 IL 복귀(`DevResumeIL`), CBS 호출 에피소드당 최대 1회.
"IL_RESUME 도달"(`CBS_EXEC → IL_RESUME` 전이)과 "실제 IL 재실행"(복귀 후 `policy.act()`·`sim.step()` 호출 수)을 따로 기록한다.

| 항목 | Corridor | Intersection | Bottleneck |
|---|---|---|---|
| Trigger / S_t | t=5, 7.1과 동일 | t=2, 7.1과 동일 | t=2, 7.1과 동일 |
| CBS / latency | success / 290 ms | success / 277 ms | success / 2,800 ms |
| CBS 실행 (창 / 경로 길이) | 3 / 8 | 3 / 4 | 3 / 13 |
| expected=actual, overlap | 3/3 일치, 0 | 3/3 일치, 0 | 3/3 일치, 0 |
| IL_RESUME 도달 | t=8 | t=5 | t=5 |
| 복귀 후 IL step (act / sim.step) | 5 (5 / 5), 첫 t=9 | 1 (1 / 1), 첫 t=6 | 155 (155 / 155), 첫 t=6 |
| 복귀 후 재정체 | 없음 | 없음 | **t=7 (first_seen=6, period 1)** |
| Monitor 재Trigger | 없음 | 없음 | 없음 (첫 Trigger에서 고정) |
| 종료 | **SUCCESS** (13 step) | **SUCCESS** (6 step) | **TIMEOUT** (160 step, all_at_goal=False) |
| 판정 | PASS (14/14) | PASS (14/14) | **FAIL** (`all_at_goal`) |

**Bottleneck IL 복귀 실패 원인**: CBS 경로는 agent 2가 (0,5)에서 비켜 대기하는 동안(k=3~4) agent 0·1이
문을 지나 동쪽으로 빠지는 대기·우회 경로였다. H_exec=3에서 이 경로가 끝나기 전(k=3, t=5)에 IL로 복귀했고,
IL은 t=6에 agent 2를 (1,5)로 다시 내려 계획(k=4: (0,5) 대기)과 달라졌다. 그 결과 복귀 직후(t=6) 네 agent가 문 동쪽 칸 (2,5) 주변에 모인다
(0:(2,5), 1:(2,4) 문, 2:(1,5), 3:(3,5)). t=7부터 agent 1·2·3은 (2,5)로, agent 0은 (1,5)로 가려 하고
모든 목표 칸이 점유되어 전원 vertex WAIT → 같은 상태가 160 step까지 반복된다. Monitor는 첫 Trigger(t=2)
이후 고정되어 이 재정체를 Trigger하지 않으며, DEV 재호출 제한(1회)도 있어 CBS가 다시 불리지 않는다.
즉 H_exec=3은 Bottleneck에서 IL이 혼자 통과할 수 있는 상태까지 데려가지 못했다.

### 7.3 이 실험에서 드러난 합의 필요 항목 / 한계
- **Monitor 재무장**: 현재 Monitor는 에피소드당 한 번만 Trigger한다. IL 복귀 후 재정체는 사후 분석(`find_repeat_after`)으로만 확인된다. 재호출 정책을 허용해도 재Trigger가 없으므로 실제 호출은 1회뿐이다.
- **H_exec**: 경로 전체 실행은 세 시나리오 모두 SUCCESS, H_exec=3은 Bottleneck에서 실패. H_exec와 복귀 조건(예: 병목 통과 여부) 결정이 필요하다.
- **CBS 경로 차이**: 동일한 Bottleneck S_t에서 7.1과 7.2의 CBS 결과로 서로 다른 유효 경로가 관찰됐다(Path[1]의 agent 3: (3,6) vs (4,7), 둘 다 Path Guard 통과). 원인은 미확인이다.
- 각 결과는 1회 실행이다. 세 시나리오의 Agent ID는 0..n-1이라 ID/인덱스 혼동은 실제 실행으로 구별되지 않으며, 단위 테스트(`test_cbs_runtime_bridge.py`, id 3·7)로만 검증된다.
