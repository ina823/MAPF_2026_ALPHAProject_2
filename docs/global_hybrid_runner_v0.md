# Global Hybrid Runner v0 (Mock CBS)

## 1. 범위
- IL → TRIGGER → CBS_REQUEST → CBS_EXEC → IL_RESUME / SAFE_STOP 상태기계 1차 구현.
- 실제 CBS(`feature/cbs-path-guard`)는 **연결하지 않음**. Mock planner로만 검증.
- 기존 `scripts/run_il.py`, `src/simulator/`, `src/common/`는 변경하지 않음 (import만 사용).

| 파일 | 역할 |
|---|---|
| `src/hybrid/replan_boundary.py` | CBS 결과 수용 경계 (Path Guard 알고리즘 아님) |
| `src/hybrid/global_hybrid.py` | 상태기계 `run_global_hybrid_episode()` |
| `tests/hybrid_mocks.py` | Mock planner, test-only validator, 정책 stub, 오류 주입 Simulator |
| `tests/test_hybrid_replan_boundary.py` | 경계 단위 테스트 |
| `tests/test_global_hybrid_runner.py` | 상태기계 테스트 |

## 2. CBS 계약 (CBS 브랜치 handoff 기준, 실제 브랜치 미확인)
```
planner(current_positions, goals, grid) -> ReplanResult(status, paths, latency_sec, error)
status ∈ {success, timeout, no_solution, invalid_path, input_error, solver_error}
paths[i] ↔ 입력 리스트 i번째 agent  (입력 순서 = sorted(agent_id))
```

## 3. 수용 경계 (`accept_replan_result`)
순서대로 검사하고 하나라도 실패하면 계획 전체를 거절한다 (이동 0회).

| 단계 | 실패 코드 |
|---|---|
| status == success | `CBS_<STATUS>`, `CBS_UNKNOWN_STATUS` |
| paths 형식/개수, 인덱스 0..n-1 | `BOUNDARY_PATHS_TYPE`, `BOUNDARY_PATH_COUNT` |
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

## 6. 실제 CBS 통합 시 확인할 것
1. `replan_with_status` 시그니처와 `paths` 타입(list/dict)
2. 실패 시 `paths`/`latency_sec` 값
3. `replan_with_status` 내부의 Path Guard 수행 여부 → 경계에서 validator 이중 호출 여부 결정
4. `src/path_guard.py` 함수명·반환 형식 → `(ok, reason)` 어댑터 작성
5. PyYAML, `third_party` submodule의 테스트 환경 포함 여부
