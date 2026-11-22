# 11/23 Pure CBS / Safety-only 실행·계약 검증 결과

- **담당 작업일:** 2026-11-23 (실제 검증 실행: 2026-11-22, KST)
- **검증 범위:** 개발용 Pure CBS·Safety-only 비교, CBS 수용 경계·오류 상태 계약 점검
- **환경:** Windows PowerShell, Python `.venv`, 실제 atb033 CBS Solver, `PYTHONHASHSEED=0`
- **데이터 성격:** Development 시나리오 2개, 각 방법 1회 실행. Held-out Validation/Test 아님.
- **실행 배치:** `outputs/logs/compare_20261122_235630`

## 1. 계약 및 단위 테스트

실행 명령:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_hybrid_replan_boundary `
  tests.test_cbs_runtime_bridge `
  tests.test_corridor_hybrid_e2e `
  tests.test_safety_only `
  -v
```

- 사용자 확인: `OK` (실행한 테스트의 정확한 건수 및 SKIP 개수는 출력 미제공).
- CBS 상태 계약: `success`, `timeout`, `no_solution`, `invalid_path`, `input_error`, `solver_error`.
- CBS 경로 수용 경계: 경로 형식·agent ID·시작 상태·이동 유효성·Path Guard를 검증하고 거부된 경로는 실행하지 않는 구조.
- Safety-only v0: 최초 유효 Trigger 후 가장 낮은 ID의 미도착 로봇이 2 step WAIT, CBS 호출 없음. `hold_steps=2`는 DEV 설정.

## 2. 실제 비교 배치

비교 조건:

- 시나리오: `scenario_corridor_bay_n2.json`, `scenario_intersection_n2.json`
- 방법: `PURE_CBS_DEV_SINGLE_PLAN`, `SAFETY_ONLY_DEV_K2`
- `max_steps=160`, CBS Solver 호출 제한 `timeout_sec=30` (Pure CBS에만 적용)
- Pure CBS: `t=0`에서 CBS 1회 호출, 승인된 전체 경로 실행
- Safety-only: IL 정책 + Trigger 후 2 step WAIT 개입, CBS 호출 0회

| Scenario | Method | Termination | Steps | Stop reason |
| --- | --- | --- | ---: | --- |
| Corridor | PURE_CBS_DEV_SINGLE_PLAN | SUCCESS | 12 | None |
| Corridor | SAFETY_ONLY_DEV_K2 | TIMEOUT | 160 | None |
| Intersection | PURE_CBS_DEV_SINGLE_PLAN | SUCCESS | 5 | None |
| Intersection | SAFETY_ONLY_DEV_K2 | SUCCESS | 7 | None |

- **실행 수:** 4 episodes
- **CBS 호출 수:** 총 2회 (Pure CBS 시나리오별 1회)
- **비교 CSV 검증:** cell checks 248, runner checks 38
- **재현성:** `reproducible=True` / `PYTHONHASHSEED=0` / parent VERIFIED / inherited OK
- **공정성:** `fair_batch=False`; 이유 `project repository dirty or unknown`

따라서 해당 배치는 기능 검증 및 DEV 비교 자료로만 사용하고 **공정한 최종 성능 비교나 일반화 성능의 근거로 사용하지 않는다**.

## 3. 예산 및 오류 코드 계약 점검

- CBS 호출 오류는 위 6개 상태 계약에 따라 처리한다. `success` 외 상태는 경로 실행 승인 대상이 아니다.
- `PURE_CBS_DEV_SINGLE_PLAN`: 최초 상태 `S_0`에서 **1회** CBS 요청, 별도 재계획 없음.
- `SAFETY_ONLY_DEV_K2`: **CBS 호출 0회**. CBS timeout 예산은 비적용이며, WAIT 개입 2 step은 DEV 설정이다.
- 에피소드 상한 `max_steps=160`과 CBS subprocess 시간 제한 `timeout_sec=30`은 **서로 다른 제한**이다.
- 실제 호출별 지연시간의 수치는 제공된 콘솔 출력에 없으므로 이 보고서에 임의 기입하지 않는다. 원본 수치는 `compare_cbs_calls.csv`에서 확인할 수 있다.

## 4. 결론 및 남은 주의사항

1. Pure CBS 두 개발용 시나리오 모두 목표 도달 (`12`, `5` step).
2. Safety-only는 Intersection 성공 (`7` step), Corridor는 에피소드 제한 도달 (`160` step).
3. 오류 상태 처리와 Safety-only WAIT 동작의 단위 테스트 `OK` 확인. 테스트 SKIP 수는 재확인 필요.
4. 공정성 실패 사유는 프로젝트 dirty 상태이며, 실행 수가 적고 DEV 시나리오이므로 비교 결과를 연구 성능 우위로 일반화하지 않는다.
5. 실제 연구 비교 전 클린 커밋 환경에서 동일 조건 재실행 및 별도 Validation/Test가 필요하다.

## 5. 원본 증거

- `outputs/logs/compare_20261122_235630/compare_episodes.csv`
- `outputs/logs/compare_20261122_235630/compare_cbs_calls.csv`
- `outputs/logs/compare_20261122_235630/batch_manifest.json`

GitHub 증거 폴더에 CSV를 복사해 커밋할 경우 원본 `outputs/` 경로는 그대로 유지한다. `batch_manifest.json`에는 로컬 경로가 포함될 수 있으므로 공유 전 내용을 확인한다.
