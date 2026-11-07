# PILOT 데이터 품질 Gate v0

작성 2026-11-07. 11/6 읽기 전용 조사 결과(REVIEW 5건, OOD_WARNING 3건, PILOT-2 CBS timeout)를 **감사 가능한 기록과 반복 가능한 검사**로 옮긴 것이다. Gate는 기록하고 표시만 한다. **판정·병합·자동 실패·정책 확정은 하지 않는다.**

| 파일 | 역할 |
|---|---|
| `src/data/quality_gate.py` | 원장 검증, OOD 분류, 해결 가능성 기록 검증, 독립성 검사, gate 보고 (읽기 전용) |
| `registry/review_ledger_v0.jsonl` | REVIEW 관계 감사 원장 (append-only, 5줄) |
| `registry/pilot_solvability_v0.json` | PILOT별 CBS 결과와 CBS가 아닌 근거를 분리한 기록 |
| `scripts/check_pilot_gate.py` | gate 보고 출력. Solver 실행·파일 쓰기 없음 |
| `tests/test_data_quality_gate.py` | 단위 테스트 19개 |

기존 `instances_v1.jsonl`, `families_v1.json`, `il_input_reference_v1.json`, `scenarios/`, `checkpoints/`, Solver, Simulator, `outputs/`, 기존 `src/data/*.py`는 바꾸지 않았다.

## 1. REVIEW 감사 원장 (`review_ledger_v0.jsonl`)

한 줄 = REVIEW 관계 1건. 필드:

| 필드 | 뜻 |
|---|---|
| `review_id` | `RV0001` 형식, 중복 불가 |
| `a`, `b`, `a_instance_fp`, `b_instance_fp` | 두 instance ID와 instance identity(`registry.instance_fp`). ID가 바뀌거나 다른 instance를 가리키면 검증 실패 |
| `relation` | REVIEW 관계 유형 (`SUBMAP` / `NEAR_MAP` / `STRUCT_MATCH`) |
| `evidence` | 재계산 가능한 근거. `SUBMAP`이면 `smaller_shape`, `larger_shape`, `submap_placements`가 재계산 값과 같아야 한다 |
| `basis` | 사람이 읽는 검토 근거 (비어 있으면 실패) |
| `status` | v0에서는 **`HOLD`만** 허용 |
| `final_decision`, `decision_owner` | v0에서는 **반드시 `null`** (최종 판정 미완료) |
| `recorded_at`, `recorded_by` | 기록 날짜·기록자 |

검증 규칙(`validate_review_ledger`): ID·`instance_fp`가 레지스트리와 같다 / 그 관계가 `relate()`로 실제 재계산된다 / 근거 수치가 재계산 값과 같다 / 미해결 REVIEW 관계마다 원장 항목이 정확히 하나 있다.

**원장은 `ReviewBook`이 아니다.** `Registry.evaluate()`·`consistency_errors()`는 원장을 읽지 않으므로 HOLD를 적어도 관계는 계속 미해결이고, family는 합쳐지지 않으며, `UNRELATED`로 확정되지 않는다. RELATED/UNRELATED 판정은 승인된 별도 단계(원장 v1 + `ReviewBook` 연결)에서만 한다.

현재 5건 (`RV0001`~`RV0005`): `DEV_fcorr01_001…005` ↔ `PILOT_fint02_001`, `SUBMAP`, `HOLD`, 판정 없음.
- 2×11 trim 지도(대피칸 행 + 폭 1 복도 행)가 9×11 교차로 안에 **서로 다른 배치 4곳**으로 들어 있다. 11/6 조사의 "8곳"은 대칭 때문에 같은 격자가 되는 변환을 중복으로 센 값이라 원장에는 4로 기록했다(`evidence.investigation_ref`).
- `DEV_fcorr01_001…005`는 `grid_fp`가 같다(시작·목표만 다름). 즉 실제로는 **지도 단위 관계 1건이 5번 집계**된 것이다. 판정할 때 5건을 함께 다루는 것이 자연스럽지만, 그 결정도 승인 대상이다.

## 2. OOD_WARNING 기록·분류 (`classify_ood`)

`il_compat.check_instance()`의 경고마다 학습 분포(`real_v03_fd.npz`, n = 45,730) 기준으로 두 가지를 붙인다.

| 붙이는 값 | 정의 |
|---|---|
| `severity = TAIL` | [p0.5, p99.5] 밖, 학습에서 관측된 [min, max] 안 |
| `severity = BEYOND_OBSERVED` | 학습 [min, max] 밖 (학습 샘플에 한 번도 없던 값) |
| `cause` | `wall_count` → `DENSE_WALL_VIEW`, `goal_col` → `LONG_HORIZONTAL_GOAL_HINT`, `goal_row` → `LONG_VERTICAL_GOAL_HINT`, `robot_count`·`other_goal_count` → `CROWDED_VIEW` |

instance 등급: `IN_RANGE` / `OOD_TAIL` / `OOD_BEYOND_OBSERVED`(항목 중 가장 심한 것). **`auto_fail`은 항상 False**이고 출력에 `threshold_status: PROVISIONAL …`이 붙는다. 두 경계는 학습 분포를 설명하는 값일 뿐 **Validation 제외 정책이 아니다**(정책은 `data_split_plan_v1.md` 7.3에 따라 Test 생성 전에 따로 승인).

| instance | 등급 | 원인 | 근거 |
|---|---|---|---|
| `PILOT_fcorr02_001` | `OOD_BEYOND_OBSERVED` | `DENSE_WALL_VIEW`, `LONG_HORIZONTAL_GOAL_HINT` | 시작 wall_count 22 (학습 max 20), goal_col −14 (tail) |
| `PILOT_fbot02_001` | `OOD_TAIL` | `LONG_HORIZONTAL_GOAL_HINT` | goal_col −14, 도달 범위 최대 15 (모두 학습 [−38, 44] 안) |
| `PILOT_fint02_001` | `OOD_BEYOND_OBSERVED` | `DENSE_WALL_VIEW`, `LONG_HORIZONTAL_GOAL_HINT` | 시작 wall_count 21 (학습 max 20), 도달 goal_col ±14 |

## 3. PILOT-2 (`PILOT_fbot02_001`) — 두 사실을 따로 보존

`pilot_solvability_v0.json`에서:

- `cbs`: 어댑터가 낸 그대로 — `status: "timeout"`, 120 s, 120.07 s, Solver `89f69e1`(clean), `PYTHONHASHSEED=0`, 원시 출력 `outputs/pilot_checks_v1/20261105_173041/pilot_checks.json`(sha256 `a0ac9281…`)의 `pilot_rooms_door_6x14` 항목.
- `other_methods`: 2026-11-06 읽기 전용 조사의 우선순위 시공간 planner — 순서 (0,1,2,3), 경로 길이 합 73, makespan 21, Path Guard 통과, **`archived: false`(경로 파일 없음, 메모리 내 실행)**.

코드가 계산하는 파생값(파일에 저장 금지): `cbs_status = timeout`, `cbs_solved = False`, `solution_existence = CONFIRMED_BY_OTHER_METHOD`, `witness_evidence = UNARCHIVED`.

막아 둔 것: 파생 필드(`cbs_solved` 등)를 파일에 쓰면 실패 / `other_methods`에 CBS 이름이 들어가면 실패 / `cbs.status`가 어댑터 6개 status가 아니면 실패 / Path Guard 없는 `FOUND_VALID_PLAN`은 실패. `check_pilot_gate`는 `cbs` 블록을 원시 출력 파일과 비교한다(sha256·status·latency·경로 길이). 원시 파일이 없는 환경에서는 실패가 아니라 NOTE로 표시한다(`outputs/`는 추적되지 않음).

## 4. Gate 보고

`python -m scripts.check_pilot_gate [--json]` — PILOT마다 flag를 붙인다. `FLAGGED` = 사람이 볼 것, `CLEAR` = flag 없음. **gate는 거부하지 않는다**(`auto_reject: false`). 종료 코드 1은 기록이 레지스트리·원시 출력과 어긋날 때만.

| instance | gate | flags |
|---|---|---|
| `PILOT_fcorr02_001` | FLAGGED | `OOD_BEYOND_OBSERVED` |
| `PILOT_fbot02_001` | FLAGGED | `OOD_TAIL`, `CBS_TIMEOUT`, `WITNESS_UNARCHIVED` |
| `PILOT_fint02_001` | FLAGGED | `REVIEW_HOLD`(RV0001~0005), `OOD_BEYOND_OBSERVED` |

## 5. 11/8 신규 PILOT 후보 조건 (제안 — 승인 전)

### 5.1 사전 등록
생성·검사 전에 후보 목록을 먼저 적는다: slug, **등급(NORMAL / STRESS)**, 구조·파라미터, seed, 의도와 예상 flag. 검사 결과를 본 뒤 등급을 바꾸거나 후보를 조용히 빼지 않는다(실패도 기록). 수량 제안: NORMAL 2개 + STRESS 1~2개.

### 5.2 기존 지도와 독립
- `independence_check(registry, candidate)`가 `independent: true` — 기존 19개와 **어떤 관계도 없어야** 한다(confirmed, `SUBMAP`, `NEAR_MAP`, `STRUCT_MATCH` 모두). 레지스트리는 같은 development 도메인이면 REVIEW가 있어도 ACCEPT하므로 이것이 더 엄격하다. 후보끼리도 같은 검사를 한다.
- 오늘의 SUBMAP 원인: 벽 행에 붙은 전체 폭 1칸 복도는 DEV Corridor(2×11)를 부분으로 포함하기 쉽다. 이런 배치는 NORMAL 후보에서 피한다.
- 기존 생성 구조(`corridor_bays`, `rooms_door`, `crossing`)의 파라미터만 바꾼 것은 독립 구조로 보지 않는다. 새 구조(예: 기둥이 있는 열린 방, 폭 2 순환 통로, 창고 선반형)를 쓴다.
- **`src/data/generator.py`는 고치지 않는다.** 코드 sha256이 시나리오 JSON에 들어가므로 수정하면 기존 PILOT 재생성 비교가 실패한다. 새 구조는 별도 모듈(예: `generator_v2.py`, `GENERATOR_VERSION` 분리)에 둔다.
- seed는 0~99 중 기존 11·12·13을 제외. 1000~2999(Validation·Test 예정 범위)는 쓰지 않는다. 새 PILOT family도 Validation·Test에서 쓰지 않는다.

### 5.3 NORMAL (학습 분포에 가까운 일반 후보)
목표: 시작 상태 `classify_ood` = `IN_RANGE`. 설계 지침(학습 분포 기준, 정책값 아님):
- 모든 도달 가능 칸에서 목표까지 BFS 거리 ≤ 12이면 `goal_row`·`goal_col` 경고가 생기지 않는다(충분조건: 단위벡터 × 거리이므로 |성분| ≤ 거리, 학습 범위 goal_col [−12, 13], goal_row [−22, 25]).
- 시작 칸의 5×5 시야 안 벽+격자 밖 칸 ≤ 17. 모서리 칸은 시야 밖만 16칸이므로 시작을 가장자리·모서리에서 한 칸 이상 띄우거나 통로 폭을 2 이상으로 한다.
- 시작 시 시야 안 다른 robot·다른 목표 각각 ≤ 4.

### 5.4 STRESS (어려운 스트레스 후보)
의도적으로 분포 밖이거나 CBS가 어려운 경우: 긴 폭 1 통로(`DENSE_WALL_VIEW`), 가로 13칸 이상 이동(`LONG_HORIZONTAL_GOAL_HINT`), 좁은 문에 양방향 4개 이상. 예상 flag를 5.1에 미리 적는다. STRESS 결과는 NORMAL과 섞어 집계하지 않는다.

### 5.5 누수 검사
1. `Registry.evaluate(candidate, "PILOT")`가 ACCEPT, 2. `independence_check` 독립, 3. 같은 지도에서 seed만 바꾼 후보는 새 instance가 아니다(`data_split_plan_v1.md` 4절) — family 상한 3, 4. 후보 파라미터를 기록해 이후 Validation·Test 파라미터 범위와 겹치지 않게 한다(6.3절).

### 5.6 해 존재 검증
1. 실제 CBS(atb033 `89f69e1`, 120 s, `PYTHONHASHSEED=0`, clean, 순차)로 S_0 검사 → `cbs` 블록에 그대로 기록. 결과·Solver 파일은 새 `outputs/` 폴더.
2. CBS가 `timeout`이면 해결 불가로 세지 않는다. 필요하면 별도 witness planner를 돌리되 **경로를 파일로 저장하고 Path Guard로 재검증**해 `other_methods`에 `archived: true`로 기록한다. CBS success로 바꾸지 않는다.
3. 사전 지표(개별 최단 거리 합·makespan 하한)를 함께 기록한다.
4. 후보 완료 조건: 레지스트리 ACCEPT + 독립 + 해결 가능성 기록 존재 + `check_pilot_gate` 문제 0건.

참고: `scripts/check_pilot_instances.py`는 `PILOT_SPECS`(기존 3개)에 묶여 있다. 신규 후보를 검사하려면 스크립트를 바꾸지 않고 새 spec 목록을 받는 별도 진입점이 필요하다(11/8 설계 항목).

## 6. 남은 문제 (Gate v0에서 해결하지 않음)

| 항목 | 상태 |
|---|---|
| REVIEW 5건 최종 판정 | HOLD. 판정자(`decision_owner`)와 판정 절차 미정 |
| PILOT-2 witness | 미보관(`WITNESS_UNARCHIVED`). 재현 가능한 경로 파일 보관 필요 |
| PILOT-2 CBS timeout 원인 | 미확인. 11/6 코드 읽기 가설: wait 행동 때문에 low-level A*에 시간 상한이 없어 끝나지 않을 수 있음 — 검증 안 됨 |
| OOD 자동 실패 규칙·경계값 | 미정. v0 경계는 설명용 |
| `outputs/pilot_checks_v1/` | 추적되지 않음. 기록은 sha256으로 원본을 가리키며, 원본이 없으면 NOTE |

## 7. 다시 확인하는 방법

```
python -m scripts.check_registry          # 레지스트리 (변경 없음 확인)
python -m scripts.check_pilot_gate        # gate 보고 + 기록 일관성
python -m unittest tests.test_data_quality_gate
```
