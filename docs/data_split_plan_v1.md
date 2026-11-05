# 데이터 분리 실행 계획 v1 (Development / Validation / Test)

작성 2026-11-05. 상태: **계획 초안, 승인 대기**. 이 문서는 데이터를 만들거나 실험을 실행하지 않는다.
숫자 중 "제안"이라고 적은 것은 확정값이 아니다. 확인하지 못한 사항은 **[확인 필요]**로 표시한다.

관련 문서:
- [`validation_test_split_v1.md`](validation_test_split_v1.md) — 역할 정의, family 정의, seed 재실행 규칙, 동결 원칙. **이 문서는 그 규칙을 바꾸지 않고 실행 절차로 구체화한다.**
- [`deadlock_definition_v1.md`](deadlock_definition_v1.md) — 6절(근거 instance), 8.3·8.4절(분리·재실행 원칙).
- [`reproducibility_v1.md`](reproducibility_v1.md) — PYTHONHASHSEED, `fair_batch` / `reproducible_batch`, 재현성 체크리스트(10절).
- [`global_hybrid_runner_v0.md`](global_hybrid_runner_v0.md) — DEV 정책과 CSV 필드.

---

## 0. 기존 규칙과의 비교 (작성 전에 확인한 내용)

| 항목 | `validation_test_split_v1.md` | 이 문서 | 관계 |
|---|---|---|---|
| 역할 3종과 허용·금지 사항 | 2절 | 그대로 따른다 | 일치 |
| 현재 16개 파일 = 전부 Development, Validation·Test 0개 | 3.1·3.2절 | 그대로 따른다 | 일치 |
| family 정의 (같은 기본 지도 + 같은 구조 + 파라미터 변형) | 4.1절 | 그대로 쓰고, 생성 지도용 **추가 검사**(3.3절)를 덧붙인다 | 확장 (정의 변경 없음) |
| seed만 바꾼 재실행은 새 instance가 아님 | 4.2절 | 4절에서 다시 명시 | 일치 |
| 파라미터는 Validation에서만 선택, Test 후 재조정 금지 | 5·6절 | 9절에서 절차화 | 일치 |
| 목표 instance 수를 확정하지 않음 | 7절 | 45 / 75는 **제안 수치**로만 적는다 | 일치 |
| KI-1 변형 4개 | Development, 라벨 통계에서 제외 | 역할은 **Development 유지**, 성능 지표 사용은 **보류** | 일치 (보류는 역할이 아니라 사용 상태) |
| 저장소에 CBS 없음, 브랜치 `feature/common-logging` | 3.2·7절 (10/12 시점) | 현재 브랜치 `feature/ina-corridor-e2e-v0`에서는 외부 atb033 `89f69e1`을 `--solver-root`로 호출한다 | **사실 갱신**이며 규칙 변경이 아님. 기존 문서는 수정하지 않았다 |

모순되는 규칙은 발견하지 못했다. 위의 "사실 갱신"은 기존 문서 작성 이후 상황이 바뀐 것이며, 기존 문서의 본문 개정 여부는 별도 승인 사항으로 남긴다(11절).

---

## 1. 현재 데이터 분류표

지도 5개(`.npy`), 시나리오 JSON 16개(연구용 15 + smoke 1). 좌표는 JSON `[x, y]` = `[col, row]`.

| 파일 (`scenarios/…`) | agent | grid | 구조 | family | 사용 이력 | 역할 | 사용 상태 |
|---|---|---|---|---|---|---|---|
| `controlled/scenario_corridor_bay_n2` | 2 | 5×11 | 폭 1 복도 + 대피칸 1 | F-corridor | Trigger 규칙 도출, Monitor 테스트, Safety K=2, Global E2E·H3·REARM, 비교 배치, E2 | Development | 메커니즘 시연. **DEV 파라미터 조정에 사용됨** |
| `controlled/scenario_intersection_n2` | 2 | 7×7 | 십자 교차로 | F-intersection | 위와 같음 | Development | 위와 같음 |
| `controlled/scenario_bottleneck_n4` | 4 | 5×9 | 방 2개 + 문 1칸 | F-bottleneck | 위와 같음 + E1·E3 seed 조사, KI-1 회귀, H3 실패를 계기로 REARM 설계 | Development | **가장 많이 조정에 사용됨** |
| `controlled/scenario_cycle_ring_n4` | 4 | 3×3 | 벽을 둘러싼 고리 | F-cycle (단독) | 오탐 없음 회귀 확인 | Development | 음성 대조군 |
| `smoke/scenario_empty_s11_n3` | 3 | 11×11 | 빈 지도 | F-smoke (단독) | Strict Baseline 9/9 | Development | 파이프라인 회귀 기준점, 연구용 아님 |
| `controlled_variants/…corridor_bay_n2_offset` | 2 | 5×11 | 기본 지도 재사용 | F-corridor | Trigger 근거 (Category-A) | Development | 변형 |
| `controlled_variants/…intersection_n2_4way` | 4 | 7×7 | 기본 지도 재사용 | F-intersection | Trigger 근거 (Category-A) | Development | 변형 |
| `controlled_variants/…intersection_n2_lshape` | 2 | 7×7 | 기본 지도 재사용 | F-intersection | Trigger 근거 (Category-A) | Development | 변형 |
| `controlled_variants/…intersection_n2_tight` | 2 | 7×7 | 기본 지도 재사용 | F-intersection | Trigger 근거 (Category-A) | Development | 변형 |
| `controlled_variants/…bottleneck_n4_tightdoor` | 4 | 5×9 | 기본 지도 재사용 | F-bottleneck | Trigger 근거 (Category-A) | Development | 변형 |
| `controlled_variants/…corridor_bay_n2_3agent` | 3 | 5×11 | 기본 지도 재사용 | F-corridor | KI-1 시연 (Category-B) | Development | **보류** — KI-1 수정(`32e2158`) 후 재검토 안 됨 [확인 필요] |
| `controlled_variants/…corridor_bay_n2_4agent` | 4 | 5×11 | 기본 지도 재사용 | F-corridor | KI-1 시연 (Category-B) | Development | **보류** (위와 같음) |
| `controlled_variants/…bottleneck_n4_convoy` | 4 | 5×9 | 기본 지도 재사용 | F-bottleneck | KI-1 시연, Monitor 테스트 fixture | Development | **보류** (위와 같음) |
| `controlled_variants/…bottleneck_n4_imbalanced` | 4 | 5×9 | 기본 지도 재사용 | F-bottleneck | KI-1 시연 (Category-B) | Development | **보류** (위와 같음) |
| `experimental_v1/…corridor_bay_n2_3rdagent_perturbation` | 3 | 5×11 | 기본 지도 재사용 | F-corridor | 유일한 edge형 정체 관찰, Monitor 테스트 | Development | 임시 실험 |
| `experimental_v1/…bottleneck_n4_tightdoor_1agent_control` | 1 | 5×9 | 기본 지도 재사용 | F-bottleneck | 단일 agent NAV 대조 | Development | 임시 실험 |

부분 중복(시작·목표 쌍 공유) — 서로 독립 instance로 세지 않는다:
- `tightdoor_1agent_control`의 agent = `tightdoor`의 agent 1.
- `corridor_bay_n2_offset`의 agent 0 = 기본 Corridor의 agent 0.
- `corridor_bay_n2_3agent`와 `_4agent`의 agent 0·1이 같다.

모든 변형의 `CBS_verified`는 "no"다(수동 joint plan 재생으로만 해결 가능성 확인). 실제 CBS 실행은 기본 3개에서만 했다.

---

## 2. 기존 데이터를 Validation·Test에 재사용하면 안 되는 이유

1. **Monitor 규칙이 이 데이터에서 도출됐다.** `deadlock_definition_v1.md` 6절의 근거가 바로 이 instance들이다(`validation_test_split_v1.md` 2절).
2. **DEV 파라미터가 이 데이터를 보고 정해졌다.** H_exec=3, Safety K=2, REARM `max_cbs_calls=2`·`cooldown_steps=2`·`cbs_time_budget_ms=60000`은 기본 3개 시나리오에서 결과를 보며 정했고, REARM 자체가 Bottleneck H3 TIMEOUT을 보고 만든 방식이다.
3. **family가 3개뿐이다.** 기존 지도 위에 시작·목표만 새로 정해도 4.1절 정의상 같은 family이므로 Test가 될 수 없다.
4. **반복 관찰됐다.** E1~E3, 비교 배치, 렌더링 등으로 결과를 여러 번 열어 보았다.

용어 주의: `safety_only_v0_results.md` 제목의 "Development Validation"과 2026-11-05 Corridor 배치 `d2`에 붙은 "validation" 표현은 **Development 역할의 기능 확인**을 뜻한다. Validation 역할의 데이터가 아니다.

---

## 3. 중복·family 누수 판정 규칙

### 3.1 판정 대상
새 instance를 등록할 때마다, 기존 **모든** instance(Development + 이미 등록된 Validation·Test)와 비교한다.

### 3.2 판정 단계 (위에서부터, 하나라도 걸리면 그 단계의 결론을 따른다)

| 단계 | 검사 | 결론 |
|---|---|---|
| D1 | `instance_fp`가 같음 (지도 + 시작·목표 전체 동일) | **중복** — 등록 거부 |
| D2 | `map_fp_canon`이 같음 (회전·반사·평행이동 정규화 후 같은 지도) | **같은 family** — 기존 instance와 같은 역할로만 등록 가능 |
| D3 | 기존 지도를 잘라 내거나(crop) 테두리를 덧붙인(pad) 것 (부분 일치) | **같은 family** |
| D4 | 시작·목표 쌍 중 하나라도 정규화 후 다른 instance와 같고 지도도 D2에서 같음 | 같은 family (부분 중복으로 기록) |
| D5 | 생성 파라미터 벡터가 Development 지도의 값과 같음 (예: 5×11 복도 + 대피칸 위치 1, 5×9 + 문 1칸) | **Test 등록 금지**, Validation은 승인 시에만 |
| D6 | 위 어디에도 해당 없음 | 새 family |

- D2의 정규화: 지도를 8가지 대칭(회전 4 × 반사 2)으로 변환하고, 벽 테두리를 제거한 뒤 행 우선 바이트열의 sha256 중 사전순 최솟값을 `map_fp_canon`으로 쓴다. 시작·목표도 같은 변환을 적용한다.
- D5는 `validation_test_split_v1.md` 4.1절 정의를 바꾸지 않는다. 생성기가 우연히 Development 지도와 사실상 같은 지도를 만드는 경우를 막는 **추가 검사**다.
- 같은 구조 **유형**(복도·교차로·병목)이라는 이유만으로 같은 family가 되지는 않는다. 지도 배치가 달라야 새 family다.

### 3.3 Test 전용 추가 규칙
- Test instance는 Development **및** Validation과 family가 겹치면 안 된다(기존 4.2절).
- Test 지도의 생성 파라미터 범위는 Validation 범위와 겹치지 않게 잡는다(6.3절). 범위가 겹치는 설계가 필요하면 그 이유를 등록 전에 적고 승인받는다.

---

## 4. seed 변경은 새 instance가 아니다

- 같은 지도·시작·목표에서 IL은 결정적이다. `PYTHONHASHSEED`나 RNG seed만 바꾼 실행은 **같은 instance의 재실행**이다(`validation_test_split_v1.md` 4.2, `deadlock_definition_v1.md` 8.4).
- `PYTHONHASHSEED`는 CBS가 동점 해 중 무엇을 고르는지에만 영향을 준다(`reproducibility_v1.md` 3절). 이런 결과는 `SEED_SWEEP` 민감도 분석으로만 보고하고, 성공률 표본 수에 더하지 않는다.
- 예: E3 Bottleneck 40회(seed 10 × 방식 4)는 instance **1개**에 대한 조사다.
- 생성기의 **생성 seed**는 다르다. 생성 seed가 달라서 지도나 시작·목표가 실제로 달라지면 새 instance 후보이며, 3절 판정을 통과해야 새 instance로 인정한다.

---

## 5. instance 기록 규칙 (레지스트리)

새 instance마다 레지스트리에 한 행을 기록한다. 파일 형식(JSON 또는 CSV)과 위치는 승인 후 정한다(11절).

| 필드 | 내용 |
|---|---|
| `instance_id` | `<split>_<family_id>_<NNN>` (예: `VAL_fcorr03_002`). split ∈ {`DEV`, `VAL`, `TEST`, `PILOT`} |
| `family_id` | `f<구조약어><번호>` (예: `fcorr03`). 3.2절 D2~D5에서 같은 family이면 같은 id |
| `split` | 역할. 등록 후 바꾸지 않는다 (Development → Validation/Test 승격 금지) |
| `map_fp` | 지도 `.npy` 파일의 sha256 (원본 그대로) |
| `map_fp_canon` | 3.2절 정규화 지도 fingerprint |
| `grid_fp` | 0/1 격자 배열(dtype int64, 행 우선) 바이트열의 sha256 — `.npy` 헤더 차이와 무관 |
| `agents_fp` | agent id 순서대로 `(start_row, start_col, goal_row, goal_col)`을 이은 문자열의 sha256 |
| `agents_fp_canon` | 정규화 변환 후, agent 순서와 무관하게 정렬한 시작·목표 쌍의 sha256 |
| `instance_fp` | `sha256(grid_fp + agents_fp_canon)` |
| `scenario_json_sha256` | 시나리오 JSON 파일 sha256 |
| `generator` | 생성기 이름, 생성기 코드 commit SHA, 생성 seed, 생성 파라미터 전체 |
| `structure_type` | `corridor` / `intersection` / `bottleneck` / 기타 |
| `num_agents`, `grid_shape`, `free_cells` | 기본 속성 |
| `solvability` | 8절 결과 (`CBS_SOLVED` / `CBS_TIMEOUT` / `CBS_NO_SOLUTION` / `CBS_ERROR`), CBS cost, 소요 시간, solver commit |
| `il_compat` | 7절 검사 결과 |
| `created_at`, `registered_by` | 날짜, 등록자 |
| `notes` | 부분 중복, 제외 사유 등 |

규칙:
- 좌표는 레지스트리에서 **내부 `(row, col)`**로 적는다(JSON은 `[x, y]`). 변환은 `scenario_loader`와 같은 방식으로 한다.
- fingerprint 계산 코드는 하나만 두고 Development 16개에도 같은 코드로 계산해 레지스트리에 `split=DEV`로 넣는다. 그래야 3절 비교가 가능하다.
- Validation·Test 목록이 확정되면 레지스트리 파일의 sha256과 commit SHA를 함께 고정한다(9절).

---

## 6. 신규 지도 생성 조건과 최소 다양성

### 6.1 수량 (제안 — 확정 아님)
- **Validation 약 45개 (제안)**: 구조 3종 × 규모 3단계 × 5개.
- **Test 약 75개 (제안)**: 구조 3종 × 약 25개. 규모 단계는 Validation과 같은 수를 쓰되 파라미터 범위는 6.3절처럼 나눈다.
- 실제 수량은 파일럿(6.4절) 결과로 CBS 해결 가능 비율과 실행 시간을 확인한 뒤 결정한다. Notion 마스터플랜의 "조건당 ~25개"는 직접 확인하지 못했다 [확인 필요].

### 6.2 최소 다양성 (Validation·Test 각각에 적용, 제안)
- 구조 유형: 복도(대피칸 있음), 교차로, 병목(문) 3종 모두 포함.
- 한 family에서 나온 instance는 split 안 instance 수의 일정 비율 이하로 제한(예: family당 최대 3개). 한 지도에 시작·목표를 여러 개 두면 그만큼 독립성이 낮아지기 때문이다.
- agent 수: 최소 3단계(예: 2 / 4 / 6). 6 이상은 파일럿에서 CBS 시간을 보고 결정한다.
- 지도 크기: Development 지도(3×3~11×11)보다 큰 것을 포함하되, 7절 IL 호환성 검사를 통과하는 범위로 제한한다.
- 시작·목표 배치: 정면 충돌형, 직각 교차형, 같은 방향 행렬(convoy)형, 비대칭 흐름형을 섞는다.
- 난이도 분포: 각 instance에 "직선 거리 대비 BFS 거리", "시작 시 최단 경로 겹침 수" 같은 사전 지표만 기록한다. 이 값은 **방법을 실행하지 않고** 계산하며, IL 실행 결과로 instance를 고르거나 버리지 않는다(선택 편향 방지).

### 6.3 Validation·Test 생성 분리
- 생성 seed 범위를 나눈다 (예: Validation 1000~1999, Test 2000~2999 — 제안).
- 생성 파라미터 범위를 겹치지 않게 나눈다 (예: 복도 길이, 문 위치, 방 크기). 구체 범위는 파일럿 후 정한다.
- Test 생성은 Validation 동결(9절 단계 V4) 이후에 하고, 생성 직후 레지스트리에 등록·고정한다. 생성된 Test instance는 해결 가능성 검사(8절)와 IL 호환성 검사(7절) 외에는 실행하지 않는다.
- 기존 Development 지도 family(F-corridor, F-intersection, F-bottleneck, F-cycle, F-smoke)는 Validation·Test 어디에도 쓰지 않는다.

### 6.4 개발용 파일럿 (2~3개)
실제 Validation·Test 생성 전에 생성기와 실행 경로를 검증한다.
- 파일럿 instance는 `split=PILOT`으로 등록하고 **Development로 취급**한다. 이후 Validation·Test로 옮기지 않으며, 파일럿 family는 Validation·Test에서 쓰지 않는다.
- 파일럿 생성 seed는 Validation·Test 범위 밖(예: 0~99)에서 고른다.
- 확인 항목:
  1. 생성기가 같은 seed로 같은 지도를 바이트 단위로 다시 만드는가.
  2. `scenario_loader`가 오류 없이 읽고 좌표 변환이 맞는가.
  3. fingerprint 계산과 3절 판정이 Development 16개에 대해 기대대로 나오는가(예: 변형들이 기본과 같은 family로 판정).
  4. 실제 CBS로 해결 가능성 검사가 timeout 안에 끝나는가, 소요 시간은 얼마인가.
  5. 7절 IL 호환성 검사가 실행되는가.
  6. 비교 배치(`run_compare_batch`)가 경로 길이 사전 검사(MAX_PATH)를 통과하는가.
- 파일럿에서 IL·Hybrid를 실행해 볼 수는 있지만, 그 결과를 정책 파라미터 선택에 쓰지 않는다. 실행 가능성 확인용이다.

---

## 7. IL checkpoint 호환성 확인 방법

### 7.1 알고 있는 것 (`checkpoints/MANIFEST.json`, `src/il/`)
- 모델: `NavHintCNN`, 입력 local grid `3×5×5`(채널 3, 5×5 시야), `goal_dim=2`, 행동 5개.
- `hint_mode="flowdist"`: goal 입력 = **흐름 방향 단위벡터 × BFS 거리**(`src/il/nav_hint.py`). 즉 입력 크기가 목표까지의 거리에 비례한다.
- 정규화: `(goal_dir - goal_mean) / goal_std`, `goal_mean ≈ (-0.042, 0.016)`, `goal_std ≈ (6.58, 2.57)` — `(row, col)` 순서.
- 학습 명령: `train_navhint.py --npz real_v03_fd.npz --hint-mode flowdist --epochs 55`, best_val_acc 0.8255.

### 7.2 모르는 것 [확인 필요]
- 학습 데이터 `real_v03_fd.npz`는 Alpha 2로 옮겨지지 않았다(`training_data_migrated: false`).
- **학습에 쓰인 지도 종류, 지도 크기, agent 수, 장애물 밀도, 시작·목표 분포는 확인되지 않았다.**
- 학습 지도에 MovingAI 지도 또는 이번에 생성할 구조와 비슷한 지도가 있었는지 알 수 없다.
- `goal_std`가 row 6.58, col 2.57로 비대칭인 이유도 알 수 없다. 이 값에서 학습 분포를 추측하지 않는다.

### 7.3 확인 절차
1. **원본 확인**: Alpha 1 저장소(`qwedus/MAPF_2026_AlphaProject`, `analysis` 브랜치)와 Alpha 1 문서에서 `real_v03_fd.npz` 생성 방법과 사용 지도를 찾는다. 담당자에게 확인한다. 찾지 못하면 "미확인"으로 기록하고 MovingAI 등 외부 지도를 Test에 쓰지 않는다.
2. **입력 형식 검사** (실행 없이 계산만): 각 후보 instance에서
   - 모든 agent·모든 칸에서 BFS 거리가 유한한가(시작에서 목표로 갈 수 있는가),
   - local grid가 5×5 시야와 지도 경계 패딩 규칙에 맞게 만들어지는가,
   - 시작 위치의 `flowdist` 값을 정규화했을 때 z 값이 얼마인가.
3. **분포 밖 입력 표시**: 시작 시점의 정규화된 goal 입력 |z|가 기준값(제안: 3)을 넘는 instance에 `il_compat=OUT_OF_RANGE`를 기록한다. 기준값은 승인 대상이다. 이 표시는 instance를 **빼는 기준이 아니라 기록**이며, 결과 보고 때 따로 집계한다. 빼려면 Test 생성 전에 규칙으로 정한다.
4. 이 검사는 IL 정책을 실행하지 않고 입력만 계산하므로 Validation·Test 결과를 미리 보는 것이 아니다.

---

## 8. CBS 해결 가능성 검사와 timeout 처리

- 사용 Solver: atb033 `89f69e1`, 명시적 `solver_root`(submodule 미사용), PyYAML 6.0.3, `alpha2_test_env`.
- 검사 대상: 각 instance의 **시작 상태 S_0**. 이는 `joint_plan_exists`(시작 상태 해결 가능성)이며, 실행 중 정체 상태에서의 회복 가능성(`candidate_state_recoverable`)과 다르다(`deadlock_definition_v1.md` 5.2).
- 실행 조건: `PYTHONHASHSEED` 고정, 순차 실행(동시 실행 금지 — cwd 전환 때문), clean solver.
- 검사용 timeout은 실험용 CBS timeout(30 s)과 **따로** 둔다(제안: 120 s). 검사 결과 기록:

| 결과 | 처리 |
|---|---|
| `CBS_SOLVED` | 등록 가능. cost와 소요 시간 기록 |
| `CBS_NO_SOLUTION` | 해결 불가. 성능 평가 instance에서 제외하고 레지스트리에 사유 기록 |
| `CBS_TIMEOUT` | 해결 여부 **미확인**. 평가 instance에서 제외하고 따로 기록. 해결 불가로 세지 않는다 |
| `CBS_ERROR` | 원인 확인 전까지 등록 보류 |

- 출력 경로가 Path Guard(어댑터 내부 + `validate_path_guard`)를 통과해야 `CBS_SOLVED`로 인정한다.
- 해결 가능성 검사에서 나온 CBS cost는 Pure CBS 결과로 보고하지 않는다. 검사용 기록이다.
- 제외 비율이 높으면(예: 구조 유형별 20% 초과 — 제안) 생성 조건을 고친다. 고치는 것은 Test 생성 **전**에만 허용한다.
- 실험 중 CBS timeout(30 s)은 해당 방식의 결과로 기록한다(SAFE_STOP `CBS_TIMEOUT` 등). instance를 사후에 제외하는 근거로 쓰지 않는다.

---

## 9. Validation에서만 정책을 고르고 Test는 마지막까지 쓰지 않는 절차

| 단계 | 내용 | 산출물 |
|---|---|---|
| P0 | 이 계획 승인 | 승인 기록 |
| P1 | fingerprint·레지스트리·생성기 구현, Development 16개 등록 | 레지스트리 (DEV만) |
| P2 | 파일럿 2~3개 생성·검증 (6.4절) | 파일럿 보고 |
| V1 | Validation instance 생성 → 7·8절 검사 → 3절 판정 → 등록 | Validation 목록 + sha256 |
| V2 | **Validation 실험 계획 사전 등록**: 후보값, 선택 기준, 실행 횟수, 커밋 SHA | 계획 문서 |
| V3 | Validation 실행·분석, 파라미터 선택 | Validation 결과 |
| V4 | **정책 동결**: H_exec, IL 복귀 조건, cooldown, CBS 재호출 상한, 누적 CBS 시간 예산, Safety 파라미터, 비교 방식 목록 | 동결 문서 + commit SHA |
| T1 | Test instance 생성 → 7·8절 검사 → 3절 판정 → 등록·고정 | Test 목록 + sha256 |
| T2 | **Test 프로토콜 사전 등록**: instance 목록, 방식, 지표, 실패 처리, 반복 횟수, seed, 커밋 SHA | 프로토콜 문서 |
| T3 | Test 1회 실행 | 원시 출력 (수정 금지) |
| T4 | 보고 | 결과 문서 |

Validation에서 고를 항목과 후보 (제안 — 확정 아님):
- H_exec: 1, 3, 5, 8, FULL
- IL 복귀 조건: 항상 복귀 / 경로 끝까지 실행 / 병목 통과 후 복귀(정의 필요)
- cooldown: 0, 2, 4
- CBS 재호출 상한(첫 호출 포함): 2, 3, 5
- 누적 CBS 시간 예산: 10, 30, 60 s
- 조합 폭발을 막기 위해 H_exec → 복귀 조건 → 재호출 3종 순서로 하나씩 정하는 방식을 제안한다. 순서와 선택 기준은 V2에서 고정한다.
- 선택 기준 예: 성공률 최대 → SoC 최소 → CBS 호출 수 최소 → latency 순. V2에서 고정한다.

금지 사항:
- V4 이전에 Test instance를 만들거나 실행하지 않는다.
- Test 결과를 본 뒤 파라미터, instance 목록, 지표 정의를 바꾸지 않는다. 바꿔야 하면 날짜가 적힌 프로토콜 개정 후 새 Test 세트로 다시 한다(`validation_test_split_v1.md` 6절).
- Validation에서 정한 값을 Development 결과로 덮어쓰지 않는다.
- 일정이 밀리면 Test 실행도 함께 늦춘다.

---

## 10. 평가 지표, 실패 처리, 재현성·공정성 규칙

### 10.1 비교 방식
IL-only, Pure CBS, Safety-only, Global FULL, H3, H3_REARM. Selective는 구현 후 추가한다(현재 `NOT_IMPLEMENTED`). Test에서 쓸 방식 목록은 V4에서 고정한다.

### 10.2 지표

| 지표 | 정의 | 비고 |
|---|---|---|
| 성공률 | 최종 시점 전원 목표 도달(`all_at_goal`) instance 비율 | 분모 = 평가 instance 수 (seed 수 아님) |
| makespan | 성공 episode의 마지막 step | 실패 처리 방식은 승인 필요 (10.3) |
| SoC | agent별 목표 최종 도달 시점의 합 | **정의 확인 필요**: 목표에 도착한 뒤 떠났다 돌아온 경우 계산법 |
| CBS 호출 수 | `cbs_calls` 길이 (거절 포함) | 수용·거절을 나눠서도 보고 |
| latency | `ReplanResult.latency_sec` 기준, 호출별·합계 | 순수 탐색 시간이 아님(`global_hybrid_runner_v0.md` 8.2). seed 고정과 관계없이 결정적이지 않으므로 별도 표 |
| 위치 겹침 | `overlap_any_step`, `INTEGRITY_ABORT` | 실제 충돌. 0이어야 함 |
| 충돌 시도 | 시뮬레이터 vertex/edge 충돌 수 | 실제 충돌이 아니라 막힌 이동 시도 |
| 정체 | Trigger 수, IL 복귀 후 재정체 여부 | |
| SAFE_STOP | 횟수와 `stop_reason`별 분포 | |

### 10.3 실패 처리
- 종료 상태는 SUCCESS / TIMEOUT / SAFE_STOP / INTEGRITY_ABORT 네 가지로 나눠 모두 보고한다.
- makespan·SoC는 (a) 성공 instance만 집계하거나 (b) 실패를 `max_steps`(=160)로 넣는 방식 중 하나를 T2 전에 고른다. 두 방식을 모두 표로 남기는 것을 제안한다.
- 방식 간 비교는 **모든 방식이 성공한 instance 교집합**에서 따로 계산한다.
- 8절에서 제외된 instance는 모든 방식에서 똑같이 제외한다. 실행 후 특정 방식에 불리한 instance를 빼지 않는다.

### 10.4 재현성 (`reproducibility_v1.md` 10절 체크리스트를 그대로 따른다)
- clean project commit, clean solver commit, checkpoint sha256 = MANIFEST.
- `PYTHONHASHSEED`를 batch 시작 명령에서 고정하고 `--require-hash-seed` 사용. manifest에서 `fair_batch=True`, `reproducible_batch=True`, `parent_verified=VERIFIED`, `inherited_probe_status=OK`, `cbs_calls_seed_consistent=True` 확인.
- 같은 seed로 최소 2회 실행해 결정적 필드가 같은지 확인한다(latency·run_id 제외). 2회째는 재현성 확인용이며 표본 수에 더하지 않는다.
- 원시 출력은 수정하지 않고 보존한다. 출력 폴더는 새로, 짧게 만든다.
- Python·OS·패키지 버전이 바뀌면 재현을 가정하지 않고 다시 검증한다.

### 10.5 공정성
- 모든 방식이 같은 instance 목록, 같은 checkpoint, `max_steps=160`, CBS timeout 30 s, 같은 solver, 같은 seed로 실행한다(값은 V4에서 다시 확인·고정).
- 한 프로세스에서 순차 실행한다. CBS 동시 호출 금지.
- Pure CBS(`INITIAL_STATE`, t=0)와 Global(`IL_STALL_TRIGGER`, t≥1)은 다른 계획 문제임을 결과 표에 명시한다.
- seed 여러 개 실험은 `SEED_SWEEP`으로 따로 보고한다.
- Test에서 반복 측정이 필요하면(예: latency 분산) 목적과 횟수를 T2에서 미리 정한다.

---

## 11. 승인이 필요한 결정

1. Validation·Test instance 수 (제안: 45 / 75)와 구조·규모 단계 구성.
2. 생성 파라미터 범위와 Validation·Test 범위 분리 방법, 생성 seed 범위.
3. family당 최대 instance 수 (제안: 3).
4. agent 수 상한 (파일럿 CBS 시간 확인 후).
5. IL 학습 데이터 확인 담당과 방법. 확인되지 않을 때 외부 지도(MovingAI) 사용 여부.
6. IL 분포 밖 입력 기준값 (제안: |z| > 3)과 제외 여부.
7. 해결 가능성 검사 timeout (제안: 120 s)과 제외 비율 기준 (제안: 20%).
8. KI-1 변형 4개를 재검토할지, 계속 보류할지.
9. Validation 후보값, 선택 순서, 선택 기준.
10. makespan·SoC 실패 처리 방식과 SoC 정의.
11. 레지스트리 파일 형식과 위치.
12. `validation_test_split_v1.md`의 사실 갱신(브랜치, CBS 존재 여부)을 본문에 반영할지.
13. H_exec·복귀·재호출 정책의 CBS 담당(윤지)과의 합의 시점.
