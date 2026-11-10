# 창고형 지도 생성기 v3 설계안 (초안)

- 상태: **DRAFT — 설계만.** 생성기 코드, 시나리오, 레지스트리 기록, CBS 실행은 아직 없다. 이 문서의 수치 기준(임계값, 예산)은 실행 전에 사전등록(prereg)으로 확정할 항목이며 지금은 정하지 않는다.
- 작성 기준: 브랜치 `feature/ina-corridor-e2e-v0`, HEAD `ce1c104`, 2026-11-10
- 최종 목표: 다수 물류 로봇 환경에서 IL_ONLY, Global Hybrid, (연결 이후) Selective Hybrid의 **안전성·정체 복구·계산 비용**을 비교한다.
- **이 문서는 성능 개선을 주장하지 않는다.** Selective Runner는 연결되지 않았고(`docs/selective_hybrid_contract_v0.md`), 로봇 수별 성공률이나 실행 시간은 아직 측정한 적이 없다.
- 표기: **[확인]** 코드·저장 기록 근거, **[계산]** 이 문서를 위해 메모리에서만 수행한 계산(파일 생성 없음), **[제안]** 설계안, **[PENDING]** 결정 전.

---

## 1. 기존 구조에서 확인한 사실

| 항목 | 내용 | 근거 |
|---|---|---|
| 생성 방식 | `(params, seed)`의 순수 함수. 지도 배치는 params가 정하고, seed는 고정 zone 안에서 Agent 칸만 `numpy.random.default_rng(seed)`로 고른다 | [확인] `src/data/generator.py:1-12`, `generator_v2.py:1-10` |
| 유효성 | 0/1 grid, start·goal 중복 없음, 벽 위 아님, 모든 goal이 start에서 BFS로 도달 가능 | [확인] `generator.validate_instance` |
| 파일 형식 | `map_<slug>.npy` + scenario JSON(`[x, y] = [col, row]`, `agent_id` = 순번, `generator{name, version, seed, code_sha256, descriptor}`) | [확인] `generator.scenario_files` |
| v1/v2 family | v1: corridor_bays, rooms_door, crossing. v2: staggered_blocks, loop_corridor, hub_spokes. v2는 v1을 import만 하고 수정하지 않음 | [확인] |
| 사전등록 | 후보 생성 **전에** grade, seed, params, 예측값을 기록. 결과 때문에 교체·재시드하지 않음(no replacement) | [확인] `registry/pilot_v2_prereg_v0.json` |
| width-1 칸 정의 | 위아래가 모두 벽/맵밖이거나, 좌우가 모두 벽/맵밖인 빈 칸 | [확인] 같은 prereg `width1_cell_definition` |
| IL 입력 기준 | 학습 표본 45,730개(`real_v03_fd.npz`). p0.5~p99.5: `goal_row` [−22, 25], `goal_col` [−12, 13], `wall_count` [0, 17], `robot_count` [0, 4], `other_goal_count` [0, 4]. 학습 최솟값~최댓값: `goal_row` [−42, 46], `goal_col` [−38, 44], `wall_count` [0, 20], `robot_count`·`other_goal_count` [0, 7] | [확인] `registry/il_input_reference_v1.json` |
| OOD 판정 | 시작 상태 5개 특징 + **goal에 도달 가능한 모든 칸**의 목표 힌트 범위를 [p0.5, p99.5]와 비교. 판정 등급은 IN_RANGE / OOD_TAIL / OOD_BEYOND_OBSERVED. `auto_fail`은 항상 False | [확인] `src/data/il_compat.check_instance`, `quality_gate.classify_ood` |
| 목표 힌트 | `flowdist` = (BFS 다음 한 칸 방향) × (BFS 남은 거리). 거리가 같으면 상·하·좌·우 순으로 고르므로 세로 방향 힌트가 우세하다. **가로 방향 이동 칸에서는 \|goal_col\|이 남은 전체 거리와 같다** | [확인] `src/il/nav_hint.py:71-89`, `il_compat.py:71-73` |
| 레지스트리 | `instance_fp`, 관계(EXACT, SAME_MAP, SUBMAP, NEAR_MAP, STRUCT_MATCH), `FAMILY_CAP = 3`, family ID는 `f<약어><NN>`. `STRUCT_ABBR`에 없는 structure_type은 앞 4글자로 축약(`warehouse` → `ware`) | [확인] `src/data/registry.py:85-93, 353-356`, `docs/fingerprint_registry_v1.md` |
| Simulator | **1회성 MAPF.** 모든 Agent가 goal에 도착하면 종료한다. 도착한 로봇은 그 칸에 계속 남는다. 작업 재할당(lifelong)은 없다 | [확인] `mapf_step_simulator.py:327-330` |
| CBS 선례 | 120초 timeout: 6×14 rooms_door 4대 → `timeout`(다른 방법으로 해 존재 확인), 19×19 hub_spokes 6대 → `timeout`(해 존재 미확인) | [확인] `registry/pilot_solvability_v0.json`, `docs/pilot_v2_candidates.md` |

---

## 2. v3 family 정의 [제안]

- 모듈: 새 파일 `src/data/generator_v3.py`. v1·v2는 import만 하고 수정하지 않는다. `GENERATOR_NAME = "alpha2_warehouse"`, `GENERATOR_VERSION = 3`.
- structure_type: `"warehouse"` → family ID `fware<NN>`(레지스트리 코드 변경 없이 생성됨).
- 빌더 1개로 시작: `warehouse_aisles(*, n_racks, rack_len, rack_w, aisle_w, cross_w, station_rows, flows, slots, seed)`

### 2.1 배치

```text
행 0 .. cross_w-1          상단 교차 통로 (입고/대기 zone)
행 cross_w .. +rack_len-1  선반(rack, 벽) 세로 열과 통로(aisle) 세로 열이 번갈아 배치
다음 cross_w 행            하단 교차 통로
마지막 station_rows 행     피킹·출고 스테이션 band
열: aisle, rack, aisle, rack, ..., aisle   (W = (n_racks+1)*aisle_w + n_racks*rack_w)
```

| 요소 | 매개변수 | 의미 |
|---|---|---|
| 선반 | `n_racks`, `rack_len`, `rack_w` | 세로로 긴 벽 블록. 긴 축을 **세로(행)**로 둔다. 학습 분포에서 세로 힌트 범위가 가로보다 넓기 때문이다(1절) |
| 좁은 통로 | `aisle_w = 1` | width-1 칸(병목·대향 통행) 생성 |
| 넓은 통로 | `aisle_w = 2` | width-1 칸 없음 |
| 교차로 | 통로 열 × 교차 통로 행의 만나는 칸 | 개수는 descriptor에 기록 |
| 스테이션 | `station_rows` | 하단 band. goal을 두는 곳 |
| 시작·목표 | `slots`(params에 고정된 후보 칸 목록), `flows`(예: 상단→스테이션 k대, 스테이션→상단 k대) | seed는 slots 안에서만 고른다 |

### 2.2 시작·목표 규칙 [제안]

- goal은 **통로(aisle) 안에 두지 않는다.** 1회성 MAPF에서는 도착한 로봇이 그 칸을 막기 때문이다. goal은 스테이션 band나 상단 zone의 slot에만 둔다.
- slot은 간격을 두고 고정한다. 목표는 어떤 5×5 시야에도 다른 로봇이나 다른 goal이 4개 이하인 것이다(학습 p99.5 = 4). 실제 값은 생성 후 `il_compat.check_instance`로 기록하고, 예측은 생성 전에 prereg에 적는다.
- 대향 흐름(위→아래, 아래→위)을 섞어 통로 안의 경합을 만든다.
- `agent_id` = 순번, JSON 좌표는 `[x, y]`로 쓰고 `scenario_loader`로 되읽어 확인한다(v2와 동일).

### 2.3 재현성 [제안]

- 같은 params와 seed로 생성하면 `.npy`와 JSON 바이트가 같아야 한다. `PYTHONHASHSEED` 0과 1에서 모두 확인한다.
- scenario JSON의 `generator.code_sha256`에는 `generator_v3.py`의 sha256을 넣는다.
- prereg 파일을 생성 전에 커밋하고, no replacement 규칙을 적용한다.

---

## 3. 첫 검증 지도 후보: 8대 [계산]

### 3.1 방법

2.1의 배치를 192가지 매개변수 조합(`n_racks`∈{2,3,4}, `rack_len`∈{3,4,6,8}, `aisle_w`·`rack_w`·`cross_w`·`station_rows`∈{1,2})으로 메모리에서만 만들었다. 각 지도에 대해 다음을 계산했다.

- **목표 힌트 범위:** goal 후보를 **상단 행과 스테이션 행의 모든 칸**으로 놓고(최악의 경우), 각 goal에 도달 가능한 모든 칸의 `flowdist` 힌트 범위를 구했다. `il_compat.reachable_hint_range`와 같은 계산이다.
- **벽 칸 수:** 모든 빈 칸의 5×5 시야 안 벽 칸 수
- **width-1 칸 수**

Agent는 배치하지 않았고, 파일도 쓰지 않았다(스크립트는 scratchpad에만 있음).

### 3.2 결과

- 목표 힌트와 벽 칸 수가 모두 IN_RANGE인 배치는 192개 중 **11개**뿐이었다. 가장 큰 것도 9×8(빈 칸 66개)이다.
- IN_RANGE가 되려면 상하단 교차 통로 폭이 2(`cross_w = 2`)여야 했다. `cross_w = 1`인 배치는 이 조건을 만족하지 못했다.
- 빈 칸이 110~124개인 배치(9×14, 10×14, 13×11)는 `goal_col`이 ±18~19로 **OOD_TAIL**이다. 학습 최댓값 안이므로 OOD_BEYOND_OBSERVED는 아니다.
- 192개 중 학습 최솟값~최댓값을 넘는 배치(BEYOND)는 없었다.
- 원인: 가로 방향으로 이동하는 칸에서는 \|goal_col\|이 남은 전체 BFS 거리와 같다. 그래서 지도 폭과 교차 통로 길이가 조금만 늘어도 가로 p99.5(13)를 넘는다.

### 3.3 8대 후보

| ID | params | 크기 | 빈 칸 | width-1 칸 | 8대 밀도 | goal_row / goal_col 범위 (최악) | wall_count 최대 |
|---|---|---|---|---|---|---|---|
| **WH8-A** (넓은 통로) | n_racks 2, rack_len 3, rack_w 1, aisle_w 2, cross_w 2, station_rows 2 | 9×8 | 66 | 0 | 0.121 | ±15 / ±12 | 17 |
| **WH8-B** (좁은 통로) | n_racks 3, rack_len 3, rack_w 1, aisle_w 1, cross_w 2, station_rows 2 | 9×7 | 54 | 12 | 0.148 | ±14 / ±12 | 17 |

```text
WH8-A (9x8)      WH8-B (9x7)
........         .......
........         .......
..#..#..         .#.#.#.
..#..#..         .#.#.#.
..#..#..         .#.#.#.
........         .......
........         .......
........  <- 스테이션 band (2행)
........         .......
```

**경계값 주의:**
- 두 후보 모두 `goal_col` 최솟값이 정확히 −12로, p0.5 경계선 위에 있다(검사는 이상·이하로 포함).
- `wall_count` 최댓값 17도 p99.5 경계선이다.
- 실제 instance의 goal 집합은 이 최악의 경우보다 작으므로 범위가 더 좁을 수 있다. 확정은 생성 후 instance별 검사로 한다.
- `robot_count`·`other_goal_count`는 Agent 배치에 달려 있어 아직 계산하지 않았다(2.2 slot 규칙으로 통제).

**해석:**
- 두 후보는 "실제 창고 규모"가 아니다. **IL 학습 분포 안에 있는 최소 창고형 구조**다. 첫 단계의 목적은 v3 생성기·검사·로그 파이프라인과 IL_ONLY 동작을 학습 분포 안에서 확인하는 것이고, 창고 성능을 주장하는 것이 아니다.
- WH8-A의 통로는 넓어(width-1 칸 0) 통로 안에서 서로 비켜 갈 수 있다. WH8-B는 1칸 통로를 대향 통행이 지나야 해서 병목·정체 사례가 된다.

---

## 4. 단계별 확장 조건과 중단 기준 [제안]

로봇 수별 성공률이나 시간은 **측정 전이라 쓰지 않는다.** 아래 수치 임계값은 각 단계의 prereg에서 사용자가 확정한다.

| 단계 | 대상 | 들어가기 위한 조건 | 비고 |
|---|---|---|---|
| S8 | WH8-A/B, IL_ONLY | v3 생성기 단위 테스트(재현성, 유효성, 좌표, `agent_id`) 통과, prereg 커밋, instance 검사(OOD·관계) 기록 | Global Hybrid는 instance마다 CBS 사전 검사를 1회 **별도 승인**받은 뒤에만 실행 |
| S16 | 더 큰 v3 지도 | S8에서 안전 위반 0, 결정성 확인, 모든 결과(실패 포함) 보고 완료 | 3.2절에 따르면 **같은 배치 family에서 밀도를 유지하면 OOD_TAIL을 피할 수 없다.** 밀도를 올리거나(빈 칸 66개에 16대 = 0.242) OOD_TAIL을 STRESS 등급으로 명시해야 한다 |
| S32 | | S16 조건 + Trigger 미발생과 CBS timeout 비율을 진단 | 전체 CBS를 쓰는 Global Hybrid는 6절 위험이 지배적일 수 있음 |
| S64+ | | Selective 연결, Partial Trigger, solver 대안 중 최소 하나가 결정·검증된 뒤 | 그 전에는 IL_ONLY 관찰 실험으로만 한정 |

**중단 기준 (어느 단계에서든 하나라도 해당하면 그 단계에서 멈추고 보고):**
1. 겹침 발생: `overlap_any_step = True` 또는 `INTEGRITY_ABORT`
2. CBS 경로가 Path Guard에서 거부되었는데 실행됨(있어서는 안 되는 경우)
3. 같은 입력·seed의 재실행 결과(trajectory hash)가 다름
4. 생성기 재현성 실패: 같은 seed에서 바이트가 다름
5. prereg에 적은 실행 시간 예산 초과
6. prereg에 적은 OOD 등급과 실제 등급이 다름 → 해석을 보류하고 검토
7. registry 검사에서 EXACT, FAMILY_LEAK, FAMILY_CONFLICT 판정

---

## 5. 검증 계획 [제안]

| 항목 | 방법 | 기존 도구 |
|---|---|---|
| 충돌 안전성 | 매 step 위치 중복 검사, Simulator vertex/edge 취소 사건 수, CBS 경로는 Path Guard | `run_global_hybrid_episode` overlap 검사, `_last_*_conflict_count`, `validate_path_guard` |
| 목표 도달 | episode `all_at_goal`, Agent별 도착 여부와 도착 시각 | `result.json` `all_at_goal`, trajectory |
| 실제 SoC·makespan | 성공 episode의 실제 값과 BFS 하한(`soc_lower_bound`, `makespan_lower_bound`)을 함께 기록. 실패 episode는 SoC 대신 미도착 수를 기록 | `check_pilot_v2_candidates.agent_metrics` 계산 방식 |
| OOD 경고 | 생성 직후 instance별로 `check_instance` + `classify_ood` 실행. 등급과 원인 기록, `auto_fail = False` | `il_compat`, `quality_gate` |
| 정체 발생 | Monitor Trigger 수와 시각, `persistent_conflict` 연속 길이, Agent별 `no_progress` 연속 길이, 종료 사유 | `deadlock_monitor`, monitor CSV |
| CBS | status를 6종 그대로 집계하고 **`timeout`과 `no_solution`을 섞지 않는다.** timeout을 "해 없음"으로 해석하지 않는다(atb033에는 시간 상한이 없다). latency와 호출 수 기록 | `replan_status`, `quality_gate.CBS_STATUSES` |
| fingerprint·held-out | 모든 v3 후보를 기존 19개 record와 `registry.relate` + `Registry.evaluate`로 메모리에서만 대조. v3 후보끼리도 쌍으로 대조. 작은 지도가 큰 지도의 일부(SUBMAP)일 가능성이 높으므로 REVIEW는 HOLD로 기록. 독립 승인은 사람이 별도 단계에서 함 | `registry.py`, `check_pilot_v2_candidates.independence_summary` |
| held-out 설계 | Development와 Test를 **배치 매개변수 family 단위**로 분리한다(예: Test는 랙 수·통로 폭 조합이 다른 family). `FAMILY_CAP = 3`, TEST의 STRUCT_MATCH 금지 규칙을 따른다 | `docs/fingerprint_registry_v1.md` 4절 |

**비교 원칙:**
- 같은 instance, 같은 checkpoint, 같은 `max_steps`, 같은 CBS timeout으로 비교한다.
- 방법별 실행은 순차로 하고, `PYTHONHASHSEED`와 solver 커밋을 고정한다.
- 종료 사유(SUCCESS / TIMEOUT / SAFE_STOP(사유별) / INTEGRITY_ABORT)는 분리해 집계한다.
- 처리량은 1회성 기준(step당 도착 로봇 수)으로만 정의한다. lifelong 처리량은 Simulator에 작업 재할당이 없어 측정할 수 없다.

---

## 6. 해결되지 않은 한계 (별도 정리)

> 아래 두 문제는 **아직 해결되지 않았다.** 이 설계는 이를 해결하지 않으며, 대규모 결과를 해석할 때 반드시 함께 보고한다.

### 6.1 Monitor의 부분 정체 감지 한계

- [확인] `DeadlockMonitor`는 **모든 Agent의 위치가 정확히 반복될 때만** Trigger를 낸다(`deadlock_monitor.py:8-10, 321-326`). 일부 Agent만 반복하는 경우를 잡는 Trigger-Partial은 구현되지 않았다.
- [추정] 로봇 수가 늘면 일부 로봇이 교착에 빠져 있어도 다른 로봇이 움직이는 한 전체 위치는 반복되지 않는다. 그러면 Trigger가 나지 않고 episode가 `TIMEOUT`으로 끝날 수 있다. 이때 Global Hybrid는 IL_ONLY와 구별되지 않는다.
- 측정 방법: Trigger가 없었던 TIMEOUT episode에서 `no_progress`와 `persistent_conflict` 연속 길이를 따로 집계해 "감지되지 않은 정체" 후보로 기록한다(판정이 아니라 기록).

### 6.2 전체 CBS의 계산 비용

- [확인] Global Hybrid는 Trigger마다 **전체 Agent**로 CBS를 호출한다(`global_hybrid.py:320-338`).
- [확인] atb033 CBS는 탐색 노드마다 모든 Agent의 A\*를 다시 계산하고(`cbs.py:226-234`), A\*는 확장할 때마다 open set 전체를 훑으며(`a_star.py:43`), 시간 상한이 없다.
- [확인] 120초 timeout 선례: 4대(6×14), 6대(19×19)
- 따라서 큰 단계에서는 CBS `timeout`과 그에 따른 SAFE_STOP이 결과를 지배할 수 있다. Selective(그룹 단위 CBS)는 이 문제의 후보 해법이지만, P1~P5가 PENDING이고 Runner에 연결되지 않아 **현재 해법이 아니다.**

---

## 7. 목적 분리

| | 기존 4대 시연 | 창고형 v3 실험 |
|---|---|---|
| 목적 | 알려진 교착 메커니즘에서 Trigger → CBS → 복귀의 정확성과 안전 불변식 확인 | 통제된 크기·밀도와 held-out 분포에서 방법별 안전성·정체·계산 비용을 비교 |
| 지도 | 손으로 만든 controlled 지도 | 생성기 v3 + prereg |
| 주장 | 기능 검증(성능 주장 없음) | 사전등록한 지표만 보고. 시연 결과를 외삽하지 않음 |

---

## 8. 구현 전 확인할 사항 (PENDING)

1. 2.2의 slot 배치로 시작 상태의 `robot_count`·`other_goal_count`가 4 이하로 유지되는지. 생성 전에 손으로 예측하고 prereg에 기록한다.
2. `goal_col` −12와 `wall_count` 17이 경계값이다. 실제 goal 집합에서 IN_RANGE가 유지되는지 확인하고, 아니면 등급을 TAIL로 사전등록한다.
3. 스테이션 band를 goal 전용으로 할지, 스테이션 → 상단 흐름의 start로도 쓸지(대향 통행 설계)
4. 각 단계의 중단 임계값(실행 시간 예산 등)을 prereg에서 확정
5. IL_ONLY 실행 결과를 저장할 위치(새 `outputs/` 하위 폴더, 커밋하지 않음)와 실행 승인
6. Global Hybrid의 CBS 사전 검사는 instance별로 별도 승인
7. family ID `fware` 자동 축약을 그대로 쓸지, 아니면 `STRUCT_ABBR`에 추가할지(추가하면 레지스트리 코드가 바뀌므로 별도 승인)
