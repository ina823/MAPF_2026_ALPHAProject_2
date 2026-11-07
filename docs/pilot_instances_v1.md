# PILOT instances v1

작성 2026-11-05. PILOT은 **Development 쪽 데이터**다([`data_split_plan_v1.md`](data_split_plan_v1.md) 6.4절). 생성기와 검사 경로를 확인하는 데만 쓴다. Validation·Test로 옮기지 않고, PILOT family는 Validation·Test에서 쓰지 않는다. 여기서 나온 결과로 정책 파라미터를 고르지 않는다.

## 1. 생성

- 생성기: `src/data/generator.py` (`alpha2_structured` v1, 코드 sha256 `a2e10319…`). 레이아웃은 파라미터로 정해지고, seed는 정해진 구역 안에서 agent 칸만 고른다(`numpy.random.default_rng`).
- 명령: `python -m scripts.generate_pilot_instances --registered-at 2026-11-05` (`--dry-run`이면 아무것도 쓰지 않는다).
- 파일: `scenarios/pilot_v1/` (새 폴더, 6개). 형식은 `scenarios/controlled/`와 같다(JSON `[x, y]`, 지도 `map_<slug>.npy`는 시나리오와 같은 폴더, int64). JSON에는 `generator` 블록(seed, 코드 sha, 파라미터)이 더 들어 있다.
- 같은 seed로 두 번 생성해 바이트가 같은지 확인한 뒤에만 썼다. 쓴 파일은 `scenario_loader`로 다시 읽어 fingerprint가 메모리 값과 같은지 확인했다. 실패하면 새 파일을 지우고 레지스트리를 되돌린다.
- 레지스트리: 기존 16줄은 바이트 단위로 그대로 두고 3줄만 추가했다(`git diff`: 3줄 추가, 0줄 삭제).

| instance_id | 파일 (`scenarios/pilot_v1/`) | 구조 · 파라미터 | 격자 | 빈칸 | agent | seed |
|---|---|---|---|---|---|---|
| `PILOT_fcorr02_001` | `scenario_pilot_corridor_bays_l15.json` | 복도 길이 15, 위쪽 대피칸 col 5·9, 동쪽 2 + 서쪽 1 | 2×15 | 17 | 3 | 11 |
| `PILOT_fbot02_001` | `scenario_pilot_rooms_door_6x14.json` | 6×6 방 2개 + 길이 2의 문 통로(row 2), 동쪽 2 + 서쪽 2 | 6×14 | 74 | 4 | 12 |
| `PILOT_fint02_001` | `scenario_pilot_crossing_9x11.json` | 가로 통로 row 2·6 + 세로 통로 col 5, 남·동·서 방향 각 1 | 9×11 | 29 | 3 | 13 |

시작·목표 (내부 `(row, col)`, agent ID = 순서 = JSON `agent_id`):

| instance_id | starts | goals |
|---|---|---|
| `PILOT_fcorr02_001` | (1,0) (1,2) (1,14) | (1,12) (1,13) (1,0) |
| `PILOT_fbot02_001` | (3,0) (0,1) (2,12) (5,13) | (3,12) (5,12) (0,0) (4,1) |
| `PILOT_fint02_001` | (1,5) (2,1) (6,9) | (8,5) (2,10) (6,1) |

| instance_id | JSON sha256 | 지도 sha256 | `instance_fp` | `map_canon_fp` |
|---|---|---|---|---|
| `PILOT_fcorr02_001` | `5454001d…` | `624e2551…` | `681803d4…` | `290a349e…` |
| `PILOT_fbot02_001` | `8853781e…` | `a24c2dd2…` | `84215a9c…` | `956440d7…` |
| `PILOT_fint02_001` | `9c3383ab…` | `52d9463d…` | `051ff922…` | `b07f6926…` |

## 2. 중복·family·유효성

- 세 PILOT 모두 **새 family**(`fcorr02`, `fbot02`, `fint02`)다. Development를 포함한 어떤 record와도 EXACT·RELABEL·TRIM·SYMMETRY·SAME_MAP 관계가 없다.
- 유효성(생성기와 fingerprint 양쪽에서 검사): 모든 시작·목표가 격자 안의 빈칸이고, 시작끼리·목표끼리 겹치지 않으며, 각 agent의 목표가 시작에서 도달 가능하다(BFS).
- **REVIEW 5건 (모두 미해결)**: `PILOT_fint02_001` ↔ `DEV_fcorr01_001…005`, `SUBMAP`. 9×11 교차로의 1·2행(`11111011111` / `00000000000`)이 Development Corridor를 줄인 2×11 지도(대피칸 1개 + 복도)와 정확히 같다. 실제로 있는 구조 유사성을 잡은 것이다. PILOT과 DEV는 같은 development 도메인이라 등록은 허용됐고, family는 합치지 않았다.
  - 이 REVIEW가 미해결로 남아 있으면 `PILOT_fint02_001`과 SUBMAP 관계인 Validation·Test 후보는 자동 등록되지 않는다(HOLD). PILOT 자체는 Validation·Test로 옮기지 않는다.

## 3. CBS 해결 가능성 (시작 상태 S_0)

- 실행 전 계획: 호출 3회(PILOT마다 1회, `INITIAL_STATE`), timeout 120 s/호출(최악 360 s), 출력 `outputs/pilot_checks_v1/20261105_173041/`, Solver atb033 `89f69e1`(clean), `PYTHONHASHSEED=0`, 가장 긴 출력 경로가 MAX_PATH 이내.
- 실행: `PYTHONHASHSEED=0 python -m scripts.check_pilot_instances --run-solver --timeout-sec 120`. 기존 `CbsRuntimePlanner`와 `PathGuardValidator`를 수정 없이 썼다.

| instance_id | 어댑터 status | 시간 | Path Guard | 경로 길이 합 | makespan | 판정 |
|---|---|---|---|---|---|---|
| `PILOT_fcorr02_001` | `success` | 1.08 s | 통과 | 57 | 19 | 해 존재 |
| `PILOT_fbot02_001` | **`timeout`** | 120.07 s | — | — | — | **미확인** (해가 없다는 뜻이 아님) |
| `PILOT_fint02_001` | `success` | 0.40 s | 통과 | 27 | 9 | 해 존재 |

- `PILOT_fbot02_001`은 방이 넓어서 해는 있을 것으로 보이지만, 이 검사로는 확인되지 않았다. `no_solution`으로 세지 않는다. timeout을 늘려 다시 실행하는 것은 승인 범위 밖이라 하지 않았다.
- 원시 결과: `outputs/pilot_checks_v1/20261105_173041/pilot_checks.json`과 호출별 Solver YAML. `outputs/`의 다른 폴더는 건드리지 않았다. 이 폴더는 `.gitignore` 대상이 아니므로 커밋에서 뺄지 결정해야 한다.

## 4. IL 입력 호환성 (OOD)

reference: `registry/il_input_reference_v1.json` (학습 npz sha256 `425c3f2a…`, [0.5%, 99.5%]). `auto_fail`은 모두 False다(기록만 함).

| instance_id | 결과 | 항목 |
|---|---|---|
| `PILOT_fcorr02_001` | `OOD_WARNING` | START: agent 0·2 `wall_count` 22, agent 1 `wall_count` 20 (> 17); agent 2 `goal_col` −14 (< −12); REACHABLE_RANGE: agent 2 `goal_col` [−14, 0] |
| `PILOT_fbot02_001` | `OOD_WARNING` | START: agent 2 `goal_col` −14; REACHABLE_RANGE: agent 1 `goal_col` 최대 15, agent 2 최소 −15, agent 3 최소 −14 |
| `PILOT_fint02_001` | `OOD_WARNING` | START: agent 1·2 `wall_count` 21; REACHABLE_RANGE: agent 1 `goal_col` [−14, 14], agent 2 [−13, 13] |

비교용으로 Development 16개에도 같은 검사를 했다(정책은 실행하지 않음). Corridor 계열 5개와 Intersection 2개(`4way`, `lshape`)가 `START wall_count` 때문에 `OOD_WARNING`이고, 나머지 9개는 `IN_RANGE`다. 즉 **폭 1 통로(시야 안 벽 > 17칸)와 가로로 13칸이 넘는 이동이 학습 분포 밖의 주요 원인**이다. 학습 pool은 정사각형 random·rooms·maze 지도 위주로 보이며, 이 부분은 [확인 필요]다.

## 5. 개발 단계 미해결 항목 (해결되지 않음)

| 항목 | 상태 | 남은 결정 |
|---|---|---|
| `PILOT_fbot02_001` CBS `timeout` (120 s) | **미해결** — 해 존재 여부 미확인 (`no_solution` 아님) | timeout을 늘려 다시 확인할지, 병목 생성 조건(통로 길이, agent 수)을 바꿀지 |
| REVIEW 5건 (`PILOT_fint02_001` ↔ `DEV_fcorr01_001…005`, `SUBMAP`) | **미해결** — RELATED/UNRELATED 판정 없음 | 판정과 판정 기록 파일 형식 |
| `OOD_WARNING` 3건 (PILOT 3개 모두) | **미해결** — 기록만 함, `auto_fail = False` | 자동 실패 규칙을 둘지. 폭 1 통로는 Development에서도 OOD다 |

이 항목들은 Validation·Test 생성 전에 결정한다. PILOT 결과로 정책 파라미터를 고르지 않는다.
감사 기록(REVIEW 원장 HOLD, OOD 분류, CBS/별도 근거 분리)과 검사 방법은 [`pilot_quality_gate_v0.md`](pilot_quality_gate_v0.md)에 있다(판정은 여전히 미해결).

## 6. 다시 확인하는 방법

```
python -m scripts.check_registry                        # 레지스트리·원본 sha·관계·family
PYTHONHASHSEED=0 python -m scripts.check_pilot_instances   # 재생성 바이트 비교 + OOD + Solver 계획 (Solver 실행 안 함)
```
`src/data/generator.py`를 고치면 코드 sha가 바뀌어 JSON이 달라지고, 재생성 비교가 실패한다(의도된 동작). 이 경우 PILOT을 새 버전으로 다시 만들어야 한다.
