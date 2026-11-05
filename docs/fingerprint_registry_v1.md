# Fingerprint 레지스트리 v1 (`fp_schema = 1`)

작성 2026-11-05. 상태: **1차 구현**. Development 16개만 등록했다. Validation·Test instance는 만들지 않았다.

관련 문서: [`data_split_plan_v1.md`](data_split_plan_v1.md)(분리 계획), [`validation_test_split_v1.md`](validation_test_split_v1.md)(역할·family 규칙). 이 문서는 두 문서의 규칙을 바꾸지 않는다. `data_split_plan_v1.md` 3·5절의 판정 단계와 필드 이름을 이 문서의 기준으로 구체화했으며, 차이는 8절에 적었다.

## 1. 파일

| 파일 | 역할 |
|---|---|
| `src/data/fingerprint.py` | 순수 함수. 격자 정규화, 테두리 제거, 회전·반사, fingerprint 계산, 시나리오 읽기(읽기 전용) |
| `src/data/registry.py` | 관계 판정, family, 등록 결정(ACCEPT/HOLD/REJECT), REVIEW 결정, 안전한 파일 쓰기 |
| `scripts/register_dev_scenarios.py` | Development 16개를 **새** 레지스트리에 등록 (기존 파일이 있으면 거부) |
| `scripts/check_registry.py` | 레지스트리와 원본 파일 대조 (읽기 전용) |
| `registry/instances_v1.jsonl` | instance당 1줄 |
| `registry/families_v1.json` | family별 개수, 관계 수, REVIEW 항목 (instances에서 다시 만들 수 있는 요약) |
| `tests/test_data_fingerprint.py`, `tests/test_data_registry.py`, `tests/test_register_dev_scenarios.py` | 단위·통합 테스트 |

## 2. Fingerprint 계산 규칙 (`fp_schema = 1`)

- **좌표**: 모든 필드는 내부 `(row, col)`이다. 시나리오 JSON `[x, y]`는 `src/common/scenario_loader.load_map_generator_scenario`(`xy_to_rowcol`)로만 변환한다.
- **agent ID**: loader와 같이 **리스트 순서 = agent ID**. JSON의 `agent_id` 값은 `agent_ids_in_file`에 그대로 기록하고 순서와 같은지(`agent_id_matches_index`)만 비교한다.
- **해시**: sha256, UTF-8 텍스트.

| 단계 | 텍스트 | 필드 |
|---|---|---|
| N1 격자 | 정수·bool dtype, 값 {0,1} 검사 → `"H,W:" + 행('0'/'1')을 '/'로 연결` | `grid_fp` (`.npy` 헤더·dtype과 무관) |
| N2 테두리 제거 | 전부 벽인 바깥 행·열 제거, 좌표에서 `grid_trim_offset` 차감 | `grid_trim_fp`, `grid_trim_rows`, `grid_trim_shape` |
| N3 정규형 | N2 격자에 8가지 변환(`np.rot90` k번 + 선택적 `fliplr`)을 적용해 텍스트가 가장 작은 것 | `map_canon_fp`, `map_canon_transforms`(같은 최솟값을 내는 변환 전부) |

agent는 각 단계마다 두 가지로 기록한다. `[sr, sc, gr, gc]` 목록의 compact JSON이다.

| 필드 | 의미 |
|---|---|
| `agents_ordered_fp` / `agents_set_fp` | 원래 좌표. ID 순서대로 / (start, goal) 쌍을 정렬해 ID 무시 |
| `agents_trim_ordered_fp` / `agents_trim_set_fp` | N2 좌표 |
| `agents_canon_ordered_fp` / `agents_canon_set_fp` | N3: **지도와 같은 변환으로 시작·목표도 변환**한 뒤, 정규형을 내는 변환들 중 최솟값 |

start와 goal은 쌍으로 묶여 있어, 같은 칸들이라도 짝이 바뀌면 다른 값이 된다. 유효성 검사: 지도 범위 안, 벽이 아님, 시작끼리·목표끼리 중복 없음.

## 3. 관계 판정

`registry.relate(a, b)`는 위에서부터 처음 맞는 **CONFIRMED 관계 1개**를 내거나, 없으면 **REVIEW 관계 0개 이상**을 낸다.

| 관계 | 조건 | 의미 |
|---|---|---|
| `EXACT` | `grid_fp` 같음 + `agents_ordered_fp` 같음 | **동일 instance**. 유일하게 같은 instance로 보는 관계 |
| `RELABEL_EQUIV` | `grid_fp` 같음 + `agents_set_fp` 같음 (순서만 다름) | 관련 instance. 시뮬레이터 충돌 처리·Safety-only가 ID 순서를 쓰므로 실행이 같다고 가정하지 않는다 |
| `TRIM_EQUIV` | `grid_fp` 다름 + `grid_trim_fp` 같음 + `agents_trim_set_fp` 같음 | 관련 instance. 시뮬레이터·IL 시야·BFS는 맵 밖을 벽으로 취급하지만, 좌표가 달라 CBS 동점 선택(좌표 hash)이 달라질 수 있다 |
| `SYMMETRY_EQUIV` | `map_canon_fp` 같음 + `agents_canon_set_fp` 같음 | 관련 instance. IL은 회전에 대해 같은 행동을 내지 않으므로 실행상 같은 instance가 아니다 |
| `SAME_MAP` | `map_canon_fp` 같음, 시작·목표 다름 | 같은 지도 family (`validation_test_split_v1.md` 4.1·4.2절) |
| `SUBMAP` (REVIEW) | 작은 쪽 N2 지도(8가지 변환)가 큰 쪽 N2 지도의 일부 창과 같음, 작은 쪽 넓이 ≥ `SUBMAP_MIN_AREA`=9 | 사람이 판정 |
| `NEAR_MAP` (REVIEW) | N2 shape이 같고(8가지 변환) 다른 칸이 1~`NEAR_MAP_MAX_DIFF`=2개 | 사람이 판정 |
| `STRUCT_MATCH` (REVIEW) | 둘 다 `structure_descriptor`가 있고 같음 | 사람이 판정. Development는 descriptor가 없어(`null`) 현재 발생하지 않는다 |

- 관계 상세(`details`)에 `id_order_equal`(ID 순서까지 같은지), `shared_start_goal_pairs`, `num_agents`를 기록한다. 즉 **ID 순서를 따른 비교와 ID를 무시한 비교를 따로** 남긴다.
- RELABEL·TRIM·SYMMETRY·SAME_MAP은 instance_id를 합치지 않는다. 각자 자기 instance_id를 가진다.
- 시작·목표만 돌리고 지도는 그대로 두거나, 지도만 돌린 경우는 `SYMMETRY_EQUIV`가 아니다(테스트로 확인).

## 4. family와 등록 결정

- family는 **CONFIRMED 관계**와 **사람이 RELATED로 판정한 REVIEW 관계**로만 묶는다. 미해결 REVIEW는 family를 합치지 않는다.
- 도메인: `DEV`·`PILOT`·`EXTERNAL_REF` = development, `VAL` = validation, `TEST` = test.
- `family_id`: `f<약어><NN>` (corr, int, bot, cyc, emp). `instance_id`: `<SPLIT>_<family_id>_<NNN>`. 한 번 기록하면 바꾸지 않는다.

| 결정 | 조건 |
|---|---|
| REJECT | 유효하지 않은 instance · 같은 파일(경로 또는 JSON sha256)이 이미 등록됨(같은 split: `ALREADY_REGISTERED`, 다른 split: `SPLIT_CHANGE`) · 어떤 record와든 `EXACT`(`DUPLICATE_INSTANCE`) · 다른 도메인 record와 CONFIRMED 또는 RELATED 관계(`FAMILY_LEAK`) · 서로 다른 두 family와 관련(`FAMILY_CONFLICT`) · VAL/TEST인데 `agent_id` ≠ 순서 |
| HOLD | 미해결 REVIEW 관계가 있고, 대상이 VAL/TEST이거나 상대 record가 다른 도메인 (`UNRESOLVED_REVIEW`). 등록하지 않는다 |
| ACCEPT | 그 외. 관련 family에 들어가거나 새 family를 만든다 |

REVIEW 판정은 `ReviewBook.add(key_a, key_b, relation, RELATED|UNRELATED, reviewer, reason)`으로 넣는다. key는 `instance_key` = `grid_fp:agents_ordered_fp`이며, 등록 전 후보도 판정할 수 있다. 1차 구현에서는 판정을 저장하는 파일을 만들지 않았다(REVIEW 0건). 파일 형식은 첫 REVIEW가 생길 때 정한다.

## 5. 안전한 기록

- JSONL 한 줄 = `json.dumps(sort_keys=True, separators=(",", ":"))` + `\n`, UTF-8, BOM 없음. 같은 입력이면 같은 바이트가 나온다.
- 새로 만들 때(`create_files_exclusive`): 같은 폴더 임시 파일에 전부 쓰고 fsync한 뒤 `os.link`로 게시한다. 대상이 이미 있으면 실패하고 덮어쓰지 않는다. 두 번째 파일이 실패하면 첫 번째도 지운다. 즉 둘 다 생기거나 둘 다 안 생긴다.
- 갱신할 때(`replace_file_checked`): 불러온 시점의 sha256과 현재 파일이 같을 때만 `os.replace`로 교체한다. 다른 곳에서 바뀌었으면 거부한다.
- 불러올 때 거부하는 경우: 마지막 줄이 끝나지 않음(일부만 기록됨), 빈 줄, 잘못된 JSON, 중복 `instance_id`, 같은 source 중복, `fp_schema` 불일치, `instance_id`와 split·family 불일치(split만 바꾼 경우 검출).

## 6. 실행 방법

```
../alpha2_test_env/Scripts/python.exe -m scripts.register_dev_scenarios --registered-at 2026-11-05
../alpha2_test_env/Scripts/python.exe -m scripts.check_registry
```
- `register_dev_scenarios`: 0 등록, 1 ACCEPT가 아닌 시나리오가 있거나 원본이 바뀜, 2 실행 전 거부(레지스트리가 이미 있음, 시나리오가 16개 집합과 다름).
- `check_registry`: 0 일치, 1 문제 발견. 원본 sha256, fingerprint 재계산, 관계 재계산, family·도메인 일관성, 미해결 REVIEW, `families_v1.json` 재생성 결과를 확인한다.

## 7. Development 16개 등록 결과 (2026-11-05)

원본 21개 파일(JSON 16 + 지도 5)의 sha256은 등록 전후로 같았다. `check_registry`: consistent.

| family | 수 | 구성 |
|---|---|---|
| `fcorr01` | 5 | corridor_bay_n2, _offset, _3agent, _4agent, 3rdagent_perturbation |
| `fbot01` | 5 | bottleneck_n4, _tightdoor, _convoy, _imbalanced, tightdoor_1agent_control |
| `fint01` | 4 | intersection_n2, _4way, _lshape, _tight |
| `fcyc01` | 1 | cycle_ring_n4 |
| `femp01` | 1 | empty_s11_n3 (smoke) |

- 관계: `SAME_MAP` 26건(5×4/2 + 5×4/2 + 4×3/2). EXACT·RELABEL·TRIM·SYMMETRY 0건.
- **REVIEW 0건**. 다섯 지도의 N2 shape(2×11, 5×9, 7×7, 3×3, 11×11)이 모두 달라 NEAR_MAP이 생기지 않는다. 일부분이 겹치는 지도(SUBMAP)도 없다.
- 모든 시나리오가 `agent_id == 순서`.
- 지도 자체가 대칭인 경우: intersection·cycle ring·empty는 8가지 변환이 모두 같은 정규형을 낸다. corridor는 2가지, bottleneck은 4가지다.
- `dev_role`과 `usage_status`는 `data_split_plan_v1.md` 1절 표를 옮겼다. KI-1 변형 4개는 `variant_ki1` / `HOLD`다.

## 8. `data_split_plan_v1.md`와의 차이 (규칙 변경 아님, 구체화)

- 계획 5절의 `instance_fp = sha256(grid_fp + agents_fp_canon)`은 회전·반사와 ID 순서를 정규화한 agent 값을 써서, 엄격한 instance 판정과 family 판정이 섞인다. 이 구현은 instance 판정을 `EXACT`(원래 격자 + ID 순서)로 하고, 정규화된 값은 관계 판정에만 쓴다.
- 계획 3.2절 D2("정규화 지도가 같으면 같은 family")와 D4는 `SAME_MAP`/`SYMMETRY_EQUIV`로, D3("잘라 내거나 덧붙인 지도")는 `TRIM_EQUIV`(벽 테두리)와 `SUBMAP`(REVIEW)으로 나누었다. D5는 `STRUCT_MATCH`(REVIEW)다. 계획은 D3를 "같은 family"로 적었지만, 여기서는 확인 전까지 REVIEW로 두고 VAL/TEST 등록을 보류한다. 계획보다 엄격한 쪽이다.
- 계획 문서 본문을 개정할지는 승인 대상이다.

## 9. 한계와 승인 필요 사항

- `NEAR_MAP_MAX_DIFF = 2`, `SUBMAP_MIN_AREA = 9`는 1차 값이다(승인 필요). 값이 바뀌면 `fp_schema`를 올리고 레지스트리를 다시 계산한다.
- NEAR_MAP은 테두리를 제거한 shape이 같을 때만 비교한다. 크기가 다르면서 거의 같은 지도는 SUBMAP으로만 잡힌다.
- `structure_descriptor`는 생성기가 아직 없어 정의하지 않았다.
- Alpha 1 지도(`EXTERNAL_REF`)는 등록하지 않았다(별도 승인).
- 모든 쌍을 비교하므로 비교량이 instance 수의 제곱으로 늘어난다(수백 개 규모에서는 문제없음).
- REVIEW 판정 파일 형식, 판정 담당자는 미정이다.
