# PILOT v2 후보 (사전 등록 + 정적 검사)

작성 2026-11-08, CBS 결과 추가 2026-11-09. 상태: **후보. 레지스트리에 등록하지 않았고, 독립으로 승인하지 않았다. CBS 사전 검사는 후보당 1회 끝났다(5절).** 조건과 수동 HOLD는 [`pilot_quality_gate_v0.md`](pilot_quality_gate_v0.md) 5절의 제안을 따른다.

## 1. 순서와 그 한계

1. `registry/pilot_v2_prereg_v0.json` 작성 (sha256 `67a748ef…`, 2026-11-08 13:16 UTC 확인). 이때 생성 코드는 없었다.
2. `src/data/generator_v2.py`와 스크립트 작성. 생성 스크립트는 위 sha256을 상수로 고정해, 파일이 달라지면 생성을 거부한다.
3. 2026-11-08 22:21 KST(13:21 UTC) 생성, 이어서 정적 검사.

**한계:** commit 없이 진행했으므로 순서의 근거는 파일 sha256, 시나리오 JSON의 `prereg_sha256`, 작업 기록뿐이다. 제3자가 검증할 수 있는 시각 증거는 아니다.

## 2. 후보 (`scenarios/pilot_v2/`)

| ID | slug | 등급 | seed | 격자 | 빈칸 | agent | 시작 → 목표 (내부 `(row, col)`) |
|---|---|---|---|---|---|---|---|
| `PV2C-001` | `pilot2_staggered_blocks_9x9` | NORMAL | 21 | 9×9 | 65 | 4 | (3,0)→(4,8), (4,1)→(5,8), (0,3)→(8,4), (8,5)→(0,5) |
| `PV2C-002` | `pilot2_loop_w2_7x9` | NORMAL | 22 | 7×9 | 48 | 4 | (0,5)→(5,3), (6,5)→(1,3), (2,1)→(2,7), (3,7)→(3,1) |
| `PV2C-003` | `pilot2_hub_spokes_19x19` | STRESS | 23 | 19×19 | 89 | 6 | (0,7)→(16,10), (2,7)→(17,10), (7,1)→(10,16), (9,2)→(11,18), (13,9)→(1,8), (9,13)→(8,0) |

생성기 `alpha2_structured_v2` v2 (코드 sha256 `b60e272f…`). 기존 `generator.py`(`a2e10319…`)는 import만 하고 그 hash를 JSON에 함께 기록한다.

## 3. 정적 검사 결과

원본: `outputs/pilot_v2_checks/static_20261108_222153/static_checks.json`.

| ID | 유효성 | BFS 도달 거리 | Agent ID | 좌표 | seed 재현 | OOD | 목표 달성 | 경로 길이 합 하한 | makespan 하한 |
|---|---|---|---|---|---|---|---|---|---|
| `PV2C-001` | 통과 | 9, 8, 9, 10 | 통과 | 통과 | 통과 | `IN_RANGE` | **미달** | 36 | 10 |
| `PV2C-002` | 통과 | 11, 11, 8, 10 | 통과 | 통과 | 통과 | `IN_RANGE` | 달성 | 40 | 11 |
| `PV2C-003` | 통과 | 19, 18, 18, 18, 13, 14 | 통과 | 통과 | 통과 | `OOD_TAIL` | 달성 | 100 | 19 |

- **`PV2C-001`:** 폭 1 칸 8개 때문에 NORMAL 목표에 미달한다. 블록 (1,1)과 (6,6)이 지도 가장자리와 2칸 틈을 만들기 때문이고, 사전 등록에 예측되어 있었다. 등급은 NORMAL로 유지한다. 잘라내기 칸(articulation)은 0개라 이 틈이 지도를 나누지는 않는다.
- **`PV2C-003`:** `wall_count` 20(`DENSE_WALL_VIEW`)과 `goal_col` 절댓값 최대 20(`LONG_HORIZONTAL_GOAL_HINT`)이 모두 TAIL이다. 학습 관측 범위는 벗어나지 않는다. `auto_fail`은 False다.
- OOD 등급과 목표 달성 여부는 사전 예측과 모두 일치했다. 다만 예측은 사람이 쓴 것이고, 이 일치가 지도의 품질을 뜻하지는 않는다.
- BFS 도달성은 agent별 **단독** 경로만 확인한다. 여러 agent가 함께 풀 수 있는지는 확인되지 않았다.

## 4. 관계와 HOLD

원장: `registry/pilot_v2_candidate_review_v0.jsonl` (4건, 모두 `HOLD`, `final_decision`과 `decision_owner`는 null). 기존 `review_ledger_v0.jsonl`은 등록된 ID만 받으므로 별도 파일이다.

| ID | 기존 19개와 자동 관계 | 후보 간 자동 관계 | 수동 HOLD | 상태 |
|---|---|---|---|---|
| `PV2C-001` | 0 | 0 | `PV2R0001`: `DEV_femp01_001` (열린 방) | HOLD |
| `PV2C-002` | 0 | 0 | `PV2R0002`: `DEV_fcyc01_001` (벽을 둘러싼 고리) | HOLD |
| `PV2C-003` | 0 | 0 | `PV2R0003`: `fint01`, `fint02` (십자형) / `PV2R0004`: `fbot01`, `fbot02` (방과 폭 1 통로) | HOLD |

자동 관계 0건은 **검사기가 잡는 관계가 없다는 뜻이지 독립 승인이 아니다.** `independence_approved`는 코드에서 항상 False다. `Registry.evaluate`는 메모리 안에서 세 후보 모두 `ACCEPT`를 냈지만, 수동 유사성을 보지 않는 값이고 실제 등록은 하지 않았다.

## 5. CBS 사전 검사 (S_0, 후보당 1회)

조건: atb033 `89f69e1`(clean), `PYTHONHASHSEED=0`, timeout 120 s, 순차 실행, 재실행 없음. 명령: `PYTHONHASHSEED=0 python -m scripts.run_pilot_v2_cbs --candidate <ID> --run`.

| ID | 등급 | CBS status | latency | Path Guard | 실제 SoC (하한) | makespan (하한) | Solver `cost` | 결과 폴더 (`outputs/pilot_v2_checks/`) |
|---|---|---|---|---|---|---|---|---|
| `PV2C-001` | NORMAL (`target_met=false`) | `success` | 1.375 s | 통과 | 39 (36) | 11 (10) | 43 | `cbs_20261108_231954/` |
| `PV2C-002` | NORMAL (`target_met=true`) | `success` | 0.338 s | 통과 | 40 (40) | 11 (11) | 44 | `cbs_20261108_230200/` |
| `PV2C-003` | STRESS (`OOD_TAIL`) | **`timeout`** | 120.105 s | 해당 없음 | 없음 (100) | 없음 (19) | 없음 | `cbs_20261109_010839/` |

- **결과 파일:** `outputs/`는 커밋하지 않으므로 위 결과 폴더는 이 작업을 실행한 로컬 환경에만 있다.
- **`PV2C-003`:** 해 존재 여부 **미확인**이다. `no_solution`이 아니다. 경로가 없어 SoC와 makespan도 없다. `output.yaml`은 없고 Solver stdout/stderr는 비어 있다. 재실행하지 않았고 지도와 Agent 수도 바꾸지 않았다.
- **`PV2C-002`의 SoC:** 이 실행의 `result.json`에는 패딩 값 `sum_of_path_lengths` 44가 남아 있다. 실제 SoC 40의 근거는 `recompute_soc_20261108_231954/pv2c002_soc.json`이다.
- **SoC 정의:** Agent별로 목표에 끝까지 머무는 첫 시각의 합이다. Solver `cost`와 패딩 합은 SoC가 아니다([`pilot_v1_soc_correction_note.md`](pilot_v1_soc_correction_note.md)).
- `success`는 S_0에서 해가 있다는 뜻일 뿐이다. 독립성, IL·Hybrid 동작, 성능을 뜻하지 않는다. 등급, `target_met`, OOD, HOLD는 CBS 결과로 바꾸지 않았다.

## 6. 미해결 항목

| 항목 | 상태 |
|---|---|
| `PV2C-003` CBS `timeout` | 해 존재 미확인. timeout 연장이나 별도 witness는 별도 승인 사항 |
| 수동 HOLD 4건 | 판정자와 절차 미정 |
| `PV2C-001` NORMAL 목표 미달 | 기록만 함. 등급 유지 |
| 독립성 | **승인 안 됨.** 자동 관계 0건은 독립 승인이 아니며, 수동 HOLD 4건이 남아 있다 |
| 레지스트리 등록 여부 | 미정. 등록하면 해결 가능성 기록과 REVIEW 원장도 함께 처리해야 gate가 통과한다 |
| Validation·Test | **등록 안 됨.** 세 후보 모두 Development 쪽 PILOT 후보다 |

## 7. 다시 확인하는 방법

```
python -m scripts.generate_pilot_v2_candidates --dry-run   # 파일 hash만 출력, 쓰기 없음
python -m scripts.check_pilot_v2_candidates                # 정적 검사. Solver 없음. 새 outputs 폴더에 기록
python -m unittest tests.test_pilot_v2_candidates
```

`generator_v2.py`나 사전 등록 파일을 고치면 재생성 비교 또는 sha256 고정 검사가 실패한다(의도된 동작).
