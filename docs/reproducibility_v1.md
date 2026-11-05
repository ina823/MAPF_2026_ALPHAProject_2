# 재현성·비교 공정성 v1 (PYTHONHASHSEED, fair_batch / reproducible_batch)

작성 2026-11-05. 이 문서에는 **실제로 실행해 확인한 결과만** 적는다. 구현하지 않았거나 검증하지 않은 항목은 9절에 따로 모았다.

관련 문서:
- [`global_hybrid_runner_v0.md`](global_hybrid_runner_v0.md): Runner 상태기계, DEV 정책, CSV 필드 정의. 7.3절은 "같은 S_t에서 CBS 경로가 달라지는" 현상을 처음 기록한 곳이며, 그 원인을 이 문서에서 밝힌다.
- [`validation_test_split_v1.md`](validation_test_split_v1.md): Development / Validation / Test 역할 구분. 4.2절은 seed만 바꿔 다시 실행한 것을 새 instance로 세지 않는다는 규칙이다.
- [`alpha1_baseline.md`](alpha1_baseline.md): Strict Baseline(IL 기준선)의 정의.

## 1. 목적

- 같은 조건에서 실험을 다시 돌렸을 때 결과가 같은지 확인한다.
- 방식 간 비교가 같은 조건에서 이루어졌는지(공정성) 확인한다.
- 위 두 가지를 **실행 기록만으로 판정**할 수 있게 하는 절차와 기준을 정한다.

## 2. 실험 환경 (이 문서의 결과가 나온 환경)

| 항목 | 값 |
|---|---|
| 프로젝트 커밋 | E1~E3: `729f391` (clean). 최종 검증(7절): `ae9aa50` (clean) |
| Solver | atb033 `multi_agent_path_planning` `89f69e1` (clean), `solver_root` = `MAPF_ALPHA2_CBS_CHECK/third_party/multi_agent_path_planning/centralized/cbs` |
| Python / 패키지 | Python 3.14.6, torch 2.15.0+cpu, numpy 2.5.4, PyYAML 6.0.3 (`alpha2_test_env`) |
| OS | Windows-11-10.0.26200-SP0, `LongPathsEnabled=0` |
| checkpoint | `checkpoints/cnn_navhint.pt` sha256 `eb92f2f7487342d958fd8c6b23dc951cafe36f1109a2293c2e37907313cf3471` (`checkpoints/MANIFEST.json`과 일치) |
| 시나리오 sha256 | `scenario_corridor_bay_n2.json` `114d5d30…`, `scenario_bottleneck_n4.json` `4cea50b7…`, `map_bottleneck_n4.npy` `aeee5555…` |
| 실행 조건 | `max_steps=160`, CBS timeout 30 s, DEV 정책은 `src/compare/common_runner.DEV_SETTINGS` 그대로 |

## 3. 원인: PYTHONHASHSEED가 CBS 경로 선택에 미치는 영향

### 3.1 발견 경위
- Bottleneck H3의 `vertex_conflicts`가 11/03 배치에서는 312, 11/05 재실행에서는 310이었다.
- 두 실행의 `solver_input.yaml`은 sha256 `99a0b8e8…`로 **바이트 단위까지 같았다**.
- 그런데 `output.yaml`은 서로 달랐다. cost는 둘 다 44였다.
- 경로가 달라지자 IL로 복귀하는 t=5의 상태가 달라졌고, 첫 IL step(t=6)의 vertex 충돌 수가 2와 0으로 갈렸다. 312와 310의 차이는 전부 이 한 step에서 나왔다.

### 3.2 코드상 원인 (atb033 `89f69e1`)
- `a_star.py`는 `open_set`을 `set`으로 두고, `min(temp_dict, key=f_score)`로 다음 노드를 고른다. f값이 같은 노드가 여럿이면 **set을 순회하다 처음 만난 노드**가 선택된다.
- `cbs.py`의 `State.__hash__`는 `hash(str(...))`, 즉 문자열 hash를 쓴다. Python 문자열 hash는 `PYTHONHASHSEED`가 없으면 **프로세스마다 무작위**다.
- `src/cbs_adapter.py`는 `subprocess.run(..., cwd=solver_root)`을 `env=` 없이 호출한다. 따라서 cbs.py 프로세스는 부모의 환경변수를 그대로 물려받는다.
- 결론: seed가 없으면 cbs.py 호출마다 hash seed가 달라지고, cost가 같은 최적 해 중 어느 것이 선택될지가 매번 달라진다.

### 3.3 E1: Solver 단독 실험 (90회)
- 저장된 `solver_input.yaml` 3종을 실행마다 따로 복사해 `cbs.py`를 독립 subprocess로 순차 실행했다.
- seed 조건은 미설정 / 0 / 1이고, 각 10회씩 돌렸다.

| 입력 | 전체 경로 종류 | 미설정 | seed 0 | seed 1 | cost |
|---|---|---|---|---|---|
| Corridor (Trigger 시점 S_t, `5621b9bf`) | 1 | 1종 | 1종 | 1종 (seed 0과 같음) | 17 |
| Intersection (S_t, `315230f6`) | 1 | 1종 | 1종 | 1종 | 9 |
| Bottleneck (S_t, `99a0b8e8`) | **12** | **10종 (10회 모두 다름)** | **1종** | **1종 (seed 0과 다름)** | 44 |

- 90개 출력은 모두 유효했다. 시작점과 목표, 맵 범위, 장애물, 한 칸 이동, vertex·edge 충돌을 직접 검사했다.
- **seed가 같으면 출력이 바이트 단위로 같았고, seed가 다르면 경로가 달라졌다.**
- 이후 E2에서 **Corridor의 S_0 입력(`0da5667e`, Pure CBS가 쓰는 입력)도 seed에 따라 경로가 달라진다**는 것을 확인했다. "Corridor는 seed와 무관하다"는 결론은 Trigger 시점의 입력에만 해당한다.

### 3.4 E2: seed 0으로 같은 배치를 두 번 실행
- 18개 run(기본 5방식 + REARM, 3개 시나리오)을 seed 0으로 두 번 돌렸다. 두 배치 모두 `fair_batch=True`였다.
- **18/18이 완전히 같았다.** 비교한 항목은 trajectory, actions, action_sources, transitions, Monitor, Trigger, CBS 호출과 경로, `cbs_exec_trace`, overlap, conflicts, steps/summary/monitor CSV, Solver YAML의 sha256이다.
- 차이가 난 것은 latency, run_id, 시각, 작업 폴더 경로 같은 메타데이터뿐이었다.
- Bottleneck의 첫 CBS 출력은 E1에서 seed 0으로 단독 실행한 출력과 같았다. 이것으로 seed가 Solver subprocess까지 전달됐음을 확인했다.
- **실행 중 발생한 문제(Windows MAX_PATH):** 처음 실행했을 때 REARM run에서 `…_summary.csv` 경로가 261자가 되어 `FileNotFoundError`로 멈췄다. 출력 폴더 이름을 짧게 바꿔 다시 실행했다. 이후 6절에서 설명하는 사전 검사를 추가했다.

### 3.5 E3: Bottleneck seed 민감도 (seed 0~9 × 4방식 = 40회)
- seed마다 batch 프로세스를 새로 띄우고, 시작할 때부터 `PYTHONHASHSEED`를 설정했다. 10개 배치 모두 `rc=0`, `fair_batch=True`였다.
- 각 배치의 첫 CBS 입력을 같은 seed로 단독 실행한 출력이 배치 출력과 20/20 일치했다.

| 방식 | SUCCESS / TIMEOUT | steps | CBS 호출 | 첫 CBS 경로 종류 |
|---|---|---|---|---|
| PURE_CBS_DEV_SINGLE_PLAN | 10 / 0 | 14 (전부) | 1 | 10 |
| GLOBAL_HYBRID_DEV_FULL | 10 / 0 | 15 (전부) | 1 | 10 |
| GLOBAL_HYBRID_DEV_H3 | 0 / 10 | 160 (전부) | 1 | 10 |
| GLOBAL_HYBRID_DEV_H3_REARM | 10 / 0 | 16 ×8, 17, 19 | 2 | 10 |

- 같은 seed 안에서 H3와 REARM은 첫 CBS 경로, 복귀 상태, 재정체 시점이 같았다.
  - 재정체는 대부분 t=7이었고, seed 7은 t=8, seed 9는 t=10이었다.
  - H3는 다시 CBS를 부르지 못해 TIMEOUT으로 끝났다.
  - REARM은 두 번째 CBS 호출로 10개 seed 모두 회복했다.
- overlap은 40회 모두 없었고, 모든 CBS step에서 expected와 actual이 같았다.
- seed는 경로와 재정체 시점에 영향을 줬지만, 이 표본에서는 **어느 방식의 종료 결과도 바꾸지 않았다**.
- REARM의 두 번째 Trigger가 복귀 시점과 떨어진 간격은 seed 10개 중 8개에서 DEV cooldown 값(2)과 정확히 같았다. 경계값에 매우 가깝다.
- 원시 결과: 40행 기록과 실행 로그는 2026-11-05 세션의 임시 scratchpad(`e1_20261105_run1/`, `e2s/`, `e3/`)에만 있고 저장소에는 없다. 이 문서의 표가 그 요약이다.

## 4. seed 고정과 전달 검증 (`ae9aa50`, `src/compare/provenance.py`)

### 4.1 고정 방법
`PYTHONHASHSEED`는 **인터프리터가 시작할 때** 한 번 적용된다. 실행 도중 `os.environ`을 바꾸면 그 프로세스 자신의 hash는 바뀌지 않고, 이후 자식 프로세스가 물려받는 값만 바뀐다. 따라서 batch 프로세스를 시작하는 명령에서 설정해야 한다.

```
PYTHONHASHSEED=0 python -m scripts.run_compare_batch --solver-root <atb033>/centralized/cbs \
    --out-dir <새 짧은 폴더> --require-hash-seed [--scenario ...] [--method ...]
```

| 값 | 기록되는 모드 | 의미 |
|---|---|---|
| 미설정, `random` | `UNSET_RANDOM` | 프로세스마다 무작위 |
| `0` | `FIXED_DISABLED` | **hash 무작위화가 꺼진 고정 설정** (`sys.flags.hash_randomization=0`) |
| 1~4294967295 | `FIXED` | 그 값으로 고정 |
| 그 밖의 값 | `INVALID` | — |

### 4.2 세 계층을 따로 기록하고 따로 검증
| 계층 | 기록 위치 | 검증 방법 |
|---|---|---|
| ① 부모 프로세스 | `environment.hash_seed`(manifest와 각 `result.json`): 시작 시점 값, 현재 값, 모드, `hash_randomization`, probe hash | 기록된 seed를 **명시적으로 넣은** 자식의 `hash(probe)`가 부모와 같으면 `parent_verified=VERIFIED` |
| ② CBS 호출 | `planner_calls[i].pythonhashseed_env`: 호출 시점에 cbs.py가 물려받을 값 | 배치의 모든 호출이 부모의 시작 시점 값과 같아야 함 (`cbs_calls_seed_consistent`) |
| ③ 상속 probe | `reproducibility.inherited_probe` | cbs.py와 똑같이 `env=` 없이 띄운 자식이 같은 seed와 같은 hash를 보면 `OK` |

- 시작 후 환경변수만 바꾼 경우는 "changed after start-up"으로 검출되며 VERIFIED가 되지 않는다(테스트로 확인).
- `cbs_adapter`가 `env=` 없이 subprocess를 부른다는 전제는 계약 테스트가 지킨다. 이 전제가 깨지면 ②의 기록이 자식의 실제 환경을 나타내지 못한다.

## 5. `fair_batch`와 `reproducible_batch`

| | `fair_batch` | `reproducible_batch` |
|---|---|---|
| 묻는 것 | 이 배치의 방식들이 같은 조건에서 실행됐는가 | 같은 seed로 다시 돌리면 같은 결과를 기대할 수 있는가 |
| 기준 | `check_fairness`: scenario·grid sha, checkpoint sha(MANIFEST), solver root·commit·clean, `max_steps=160`, CBS timeout 30 s, project clean·같은 commit, Python·패키지·platform, scenario×method 조합. 여기에 **배치 안에서 seed 값이 바뀌지 않았을 것**이 추가됨 | seed 모드가 `FIXED` 또는 `FIXED_DISABLED`, ①이 `VERIFIED`, ③이 `OK`, ②가 일관, 이유 목록이 비어 있음 |
| seed 미설정 배치 | **판정이 이전과 같음** | `False` |
| 종료 코드 | 0 = fair, 1 = 쓰기는 했지만 unfair, 2 = 실행 전 거부 (이전과 같음) | 종료 코드에 영향 없음 |

**배치 사이 비교:** `provenance.compare_batch_provenance()`가 아래 중 하나로 판정한다.

| 판정 | 의미 |
|---|---|
| `NOT_COMPARABLE` | 실행 조건이 다름 |
| `LEGACY_UNSEEDED` | seed 기록이 없는 배치가 섞여 있음 (`NOT_RECORDED`) |
| `UNSEEDED_OR_UNVERIFIED` | seed가 미설정이거나 검증되지 않은 배치가 있음 |
| `SEED_SWEEP` | 모두 검증됐지만 seed가 서로 다름. **같은 seed 재현성 비교가 아니라 민감도 분석이다** |
| `SAME_SEED_COMPARABLE` | 모두 검증됐고 seed가 같음 |

## 6. Pure CBS `INITIAL_STATE`와 Global Hybrid `IL_STALL_TRIGGER`

- **Pure CBS**는 t=0에 시작 상태 S_0을 CBS에 넣는다(`INITIAL_STATE`).
- **Global Hybrid**(FULL, H3, REARM)는 IL이 정체한 t≥1의 S_t를 넣는다(`IL_STALL_TRIGGER`).
- 두 경우는 **Solver에 들어가는 시작 상태가 달라서 서로 다른 CBS 계획 문제**다. 기록된 경로 길이 합(`planner_calls` 경로 길이의 합)도 다르다. 예를 들어 E3 Bottleneck에서는 Pure가 60, Global이 56이었다.
- manifest의 `cbs_request_states`에 호출마다 요청 종류, 시점, 시작 상태 sha256, solver input sha256을 기록하고, `cbs_request_states_note`에 이 구분을 명시한다.
- CSV에서는 기존 열 `cbs_request_kind`와 `request_timestep`으로 구분할 수 있다.

**출력 경로 사전 검사:** 실제 `make_run_id` 형식과 실제 파일 목록(steps/summary/monitor CSV, result.json, `cbs/cbs_call_NN/outputs/paths/replan/{input,solver_input,output}.yaml`)으로 가장 긴 경로를 계산한다. Windows에서 `LongPathsEnabled`가 아니고 경로가 259자를 넘으면, **폴더를 만들기 전에** exit 2로 거부한다.

## 7. 비교 가능 범위

| 데이터 | seed | 판정 |
|---|---|---|
| 기존 15개 Development 배치 `compare_20261103_215010` (`08d20c4`, `fair_batch=True`) | **기록 없음 (`NOT_RECORDED`)**. 당시 seed는 고정되지 않았고 값을 추정하지 않는다 | 당시 기준으로 내린 `fair_batch=True` 판정은 그대로 둔다. `reproducible_batch`는 판정할 수 없다(`NOT_RECORDED`). Bottleneck의 CBS 경로는 무작위 표본 1개이며 **다시 만들 수 없다** |
| REARM 첫 실행 `compare_rearm_dev_20261105` (`8038bad`, dirty) | 미설정 (기록 없음) | 개발 확인용. 공정 비교에 쓰지 않는다 |
| E2 (`729f391`, seed 0, 2회) | 실제로는 0이지만 provenance 기능이 생기기 전이라 manifest에는 **기록 없음** | 18/18 재현을 확인한 근거. 세션 기록(`e2_meta.json`)으로만 seed를 확인할 수 있다 |
| E3 (`729f391`, seed 0~9) | 위와 같음 | `SEED_SWEEP` 성격의 민감도 조사. 같은 seed 비교가 아니다 |
| 기존 15개 vs E2 seed 0 | — | trajectory와 CBS 경로가 같았던 것: 세 시나리오의 IL_ONLY·SAFETY, Corridor의 FULL·H3, Intersection의 PURE·FULL·H3. **경로가 달랐던 것**: Corridor PURE, Bottleneck PURE·FULL·H3. 경로가 달랐던 run은 **지표만 참고**한다 |

**`ae9aa50` clean 검증 (출력 `d2`, 2026-11-05 13:05)**
- 실행 조건: Corridor × 6방식, `PYTHONHASHSEED=0 … --require-hash-seed`, `--allow-dirty` 없이.
- 결과: 종료 코드 0, **`fair_batch=True`**(이유 없음), **`reproducible_batch=True`**.
  - seed: `FIXED_DISABLED`, `'0'`
  - 부모 VERIFIED, 상속 OK, CBS 호출 4건 일관
  - project `ae9aa50` clean, solver `89f69e1` clean
- 방식별 결과: IL_ONLY와 SAFETY는 TIMEOUT 160, PURE는 SUCCESS 12, FULL·H3·REARM은 SUCCESS 13이었다.
- `cbs_request_states`: Pure는 `INITIAL_STATE` t=0, FULL·H3·REARM은 `IL_STALL_TRIGGER` t=5.
- 이 배치를 다른 배치와 바이트 단위로 비교하지는 않았다.
- 참고로 같은 코드가 커밋 전 dirty 상태였을 때 돌린 `d1` 배치(6 run)는, 결정적인 필드가 모두 E2 seed 0 결과와 같았다. CSV 헤더도 같았다.

## 8. 테스트 결과
- 전체 unittest **468/468 PASS**. 신규 `tests/test_hash_seed_provenance.py` 24개를 포함하며, `PYTHONHASHSEED=0`으로 실행해도 468/468이었다.
- Strict Baseline(`scripts.smoke_test_il --strict-baseline`) **9/9 PASS**.
- 이 두 결과는 커밋 직전, `ae9aa50`과 내용이 같은 작업 트리에서 마지막 코드 수정 이후에 실행한 것이다. 커밋 후에는 다시 실행하지 않았다.

## 9. 한계와 미완료 항목
- **E1~E3는 Development 시나리오에서 hash seed가 결과에 주는 영향을 본 조사다.** held-out Validation이나 Test 평가가 아니고, 최종 성능 검증도 아니다. seed만 바꾼 재실행은 새 instance가 아니다([`validation_test_split_v1.md`](validation_test_split_v1.md) 4.2절).
- Bottleneck 결과는 시나리오 1개, seed 10개다. 10/10이라는 수치를 일반적인 성공률로 해석하면 안 된다.
- 결정성은 **이 PC, Windows, Python 3.14.6, Solver `89f69e1`**에서만 확인했다. 다른 PC, OS, Python 버전에서는 seed가 같아도 같은 경로가 나온다고 보장할 수 없다.
- CBS latency는 seed를 고정해도 결정적이지 않다. 비교에서는 메타데이터로만 다룬다.
- **아직 없는 것:**
  - Solver seed 사후 재실행 검증 도구(`scripts/verify_solver_seed.py`)는 구현하지 않았다. E3에서는 같은 검증을 수동 스크립트로 했다.
  - Global 방식의 `planner_wall_ms`는 기록되지 않는다.
  - provenance 기능 이전의 manifest에는 seed가 기록되어 있지 않다.
- Solver의 동점 처리를 결정적으로 만드는 일은 atb033 코드의 문제라 CBS 담당(윤지)과 상의할 사항이다. 이번 작업에서는 Solver를 수정하지 않았다.

## 10. Validation / Test 실험 재현성 체크리스트
1. 작업 트리가 clean이고 커밋 SHA가 정해져 있다. `--allow-dirty`를 쓰지 않는다.
2. Solver 커밋이 정해져 있고 clean이다. checkpoint sha256이 `MANIFEST.json`과 일치한다.
3. scenario와 grid의 sha256을 기록한다. Validation·Test instance의 선정과 사용 규칙은 [`validation_test_split_v1.md`](validation_test_split_v1.md)를 따른다.
4. `max_steps`, CBS timeout, H_exec 등 DEV 파라미터는 **실험 전에 정하고**, Test 결과를 보고 바꾸지 않는다(같은 문서 5~6절).
5. batch를 시작하는 명령에서 `PYTHONHASHSEED`를 설정하고 `--require-hash-seed`를 붙인다.
6. manifest에서 `fair_batch=True`, `reproducible_batch=True`, `parent_verified=VERIFIED`, `inherited_probe_status=OK`, `cbs_calls_seed_consistent=True`인지 확인한다.
7. 출력 폴더는 새로 만들고 경로를 짧게 한다. 기존 폴더는 재사용하지 않는다(사전 검사가 거부한다).
8. 같은 seed로 최소 두 번 실행해 결정적인 필드가 같은지 확인한다(E2 방식). latency와 run_id는 비교하지 않는다.
9. seed를 여러 개 돌리는 실험은 `SEED_SWEEP`으로 보고하고, 같은 seed 재현성 결과와 섞지 않는다.
10. Pure CBS(`INITIAL_STATE`)와 Global(`IL_STALL_TRIGGER`)을 같은 계획 문제로 묶지 않는다.
11. Python, OS, 패키지 버전이 달라진 결과끼리는 같은 seed라도 재현을 가정하지 않고 다시 검증한다.
12. 원시 출력(run 폴더, manifest, CSV)은 수정하지 않고 보존한다. 요약은 별도 문서로 남긴다.
