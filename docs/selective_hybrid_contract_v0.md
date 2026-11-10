# Selective Hybrid 최소 API 계약 v0 (초안)

- 상태: **DRAFT — 윤지와 합의 전. 승인된 구현 명세가 아니다.** 이 문서의 어떤 값도 운영 정책으로 확정되지 않았다. [제안] 항목은 합의 전까지 구현 근거로 쓰지 않는다.
- 현재 코드 상태 [확인]: WFG는 **후보 순수 함수만** 있다(`src/hybrid/wfg.py`의 `build_wfg`, 규칙 `wfg-candidate-v0`, 3.2.1절). Runner, Monitor, CBS 어디에서도 호출하지 않으며 Selective Runner에 연결되지 않았다. **`group_selector`와 Selective Runner는 구현되어 있지 않다.** Selective Hybrid는 실행된 적이 없다. 실행 가능한 hybrid 경로는 Global Hybrid뿐이다.
- 작성 기준: 브랜치 `feature/ina-corridor-e2e-v0`, HEAD `c635276`, 2026-11-10
- 범위: 계약(인터페이스·불변식)과 DEV 로그 기반 입력 검증 설계를 다룬다. 이후 WFG 후보 순수 함수와 단위 테스트(`tests/test_wfg.py`)가 추가되었다. Selective Runner는 만들지 않았다.
- 표기 규칙
  - **[확인]** 코드나 저장된 로그에서 직접 읽은 사실 (파일·줄 근거 병기)
  - **[추정]** 원본 로그·코드에 없는 내용을 추론한 것
  - **[제안]** 이 문서가 내놓는 계약 후보. 합의 전까지 구속력 없음
  - **[PENDING]** 의도적으로 확정하지 않은 항목 (4장)

---

## 1. 용어, 좌표, 시간축

### 1.1 좌표계

| 계층 | 좌표 | 변환 위치 | 근거 |
|---|---|---|---|
| Scenario JSON (`start`, `goal`) | `[x, y] = [col, row]` | 읽을 때 1회 `scenario_loader.xy_to_rowcol` → `(row, col)` | [확인] `src/common/scenario_loader.py:21-28, 63-64`. bottleneck agent 2 `start [7,0]` ↔ 로그 t=1 `current (0,7)` |
| Runner / Simulator / Monitor / Path Guard / 로그(step CSV, `result.json`) | `(row, col)` | 변환 없음 | [확인] step CSV `current_row/current_col`, `cbs_calls[].s_t`, `result.json` `starts/goals` |
| atb033 CBS 입력·출력 | `[x, y]` | `src/cbs_adapter.py`에서만 왕복 변환 (`internal_to_atb033`, schedule → `(row, col)`) | [확인] `src/cbs_adapter.py:3-4, 57-63, 199` |

[제안] Selective(WFG 포함)는 **`(row, col)`만** 주고받는다. WFG는 Scenario JSON을 직접 읽지 않으며, 어떤 새 모듈도 좌표 변환을 추가하지 않는다.

### 1.2 Agent ID 보존 규칙

| 항목 | 내용 | 근거 |
|---|---|---|
| 런타임 Agent ID | Scenario JSON `agents` **리스트의 index** (0..n-1). JSON의 `agent_id` 필드는 런타임 ID로 쓰이지 않는다. 검토한 controlled 시나리오 3개는 두 값이 같다. | [확인] `scripts/run_il.py:246-249` (`{i: starts[i]}`), `src/compare/common_runner.py:299` |
| 순서 | 런타임은 정렬된 ID 순서(`sim._agent_ids`, `agent_order`)를 쓴다 | [확인] `MAPFStepSimulator.reset`, `common_runner.py:394` |
| CBS 경계 | index↔ID 변환은 `run_cbs_runtime()` 한 곳에서만 한다 (정렬 후 호출하고 `paths_by_agent[agent_ids[i]]`로 복원) | [확인] `src/cbs_runtime_adapter.py`, `src/hybrid/cbs_runtime_bridge.py` 모듈 설명 |
| Selective 규칙 [제안] | 그룹은 **런타임 Agent ID의 집합**이다. 부분 집합 CBS 호출에서도 반환 경로의 키는 원래 Agent ID여야 한다(0..k-1로 다시 매기지 않는다). WFG 입출력, 로그, 검증 모두 같은 ID를 쓴다. | — |
| PENDING | JSON `agent_id`와 리스트 index가 다른 시나리오를 허용할지(현재 loader는 `agent_id`를 무시함) | — |

### 1.3 시간축

| 항목 | 계약 | 근거 |
|---|---|---|
| step CSV의 `timestep = t` 행 | `current` = step t 직전 위치(S_{t-1}), `intended` = current + IL 행동 delta(충돌 해소 전 의도), `next` = step t 직후 위치(S_t) | [확인] corridor 행 t=5의 `next` = `cbs_calls[0].s_t` = `{0:(2,4), 1:(2,6)}` |
| **S_t** | Trigger timestep t에서 step t를 끝낸 직후의 전체 Agent 위치 `{agent_id: (row, col)}`. 기존 `cbs_calls[].s_t`와 같은 정의다. | [확인] `global_hybrid.py`의 CBS_REQUEST 기록 |
| CBS 경로 index k | `path[0] = S_t`, `path[k]` = timestep t+k에 기대하는 위치 | [확인] `cbs_exec_trace` (t=6, k=1, …) |

> 주의 [확인]: step CSV의 `trigger_reason`은 Monitor가 rearm될 때까지 CBS 실행 구간 내내 `TRIGGER_FULL_REPEAT`로 남는다. 새 Trigger인지 이전 Trigger의 잔존값인지는 이 열만으로 구분할 수 없으므로, Trigger 시점의 기준은 `result.json`의 `transitions`(IL→TRIGGER)와 `cbs_calls[].trigger_timestep`으로 한다.

---

## 2. 현재 코드가 지원하는 것 / 아직 없는 것

| 구성요소 | 지원함 [확인] | 아직 없음 [확인] |
|---|---|---|
| DeadlockMonitor (`src/common/deadlock_monitor.py`) | 전체 joint-state 반복(FULL) Trigger, `blocking_type`, persistent_conflict / no_progress 관찰, `rearm()` | Trigger-Partial(부분 집합 반복), 그룹 정보 출력, Coordination-stall 라벨 |
| WFG | — | **런타임 연결 없음.** 후보 순수 함수 `build_wfg`(`src/hybrid/wfg.py`, 3.2.1절)만 있고 어떤 Runner도 호출하지 않는다. `common_runner.NOT_IMPLEMENTED = ("SELECTIVE_HYBRID", "SELECTIVE_WFG", "WFG")`이고, 이 메서드는 실행 전에 거부된다 (`src/compare/common_runner.py:26,70`) |
| `group_selector` | — | **파라미터 자체가 없다.** `src/`, `scripts/`, `tests/` 어디에도 정의가 없고 grep 결과 0건이다. 3.8절의 "None 보존"은 새로 도입할 때의 계약이다. |
| Global Hybrid Runner (`src/hybrid/global_hybrid.py`) | IL→TRIGGER→CBS_REQUEST→CBS_EXEC→IL_RESUME/SAFE_STOP, h_exec, rearm 모드(호출 수, 시간 예산, cooldown, same-state) | 부분 집합 CBS 호출, 그룹 밖 Agent 행동 정책, L0/L1/G 사다리 |
| CBS 경로 (`run_cbs_runtime` → `replan_with_status` → `replan` → `CBSAdapter`) | 넘겨준 Agent 집합만으로 CBS 호출(부분 집합 입력 자체는 형식상 가능), ID 복원, 6종 status | 그룹 밖 Agent를 **시간 의존** 장애물·예약으로 넘기는 입력 경로 |
| atb033 CBS (`../MAPF_ALPHA2_CBS_CHECK/.../centralized/cbs/cbs.py`, 89f69e1) | 입력 YAML: `map.dimensions`, `map.obstacles`(정적), `agents[name,start,goal]` (L311-328). 내부 Vertex/EdgeConstraint | 외부 입력으로 시간축 제약을 넣는 경로 없음. root 노드 제약이 항상 빈 `Constraints()`로 초기화된다 (L260-265). 탐색 시간 상한 없음 |
| Path Guard (`src/path_guard.py`) | `validate_path_guard`(끝점 + 이동 + 지도 + vertex/edge + 동일 길이), 개별 함수 `validate_path_map_and_conflicts(paths, grid_map)`는 끝점 검사 없이 vertex/edge를 검사 | Selective용 "그룹 경로 + 그룹 밖 예약" 합성 검증 연결. `validate_path_guard`는 Agent 집합이 `paths`/`current`/`goals`에서 정확히 일치해야 하므로 그룹 호출에서는 **그룹 내부만** 검증한다 |
| Replan boundary (`src/hybrid/replan_boundary.py`) | status, keying, 시작점=S_t, 인접 이동, 주입된 validator | `agent_order`가 전체 Agent 기준이다. 부분 집합 계획을 받는 경로는 없다 |
| Simulator (`MAPFStepSimulator`) | 매 step 전체 Agent의 vertex/edge 충돌을 **취소 방식**으로 해소(겹침이 생기지 않음), `_last_conflict_types` | 계획 위반을 오류로 올리지 않는다. 위반은 Runner의 expected/actual 비교(EXEC_MISMATCH)로만 드러난다 |
| 로그 | step CSV 25열, monitor CSV(최종 상태 1행), `result.json`의 `cbs_calls`/`planner_calls`/`cbs_exec_trace`/`transitions` | 그룹 ID, WFG 간선, 선택 레벨(L0/L1/G), 그룹 밖 Agent 처리 기록 |

---

## 3. 계약 v0

### 3.1 Trigger 입력 (Runner → WFG)

[제안] Trigger 1회당 WFG에 넘기는 입력은 다음과 같다.

```text
WfgInput:
  trigger_timestep: int                       # t
  agent_ids: tuple[int, ...]                  # 정렬된 전체 ID
  positions: {agent_id: (row, col)}           # S_t (전체 Agent)
  intended:  {agent_id: (row, col)}           # step t에서 IL이 시도한 의도 셀 (충돌 해소 전)
  conflict_types: {agent_id: "none"|"wall"|"vertex"|"edge"}   # step t의 sim._last_conflict_types
  goals: {agent_id: (row, col)}
  grid: 2D (0=free, 1=obstacle)
  monitor: {trigger_type, first_seen, period, blocking_type}  # 기존 Monitor 출력 그대로
```

- [확인] 위 필드는 모두 현재 Runner가 step t 시점에 이미 들고 있다 (`run_step`이 intended/current를 Monitor에 넘김).
- [추정] "S_t 이후 IL이 다시 시도할 의도"는 아직 계산되지 않은 값이다. v0는 **step t에서 차단된 의도**를 대신 쓴다. IL_ONLY 로그에서는 같은 S_t에서 IL이 같은 행동을 반복했다 (5장). 다만 이것이 IL의 결정성을 보장하지는 않는다.

### 3.2 WFG 반환 (WFG → Runner)

[제안, 형식은 PENDING P5]

```text
WfgResult:
  edges: list[(src_id, dst_id, kind)]   # kind ∈ {"occupy", "contest", "swap"}
  groups: list[frozenset[int]]          # 서로소, 정렬 규칙 고정(최소 ID 오름차순)
  trigger_group_ids: list[int]          # groups 중 Trigger 원인을 포함하는 그룹 index
  meta: {wfg_version: str, rule: str}
```

간선 정의 후보 [제안]:

- `occupy`: i의 intended 셀을 S_t에서 j가 점유하고 있으면 i→j
- `contest`: i와 j의 intended 셀이 같은 빈 셀이면 i↔j
- `swap`: i의 intended = j의 위치이고 j의 intended = i의 위치이면 i↔j

그룹은 위 간선의 무향 연결 성분 가운데 **차단된(blocked) Agent를 하나 이상 포함하는 성분**으로 한다.

#### 3.2.1 현재 후보 구현 `wfg-candidate-v0` (DRAFT, P5 미합의)

`src/hybrid/wfg.py`의 `build_wfg(current, intended, blocked, grid) -> WfgResult`는 위 제안 규칙을 순수 함수로 옮긴 **후보 구현**이다. 합의된 형식이 아니며, 런타임에서 호출되지 않는다.

- **결과의 의미**: 출력은 한 step의 이동 의도에서 계산한 **후보 간선과 후보 그룹**이다. 그룹 CBS가 성공하거나, 해당 그룹만 계획하는 것이 안전하다는 뜻이 **아니다**.
- **입력**: `(row, col)` 좌표와 런타임 Agent ID만 받는다. 로그의 `conflict_type`과 Monitor `blocking_agents`는 입력이 아니다. swap은 current/intended에서 다시 계산한다(5.4절: 로그에는 vertex로 남음).
- **3.2 제안 형식과의 차이**:
  - `edges`: `WfgEdge(src, dst, kind, cell, occupant_moves)`의 정렬된 tuple이다. contest·swap은 `src < dst`, occupy는 방향이 있다.
  - `groups`: 정렬된 tuple의 tuple이다.
  - 메타 정보는 `rule_version`뿐이다. `trigger_group_ids`는 아직 없다.
  - contest는 목표 셀이 비어 있지 않아도 만든다. 점유된 셀이면 occupy 간선이 함께 생긴다.
  - 벽이나 맵 밖으로 향하는 의도는 "이동 없음"으로 처리한다.
- **T7 (미합의 정책)**: 점유자가 떠나는 칸으로 다른 Agent가 진입하려는 경우, 현재 구현은 `occupy` 간선을 만들고 그룹 연결에도 포함한다. 이것은 그룹을 크게 만드는 방향의 **임시 선택이며, 검증된 규칙이 아니다**. 해당 간선은 `occupant_moves=True`로 표시된다. T7 단위 테스트는 기대값 없이 **SKIP**으로 둔다. 저장된 네 사례 중 이 상황은 Bottleneck t=7에만 있으며, 이 경우 그룹 결과는 T7 규칙과 무관하다(contest·swap만으로 연결됨, `tests/test_wfg.py` T4b).

### 3.3 그룹 선택 사다리: L0 → L1(확장) → G → SAFE_STOP

```text
for level in [L0, L1, ..., G]:
    if 예산 소진(3.6): break
    req = 그룹(level) 기준 CBS 요청
    res = CBS(req)
    if res.status == success and 전체 로봇 검증(3.5) 통과:
        CBS_EXEC(그룹 경로 + 그룹 밖 정책 3.4) → return
SAFE_STOP(reason = 마지막 실패 사유)
```

- 전제: 그룹 밖 Agent의 안전한 처리(3.4, P2)와 전체 로봇 검증(3.5, P3)이 검증되기 전까지는 L0/L1 단계를 실제 실행 경로로 쓰지 않는다. **G(Global fallback)를 우선 경로로 고려한다.** 아래 사다리는 합의 대상인 구조 제안이다.
- **L0** [제안]: `trigger_group_ids`에 해당하는 그룹들의 합집합. 독립 그룹이 여럿이면 합집합 1회 호출로 할지, 그룹별로 호출할지는 PENDING(P4, P5).
- **L1..** [제안, 규칙 PENDING P5]: L0에 "L0 경로가 지나는 셀이나 그 인접 셀을 점유한 Agent", 또는 "L0 Agent의 goal을 점유한 Agent"를 더한다.
- **G**: 전체 Agent. 기존 Global Hybrid의 CBS 요청과 **같은 입력**이어야 한다(회귀 기준).
- **SAFE_STOP**: G까지 실패했거나 예산이 소진된 경우. 기존 SAFE_STOP 의미(이후 이동 없음, I-4)를 그대로 쓴다.
- 각 레벨의 실패 사유(`CBS_*`, `VALIDATOR_REJECT:*`, `SELECTIVE_*`)는 모두 기록한다.

### 3.4 그룹 밖 Agent의 동작과 예약

**[PENDING P2] 안전한 방식이 검증되지 않았다. 검증 전까지는 Global fallback(G, 전체 Agent CBS)을 우선 경로로 고려한다.**

- 후보 A (검토용): 실행 구간(h_exec step) 동안 그룹 밖 Agent를 WAIT으로 고정하고, CBS에는 S_t 위치를 정적 장애물로 넘긴다.
- 후보 B (검토용): 그룹 밖 Agent는 IL을 계속 따르고, 그룹 경로와 충돌하는 step만 WAIT으로 대체한다. 예약 테이블이 필요하며 v0 범위 밖이다.

후보 A(WAIT 고정)의 알려진 한계:

1. **점유 칸 통과와 목표 점유**: 다른 그룹의 경로가 고정된 Agent의 점유 칸을 지나야 하거나, 그 칸을 목표로 삼을 수 있다. 이때 그룹 CBS는 실패하거나(정적 장애물), 고정 Agent와 충돌하는 계획을 낸다(장애물로 넘기지 않은 경우).
2. **그룹 내부 검사만으로는 전체 안전성을 보장하지 못한다**: 그룹 CBS와 현재 Path Guard 호출은 그룹 안의 vertex/edge만 본다. 그룹 경로와 그룹 밖 Agent 사이의 충돌, 동시에 실행되는 다른 그룹과의 충돌은 별도의 전체 검증(3.5, P3) 없이는 잡히지 않는다.
3. **실행 중 그룹 밖 Agent의 동작 제어가 필요하다**: 현재 Runner는 CBS_EXEC 중 전체 Agent에 CBS 행동을 넣는다. 그룹 밖 Agent에 WAIT을 강제하는 경로는 없다 [확인]. 고정을 하려면 Runner가 실행 창 동안 그룹 밖 Agent의 행동을 직접 결정해야 하고, 고정 중 no_progress가 쌓이는 것과 재Trigger의 상호작용도 정해야 한다.
4. **solver 제약**: atb033 코드 판독상 시간 한정 장애물을 넣을 수 없다(2장, P1). 따라서 현재 solver로는 고정 Agent가 실행 창(h_exec) 동안이 아니라 **계획 전체 horizon 동안의 정적 장애물**로만 표현될 수 있다. 이 판단은 실행으로 검증하지 않았다.
- [추정, CBS 미실행] 정적 장애물 방식의 한계는 bottleneck t=2 사례에서 이미 보인다(5.3절). {0,1}만 계획하면 agent 0의 goal (1,7)과 agent 1의 goal (3,7)을 그룹 밖 agent 2, 3이 점유하고 있다. 또 `cbs_adapter`는 start/goal이 막힌 셀이면 `ValueError`를 던진다(`src/cbs_adapter.py:341-348`). 그 결과 `input_error`나 `no_solution`이 날 가능성이 높다. 이 경우 "부분 goal(창 끝 목표)" 또는 L1 확장이 필요하다.

### 3.5 전체 로봇 vertex/edge 충돌 검증

[제안. 그룹 밖 Agent 검증 방식과 책임 분담은 PENDING P3]

- 그룹 CBS 결과를 실행하기 전에 **전체 Agent**에 대해 검증한다. 검증 대상은 `그룹 경로[0..H] ∪ 그룹 밖 Agent의 예정 위치[0..H]`(후보 A라면 S_t 반복)이다. H = min(h_exec, horizon).
- 사용 가능한 기존 함수 [확인]: `validate_path_movements(paths)`, `validate_path_map_and_conflicts(paths, grid_map)`. 끝점 검사가 없으므로 창 단위의 합성 경로에 쓸 수 있다. `validate_path_guard`는 path[-1]이 goal과 같아야 하므로 창 단위 합성 경로에 그대로 쓸 수 없다.
- 실행 중에는 기존 EXEC_MISMATCH 검사를 **전체 Agent**의 expected/actual에 그대로 적용한다. Simulator는 충돌을 취소로 해소하므로 위반이 겹침으로 나타나지 않고 불일치로 나타난다 [확인].

### 3.6 CBS 호출 횟수와 총 시간 예산

[PENDING P4] 참고용 현재 값은 아래와 같다.

- [확인] DEV rearm: `max_cbs_calls=2`, `cooldown=2`, `cbs_time_budget_ms=60000`, solver `timeout_sec=30` (`scripts/run_hybrid_resume_dev.py`, `common_runner`). 모두 DEV 시험값이다.
- [확인] 저장 로그의 CBS latency: corridor 235.5 ms, intersection 241.1 ms, bottleneck 3117.6 ms(t=2), 383.5 ms(t=7).
- 결정할 것 (모두 PENDING P4): (a) L0/L1/G 각각을 `max_cbs_calls`에 1회로 셀지, Trigger 단위로 셀지 (b) Trigger당 레벨별 timeout (c) 레벨 하나가 timeout되었을 때 다음 레벨로 넘어갈지, 바로 SAFE_STOP할지 (d) 누적 latency에 실패한 호출도 포함할지(현재 `latency_ms`는 실패 호출도 측정됨 [확인]).

### 3.7 IL 복귀, 재Trigger, 로그

- [제안] 실행 창이 끝나면 기존과 같이 `IL_RESUME`으로 가고, rearm 모드라면 Monitor `rearm()`을 1회 호출한다. 재Trigger 거부 규칙(RECALL_*)은 기존 규칙을 그대로 재사용한다.
- [제안] 로그 추가 필드(`result.json`의 `cbs_calls[]` 항목에 append-only로 추가):
  `selective_level` (L0/L1/G), `group_agent_ids`, `wfg_edges`, `wfg_groups`, `outside_policy` ("WAIT_FREEZE" 등), `outside_agent_ids`, `full_check_ok`, `full_check_reason`.
- [제안] step CSV의 `STEP_FIELDS`는 바꾸지 않는다. 그룹 정보는 `result.json`에만 기록한다. MONITOR_FIELDS도 그대로 둔다.

### 3.8 `group_selector=None` 동작 보존

- [제안] Selective 도입 시 `GlobalHybridConfig`(또는 새 Runner config)에 `group_selector: Optional[...] = None`을 추가한다. **None이면 현재 Global Hybrid와 결과가 같아야 한다.** 비교 대상은 `transitions`, `cbs_calls`, `cbs_exec_trace`, `trajectory`, step CSV의 행 단위 내용이다.
- [확인] 현재 이 파라미터는 없다. 따라서 보존 대상은 현재 HEAD `c635276`의 Global Hybrid 동작이다. 회귀 기준은 3.5절과 6장의 저장 로그다.
- `common_runner.NOT_IMPLEMENTED` 거부 동작은 Selective 메서드가 등록되기 전까지 그대로 둔다.

---

## 4. PENDING (확정 금지 항목)

| ID | 항목 | 현재 알고 있는 것 | 결정에 필요한 것 |
|---|---|---|---|
| P1 | atb033 CBS의 동적 장애물·시간축 예약 지원 여부 | [코드 판독만] 입력 YAML에 시간 제약 필드가 없고 root 제약은 빈 상태로 시작한다. **실행으로 검증하지 않았다. 구현 가능한 기능으로 간주하지 않는다.** | 윤지 측 확인: solver 수정 없이 가능한지, wrapper나 다른 solver가 필요한지 |
| P2 | 그룹 밖 로봇을 실행 구간 동안 WAIT으로 고정하는 방안의 안전성과 한계 | 3.4의 한계 1~4. [추정] bottleneck t=2에서는 goal 점유로 실패할 가능성이 높다. 안전한 해결책이 검증되지 않았으므로 **Global fallback 우선** | 창 끝 목표 사용 여부, L1 확장 규칙, 실행 중 그룹 밖 행동 제어 방식, no_progress와 재Trigger의 상호작용 |
| P3 | Path Guard와 Simulator의 전체 Agent 검증 책임 | [확인] Path Guard는 넘겨받은 집합만 검증하고, Simulator는 취소 방식이라 오류를 내지 않는다. 그룹 밖 Agent까지 포함하는 검증 경로는 없다 | 그룹 밖 Agent 검증 방식(합성 경로의 구성, 창 길이), 위치(Runner/boundary/Path Guard), 구현 담당 |
| P4 | Trigger당 L0/L1/G 호출 예산, 전체 CBS 호출·시간 예산, 실패 상태 처리 | [확인] DEV 값만 있다 (3.6). 현재 Global Hybrid는 모든 비성공 status를 거부하고 SAFE_STOP으로 간다 | 호출 수 계산 단위, Trigger당·에피소드당 총 시간 예산, 레벨별 timeout, `timeout` / `no_solution` / `invalid_path`(+`input_error`, `solver_error`, `VALIDATOR_REJECT`) 각각에 대해 다음 레벨로 진행할지 SAFE_STOP할지 |
| P5 | 윤지의 WFG 결과 형식과 그룹 확장 규칙 | 3.2·3.3은 인아의 제안일 뿐이다 | 간선 종류, 그룹 정렬과 결정성, 독립 그룹 처리, L1 규칙 |

---

## 5. DEV 저장 로그 검토 (읽기 전용)

출처: `outputs/logs/compare_rearm_dev_20261105/runs/*_GLOBAL_HYBRID_DEV_H3_REARM/` (steps.csv, result.json), 비교용 `*_IL_ONLY/*_steps.csv`, 지도 `scenarios/controlled/map_*.npy`. CBS는 재실행하지 않았다.

### 5.1 Corridor (`corridor_bay_n2`), Trigger t=5

지도(행 2가 통로, (1,5)가 bay):

```text
###########
#####.#####
...........
###########
```

| Agent | S_4 (current) | 의도(step 5) | S_5 | conflict | blocked |
|---|---|---|---|---|---|
| 0 | (2,4) | RIGHT → (2,5) | (2,4) | vertex | True |
| 1 | (2,6) | LEFT → (2,5) | (2,6) | vertex | True |

- [확인] Monitor: FULL, first_seen=4, period=1. `cbs_calls[0].s_t = {0:(2,4), 1:(2,6)}`
- WFG 후보 간선: `contest 0↔1 @ (2,5)`. 점유 간선은 없다((2,5)가 비어 있음).
- 그룹: **{0,1}** = 전체 Agent. L0 = G이므로 Selective와 Global이 같은 요청이 된다.
- 기존 CBS 결과 [확인]: agent 1이 bay (1,5)로 비켜 준다 (`planner_calls[0].paths`).

### 5.2 Intersection (`intersection_n2`), Trigger t=2

| Agent | S_1 | 의도(step 2) | S_2 | conflict | blocked |
|---|---|---|---|---|---|
| 0 | (3,2) | RIGHT → (3,3) | (3,2) | vertex | True |
| 1 | (2,3) | DOWN → (3,3) | (2,3) | vertex | True |

- [확인] `s_t = {0:(3,2), 1:(2,3)}`, first_seen=1
- 간선: `contest 0↔1 @ (3,3)`. 그룹: **{0,1}** = 전체 (L0 = G)

### 5.3 Bottleneck (`bottleneck_n4`), Trigger t=2 — **독립 그룹 사례**

지도(열 4는 (2,4)만 열려 있음):

```text
....#....
....#....
.........
....#....
....#....
```

| Agent | S_1 | 의도(step 2) | S_2 | conflict | goal |
|---|---|---|---|---|---|
| 0 | (1,2) | DOWN → (2,2) | (1,2) | vertex | (1,7) |
| 1 | (2,1) | RIGHT → (2,2) | (2,1) | vertex | (3,7) |
| 2 | (1,7) | DOWN → (2,7) | (1,7) | vertex | (0,1) |
| 3 | (3,7) | UP → (2,7) | (3,7) | vertex | (4,1) |

- [확인] `s_t = {0:(1,2), 1:(2,1), 2:(1,7), 3:(3,7)}`. Monitor는 4개 Agent 모두 VERTEX로 차단되었다고 기록했다.
- 간선: `contest 0↔1 @ (2,2)`, `contest 2↔3 @ (2,7)`. 두 쌍 사이에는 간선이 없다.
- 그룹 후보: **{0,1}, {2,3}** — 3.2의 제안 규칙을 저장 로그에 적용해 계산한 **로그 기반 후보**다. 이후 후보 함수 `build_wfg`가 같은 값을 재현함을 단위 테스트(T3)로 확인했지만, 이것은 규칙 적용 결과일 뿐 **Selective 실행 성공 사례가 아니다.** 실제 기록된 CBS 호출은 4개 Agent 전체에 대한 Global 호출 1회다. 저장 로그 중 독립 그룹 후보를 시험할 수 있는 유일한 입력이다.
- [추정, CBS 미실행] WFG 그룹은 독립이지만 공간적으로는 독립이 아니다. 두 그룹 모두 병목 (2,4)를 지나야 하고, agent 0, 1의 goal을 agent 2, 3이 S_2에서 점유하고 있다. 따라서 {0,1}만 L0로 계획하고 {2,3}을 정적 장애물로 고정하면 실패할 가능성이 높다. 반대로 {2,3}의 goal (0,1), (4,1)은 {0,1}이 점유하지 않는다. 이 사례는 P2와 L1 확장 규칙을 시험하기에 적합하다.

### 5.4 Bottleneck, Trigger t=7 (첫 CBS 창 이후 IL 재진입)

| Agent | S_6 | 의도(step 7) | S_7 | conflict(기록) |
|---|---|---|---|---|
| 0 | (2,4) | RIGHT → (2,5) | (2,4) | vertex |
| 1 | (2,5) | DOWN → (3,5) | (2,5) | vertex |
| 2 | (2,6) | LEFT → (2,5) | (2,6) | vertex |
| 3 | (3,5) | UP → (2,5) | (3,5) | vertex |

- [확인] `cbs_calls[1].s_t = {0:(2,4), 1:(2,5), 2:(2,6), 3:(3,5)}`, latency 383.5 ms. `monitor.trigger_history[1]`: first_seen=6, period=1, VERTEX, armed_since=5(첫 rearm 이후 두 번째 Trigger)
- 간선: `occupy 0→1`, `occupy 2→1`, `occupy 3→1`, `occupy 1→3`, `contest {0,2,3} @ (2,5)`, `swap 1↔3`
- [추정] 1↔3은 기하학적으로 swap(edge) 관계다. 그러나 로그에는 4개 Agent 모두 `vertex`로 기록되어 있다. Simulator가 (2,5) 경합을 먼저 취소한 뒤 agent 1의 (3,5) 진입을 vertex로 취소하고, conflict 우선순위가 vertex > edge이기 때문으로 보인다. 따라서 `conflict_type`만으로는 swap 간선을 복원할 수 없고, current/intended에서 다시 계산해야 한다.
- 그룹: **{0,1,2,3}** 단일 그룹 (L0 = G)

### 5.5 재구성 가능성 요약

| 데이터 | 정확히 재구성 가능 | 근거 / 부족한 점 |
|---|---|---|
| S_t (전체 위치) | 예 [확인] | step CSV `next_*` = `cbs_calls[].s_t` |
| step t의 의도 셀 | 예 [확인] | `intended_*` (충돌 해소 전) |
| conflict_type, blocked | 예 [확인] | 다만 swap은 vertex로 가려질 수 있다 (5.4) |
| goals, grid | 예 [확인] | 시나리오 JSON([x,y]→(row,col) 변환 필요), `.npy` |
| first_seen / period / blocking_type | 예 [확인] | `result.json` → `monitor.trigger_history[]`, `monitor.triggers[].positions` (monitor CSV는 rearm 후 최종 1행뿐이라 사용할 수 없다) |
| **S_t 이후 IL의 의도** | 부분 [추정] | corridor/intersection/bottleneck t=2는 IL_ONLY 로그에서 같은 S_t에서 같은 행동이 반복된다. **bottleneck t=7의 S_7에 대한 IL 출력은 어느 로그에도 없다** (H3_REARM t=8은 CBS 행동) |
| WFG 간선과 그룹 | 계산으로만 가능 [추정] | 로그에 WFG 기록이 없다. 3.2의 규칙(후보 구현 `wfg-candidate-v0`)을 적용한 결과이므로 규칙이 바뀌면 함께 바뀐다 |
| 그룹 CBS 결과 | 없음 | 저장된 CBS는 모두 전체 Agent 호출이다. CBS 재실행 금지이므로 이번 단계에서는 만들 수 없다 |
| CBS 내부 탐색량 | 없음 | atb033 출력은 schedule/cost뿐이다 |

---

## 6. 다음 단계 테스트 설계 (T-WFG만 구현됨, 아래 진행 상황 참조)

| ID | 대상 | 입력 | 기대 |
|---|---|---|---|
| T-WFG-1 | 단일 contest | 5.1 corridor S_5 | edges=[contest 0↔1], groups=[{0,1}] |
| T-WFG-2 | 교차로 | 5.2 S_2 | groups=[{0,1}] |
| T-WFG-3 | **독립 그룹** | 5.3 S_2 | groups=[{0,1},{2,3}], 그룹 사이 간선 없음, 출력 순서 결정적 |
| T-WFG-4 | occupy·swap 혼합 | 5.4 S_7 | swap 1↔3 검출, groups=[{0,1,2,3}] |
| T-WFG-5 | 결정성 | 위 입력에서 dict 순서만 섞음 | 동일 결과 |
| T-NONE-1 | `group_selector=None` 회귀 | 3개 시나리오 H3_REARM | `transitions`/`cbs_calls`/`trajectory`가 저장 로그와 같음 (CBS 실행이 필요하므로 다음 단계에서 승인 후 진행) |
| T-FULL-1 | 전체 로봇 합성 검증 | 그룹 경로 + 그룹 밖 WAIT, 인위적 vertex/edge 위반 | 거부 사유가 Agent ID 쌍을 포함 |

- fixture 원천: 위 4개 시점의 step CSV 행과 시나리오 파일. 테스트는 저장 로그를 **복사하지 않고** 경로와 sha256으로 참조하는 방식을 제안한다(outputs 수정 금지 유지).
- T-WFG-*는 CBS 없이 순수 함수로 검증할 수 있다. T-NONE-1과 그룹 CBS 시험은 CBS 실행 승인이 필요하다.
- 진행 상황 [확인]: T-WFG-1~5는 `tests/test_wfg.py`의 T1~T5로 구현되었다(추가: T4b, T6, T6b, T8–T8d, T9, T10a/b). 결과는 15 PASS / 1 SKIP이며, SKIP은 T7(PENDING)이다. 저장 로그는 git에서 제외되어 있으므로, T10b는 로그가 없으면 skip하고 있으면 경로와 sha256으로 대조한다. 기대 간선·그룹은 후보 규칙의 출력이며 합의된 정답이 아니다. T-NONE-1, T-FULL-1은 아직 구현하지 않았다.

---

## 7. 윤지와 최종 합의할 질문 5개

1. **WFG 출력 형식**: `edges(kind 포함) + groups(서로소, 정렬 규칙)` 형식을 받아들일 수 있는가? 간선 종류(occupy/contest/swap)와 "차단 Agent가 있는 연결 성분만 그룹" 규칙에 동의하는가? (P5)
2. **독립 그룹 처리**: bottleneck t=2처럼 {0,1}, {2,3}이 동시에 나오면 L0를 합집합 1회 호출로 할 것인가, 그룹별 호출로 할 것인가? 그룹별 호출이라면 그룹 사이 충돌은 누가 검증하는가? (P4, P5)
3. **그룹 밖 Agent 처리와 atb033 한계**: atb033에 시간축 제약을 넣을 수 없다면(P1) 그룹 밖 Agent를 정적 장애물로 고정하는 방식을 쓸 것인가? goal 점유 문제(bottleneck t=2의 {0,1})는 창 끝 목표와 L1 확장 중 어느 쪽으로 풀 것인가? (P1, P2)
4. **전체 로봇 검증 위치**: 그룹 경로 + 그룹 밖 예정 위치의 합성 vertex/edge 검증을 Path Guard에 새 함수로 둘 것인가, Runner/boundary에 둘 것인가? 기존 `validate_path_map_and_conflicts`를 재사용해도 되는가? (P3)
5. **예산**: Trigger당 L0→L1→G 호출을 `max_cbs_calls`에 어떻게 셀 것인가? 레벨별 timeout은 얼마로 하고, timeout이 나면 다음 레벨로 갈 것인가 SAFE_STOP할 것인가? (P4)

---

## 8. 담당 작업 (제안)

| 담당 | 작업 |
|---|---|
| **인아** | (1) 이 계약서를 유지하고 합의 결과를 반영 (2) 5장 4개 시점의 WFG 입력 추출 설계를 T-WFG fixture 명세로 확정 (3) Runner 쪽 `group_selector` 연결점, 로그 필드(3.7), None 회귀(T-NONE-1) 설계 (4) 전체 로봇 합성 검증의 Runner 측 호출 지점 설계 |
| **윤지** | (1) WFG 결과 형식과 그룹 확장 규칙 확정 (P5) (2) atb033 시간축 제약 지원 여부 확인 (P1) (3) 그룹 CBS 호출 시 그룹 밖 Agent 처리 방식과 goal 점유 대응 (P2) (4) CBS 측 호출과 timeout 정책 제안 (P4) |
| 공동 | P3 책임 분담 결정, T-WFG-3(독립 그룹) 기대값 합의 |

---

## 9. 이번 단계에서 하지 않은 것

- Solver, IL, Runner, Simulator, Monitor, Path Guard, 시나리오, registry, checkpoint, outputs는 수정하지 않았다.
- CBS를 재실행하지 않았다. 5장의 그룹 CBS 관련 서술은 모두 [추정]이다.
- Selective Runner와 `group_selector`를 만들지 않았다. WFG는 후보 순수 함수와 단위 테스트까지만 추가했고, 런타임에 연결하지 않았다.
- P1~P5는 계속 PENDING이다.
