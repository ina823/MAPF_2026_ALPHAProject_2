# Alpha 1 IL-navhint baseline (재현 기준 문서)

Alpha 2 는 아래 baseline 을 **동작 변경 없이** 재현하는 것에서 출발한다. 이 문서는 무엇이 baseline 인지,
Alpha 1 의 어느 코드가 Alpha 2 의 어느 파일이 되었는지, 그리고 지금 알려진 문제(known issue)를 기록한다.
파일별 출처 blob 해시는 [migration_map.md](migration_map.md) 에 있다.

## 1. Baseline 정의

| 항목 | 값 |
|---|---|
| checkpoint | `checkpoints/cnn_navhint.pt` (134,962 B, sha256 은 `checkpoints/MANIFEST.json`) |
| model | `NavHintCNN` (`src/il/model_navhint.py`), 파라미터 32,549개 |
| goal_dim / hint_mode | 2 / `flowdist` (checkpoint 안의 `hint_mode`, label `navhint-flowdist`) |
| local observation | `(3, 5, 5)` float32 : ch0 벽/장애물/맵 밖, ch1 다른 로봇 현재 위치, ch2 다른 agent 의 목표 |
| goal feature | `(2,)` = BFS flow 단위 방향 x 현재 위치의 BFS 거리 (flowdist), (row, col) 순서 |
| 정규화 | `(goal_dir - goal_mean) / goal_std`, `goal_mean` `(1,2)` = [-0.0419, 0.0156], `goal_std` `(1,2)` = [6.575, 2.572] |
| output | logits `(1, 5)` -> argmax |
| action | 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=WAIT ; delta 는 (drow, dcol) |
| simulator | Python + NumPy 이산 grid-world `MAPFStepSimulator` (Alpha 1 `origin/feature/simulator`) |
| 학습 기록 | best_epoch 7, best_val_acc 0.8255 (학습 데이터/스크립트는 Alpha 2 로 가져오지 않음) |

flowdist 예: flow 방향이 (-1, 0) 이고 BFS 남은 거리가 9 이면 goal feature 는 (-9, 0) 이다.

## 2. Pipeline 과 파일 위치

```
scenario JSON [x,y] --scenario_loader--> grid, starts/goals (row,col)
  -> MAPFStepSimulator.reset() -> obs {grid (3,5,5), goal_dir (2,) 직선벡터}
  -> (episode 당 1회) dist = bfs_dist(grid, goal)
  -> (step 마다) goal_dir <- make_goal_dir(dist, position, goal, "flowdist")
  -> predict: (goal_dir - gmean)/gstd -> NavHintCNN -> logits -> argmax
  -> MAPFStepSimulator.step(actions) -> (obs, done, info)
```

| 기능 | Alpha 1 원본 | Alpha 2 |
|---|---|---|
| checkpoint loader | `src/model_navhint.py::load_navhint` | `src/il/model_navhint.py` (변경 없음, torch.load 그대로) |
| NavHintCNN | `src/model_navhint.py::NavHintCNN` | `src/il/model_navhint.py` |
| BFS / flow / flowdist | `src/nav_hint.py` `bfs_dist`, `flow_step`, `make_goal_dir` | `src/il/nav_hint.py` (verbatim) |
| goal_dir overwrite | `scripts/eval_flow.py::rollout` (40-47행) | `src/il/policy.py::NavHintPolicy` (그대로 조립) |
| 정규화 + 추론 + argmax | `scripts/eval_cbs_vs_il.py::predict` (60-71행) | `src/il/policy.py::predict` (verbatim 복사) |
| observation / collision / 종료 | `simulator.py::MAPFStepSimulator` | `src/simulator/mapf_step_simulator.py` (verbatim) |
| action / channel 상수 | `spec.py` | `src/common/spec.py` (verbatim) |
| 시나리오 로더 | `src/scenario_loader.py` | `src/common/scenario_loader.py` (1줄 수정) |
| episode loop | `scripts/eval_flow.py::rollout` | `scripts/run_il.py::run_episode` |

## 3. 규약

- 내부 좌표는 전부 **(row, col)**. 시나리오 JSON 과 atb033 은 `[x, y] = [col, row]` 이고 변환은 `scenario_loader.xy_to_rowcol` 에서만 한다.
  `goal_std` 가 행/열에서 다르므로(6.575 / 2.572) 좌표를 뒤집으면 오류 없이 잘못된 행동이 나온다.
- Action 표는 Alpha 1 안에서 4곳에 복제되어 있다: `spec.ACTION_DELTA`, `MAPFStepSimulator._STEP_ACTION_DELTA`,
  `nav_hint._NEIGHBORS` (up, down, left, right 순서), batch `Simulator.actions`. 현재 일치하며 `tests/test_conventions.py` 가 고정한다.
- 관측 채널 우선순위: wall > other robot > other goal, 한 칸에 최대 한 채널. 자기 자신의 위치/목표는 그리지 않는다.
- 시뮬레이터 규칙: 벽/맵 밖 이동은 제자리, 같은 칸으로 몰리면(vertex) 관련자 전원 제자리, 서로 자리를 바꾸면(edge swap) 둘 다 제자리,
  빈 칸(방금 비워진 칸 포함)으로의 이동은 허용. 종료: `all_at_goal` 이거나 `t >= max_steps`.
- 추론에는 난수가 없다. `load_navhint` 가 `model.eval()` 을 호출하므로 Dropout(0.1) 은 꺼져 있다.

## 4. Reference 결과 (관측값, hard assertion 아님)

`scenarios/smoke/scenario_empty_s11_n3.json` (11x11 빈 맵, 3 agent), `max_steps=160`:

| 관측 | 값 | 출처 |
|---|---|---|
| steps / all_at_goal / timed_out | 9 / True / False | Alpha1_legacy 에서 사용자가 torch 로 실행 (2026-09-21) |
| 동일 조건 재현 | 9 / True / False, 궤적 포함 | torch 없는 numpy 재구현 교차검증 (`tests/baselines/alpha1_il_navhint_reference.json`) |
| 매 결정의 top1-top2 logit 최소 차이 | 0.717 | 같은 교차검증 |

9 steps 는 이 시나리오의 makespan 하한(최대 BFS 거리 = 9)과 같다. agent 끼리 상호작용이 없는 빈 맵이라 이 값이 기대되지만
정책 전반의 불변식은 아니다. 그래서 smoke test 는 정확한 step 수를 assert 하지 않고
`all_at_goal`, `timed_out == False`, `final == goals`, `steps >= BFS 하한` 만 확인하며, 9 는 reference metadata 와의 비교(WARN)로만 쓴다.

참고: Alpha 1 의 다른 dev 시나리오를 같은 방식(numpy 교차검증, torch 실측 아님, 이번에 이식하지 않음)으로 돌린 결과.

| 시나리오 | agents | 하한 | 결과 |
|---|---|---|---|
| dense_s11_n3 | 3 | 7 | 7 steps 성공 |
| sparse_s12_n3_0 | 3 | 7 | 7 steps 성공 |
| rooms_s8_n2_0 | 2 | 9 | 9 steps 성공 |
| empty_s11_n3 | 3 | 9 | 9 steps 성공 (smoke) |
| empty_s14_n4_0 | 4 | 9 | timeout (2/4 도착) |
| sparse_s11_n3 | 3 | 8 | timeout (2/3 도착) |
| dense_s10_n2_0 | 2 | 10 | timeout (0/2 도착) |
| rooms_s11_n3 | 3 | 16 | timeout, 중간에 두 agent 가 같은 칸 (KI-1) |
| maze_s11_n3 | 3 | 32 | timeout, 중간에 두 agent 가 같은 칸 (KI-1) |
| maze_s9_n5_0 | 5 | 23 | timeout (2/5 도착) |

## 5. Known issues (Alpha 1 baseline 에서 그대로 가져온 것)

이번 migration 에서는 **어느 것도 고치지 않는다**. 별도 단계(simulator 개선 / 리팩터링)에서 다룬다.

**KI-1 vertex-collision 연쇄 overlap (simulator).** `MAPFStepSimulator.step()` 은 vertex 충돌을 한 번만 처리한다.
충돌로 제자리로 되돌아간 agent 가, 그 agent 의 원래 칸으로 들어오던 다른 agent 와 다시 충돌하는지는 검사하지 않는다.
결과적으로 두 agent 가 같은 칸에 놓일 수 있다.
- 1차원 예: A(col1)->col2, C(col3)->col2 가 충돌해 둘 다 제자리로, B(col0)->col1(A 의 원래 칸) => A 와 B 가 col1 을 공유.
- 2차원 예(Alpha 1 rollout 에서 관측): agent0 (9,2) UP, agent1 (8,3) LEFT, agent2 (8,2) DOWN. agent0 과 agent1 이 (8,2) 에서 충돌해 제자리로 돌아가면서
  agent0-agent2 의 swap 이 감지되지 않고 agent2 가 (9,2) 로 내려와 agent0 과 겹친다. `maze_s11_n3` t=23, `rooms_s11_n3` t=12 에서 발생.
- 영향: 위치 유일성을 가정하는 이후 단계(Wait-for graph, deadlock 판정)에 주의가 필요하다.
- 처리: `tests/test_simulator_characterization.py::TestKnownQuirkKI1` 이 **현재 동작**을 그대로 고정한다.
  simulator 를 의도적으로 고칠 때 이 두 테스트를 같은 변경에서 갱신해야 한다.
- **[Alpha 2 갱신] 32e2158 에서 수정됨.** 취소가 생기면 충돌이 없어질 때까지 다시 검사한다(연쇄 취소).
  위 내용과 기존 `outputs/logs/*` 의 overlap 기록은 수정 전 동작의 증거로 그대로 보존한다.
  수정 후 기대값은 `tests/test_simulator_characterization.py::TestKnownQuirkKI1`, `tests/test_safety_ki1.py`,
  `tests/test_deadlock_monitor.py` 의 KI-1 분리 클래스들에 반영되어 있다.

**KI-2 `timed_out` 의미.** `timed_out = (t >= max_steps)` 이므로 goal 도착이 정확히 `max_steps` 스텝일 때
`all_at_goal=True` 와 `timed_out=True` 가 동시에 나온다. 성공 판정은 `all_at_goal` 로 한다.

**KI-3 `cbs_timeout_sec`.** Alpha 1 의 `train_dagger_navhint.py` / `train_dagger.py` 는 `MAPFStepSimulator(..., cbs_timeout_sec=...)` 를 넘기지만
GitHub 의 어느 branch 의 클래스도 그 인자를 받지 않는다. 학습 스크립트는 이번에 가져오지 않았고 추론과 무관하다.

**KI-4 README 와 코드 불일치.** `origin/main` README 는 `simulator.py` 에 `MAPFStepSimulator` 가 있다고 적지만
`origin/main`, `origin/analysis` 의 `simulator.py` 에는 `MAPFStepSimulator` 가 없다 (batch `Simulator` 만 있음).
`MAPFStepSimulator` 는 `origin/feature/simulator` (커밋 8c4ac1a) 에만 있고, IL branch 는 커밋 64d4c1f 에서 simulator 를 제거했다.
Alpha 2 는 feature/simulator 의 `simulator.py` (blob c323d7d5) 를 사용한다.

**KI-5 `sim._positions` 접근.** 관측에 절대 좌표가 없어서 caller 가 simulator 의 private 속성 `_positions` 를 읽는다
(Alpha 1 `eval_flow.py` 와 동일; `scripts/run_il.py` 에서 사용). read-only property 도입은 baseline 재현 확인 이후로 미뤘다.

**KI-6 `load_navhint` (변경하지 않음).** `torch.load(path, map_location=device)` 를 `weights_only` 지정 없이 호출한다
(torch >= 2.6 에서는 기본값이 True 이고, 이 checkpoint 는 tensor 와 기본 타입만 담고 있어 로드된다).
checkpoint 에 `hint_mode` 키가 없으면 기본값이 `"both"` 가 된다. `NavHintCNN()` 의 기본 `goal_dim` 은 4 이므로
checkpoint 는 반드시 `load_navhint` 로 읽어야 한다. 정리/보안 개선은 baseline 재현 성공 이후 별도 단계.

**KI-7 사용하지 않는 코드 경로.** `get_expert_actions` 안의 lazy `from src.cbs_adapter import ...` 는 Alpha 2 에 해당 모듈이 없어
호출하면 실패한다 (baseline 추론은 호출하지 않음). 파일 상단의 `from dagger import MAPFSimulator` 는 ImportError 시 빈 base class 로
대체되어 정상 동작한다. batch `Simulator.validate_and_parse_paths` 는 이모지를 print 하므로 Windows cp949 콘솔에서 인코딩 오류가 날 수 있다.

**KI-8 입력 검증 없음.** `step(actions)` 는 모든 agent id 에 대한 action 이 필요하고(누락 시 KeyError), action 범위를 검증하지 않는다.

**KI-9 smoke 시나리오의 범위.** 이식한 시나리오는 agent 간 상호작용이 없다. 정책의 coordination / 충돌 회피 경로는
smoke test 가 아니라 simulator characterization test 로만 부분적으로 덮인다.

## 6. Provenance

### 6.1 Alpha 1 source of truth (조사 시점 2026-09-21)

원본: https://github.com/qwedus/MAPF_2026_AlphaProject

| branch | tip | 이번 migration 에서의 역할 |
|---|---|---|
| `analysis` | 89ab86c0 | `cnn_navhint.pt`, `src/nav_hint.py`, `src/model_navhint.py` (IL 과 동일 blob) |
| `IL` | 89e1f251 | `spec.py` (analysis/main 에는 없음), navhint 코드 원본 |
| `feature/simulator` | fbf2a3e9 | `MAPFStepSimulator` 가 있는 유일한 `simulator.py` |
| `main` | 75f8309 | `src/scenario_loader.py`, smoke 시나리오/맵, 최종 README |
| `feature/cbs-adapter`, `feature/map-generator` | - | 이번에는 사용하지 않음 |

analysis branch 만 checkout 하면 깨지던 import 의 실제 출처:

| 깨진 import | 출처 |
|---|---|
| `import spec` (`model_navhint.py`) | `origin/IL` 에만 있음 |
| `from simulator import MAPFStepSimulator` | `origin/feature/simulator` 에만 있음 |
| `model_cnn`, `model_mlp` | `origin/IL` 에만 있음. `scripts/eval_cbs_vs_il.py` 가 module 최상단에서 import 할 뿐 NavHintCNN 추론에는 필요 없다 (Alpha 2 로 가져오지 않음) |

`Alpha1_legacy` 의 복구 상태(branch `alpha1-recovery`, analysis 89ab86c 위에서 시작)는 commit 되지 않은 작업 트리이다:
`simulator.py` 는 analysis 버전을 feature/simulator 버전으로 교체한 modified 상태이고 `spec.py`, `model_cnn.py`, `model_mlp.py` 는 untracked 이다.
이 복구본의 내용은 (CRLF 를 LF 로 정규화하면) 위 branch blob 과 동일함을 확인했다.

### 6.2 "최종 권장 모델" 기록 위치 (`cnn_navhint.pt`)

| 위치 | 내용 |
|---|---|
| `origin/main` `README.md` (blob 30c897ef, 커밋 cb68d37 2026-07-13 "Update README.md") | 119행 `## 모델 체크포인트 (전부 analysis 브랜치)`, 129행 `\| cnn_navhint.pt \| flowdist \| **★ 최종 권장 모델** (미로/방 항법 해결) \|`, 130행 `cnn_dagger_navhint.pt` 는 비권장(참고용), 153행 사용 예 `--flow-ckpt cnn_navhint.pt` |
| `origin/analysis` `docs/navhint_dagger_verification.md` 62행 | `최종 권장 = BC-navhint (cnn_navhint.pt)` |
| `origin/analysis`, `origin/IL` `docs/navhint_prototype.md` 66-67행 (두 branch 동일) | `BC-navhint가 최종 권장 모델. DAgger는 여기선 비가산적.` (문장이 두 줄에 걸쳐 있음) |
| `origin/analysis` `README.md` (blob 379e8a81) | 모델 표 없음. 이것이 2026-09-21 현재 `Alpha1_legacy` 작업 트리에 있는 README 이다 |
| `origin/IL`, `origin/feature/*` `README.md` | `cnn_navhint` 언급 없음 |

`cnn_navhint.pt` 파일 자체는 `origin/analysis` 에만 있고 (커밋 7048e5e 2026-07-10), `origin/main` 에는 없다. main README 도 "전부 analysis 브랜치"라고 적고 있다.
