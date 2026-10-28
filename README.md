# MAPF Alpha2 — Deadlock-Aware Hybrid MAPF

> **현재 개발 기준 (2026-10-28):** 이 브랜치(`feature/safety-only-v0`)에는 IL-navhint baseline, Deadlock Monitor/Logger, Safety-only v0가 구현돼 있습니다. CBS 재계획은 별도 `feature/cbs-replanning` 브랜치에서 대표 Trigger 상태 3개의 독립 Replay까지 검증됐습니다. **자동 Global/Selective Hybrid와 Wait-for Graph는 아직 통합되지 않았습니다.**

## 1. 목적과 브랜치

평소에는 IL 정책으로 한 스텝씩 이동하고, 공동 상태 반복을 감지하면 CBS로 현재 위치부터 재계획해 복구하는 Hybrid MAPF를 연구합니다. 먼저 Global을, 이후 필요한 로봇만 재계획하는 Selective를 검증합니다. 로봇 2대 시연은 장비·안전 조건이 충족될 때 진행합니다.

| 브랜치 | 맡은 역할 |
| --- | --- |
| `main` | Alpha1 IL-navhint 기준선 |
| `feature/common-logging` | Deadlock Monitor, CSV Logger, Controlled 개발 시나리오 |
| `feature/safety-only-v0` | CBS 없는 K-step WAIT 비교군, KI-1 사후 무결성 중단 |
| `feature/cbs-replanning` | 외부 CBS Adapter, Trigger-State 경로 생성·Replay, MovingAI 로더 |

**주의:** 네 브랜치는 같은 기능을 복사한 폴더 네 개가 아닙니다. 각 브랜치에서 일부 코드가 다르며, 자동 통합은 아직 완료되지 않았습니다. 팀원 작업 중인 CBS 브랜치는 이 문서 정리 과정에서 변경하지 않습니다.

## 2. 실행 흐름

```text
Scenario JSON + Grid NPY
  → scenario_loader → NavHintPolicy(BFS flowdist + NavHintCNN)
  → (Safety-only 모드: K-step WAIT override)
  → MAPFStepSimulator.step() → DeadlockMonitor.observe_step()
  → EpisodeLogger(CSV, 옵션)

[별도 CBS Gate] 첫 Trigger 상태 S_t → replanner → CBSAdapter
  → 외부 CBS → 공동 Path → 검증 → 독립 Simulator Replay
```

- 외부 JSON/CBS 좌표 `[x,y]`, 내부 `(row,col)=(y,x)`.
- Action: `0 UP`, `1 DOWN`, `2 LEFT`, `3 RIGHT`, `4 WAIT`.
- CNN 입력: 로컬 Grid `(3,5,5)`와 2D BFS `flowdist`; checkpoint는 `checkpoints/cnn_navhint.pt`.
- Monitor의 `TRIGGER_FULL_REPEAT`는 반복 상태 **신호**이지 최종 Operational Label이 아닙니다.
- CBS Path의 0번째 점은 현재 위치입니다. 첫 이동은 `Path[0]→Path[1]`입니다.

## 3. 중요한 파일만 보기

| 위치 | 용도 |
| --- | --- |
| `scripts/run_il.py` | IL·Monitor·Logger·Safety-only CLI/실행 루프 |
| `src/il/` | BFS NavHint, CNN, Policy |
| `src/simulator/mapf_step_simulator.py` | Agent 이동, 충돌 및 CBS Path 검사 |
| `src/common/deadlock_monitor.py` | Full-repeat Trigger와 정체 후보 관찰 |
| `src/common/episode_logger.py` | Step/Summary/Monitor CSV 출력 |
| `src/common/safety_only.py` | K-step WAIT 비교군 |
| `scenarios/controlled/` | Corridor, Intersection, Bottleneck, Cycle 개발 시나리오 |
| `scenarios/controlled_variants/` | 추가 9개 개발 사례와 결과 |
| `tests/` | 단위·회귀·KI-1 특성 테스트 |
| `checkpoints/` | 모델 checkpoint 및 출처·SHA 기록 |
| `outputs/` | 로컬 로그·그래프·영상 저장 (`.gitkeep`만 Git 추적) |

CBS의 `src/replanner.py`, `src/cbs_adapter.py`, 외부 solver submodule과 검증 스크립트는 `feature/cbs-replanning`에 있습니다. 이 Safety-only 브랜치에 없다고 누락된 파일이 아닙니다.

## 4. 실행 (저장소 루트)

Python 환경에 의존성이 설치되어 있어야 합니다. Windows에서 인아 PC의 기존 가상환경을 사용하는 경우 아래의 `python` 대신 `& "..\.venv-alpha1\Scripts\python.exe"`를 붙여 실행할 수 있습니다.

```powershell
python -m scripts.smoke_test_il --strict-baseline
python -m unittest discover -s tests -t .
python -m scripts.run_il --scenario scenarios/controlled/scenario_intersection_n2.json --max-steps 20 --verbose
python -m scripts.run_il --scenario scenarios/controlled/scenario_intersection_n2.json --safety-only --safety-hold-steps 2 --max-steps 20 --log
```

- `--monitor`: 런타임 관찰. `--log`도 함께 지정하면 Monitor CSV를 남깁니다.
- `--safety-only`: 최초 유효 Trigger 이후 최소 활성 ID Agent가 2 step WAIT(기본값). CBS 호출 없음.
- 실행 결과와 분석 로그는 `outputs/` 아래에 남으며 기본적으로 Git에 업로드하지 않습니다.

**기존 검증 기록(2026-10-27):** 이 Safety-only 브랜치에서 165/165 unittest, strict smoke 9/9 PASS. 아래 문서 축약 이후의 테스트는 별도로 재실행해야 합니다.

## 5. 반드시 남겨둘 제약과 근거

- **KI-1 미해결:** 일부 연쇄 Vertex 취소 후 실제 Agent 위치가 겹칠 수 있습니다. Safety-only의 `INTEGRITY_ABORT`는 **사후 중단**일 뿐 예방이 아닙니다. 실물 로봇 실행에 그대로 사용하지 않습니다.
- **개발 결과:** Safety-only K=2는 Intersection에서 성공, Corridor에서 실패, Bottleneck에서 t=9 KI-1 중단. 일반 성능을 뜻하지 않습니다.
- **연구 검증:** Validation으로 설정을 정하고 동결한 뒤 독립 held-out Test에 적용합니다. 같은 exact instance의 seed 반복은 독립 표본이 아닙니다.
- **계획 실패:** CBS timeout은 해가 없다는 증거가 아니며, 좌표·Agent ID·Path 인접성·Vertex/Edge·Replay를 확인해야 합니다.

원본/연구 근거는 다음 문서로 유지합니다. 상세 설명을 README에 반복하지 않습니다.

- [Alpha1 baseline/known issues](docs/alpha1_baseline.md) · [이관 출처](docs/migration_map.md) · [체크포인트 SHA](checkpoints/MANIFEST.json)
- [Deadlock 정의](docs/deadlock_definition_v1.md) · [첫 Trigger 타임라인](docs/monitor_trigger_timeline_v1.md) · [Validation/Test 분리](docs/validation_test_split_v1.md)
- [Safety-only v0 결과](docs/safety_only_v0_results.md) · [9 variants 재현 결과](scenarios/controlled_variants/RESULTS.md)
- [윤지 CBS 별도 브랜치](https://github.com/ina823/MAPF_2026_ALPHAProject_2/tree/feature/cbs-replanning) · [대표 3개 CBS 원본 경로](https://github.com/ina823/MAPF_2026_ALPHAProject_2/blob/feature/cbs-replanning/docs/evidence/cbs_gate_paths.json)

Alpha1 원본: https://github.com/qwedus/MAPF_2026_AlphaProject (read-only 기준).
