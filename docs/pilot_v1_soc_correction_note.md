# PILOT v1 SoC 정정 메모

작성 2026-11-09. **이 메모는 해석을 정정할 뿐이며 원본을 고치지 않는다.** `outputs/pilot_checks_v1/`, `registry/pilot_solvability_v0.json`, `docs/pilot_instances_v1.md`, `scripts/check_pilot_instances.py`, 기존 비교 코드는 그대로다.

## 1. 요약

PILOT v1의 `sum_of_path_lengths`(57, 27)는 **실제 SoC가 아니다.** 어댑터가 같은 길이로 패딩한 경로에서 계산한 값이라 `makespan × agent 수`와 같다. **이 값을 SoC로 인용하지 않는다.**

세 값은 서로 다른 지표다.

| 지표 | 정의 | 용도 |
|---|---|---|
| 패딩 경로 길이 합 (기존 `sum_of_path_lengths`) | 모든 Agent를 가장 긴 경로에 맞춰 WAIT로 늘린 뒤 `len(path) - 1`을 합산 | SoC로 사용 금지 |
| 실제 SoC | Agent별로 목표에 끝까지 머무는 첫 시각의 합. WAIT 패딩의 영향을 받지 않는다 | SoC로 인용 |
| Solver `cost` | atb033 `output.yaml`의 값. 관찰된 값은 원본 일정 합 + Agent 수다(t=0까지 세는 방식) | 원본 기록으로만 보존 |

## 2. 비교표

재계산 근거: `outputs/pilot_v2_checks/soc_correction_20261109_011135/pilot_v1_soc_comparison.json`(로컬 전용, 커밋 안 함). 원본 `pilot_checks.json`(sha256 `a0ac9281…`)과 Solver `output.yaml`을 읽기 전용으로 사용했다.

| instance | CBS status | 기록값 `sum_of_path_lengths` | 실제 SoC | 차이 | makespan | Solver `cost` | Agent별 정착 시각 |
|---|---|---|---|---|---|---|---|
| `PILOT_fcorr02_001` | `success` | 57 (19×3) | **42** | +15 | 19 | 45 | 12, 11, 19 |
| `PILOT_fbot02_001` | `timeout` | 없음 | 없음 | — | 없음 | 없음 | 없음 |
| `PILOT_fint02_001` | `success` | 27 (9×3) | **24** | +3 | 9 | 27 | 7, 9, 8 |

- **실제 SoC:** Agent별로 목표에 끝까지 머무는 첫 시각의 합이다. 패딩 경로와 원본 일정에서 계산한 값이 같고, 두 인스턴스에서 도착 후 목표를 떠난 Agent는 없다.
- **makespan:** 기록값이 실제와 같다. 정정 대상이 아니다.
- **Solver `cost`:** 두 인스턴스 모두 원본 일정 합 + Agent 수다(t=0까지 세는 방식). 역시 SoC가 아니다. `PILOT_fint02_001`의 `cost` 27이 기록값 27과 같은 것은 우연이다.

## 3. 영향 범위

| 위치 | 내용 | 조치 |
|---|---|---|
| `scripts/check_pilot_instances.py:116-118` | 패딩 경로의 `len(p) - 1` 합을 `sum_of_path_lengths`로 저장 | 수정 안 함 |
| `scripts/check_pilot_instances.py:178` | 위 값을 `SoC=`로 출력 | 수정 안 함 |
| `registry/pilot_solvability_v0.json:13`, `:58` | 57, 27 저장 | 수정 안 함 |
| `src/data/quality_gate.py:289` | 기록값이 원시 출력과 같은지 비교 | 수정 안 함. 기록을 고치면 이 비교도 함께 다뤄야 한다 |
| `docs/pilot_instances_v1.md:45-49` | 57, 27을 "경로 길이 합"으로 표기 | 수정 안 함 |

**영향 없음:** `src/compare/compare_csv.py:114`의 `sum_of_costs`는 이미 "목표에 끝까지 머무는 첫 시각의 합"이다. 이 코드로 계산한 비교 CSV 값은 이 문제와 무관하다.

## 4. 확인하지 못한 값

`registry/pilot_solvability_v0.json:41`의 `PILOT_fbot02_001` 다른 방법 기록 73은 `makespan × agent 수`(21×4 = 84)와 다르므로 패딩 값은 아닌 것으로 보인다. 다만 경로 파일이 없어(`archived: false`) 실제 SoC인지 확인할 수 없다. 같은 파일 안에 정의가 다른 값이 섞여 있을 수 있으므로 73과 57, 27을 직접 비교하지 않는다.

## 5. 남은 결정

기존 스크립트와 기록을 고칠지, 고친다면 원시 출력과의 대조를 어떻게 유지할지는 별도 승인 단계에서 정한다.
