# Benchmark Validation / Test Split v1

## 1. 목적

MovingAI benchmark를 이용한 실험에서 validation과 final test 데이터가 섞이는 것을 방지하기 위해 instance split 규칙을 정의한다.

Validation 데이터는 parameter, threshold, interface 및 실행 조건을 확인하는 데 사용하고, Test 데이터는 모든 설정을 고정한 이후 최종 평가에만 사용한다.

---

## 2. Instance 정의

본 프로젝트에서 하나의 benchmark instance는 다음 조합으로 정의한다.

```text
map
+ num_agents
+ starts
+ goals
```

따라서 동일한 map을 사용하더라도 start-goal 배치가 다르면 서로 다른 instance로 취급한다.

반대로 동일한 map / starts / goals / agent 수를 반복 실행한 경우에는 새로운 실험 sample로 세지 않는다.

동일 exact instance 반복 실행은 latency variance 또는 재현성 확인 용도로만 사용한다.

---

## 3. Validation / Test 원칙

### Validation

Validation instance는 다음 목적으로 사용할 수 있다.

- loader 및 interface 검증
- IL-only 동작 확인
- parameter 조정
- threshold 비교
- `max_steps` 확인
- 향후 `K_label`, `H_exec` 등의 validation
- logging 및 recovery interface 확인

Validation instance는 개발 과정에서 여러 번 실행할 수 있다.

### Test

Test instance는 최종 설정이 고정된 이후에만 실행한다.

다음 항목이 확정되기 전에는 test instance를 실행하지 않는다.

```text
Deadlock Definition
Monitor threshold
K_label
Safety-only rule
H_exec
Hybrid execution rule
logging schema
```

Test 결과를 확인한 뒤 parameter나 threshold를 다시 변경하지 않는 것을 원칙으로 한다.

---

## 4. 현재 Split v1

현재 사용 중인 MovingAI `empty-8-8` random scenario를 다음과 같이 구분한다.

### Validation

```text
empty-8-8-random-1.scen
empty-8-8-random-2.scen
empty-8-8-random-3.scen
empty-8-8-random-4.scen
empty-8-8-random-5.scen
```

`empty-8-8-random-1.scen`은 이미 Alpha2 MovingAI loader 및 IL-only pipeline 검증에 사용했기 때문에 validation으로 고정한다.

### Reserve / Unassigned

```text
empty-8-8-random-6.scen
~
empty-8-8-random-20.scen
```

현재 validation/test 어디에도 포함하지 않는다.

최종 실험 조건과 필요한 sample 수가 확정된 이후 추가 validation 또는 test instance로 배정한다.

### Reserved Test

```text
empty-8-8-random-21.scen
empty-8-8-random-22.scen
empty-8-8-random-23.scen
empty-8-8-random-24.scen
empty-8-8-random-25.scen
```

위 test candidate는 split freeze 전까지 실행하지 않는다.

---

## 5. Data Leakage 방지 규칙

Validation과 Test 사이에는 동일한 exact start-goal instance를 사용하지 않는다.

파일명이 다르더라도 다음 값이 동일한 경우 동일 instance로 간주한다.

```text
map
num_agents
ordered starts
ordered goals
```

따라서 최종 split 확정 시 starts/goals 중복 여부를 확인한다.

---

## 6. Deterministic IL 반복 실행 규칙

현재 IL policy는 deterministic 실행을 기본으로 한다.

따라서 동일 exact instance를 여러 번 실행해도 각각을 독립적인 success sample로 세지 않는다.

예:

```text
empty-8-8-random-1
10회 실행
10회 success
```

라고 하더라도 실험 sample 수는:

```text
1 instance
```

로 본다.

반복 실행은 다음 확인에만 활용한다.

```text
latency variance
실행 재현성
logging consistency
```

---

## 7. Test Contamination 처리

Test로 예약한 instance를 최종 freeze 전에 실수로 실행하여 결과를 확인한 경우 해당 instance는 더 이상 순수한 final test instance로 사용하지 않는 것을 원칙으로 한다.

이 경우 reserve pool에서 다른 instance로 교체한다.

---

## 8. 현재 상태

현재 Alpha2에서 실제 실행한 benchmark instance:

```text
empty-8-8-random-1.scen
agents = 4
```

용도:

```text
Validation
```

실행 결과:

```text
status = success
steps = 6
episode_timed_out = False
final_positions == goals = True
```

현재 `random-21 ~ random-25`는 reserved test candidate로 두며 실행하지 않는다.

최종 실험의 전체 validation/test instance 수는 실험 조건과 계산량을 확인한 뒤 본 실험 전에 고정한다.
