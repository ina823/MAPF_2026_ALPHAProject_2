\# Recovery Interface v1



\## 1. 목적



Deadlock Monitor가 failure/deadlock candidate를 감지한 시점의 현재 상태를 CBS에 전달하고, 현재 위치 기준으로 새로운 충돌 없는 경로를 생성하기 위한 최소 인터페이스를 정의한다.



이번 단계에서는 Global/Selective Hybrid 전체 state machine을 구현하지 않고, Monitor와 CBS 사이의 입력·출력 규약을 고정하는 것을 목표로 한다.



\---



\## 2. Current-State Replanning Input



CBS recovery의 최소 입력은 다음과 같다.



```text

current\_positions

goals

grid\_map

```



\### current\_positions



각 agent의 recovery 요청 시점 현재 위치이다.



```text

\[(row, col), ...]

```



agent 순서는 goals와 동일해야 한다.



초기 episode start 위치가 아니라 Deadlock Monitor가 trigger된 시점의 실제 현재 위치를 사용한다.



\### goals



각 agent의 최종 목표 위치이다.



```text

\[(row, col), ...]

```



\### grid\_map



2차원 grid map을 사용한다.



```text

0 = free

1 = obstacle

```



\---



\## 3. Coordinate Convention



Alpha2 내부 좌표는 모두 다음 형식을 사용한다.



```text

(row, col)

```



atb033 CBS solver는 `\[x, y]` 좌표를 사용하므로 변환은 `CBSAdapter` 내부 solver boundary에서만 수행한다.



```text

Alpha2 (row, col)

&#x20;       ↓

CBSAdapter

&#x20;       ↓

atb033 \[x, y]

&#x20;       ↓

CBS

&#x20;       ↓

CBSAdapter

&#x20;       ↓

Alpha2 (row, col)

```



Monitor, Simulator, IL runner 및 Hybrid runner에서는 `(row,col)` 좌표를 유지한다.



\---



\## 4. Current replan() Contract



현재 구현:



```python

replan(

&#x20;   current\_positions,

&#x20;   goals,

&#x20;   grid\_map,

&#x20;   solver\_root=None,

&#x20;   timeout\_sec=30,

)

```



현재 반환형:



```text

agent\_id -> \[(row, col), ...]

```



예:



```text

{

&#x20;   0: \[(4,3), (4,4), (3,4), ...],

&#x20;   1: \[(9,2), (8,2), (7,2), ...]

}

```



각 path의 index 0은 반드시 recovery 요청 시점의 해당 agent 현재 위치와 같아야 한다.



```text

paths\[agent\_id]\[0] == current\_positions\[agent\_id]

```



마지막 위치는 해당 agent의 goal과 같아야 한다.



\---



\## 5. CBS Path Validation



CBS가 반환한 path는 Alpha2의 `Simulator.validate\_and\_parse\_paths()`를 이용해 검증한다.



검증 항목:



```text

map boundary

obstacle collision

vertex conflict

edge conflict

```



정상적인 CBS path는 모두 collision-free이어야 한다.



검증된 path는 동일 함수에서 Alpha2 action array로 변환할 수 있다.



Action convention:



```text

0 = Up

1 = Down

2 = Left

3 = Right

4 = WAIT

```



\---



\## 6. Recovery Metadata v1



현재 `replan()`은 기존 compatibility를 유지하기 위해 path dictionary만 반환한다.



향후 Hybrid runner에서는 다음 metadata가 필요하다.



```text

status

cbs\_latency\_ms

paths

cbs\_vertex\_conflicts

cbs\_edge\_conflicts

cbs\_timed\_out

error

```



현재 `replan()` 반환형을 바로 변경하지 않고, 필요 시 별도의 wrapper 또는 result object를 추가하는 방식을 우선 고려한다.



\---



\## 7. CBS Status



CBS 실행 상태는 다음 네 종류로 구분한다.



```text

success

timeout

no\_solution

error

```



\### success



CBS가 정상적으로 collision-free path를 반환한 경우.



\### timeout



설정된 CBS solver timeout을 초과한 경우.



현재 `CBSTimeoutError`로 구분한다.



\### no\_solution



CBS solver가 종료되었지만 solution을 찾지 못한 경우.



현재 `CBSNoSolutionError`로 구분한다.



\### error



solver 실행 실패, 파일 오류 등 timeout/no\_solution 이외의 실행 오류.



`timeout`과 `no\_solution`은 동일한 상태로 취급하지 않는다.



특히 `timeout`은 해당 instance가 unsolvable이라는 의미가 아니다.



\---



\## 8. CBS Latency



기존 standalone CBS 실험에서 latency는 다음 구간의 wall-clock time으로 정의하였다.



```text

CBSAdapter.plan() 호출 직전

&#x20;       ↓

CBS planning

&#x20;       ↓

success / timeout / no\_solution / error 결정

```



기존 standalone log:



```text

latency\_sec

```



Alpha2 통합 표준:



```text

cbs\_latency\_ms

```



변환 규칙:



```text

cbs\_latency\_ms = latency\_sec × 1000

```



CBS latency에는 전체 IL episode 실행 시간을 포함하지 않는다.



\---



\## 9. Timeout 구분



CBS solver timeout과 전체 episode timeout은 서로 다른 값으로 관리한다.



```text

cbs\_timed\_out

episode\_timed\_out

```



\### cbs\_timed\_out



CBS solver가 제한시간 내 solution을 반환하지 못한 경우.



\### episode\_timed\_out



IL/Simulator episode가 `max\_steps`에 도달한 경우.



따라서 두 값을 하나의 `timed\_out` 필드로 합치지 않는다.



\---



\## 10. Conflict 구분



CBS 내부 search conflict와 Simulator runtime collision은 서로 다른 지표이다.



CBS High-Level Search:



```text

cbs\_vertex\_conflicts

cbs\_edge\_conflicts

```



Simulator runtime:



```text

runtime vertex collision/event

runtime edge collision/event

```



CBS conflict count는 `cbs\_conflict\_worker.py`가 CBS의 `get\_first\_conflict()`에서 발견한 conflict event를 JSONL로 기록하고 이를 집계한다.



따라서 CBS conflict count를 실제 실행 중 로봇이 충돌한 횟수로 해석해서는 안 된다.



\---



\## 11. Conflict Logging Evidence



Alpha2에서 기존 Conflict Logging pipeline을 재실행해 정상 동작을 확인하였다.



검증 조건:



```text

scenario = empty

map\_size = 8

num\_agents = 2

seed = 0

```



실행 결과:



```text

status = success

vertex\_conflict\_count = 1

edge\_conflict\_count = 0

```



생성 파일:



```text

outputs/conflict\_experiments/.../conflicts.jsonl

outputs/logs/conflict\_summary.csv

```



문서에 기록된 과거 대표 conflict 숫자보다 실제 실행으로 생성된 CSV/JSONL을 실험 결과의 근거로 사용한다.



\---



\## 12. Failure / Fallback 원칙



현재 단계에서는 fallback state machine을 구현하지 않는다.



향후 CBS가 `timeout`, `no\_solution`, `error`를 반환하는 경우 Safety-only WAIT/yield 또는 기존 policy 실행 유지 등의 fallback 후보를 검토한다.



구체적인 fallback rule은 Deadlock Monitor 및 Safety-only baseline이 고정된 이후 통합 단계에서 결정한다.



\---



\## 13. 현재 검증 상태



Alpha2 smoke scenario를 이용해 다음 항목을 확인하였다.



```text

replan() 실행 성공

path\[0] == current\_positions

path\[-1] == goals

collision\_free == True

CBS path -> Alpha2 action\_array 변환 성공

```



따라서 Recovery Interface 자체는 Alpha2에서 정상 연결된 상태이다.



단, 현재 검증은 일반 smoke scenario를 사용한 interface 검증이며, 실제 corridor / intersection / bottleneck failure state에 대한 recovery 검증은 별도로 수행해야 한다.

