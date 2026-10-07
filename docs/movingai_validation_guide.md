\# MovingAI Benchmark Validation Guide



\## 1. 목적



MovingAI `.map/.scen` benchmark instance를 Alpha2 내부 표준 `(row,col)` 형식으로 로딩하고, IL-only policy를 `MAPFStepSimulator`에서 실행해 validation input으로 사용할 수 있는지 확인한다.



\---



\## 2. 관련 파일



\- MovingAI loader  

&#x20; `src/movingai\_loader.py`



\- IL-only benchmark runner  

&#x20; `scripts/run\_movingai\_il\_once.py`



\- Step Simulator  

&#x20; `simulator.py`

&#x20; - `Simulator`: CBS path 검증 및 path-to-action 변환

&#x20; - `MAPFStepSimulator`: IL step-by-step rollout



\- IL model  

&#x20; `model\_cnn.py`



\- Checkpoint  

&#x20; `cnn\_diverse.pt`



\- Episode log  

&#x20; `outputs/logs/episode\_log.csv`



\---



\## 3. 좌표 및 Grid 규약



\### MovingAI

\- 좌표: `(x, y)`

\- `x = column`

\- `y = row`



\### Alpha2 내부

\- 좌표: `(row, col)`

\- MovingAI loader 단계에서 `(x,y) → (row,col)` 변환



\### Grid

\- `0 = free`

\- `1 = obstacle`



\---



\## 4. Validation Instance



사용 instance:



```text

map:

C:\\Users\\LG\\MAPF\_workspace\\mapf\_bench\\empty-8-8.map



scenario:

C:\\Users\\LG\\MAPF\_workspace\\mapf\_bench\\scen-random\\empty-8-8-random-1.scen



agents:

4

```



로드 결과:



```text

grid shape = (8, 8)

walls = 0



starts =

\[(4, 1), (0, 1), (6, 1), (6, 4)]



goals =

\[(7, 4), (2, 3), (7, 6), (1, 5)]

```



\---



\## 5. IL-only 실행



프로젝트 루트에서 실행:



```powershell

python scripts\\run\_movingai\_il\_once.py `

\--map-path C:\\Users\\LG\\MAPF\_workspace\\mapf\_bench\\empty-8-8.map `

\--scen-path C:\\Users\\LG\\MAPF\_workspace\\mapf\_bench\\scen-random\\empty-8-8-random-1.scen `

\--agents 4 `

\--max-steps 64

```



검증 실행 결과:



```text

status = success

success = True

steps = 6

timed\_out = False

```



모든 agent의 final position이 goal과 일치함을 확인했다.



\---



\## 6. Logging



실행 결과는 다음 파일에 자동 추가된다.



```text

outputs/logs/episode\_log.csv

```



현재 필드:



```text

timestamp

mode

instance

map\_name

num\_agents

status

episode\_steps

episode\_runtime\_ms

episode\_timed\_out

```



주의:



\- `episode\_runtime\_ms`는 전체 IL episode 실행 시간

\- `episode\_timed\_out`는 `max\_steps` 도달 여부

\- `cbs\_latency\_ms`와 `cbs\_timed\_out`은 CBS planning 전용 값이므로 서로 합치지 않는다.



\---



\## 7. Validation / Test 원칙



현재 `empty-8-8-random-1.scen`은 validation 후보로만 사용한다.



최종 evaluation 전에 validation/test instance를 파일 단위로 분리하고, test instance는 split freeze 전까지 실행하지 않는다.

