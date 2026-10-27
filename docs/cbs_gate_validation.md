# CBS Gate Validation Results

## 1. Controlled Variants

총 9개 controlled variant에 대해 CBS solvability를 독립 검증하였다.

### Category A - Integrity Clean
- corridor_bay_n2_offset: success
- intersection_n2_4way: success
- intersection_n2_lshape: success
- intersection_n2_tight: success
- bottleneck_n4_tightdoor: success

### Category B - KI-1 Integrity Issue
- corridor_bay_n2_3agent: success
- corridor_bay_n2_4agent: timeout
- bottleneck_n4_convoy: success
- bottleneck_n4_imbalanced: success

Summary:
- success: 8
- timeout: 1
- no_solution: 0
- error: 0

`corridor_bay_n2_4agent`는 joint-plan replay로 경로 존재가 확인된 instance이므로,
CBS timeout을 unsolvable 또는 no_solution으로 해석하지 않는다.

## 2. Experimental Instances

- exp_corridor_bay_n2_3rdagent_perturbation: timeout
- exp_bottleneck_n4_tightdoor_1agent_control: success

Experimental instance는 정식 validation/test sample과 분리하여 관리한다.

## 3. Trigger-State CBS Recovery

Deadlock Monitor의 최초 TRIGGER_FULL_REPEAT 시점에서
`sim.step()` 실행 후 실제 상태 `S_t`를 CBS 입력으로 사용하였다.

### Corridor
- trigger_timestep: 5
- current_positions: [(2, 4), (2, 6)]
- goals: [(2, 10), (2, 0)]
- CBS: success

### Intersection
- trigger_timestep: 2
- current_positions: [(3, 2), (2, 3)]
- goals: [(3, 5), (5, 3)]
- CBS: success

### Bottleneck
- trigger_timestep: 2
- current_positions: [(1, 2), (2, 1), (1, 7), (3, 7)]
- goals: [(1, 7), (3, 7), (0, 1), (4, 1)]
- CBS: success

세 scenario 모두 다음 조건을 만족하였다.

- path[0] == current_positions
- path[-1] == goals
- collision_free == True
- path-to-actions conversion success

## 4. Recovery Execution Replay

CBS path를 Alpha2 action array로 변환한 뒤
MAPFStepSimulator에서 실제 recovery execution을 수행하였다.

### Corridor
- execution_success: True
- all_at_goal: True
- path_followed_exactly: True
- overlap_timesteps: 0

### Intersection
- execution_success: True
- all_at_goal: True
- path_followed_exactly: True
- overlap_timesteps: 0

### Bottleneck
- execution_success: True
- all_at_goal: True
- path_followed_exactly: True
- overlap_timesteps: 0

따라서 대표 3개 failure scenario에서 다음 연결을 실제로 검증하였다.

Deadlock Monitor Trigger
→ current state S_t
→ CBS current-state replanning
→ collision-free path
→ path-to-actions
→ Simulator recovery execution
→ all agents reach goals

## 5. CBS Interface

Execution environment:
- Branch: feature/cbs-replanning
- solver_root: None
- timeout_sec: 30

Input:
- current_positions: [(row, col), ...]
- goals: [(row, col), ...]
- grid_map: 0=free, 1=obstacle

Output:
- agent_id -> [(row, col), ...]

Status:
- success
- timeout
- no_solution
- error

Important:
- timeout != no_solution
- timeout != unsolvable
- CBS trigger success, CBS recoverability, recovery execution success,
  and final Operational Deadlock Label are separate validation stages.

## 6. Reproduction Command

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m scripts.validate_cbs_gate `
  --common-root ..\MAPF_common_logging `
  --timeout 30 `
  --json-out docs\evidence\cbs_gate_paths.json |
Tee-Object -FilePath docs\evidence\cbs_gate_recovery_log.txt
```

Generated evidence:

- `docs/evidence/cbs_gate_paths.json`
  - Full CBS paths
  - trigger payload
  - latency
  - timestep-level recovery execution data
  - final execution result

- `docs/evidence/cbs_gate_recovery_log.txt`
  - Human-readable console execution log

Validation script:

- `scripts/validate_cbs_gate.py`
