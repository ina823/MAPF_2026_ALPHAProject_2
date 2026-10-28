# CBS 9-Variant Validation (2026-10-28)

## Experiment
- Input: Original scenario starts (NOT trigger states)
- Solver: atb033 CBS
- Timeout: 30 seconds per scenario
- Cases: 9 controlled variants

## Results
- 8/9 CBS success with Simulator replay success
- 1/9 timeout: corridor_bay_n2_4agent
- 30s timeout: 30045.912 ms
- Additional 60s test: timeout at 60028.810 ms
- Timeout does NOT prove no solution exists

## Provenance
- Base HEAD at execution: 6276f5e49c6d4216eb931fb6417c004e2cf5d1a0
- Scenario SHA: 3c390627ab949fc8308986e31abb71dc381cae01
- Solver SHA: 89f69e11944b172e11c3bcbbecc57344eaa4be9c
- Audit script was uncommitted when experiments ran

## Limitations
- Initial-start CBS only, not trigger-state recovery
- No automatic Hybrid Recovery validation
- No final held-out benchmark performance claim
- Timeout cause requires further investigation

## Evidence
- cbs_variants_initial/summary.csv
- cbs_variants_initial/summary.json
- cbs_variants_initial/<variant>/result.json
- cbs_variants_initial/<variant>/solver_input.yaml
- cbs_variants_timeout60/corridor_bay_n2_4agent/result.json
