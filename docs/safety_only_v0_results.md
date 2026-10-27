
# Safety-only v0 — Development Validation

## 1. 목적
CBS를 사용하지 않고 Deadlock Monitor의 최초 Full-repeat
Trigger 이후 간단한 WAIT 규칙으로 정체가 해소되는지 확인한다.

## 2. 구현 규칙
- 브랜치: feature/safety-only-v0
- 최초 TRIGGER_FULL_REPEAT 발생 시 활성화
- Trigger 시점의 overlap integrity가 CLEAN일 때만 활성화
- 목표에 도달하지 않은 가장 낮은 ID의 로봇 선택
- 선택한 로봇을 K=2 스텝 WAIT
- 나머지 로봇은 기존 IL 행동 사용
- 개입 종료 후 IL 정책으로 복귀
- CBS 호출 없음
- K=2는 Development용 시험값이며 최종 Validation 값이 아님

## 3. 대표 시나리오 결과

| Scenario | IL-only | Safety-only v0 | Trigger |
| --- | --- | --- | --- |
| Corridor | Timeout, t=20 | Timeout, t=20 | t=5 |
| Intersection | Timeout, t=20 | Success, t=7 | t=2 |
| Bottleneck | Timeout, t=20 | Integrity abort, t=9 | t=2 |

모든 실행은 max_steps=20으로 수행했다.

## 4. 주요 관찰
- Corridor: t=6,7에 WAIT 개입했지만 목표 도착 실패
- Intersection: t=3,4에 WAIT 개입 후 t=7에 성공
- Bottleneck: t=3,4에 WAIT 개입했지만 t=9에 위치 중복 발생
- Bottleneck의 첫 중복 위치: (3,7), agent 1과 agent 3
- 기존 Simulator 충돌 취소 과정에서 연쇄 위치 중복 발생
- 기존 IL-only Bottleneck 실행에서는 위치 중복 없음

## 5. 무결성 처리
- Safety-only 실행 중 위치 중복을 감지하면 중단
- Bottleneck: integrity_abort=True
- integrity_abort_timestep=9
- all_at_goal=False
- timed_out=False
- CSV 마지막 timestep의 failure_reason=integrity_overlap
- 이 기능은 충돌 예방이 아니라 발생 이후 중단이다.
- Simulator KI-1 근본 수정은 별도 작업으로 남긴다.

## 6. 테스트 결과
- Safety-only 단위 테스트 8개 PASS
- Monitor 연동 통합 테스트 2개 PASS
- KI-1 재현 테스트 1개 PASS
- 무결성 종료 및 로깅 테스트 1개 PASS
- 전체 unittest 165/165 PASS
- Alpha1 Strict Baseline 9/9 PASS

## 7. 연구상 주의 사항
본 결과는 Development 시나리오를 사용한 기능 검증이다.
Held-out Test 성능이나 일반적인 성공률로 해석하지 않는다.

Safety-only의 실패/무결성 중단과 CBS 복구 성공을 비교할 때는
실험 조건 및 평가 기준을 명확히 구분해야 한다.

## 8. 후속 작업
- Safety-only v1 규칙 및 파라미터 Validation
- Operational Label / K_label 확정
- Simulator KI-1 근본 수정 및 회귀 검증
- 독립 Validation/Test instance 구분
- Global/Selective Hybrid 통합
