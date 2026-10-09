import argparse
import json
import sys
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_yaml")
    parser.add_argument("output_yaml")
    parser.add_argument("event_log")
    parser.add_argument("solver_root")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_yaml).resolve()
    output_path = Path(args.output_yaml).resolve()
    event_log_path = Path(args.event_log).resolve()
    solver_root = Path(args.solver_root).resolve()

    # centralized 폴더를 import 경로에 추가
    # .../centralized/cbs -> parent = .../centralized
    sys.path.insert(0, str(solver_root.parent))

    from cbs.cbs import Environment, CBS, Conflict

    # CBS input 읽기
    with input_path.open("r", encoding="utf-8") as f:
        param = yaml.load(f, Loader=yaml.FullLoader)

    dimension = param["map"]["dimensions"]

    # atb033 내부에서는 obstacle을 (x, y) tuple과 비교하므로 통일
    obstacles = [
        tuple(obstacle)
        for obstacle in param["map"]["obstacles"]
    ]

    agents = param["agents"]

    event_log_path.parent.mkdir(parents=True, exist_ok=True)

    # 기존 실험 log 제거
    if event_log_path.exists():
        event_log_path.unlink()

    class LoggedEnvironment(Environment):
        def get_first_conflict(self, solution):
            conflict = super().get_first_conflict(solution)

            if conflict:
                if conflict.type == Conflict.VERTEX:
                    conflict_type = "vertex"
                elif conflict.type == Conflict.EDGE:
                    conflict_type = "edge"
                else:
                    conflict_type = "unknown"

                event = {
                    "type": conflict_type,
                    "time": conflict.time,
                    "agent_1": conflict.agent_1,
                    "agent_2": conflict.agent_2,
                    "location_1": [
                        conflict.location_1.x,
                        conflict.location_1.y,
                    ],
                }

                if conflict.type == Conflict.EDGE:
                    event["location_2"] = [
                        conflict.location_2.x,
                        conflict.location_2.y,
                    ]

                # conflict 발생 즉시 파일에 기록
                # timeout으로 process가 죽더라도 이미 발생한 event는 남음
                with event_log_path.open(
                    "a",
                    encoding="utf-8"
                ) as f:
                    f.write(json.dumps(event) + "\n")

            return conflict

    env = LoggedEnvironment(
        dimension,
        agents,
        obstacles,
    )

    cbs = CBS(env)
    solution = cbs.search()

    if not solution:
        print("Solution not found")
        return

    output = {
        "schedule": solution,
        "cost": env.compute_solution_cost(solution),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(output, f)

    print("Solution found")


if __name__ == "__main__":
    main()