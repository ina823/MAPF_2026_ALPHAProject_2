import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from map_generator import generate_map
from src.cbs_adapter import create_atb033_input


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--map-type",
        default="maze"
    )

    parser.add_argument(
        "--size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--num-agents",
        type=int,
        default=4
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0
    )

    return parser.parse_args()


def count_conflicts(event_log_path):
    vertex_count = 0
    edge_count = 0

    if not event_log_path.exists():
        return vertex_count, edge_count

    with event_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            event = json.loads(line)

            if event["type"] == "vertex":
                vertex_count += 1

            elif event["type"] == "edge":
                edge_count += 1

    return vertex_count, edge_count


def save_summary(
    scenario,
    size,
    num_agents,
    seed,
    status,
    vertex_count,
    edge_count,
):
    summary_path = PROJECT_ROOT / "outputs/logs/conflict_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = summary_path.exists()

    with summary_path.open(
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "scenario",
                "map_size",
                "num_agents",
                "seed",
                "status",
                "vertex_conflict_count",
                "edge_conflict_count",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "scenario": scenario,
            "map_size": size,
            "num_agents": num_agents,
            "seed": seed,
            "status": status,
            "vertex_conflict_count": vertex_count,
            "edge_conflict_count": edge_count,
        })


def main():
    args = parse_args()

    # --------------------------------------------------
    # 1. Scenario 생성
    # --------------------------------------------------

    grid_map, starts, goals = generate_map(
        map_type=args.map_type,
        size=args.size,
        num_agents=args.num_agents,
        seed=args.seed,
    )

    experiment_name = (
        f"{args.map_type}_{args.size}_"
        f"{args.num_agents}agents_seed{args.seed}"
    )

    work_dir = (
        PROJECT_ROOT
        / "outputs/conflict_experiments"
        / experiment_name
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    input_yaml = work_dir / "input.yaml"
    output_yaml = work_dir / "output.yaml"
    event_log = work_dir / "conflicts.jsonl"

    create_atb033_input(
        grid_map,
        starts,
        goals,
        input_yaml,
    )

    # --------------------------------------------------
    # 2. Instrumented CBS 실행
    # --------------------------------------------------

    solver_root = (
        PROJECT_ROOT
        / "third_party"
        / "multi_agent_path_planning"
        / "centralized"
        / "cbs"
    )

    worker = (
        PROJECT_ROOT
        / "scripts"
        / "cbs_conflict_worker.py"
    )

    command = [
        sys.executable,
        str(worker),
        str(input_yaml),
        str(output_yaml),
        str(event_log),
        str(solver_root),
    ]

    print(
        f"\n=== Conflict Experiment ===\n"
        f"scenario     : {args.map_type}\n"
        f"size         : {args.size}\n"
        f"agents       : {args.num_agents}\n"
        f"seed         : {args.seed}\n"
        f"timeout      : {args.timeout} sec\n"
    )

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )

        if output_yaml.exists():
            status = "success"

        elif "solution not found" in result.stdout.lower():
            status = "no_solution"

        else:
            status = "error"

    except subprocess.TimeoutExpired:
        status = "timeout"

    # --------------------------------------------------
    # 3. Conflict Count
    # --------------------------------------------------

    vertex_count, edge_count = count_conflicts(
        event_log
    )

    print("=== Result ===")
    print(f"status                  : {status}")
    print(f"vertex_conflict_count   : {vertex_count}")
    print(f"edge_conflict_count     : {edge_count}")

    save_summary(
        scenario=args.map_type,
        size=args.size,
        num_agents=args.num_agents,
        seed=args.seed,
        status=status,
        vertex_count=vertex_count,
        edge_count=edge_count,
    )

    print(
        "\n저장:"
        "\n  event log    =", event_log,
        "\n  summary      = outputs/logs/conflict_summary.csv"
    )


if __name__ == "__main__":
    main()