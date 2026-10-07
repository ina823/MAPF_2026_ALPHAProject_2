from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_il import run_episode
from src.common.movingai_loader import load_movingai_instance
from src.il.policy import NavHintPolicy


DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "cnn_navhint.pt"


def append_episode_log(
    log_path: Path,
    *,
    mode: str,
    instance: str,
    map_name: str,
    num_agents: int,
    status: str,
    episode_steps: int,
    episode_runtime_ms: float,
    episode_timed_out: bool,
):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "mode",
        "instance",
        "map_name",
        "num_agents",
        "status",
        "episode_steps",
        "episode_runtime_ms",
        "episode_timed_out",
    ]

    file_exists = log_path.exists()

    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "mode": mode,
                "instance": instance,
                "map_name": map_name,
                "num_agents": num_agents,
                "status": status,
                "episode_steps": episode_steps,
                "episode_runtime_ms": round(episode_runtime_ms, 3),
                "episode_timed_out": bool(episode_timed_out),
            }
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--map-path", type=Path, required=True)
    parser.add_argument("--scen-path", type=Path, required=True)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--max-steps", type=int, default=64)

    args = parser.parse_args()

    # 1. MovingAI benchmark -> Alpha2 (row, col)
    grid, starts, goals = load_movingai_instance(
        args.map_path,
        args.scen_path,
        args.agents,
    )

    starts_dict = {
        agent_id: starts[agent_id]
        for agent_id in range(args.agents)
    }

    goals_dict = {
        agent_id: goals[agent_id]
        for agent_id in range(args.agents)
    }

    # 2. Alpha2 IL-navhint policy
    policy = NavHintPolicy(args.ckpt)

    # 3. Alpha2 기존 run_episode 사용
    start_time = time.perf_counter()

    result = run_episode(
        policy,
        grid,
        starts_dict,
        goals_dict,
        args.max_steps,
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # 4. Episode 결과 판정
    success = bool(result["all_at_goal"])

    if success:
        status = "success"
    elif result["timed_out"]:
        status = "timeout"
    else:
        status = "failure"

    # 5. 기존 Common episode log 형식 유지
    append_episode_log(
        PROJECT_ROOT / "outputs" / "logs" / "episode_log.csv",
        mode="il_only",
        instance=args.scen_path.stem,
        map_name=args.map_path.stem,
        num_agents=args.agents,
        status=status,
        episode_steps=result["steps"],
        episode_runtime_ms=elapsed_ms,
        episode_timed_out=result["timed_out"],
    )

    # 6. 결과 출력
    print()
    print("=== Alpha2 MovingAI IL-only Result ===")
    print("checkpoint =", args.ckpt.name)
    print("status =", status)
    print("success =", success)
    print("steps =", result["steps"])
    print("elapsed_ms =", round(elapsed_ms, 3))
    print("timed_out =", result["timed_out"])
    print("starts =", starts_dict)
    print("goals =", goals_dict)
    print("final_positions =", result["final_positions"])
    print("final_positions == goals =", result["final_positions"] == goals_dict)


if __name__ == "__main__":
    main()