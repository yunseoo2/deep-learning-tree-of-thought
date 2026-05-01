"""
Run the ToT-BFS experiment on Game of 24 problems.

Usage:
    python run_tot_experiment.py                     # full run on 50 problems
    python run_tot_experiment.py --num_problems 5    # smoke test on 5 problems
"""

import argparse
import json
import os

from dotenv import load_dotenv

from run_experiments import load_problems
from tot_prompting import run_tot_experiment


def main():
    parser = argparse.ArgumentParser(description="Run ToT-BFS on Game of 24 problems.")
    parser.add_argument(
        "--num_problems",
        type=int,
        default=50,
        help="Number of problems to run (default: 50). Use a small value for smoke testing.",
    )
    parser.add_argument(
        "--beam_size", type=int, default=5, help="BFS beam size b (default: 5)."
    )
    parser.add_argument(
        "--steps", type=int, default=3, help="BFS depth (default: 3)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Defaults to ../results/tot_results.json for full runs "
             "and ../results/tot_results_smoke_<n>.json for smoke tests.",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in .env file")

    data_path = "../data/test_puzzles_50.csv"
    problems = load_problems(data_path, num_problems=args.num_problems)

    print(f"Loaded {len(problems)} problems from dataset")
    print()

    print("=" * 60)
    print(
        f"Running ToT-BFS Experiment "
        f"(beam_size={args.beam_size}, steps={args.steps}, n={len(problems)})"
    )
    print("=" * 60)
    results = run_tot_experiment(
        problems, api_key, beam_size=args.beam_size, steps=args.steps
    )

    if args.output:
        output_path = args.output
    elif args.num_problems < 50:
        output_path = f"../results/tot_results_smoke_{args.num_problems}.json"
    else:
        output_path = "../results/tot_results.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 60)
    print(
        f"ToT-BFS Results: {results['successes']}/{results['total']} "
        f"({results['success_rate']:.2%})"
    )
    print(f"Saved to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
