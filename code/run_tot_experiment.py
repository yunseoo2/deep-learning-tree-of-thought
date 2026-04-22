"""
Run the ToT-BFS experiment on 50 Game of 24 problems.
"""

import json
import os

from dotenv import load_dotenv

from run_experiments import load_problems
from tot_prompting import run_tot_experiment


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in .env file")

    data_path = "../data/test_puzzles_50.csv"
    problems = load_problems(data_path, num_problems=50)

    print(f"Loaded {len(problems)} problems from dataset")
    print()

    print("=" * 60)
    print("Running ToT-BFS Experiment (beam_size=5, steps=3)")
    print("=" * 60)
    results = run_tot_experiment(problems, api_key, beam_size=5, steps=3)

    with open("../results/tot_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 60)
    print(
        f"ToT-BFS Results: {results['successes']}/{results['total']} "
        f"({results['success_rate']:.2%})"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
