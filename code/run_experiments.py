"""
Run experiments on 50 Game of 24 problems using IO and CoT prompting.
"""

import csv
import json
import os
from dotenv import load_dotenv
from io_prompting import run_io_experiment
from cot_prompting import run_cot_experiment


def load_problems(csv_path: str, num_problems: int = 50):
    """
    Load problems from CSV file.

    Args:
        csv_path: Path to CSV file with problems
        num_problems: Number of problems to load

    Returns:
        List of problems, each is a list of 4 integers
    """
    problems = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= num_problems:
                break

            # Parse the "Puzzles" column which has format "1 2 3 4"
            puzzle_str = row['Puzzles']
            numbers = [int(x) for x in puzzle_str.split()]

            problems.append(numbers)

    return problems


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in .env file")

    # Load 10 problems from dataset for testing
    data_path = "../data/test_puzzles_50.csv"
    problems = load_problems(data_path, num_problems=10)

    print(f"Loaded {len(problems)} problems from dataset")
    print()

    # Run IO experiment
    print("="*60)
    print("Running IO Prompting Experiment")
    print("="*60)
    io_results = run_io_experiment(problems, api_key)

    # Save IO results
    with open("../results/io_results.json", "w") as f:
        json.dump(io_results, f, indent=2)

    print()
    print("="*60)
    print(f"IO Results: {io_results['successes']}/{io_results['total']} "
          f"({io_results['success_rate']:.2%})")
    print("="*60)
    print()

    # Run CoT experiment
    print("="*60)
    print("Running CoT Prompting Experiment")
    print("="*60)
    cot_results = run_cot_experiment(problems, api_key)

    # Save CoT results
    with open("../results/cot_results.json", "w") as f:
        json.dump(cot_results, f, indent=2)

    print()
    print("="*60)
    print(f"CoT Results: {cot_results['successes']}/{cot_results['total']} "
          f"({cot_results['success_rate']:.2%})")
    print("="*60)
    print()

    # Print comparison
    print("="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"IO:  {io_results['success_rate']:.2%} ({io_results['successes']}/{len(problems)})")
    print(f"CoT: {cot_results['success_rate']:.2%} ({cot_results['successes']}/{len(problems)})")
    print("="*60)


if __name__ == "__main__":
    main()
