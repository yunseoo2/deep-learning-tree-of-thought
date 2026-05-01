"""
Evaluator ablation: zero-shot vs. few-shot evaluator on the same 20 puzzles.

This is our independent contribution beyond the paper — Yao et al. used a
few-shot evaluator throughout and did not ablate it. We hold every other
variable fixed (same puzzles, same proposer, same beam size, same depth) and
toggle only the evaluator prompt.

Outputs (does NOT touch the headline tot_results.json):
  - results/tot_eval_ablation_fewshot_20.json
  - results/tot_eval_ablation_zeroshot_20.json
"""

import json
import os

from dotenv import load_dotenv

from run_experiments import load_problems
from tot_prompting import run_tot_experiment


N_PROBLEMS = 20


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in .env file")

    data_path = "../data/test_puzzles_50.csv"
    problems = load_problems(data_path, num_problems=N_PROBLEMS)
    print(f"Loaded {len(problems)} problems for evaluator ablation\n")

    summary = {}

    for mode, label in [("fewshot", "Few-shot evaluator"),
                        ("zeroshot", "Zero-shot evaluator")]:
        print("=" * 60)
        print(f"Running ToT-BFS with {label} (b=5, steps=3, n={len(problems)})")
        print("=" * 60)

        results = run_tot_experiment(
            problems,
            api_key,
            beam_size=5,
            steps=3,
            evaluator_mode=mode,
        )

        out_path = f"../results/tot_eval_ablation_{mode}_{N_PROBLEMS}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        print()
        print("=" * 60)
        print(
            f"{label}: {results['successes']}/{results['total']} "
            f"({results['success_rate']:.2%}) "
            f"in {results['total_elapsed_s']:.1f}s "
            f"(avg {results['avg_elapsed_s']:.1f}s/problem)"
        )
        print(f"Saved to {out_path}")
        print("=" * 60)
        print()

        summary[mode] = {
            "success_rate": results["success_rate"],
            "successes": results["successes"],
            "total": results["total"],
            "total_elapsed_s": results["total_elapsed_s"],
            "avg_elapsed_s": results["avg_elapsed_s"],
        }

    print("=" * 60)
    print("EVALUATOR ABLATION SUMMARY")
    print("=" * 60)
    for mode, s in summary.items():
        print(
            f"  {mode:>8}: {s['successes']:>2}/{s['total']} "
            f"({s['success_rate']:.0%})  "
            f"total={s['total_elapsed_s']:.0f}s  "
            f"avg={s['avg_elapsed_s']:.1f}s/problem"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
