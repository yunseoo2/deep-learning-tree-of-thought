# Results

Raw experiment outputs and the figures generated from them.

## Files

| File | Source script | Description |
|---|---|---|
| `io_results.json` | `run_experiments.py` | IO prompting on 50 puzzles |
| `cot_results.json` | `run_experiments.py` | CoT prompting on 50 puzzles |
| `tot_results.json` | `run_tot_experiment.py` | ToT-BFS (b=5, depth=3) on 50 puzzles — headline run |
| `tot_results_smoke_3.json` | `run_tot_experiment.py --num_problems 3` | 3-puzzle smoke test |
| `tot_eval_comparison_fewshot_20.json` | `run_evaluator_comparison.py` | ToT-BFS with few-shot evaluator (n=20) |
| `tot_eval_comparison_zeroshot_20.json` | `run_evaluator_comparison.py` | ToT-BFS with zero-shot evaluator (n=20) |
| `figures/fig1_success_rate.png` | `make_poster_plots.py` | IO/CoT/ToT success rate, ours vs. paper |
| `figures/fig2_cost_time.png` | `make_poster_plots.py` | Cost-of-correctness comparison |
| `figures/fig3_evaluator_comparison.png` | `make_poster_plots.py` | Few-shot vs. zero-shot evaluator |
| `figures/fig5_comparison_grid.png` | `make_poster_plots.py` | Per-puzzle outcome grid (n=20) |

## Headline numbers (n=50, GPT-4o, T=0.7)

| Method | Success | $/correct |
|---|---|---|
| IO  | 10.0% (5/50)  | $0.024 |
| CoT | 24.0% (12/50) | $0.0093 |
| **ToT (b=5)** | **88.0% (44/50)** | $0.116 |

## Evaluator comparison (n=20, same proposer/beam)

| Evaluator | Success |
|---|---|
| Few-shot (ours) | 19/20 (95%) |
| Zero-shot       | 5/20 (25%) |
