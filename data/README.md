# Data

`test_puzzles_50.csv` — 50 puzzles randomly sampled (seed=42) from the official
[4nums.com Game of 24 dataset](https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/data/24/24.csv)
used by Yao et al. (2023). Schema (from the source file):

| Column | Meaning |
|---|---|
| `Rank` | Difficulty rank from 4nums.com (lower = easier) |
| `Puzzles` | Four space-separated integers, e.g. `4 9 10 13` |
| `AMT (s)` | Median human solve time on Amazon Mechanical Turk |
| `Solved rate` | Fraction of human solvers who succeeded |
| `1-sigma Mean (s)` / `1-sigma STD (s)` | Solve-time stats within one sigma |

Only the `Puzzles` column is used by our code (see
`code/run_experiments.py::load_problems`).

## Reproducing the sample

The sample lives in this repo so the same 50 puzzles are used across all
experiments. To regenerate from the upstream file, sample 50 rows with
`pandas` using `random_state=42` from the source CSV linked above.
