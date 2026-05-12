# Tree of Thoughts: Problem Solving with LLMs

**Authors:** Jane Tenecota-Villa, Jade Lee, Lauren Ah-Hot
**Course:** CS 4782: Deep Learning (Cornell University)
**Based on:** [Yao et al., NeurIPS 2023](https://arxiv.org/abs/2305.10601)

---

## 1. Introduction

This repository re-implements Tree of Thoughts (ToT) from Yao et al. (NeurIPS 2023), which frames LLM inference as a search over a tree of intermediate reasoning steps, outperforming standard Chain-of-Thought (CoT) and IO prompting on tasks requiring planning and multi-step reasoning.

---

## 2. Chosen Result

We targeted **Table 2** from the paper: the IO / CoT / ToT success rate comparison on Game of 24, which directly supports the paper's central claim that search-based reasoning substantially outperforms standard prompting.

**Original paper (GPT-4, May 2023):**

| Method | Success Rate |
|--------|-------------|
| IO (5-shot) | 7.3% |
| CoT (5-shot) | 4.0% |
| ToT BFS (b=5) | 74.0% |

---

## 3. GitHub Contents

```
.
├── code/               # re-implementation scripts
├── data/               # Game of 24 puzzle samples
├── results/            # outputs: success rates, cost/time logs, ablation results
├── poster/             # In-class poster presentation
├── report/             # Final project report
├── README.md
├── LICENSE
└── .gitignore
```

---

## 4. Re-implementation Details

- **Model:** GPT-4o via the OpenAI API (successor to the May 2023 GPT-4 in the original paper)
- **Dataset:** 50 randomly sampled puzzles from the Game of 24 dataset (hard games, indices 901-1000 from [4nums.com](https://4nums.com))
- **ToT (BFS):** depth T=3, beam size b=5; a propose prompt enumerates candidate next steps; a value prompt classifies each as `sure`/`maybe`/`impossible` sampled 3x per thought, retaining top-b candidates per level
- **Baselines:** IO and CoT with 5-shot prompting; correctness verified programmatically (each input number used exactly once, expression evaluates to 24)
- **Key modification:** we wrote our own 7-example few-shot evaluator, and added a few-shot vs. zero-shot evaluator ablation across 20 puzzles

---

## 5. Reproduction Steps

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with your API key:

```
OPENAI_API_KEY=your_key_here
```

```bash
cd code
python run_experiments.py     # runs IO and CoT; saves results to results/
python run_tot_experiment.py  # runs ToT BFS; saves results to results/
```

No GPU required; all inference runs via the OpenAI API. A full 50-puzzle ToT run costs approximately $5 and takes roughly 86 seconds per puzzle.

---

## 6. Results / Insights

**Main results (50 Game of 24 puzzles, GPT-4o):**

| Method | Success Rate | Cost / Problem | Time / Problem |
|--------|-------------|----------------|----------------|
| IO (5-shot) | 10.0% | ~$0.0024 | ~1.2s |
| CoT (5-shot) | 24.0% | ~$0.0022 | ~1.3s |
| ToT BFS (b=5) | **88.0%** | ~$0.1020 | ~86.0s |

Our results confirm the paper's key finding (ToT >> CoT >> IO); absolute numbers are higher than the paper's (88% vs. 74%), attributable to GPT-4o being a stronger model than GPT-4.

**Evaluator ablation (20 puzzles):**

| Evaluator | Success Rate | Time / Problem |
|-----------|-------------|----------------|
| Few-shot (7 examples) | 95% | ~97s |
| Zero-shot | 25% | ~70s |

Removing few-shot demonstrations caused a 70 percentage point drop, showing that evaluator quality is a major driver of ToT's performance.

---

## 7. Conclusion

Tree of Thoughts reliably and substantially outperforms CoT and IO on Game of 24, and our re-implementation with GPT-4o reproduces the qualitative ordering from Yao et al. with even higher numbers. The key takeaway from our evaluator ablation is that the few-shot evaluator is central to the search working correctly, and cost remains ToT's main practical limitation.

---

## 8. References

1. Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2023). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models.* NeurIPS 2023. [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)
2. OpenAI (2024). GPT-4o. https://openai.com/index/hello-gpt-4o/
3. Princeton NLP: Official ToT code & Game of 24 dataset. https://github.com/princeton-nlp/tree-of-thought-llm

---

## 9. Acknowledgements

This project was completed as part of **CS 4782: Introduction to Deep Learning** at Cornell University (Spring 2026). We thank the course instructors and TAs for their guidance, and Shunyu Yao et al. for making their paper and codebase publicly available.
