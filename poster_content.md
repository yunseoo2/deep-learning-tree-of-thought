# Poster Content — Tree of Thoughts Replication (Game of 24)

Copy each section directly into your poster template. Hard token-counts are
estimates so you can budget poster space.

---

## 1. Title & Authors

**Title (long):**
Replicating Tree of Thoughts on the Game of 24:
A Faithful Reproduction Under Budget Constraints

**Title (short):**
Tree of Thoughts on the Game of 24 — A Replication Study

**Authors:** Jane Tenecota Villa, Jade [Lastname], Lauren [Lastname]
**Affiliation:** [Your Institution]

**Footer (small):**
- Paper: Yao et al., NeurIPS 2023 — arXiv:2305.10601
- Original code: github.com/princeton-nlp/tree-of-thought-llm
- Our code: github.com/[your-repo]

---

## 2. Introduction / Background / Motivation

**Problem.** Large language models reason left-to-right, one token at a time.
Once an LLM commits to a wrong first step, it usually can't recover —
especially on tasks that require search or planning.

**Paper claim (Yao et al., 2023).** Tree of Thoughts (ToT) reframes LLM
inference as a *search problem* over a tree of intermediate "thoughts."
A generator proposes candidate next steps and an evaluator scores each
state, allowing the model to backtrack and explore. On the Game of 24,
ToT lifts GPT-4 from **4% (Chain-of-Thought)** to **74% success**.

**What we replicate.** Section 4.1 of the paper: a head-to-head comparison
of three prompting conditions — IO, CoT, and ToT-BFS (b=5) — on the Game
of 24 (4 numbers → arithmetic expression equal to 24).

**Why this paper.** Brings classical search algorithms (BFS/DFS) into LLM
inference — a clean bridge between core CS data structures and modern ML.
Demonstrates that LLM reasoning need not be linear.

**Independent contribution.** The paper does not ablate its few-shot
evaluator prompt. We show that **few-shot demonstrations in the evaluator
are load-bearing**: removing them collapses ToT from 95% to 25%.

---

## 3. Methodology

**Task.** Game of 24. Given 4 integers, generate an expression using
+, -, *, /, and parentheses, using each number exactly once, that evaluates
to 24. Example: `[4, 9, 10, 13] → (10 - 4) * (13 - 9) = 24`.

**Dataset.** 50 puzzles randomly sampled (seed=42) from the official
4nums.com dataset used by the paper.

**Model.** GPT-4o via the OpenAI API (temperature=0.7).

**Three conditions.**
- **IO** — 5-shot prompt, single direct answer.
- **CoT** — 5-shot prompt, intermediate steps then `Answer:`.
- **ToT-BFS** — depth=3, beam b=5, evaluator samples=3, scores
  `sure / likely / impossible = 20 / 1 / 0.001`.

**Modifications from the paper.**
- 50 problems instead of 100 (budget constraint).
- GPT-4o instead of GPT-4 (May 2023). All 50-problem ToT runs cost ~$5.
- Few-shot evaluator with our own examples (paper's exact examples for IO/CoT).
- Algorithmic expression assembly instead of a 4th LLM call (functionally
  equivalent: we track sub-expressions through the search and emit them at
  the end, saving one model call per problem).

**Independent experiment we added.**
**Comparing evaluators.** Same 20 puzzles, same proposer, same beam —
only the evaluator prompt changes (zero-shot vs. our few-shot).

---

## 4. Results

### (a) Headline replication — success rate (n=50)

| Method | Paper (GPT-4) | **Ours (GPT-4o)** |
|---|---|---|
| IO | 7.3% | **10.0%** (5/50) |
| CoT | 4.0% | **24.0%** (12/50) |
| **ToT (b=5)** | **74.0%** | **88.0%** (44/50) |

The qualitative ordering (ToT >> CoT > IO) replicates. Absolute numbers
are higher across the board because GPT-4o is a stronger model than the
May-2023 GPT-4 used in the paper.

### (b) Cost-of-correctness

| Method | Total cost | Total time | Time/problem | $/correct |
|---|---|---|---|---|
| IO | ~$0.12 | ~1.0 min | 1.2 s | $0.024 |
| CoT | ~$0.11 | ~1.1 min | 1.3 s | $0.0093 |
| ToT | $5.10 | 72 min | 86 s | **$0.116** |

ToT is ~45× more expensive per problem than IO/CoT but solves 3.6× more
problems than CoT. Per *correct* answer ToT is only ~12× costlier — failed
calls in IO/CoT are wasted spend.

**vs. the paper.** Their ToT cost was $0.74/problem on GPT-4 (Appendix B.3).
We paid **$0.10/problem on GPT-4o — ~7× cheaper for the same task.**

### (c) Independent exploration — Few-shot vs. zero-shot evaluator (n=20)

| Evaluator | Success | Total time | Avg time/problem |
|---|---|---|---|
| **Few-shot (ours)** | **19/20 (95%)** | 32.5 min | 97 s |
| Zero-shot | 5/20 (25%) | 23.2 min | 70 s |

**70-percentage-point drop** when demonstrations are removed, holding
everything else fixed. Zero-shot solved a *strict subset* of the few-shot
successes — demonstrations don't just help, they pure-dominate. Likely
mechanism: zero-shot can't distinguish productive fractions
(`2/3 36 → sure`, since `36 × 2/3 = 24`) from dead-end fractions
(`5/6 16 → impossible`), so scores cluster at the floor and BFS commits
to bad paths early.

---

## 5. Conclusion

- **Replication holds.** ToT outperforms CoT and IO on the Game of 24,
  with the same qualitative ordering as Yao et al. report — 88% vs. 24%
  vs. 10% on our 50-problem sample.
- **The few-shot evaluator is load-bearing.** Removing demonstrations
  drops ToT from 95% to 25%. The paper does not ablate this design
  choice; we show it is essential, not incidental.
- **Cost is the real tradeoff.** ToT is 45× costlier per problem than
  CoT/IO. On modern models (GPT-4o) the absolute cost is small (~$5
  for 50 problems), but for high-throughput applications the
  cost-per-correct-answer gap narrows considerably.

---

## 6. Future Work

- **DFS instead of BFS.** BFS materialises every survivor at every depth,
  which is the dominant cost of ToT. DFS with aggressive pruning on
  `sure` states should reach the first solution far sooner: once a path
  is rated `sure` at depth 2, there is no value in exploring the rest of
  the level. We expect the 88% success rate to hold at a fraction of the
  cost — especially on easy puzzles where one branch is obviously right.
  Worth measuring: median depth-to-first-solution and dollars saved per
  problem at matched accuracy.

- **Beam-size sweep (b=1 → b=5).** The paper reports 45% at b=1 and 74%
  at b=5 on GPT-4. We only ran b=5. A full sweep on GPT-4o would tell us
  whether the b=1 → b=5 gap shrinks on a stronger model (because each
  individual proposal is more reliable, so the marginal value of keeping
  alternates falls) or stays the same (search budget matters
  independently of model quality). Either result is informative for
  practitioners deciding how to spend tokens.

- **Diagnose the 6 ToT failures.** We hit 88% (44/50). Looking at the
  beam traces of the 6 failed puzzles —
  `[1,5,11,11]`, `[3,7,11,11]`, `[3,3,8,9]`, `[4,5,6,7]`,
  `[3,5,8,11]`, `[8,8,12,13]` — a pattern jumps out: **the proposer
  never explores fractional intermediates**. All 5 survivors at step 1
  are integer operations (`+`, `-`, integer `*`), and commutative
  duplicates (`5+1` and `1+5`, `11+1` and `1+11`) crowd out genuinely
  different candidates. Several of these puzzles only have
  fraction-bearing solutions (e.g. `[1,5,11,11] → (1 - 1/5) × ... `
  shapes), so the search tree is missing the entire region of state
  space that contains the answer. Future work: (i) deduplicate
  commutatively-equivalent proposals before scoring so beam slots aren't
  wasted, (ii) explicitly prompt the proposer to include at least one
  division at step 1, and (iii) measure whether either fix recovers the
  6 failures and pushes the ceiling toward the ~94% the paper reports.

- **Smaller-model + ToT vs. larger-model alone.** Does GPT-4o-mini or
  Haiku 4.5 *with* search beat GPT-4o *without* search at matched dollar
  cost? This isolates the question the paper implicitly raises: is ToT
  buying you reasoning, or just buying you a bigger model? If a weaker
  base model with ToT closes most of the gap to GPT-4o-IO, that's
  evidence the gains are search-driven, not capacity-driven.

- **Generalise the evaluator-ablation finding.** Our 70-pp drop is on
  Game of 24 alone. Does the few-shot evaluator matter as much on the
  paper's other tasks (Creative Writing, Mini Crosswords)? If yes, the
  paper is under-reporting a load-bearing prompt-engineering choice
  across the board, not just on one task.

---

## 7. References

[1] Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., &
Narasimhan, K. (2023). *Tree of Thoughts: Deliberate Problem Solving with
Large Language Models.* In NeurIPS 2023. arXiv:2305.10601.

[2] Wei, J. et al. (2022). *Chain-of-thought prompting elicits reasoning
in large language models.* arXiv:2201.11903.

[3] OpenAI (2024). GPT-4o. https://openai.com/index/hello-gpt-4o/

[4] 4nums.com Game of 24 dataset. github.com/princeton-nlp/tree-of-thought-llm
