"""
ToT (Tree of Thoughts) Prompting for Game of 24 — BFS scaffolding.

This module owns the search loop only. The two prompt-dependent pieces are
stubbed out for teammates to fill in:

  - propose_next_states  (Jade — thought proposer prompt)
  - value_state          (Lauren — evaluator prompt, sure/maybe/impossible)

Once both stubs are implemented, run_tot_experiment will work end-to-end.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import openai
from dotenv import load_dotenv

from validation import validate_24_expression


# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToTState:
    """
    A node in the BFS tree.

    remaining:   numbers still available (4 at root, shrinks by 1 per step)
    steps:       history of op strings like "2 + 8 = 10 (left: 8 10 14)"
    expression:  partial algebraic expression equivalent to the combine history;
                 used only when `remaining` has exactly one element so we can
                 feed the full expression to `validate_24_expression`.
    """

    remaining: Tuple[int, ...]
    steps: Tuple[str, ...] = ()
    expression: str = ""


# ---------------------------------------------------------------------------
# Teammate stubs — do NOT implement here.
# ---------------------------------------------------------------------------


def propose_next_states(state: ToTState, client: openai.OpenAI) -> List[ToTState]:
    """
    STUB (Jade): thought proposer.

    Given a state with k remaining numbers, prompt the LLM to propose candidate
    next operations (pick two numbers, apply + - * /). Parse each proposal into
    a child ToTState with k-1 remaining numbers and an updated step/expression
    trace.

    Expected return: list of child ToTStates (typically ~ a few to ~15).
    """
    raise NotImplementedError(
        "propose_next_states is owned by Jade (thought proposer prompt)."
    )


def value_state(
    state: ToTState,
    client: openai.OpenAI,
    n_samples: int = 3,
) -> float:
    """
    Prompt the LLM `n_samples` times to judge whether `state.remaining` can
    still reach 24 (sure / likely / impossible). Map each judgement to a number
    (paper uses sure=20, likely=1, impossible=0.001) and return the sum.

    Expected return: float score (higher is better).
    """
    if n_samples <= 0:
        return 0.0

    # Terminal checks (avoid an API call when we already know the answer).
    if len(state.remaining) == 1:
        return 20.0 * n_samples if state.remaining[0] == 24 else 0.001 * n_samples

    # Paper mapping (ToT / Game of 24): sure=20, likely=1, impossible=0.001
    weights = {"sure": 20.0, "likely": 1.0, "impossible": 0.001}

    def _parse_judgement(text: str) -> str | None:
        t = (text or "").strip().lower()
        # Prefer explicit labels first.
        for key in ("sure", "likely", "impossible"):
            if key in t:
                return key
        # Handle common synonyms / formatting variants.
        if "possible" in t:
            return "likely"
        if "maybe" in t or "uncertain" in t:
            return "likely"
        if "can't" in t or "cannot" in t or "not possible" in t:
            return "impossible"
        return None

    system = (
        "You are an evaluator for the Game of 24.\n"
        "Given a multiset of remaining integers, judge whether it can still reach 24 "
        "using only +, -, *, / and parentheses, using each number exactly once.\n"
        "Respond with exactly one token: sure, likely, or impossible."
    )
    user = f"Remaining numbers: {list(state.remaining)}\nJudgement:"

    score = 0.0
    for _ in range(n_samples):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.7,
                max_tokens=5,
            )
            content = (response.choices[0].message.content or "").strip()
            label = _parse_judgement(content)
            score += weights.get(label or "", weights["impossible"])
        except Exception:
            # If evaluation fails (rate limit / transient error), be conservative.
            score += weights["impossible"]

    return float(score)


# ---------------------------------------------------------------------------
# BFS loop
# ---------------------------------------------------------------------------


def _keep_top_b(
    candidates: List[ToTState],
    scores: List[float],
    b: int,
) -> List[ToTState]:
    """Return the top-b candidates by score, ties broken by original order."""
    indexed = list(enumerate(zip(candidates, scores)))
    indexed.sort(key=lambda ix: (-ix[1][1], ix[0]))
    return [cand for _, (cand, _) in indexed[:b]]


def tot_bfs(
    numbers: List[int],
    client: openai.OpenAI,
    beam_size: int = 5,
    steps: int = 3,
) -> Tuple[str, bool, Dict]:
    """
    Run the ToT BFS for a single Game-of-24 puzzle.

    Args:
        numbers:    4 input integers
        client:     OpenAI client (passed through to proposer/evaluator)
        beam_size:  b, the number of surviving states per step (paper: 5)
        steps:      search depth (paper: 3, reducing 4 nums -> 3 -> 2 -> 1)

    Returns:
        (final_expression, success, trace)
        - final_expression is the best surviving expression (empty if none)
        - success is True iff any surviving expression validates to 24
        - trace captures per-step frontier size + survivors for debugging
    """
    if len(numbers) != 4:
        raise ValueError("Game of 24 expects exactly 4 numbers")

    root = ToTState(remaining=tuple(numbers))
    frontier: List[ToTState] = [root]
    trace: Dict = {"beam_size": beam_size, "steps": steps, "levels": []}

    for step in range(1, steps + 1):
        # 1. Propose: expand every state in the current frontier.
        proposals: List[ToTState] = []
        for parent in frontier:
            proposals.extend(propose_next_states(parent, client))

        if not proposals:
            trace["levels"].append({"step": step, "proposals": 0, "survivors": 0})
            frontier = []
            break

        # 2. Evaluate: score every child.
        scores = [value_state(child, client) for child in proposals]

        # 3. Prune: keep top-b by score.
        frontier = _keep_top_b(proposals, scores, beam_size)

        trace["levels"].append(
            {
                "step": step,
                "proposals": len(proposals),
                "survivors": [
                    {
                        "remaining": list(s.remaining),
                        "expression": s.expression,
                        "steps": list(s.steps),
                    }
                    for s in frontier
                ],
            }
        )

    # After `steps` expansions, each survivor should have 1 number left.
    # Validate each candidate expression; a problem succeeds if any hits 24.
    best_expr = ""
    success = False
    for survivor in frontier:
        if not survivor.expression:
            continue
        result = validate_24_expression(numbers, survivor.expression)
        if result.ok:
            best_expr = survivor.expression
            success = True
            break
        if not best_expr:
            best_expr = survivor.expression

    return best_expr, success, trace


# Experiment driver
def run_tot_experiment(
    problems: List[List[int]],
    api_key: str,
    beam_size: int = 5,
    steps: int = 3,
) -> Dict:
    """
    Run ToT-BFS on a list of Game of 24 problems. Result shape matches the
    IO/CoT runners so the summary code can treat all three the same way.
    """
    client = openai.OpenAI(api_key=api_key)

    results: Dict = {
        "method": "ToT-BFS",
        "beam_size": beam_size,
        "steps": steps,
        "total": len(problems),
        "successes": 0,
        "failures": 0,
        "problems": [],
    }

    for i, numbers in enumerate(problems):
        print(f"Problem {i+1}/{len(problems)}: {numbers}")

        try:
            expression, success, trace = tot_bfs(
                numbers, client, beam_size=beam_size, steps=steps
            )
            error = None
        except NotImplementedError as e:
            # Expected until Jade + Lauren land their pieces.
            expression, success, trace, error = "", False, {}, str(e)
        except Exception as e:
            expression, success, trace, error = "", False, {}, f"Error: {e}"

        results["problems"].append(
            {
                "numbers": numbers,
                "expression": expression,
                "success": success,
                "trace": trace,
                "error": error,
            }
        )

        if success:
            results["successes"] += 1
            print("✓ Success")
        else:
            results["failures"] += 1
            print(f"✗ Failed{' (' + error + ')' if error else ''}")

    results["success_rate"] = (
        results["successes"] / results["total"] if results["total"] else 0.0
    )
    return results


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in .env file")

    test_problems = [
        [4, 9, 10, 13],
        [1, 4, 5, 6],
        [2, 3, 5, 12],
    ]
    results = run_tot_experiment(test_problems, api_key)
    print("=" * 50)
    print("ToT-BFS Prompting Results")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Successes: {results['successes']}/{results['total']}")
    print("=" * 50)

    with open("tot_results.json", "w") as f:
        json.dump(results, f, indent=2)
