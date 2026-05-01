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
import re
from dataclasses import dataclass, field
from fractions import Fraction
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
# Thought-proposer helpers
# ---------------------------------------------------------------------------


def _to_frac(n) -> Fraction:
    """Convert int, float, Fraction, or numeric string (including 'a/b') to Fraction."""
    if isinstance(n, Fraction):
        return n
    if isinstance(n, int):
        return Fraction(n)
    if isinstance(n, str):
        s = n.strip()
        if "/" in s:
            num_str, den_str = s.split("/", 1)
            try:
                return Fraction(int(num_str.strip()), int(den_str.strip()))
            except (ValueError, ZeroDivisionError):
                pass
        try:
            return Fraction(s).limit_denominator(10**6)
        except ValueError:
            return Fraction(float(s)).limit_denominator(10**6)
    # float
    return Fraction(str(round(float(n), 10))).limit_denominator(10**6)


def _fmt_num(n) -> str:
    """Format a number as a compact string: integer if whole, fraction otherwise."""
    f = _to_frac(n)
    if f.denominator == 1:
        return str(int(f))
    return f"{f.numerator}/{f.denominator}"


def _get_exprs(state: ToTState) -> List[str]:
    """
    Return the algebraic sub-expression for each element of state.remaining.

    Encoding: the expression field stores pipe-separated sub-expressions in the
    same positional order as remaining when len(remaining) > 1.  When
    len(remaining) == 1, expression holds the complete final expression directly.
    """
    if state.expression and "|" in state.expression:
        parts = state.expression.split("|")
        if len(parts) == len(state.remaining):
            return parts
    # Root state or fallback: each remaining number is its own literal.
    return [_fmt_num(n) for n in state.remaining]


# ---------------------------------------------------------------------------
# Thought proposer
# ---------------------------------------------------------------------------

# Regex matching "a OP b = result" where numbers may be integers, decimals,
# or fractions (a/b notation).
_NUM = r"-?\d+(?:/\d+|-?\d*\.\d+)?"
_LINE_RE = re.compile(
    rf"({_NUM})\s*([+\-*/])\s*({_NUM})\s*=\s*({_NUM})"
)

# Few-shot example taken from the ToT paper (Yao et al., 2023), Figure 3.
_FEW_SHOT_INPUT = "2 8 8 14"
_FEW_SHOT_OUTPUT = (
    "2 + 8 = 10 (left: 8 10 14)\n"
    "8 / 2 = 4 (left: 4 8 14)\n"
    "14 + 2 = 16 (left: 8 8 16)\n"
    "2 * 8 = 16 (left: 8 14 16)\n"
    "8 - 2 = 6 (left: 6 8 14)\n"
    "2 + 14 = 16 (left: 8 8 16)\n"
    "2 * 14 = 28 (left: 8 8 28)\n"
    "14 - 2 = 12 (left: 8 8 12)\n"
    "8 + 14 = 22 (left: 2 8 22)\n"
    "8 * 14 = 112 (left: 2 8 112)\n"
    "14 - 8 = 6 (left: 2 6 8)\n"
    "8 + 8 = 16 (left: 2 14 16)\n"
    "8 - 8 = 0 (left: 0 2 14)\n"
    "8 * 8 = 64 (left: 2 14 64)\n"
    "8 / 8 = 1 (left: 1 2 14)"
)

_SYSTEM_PROPOSER = (
    "You are solving the Game of 24.\n"
    "Given a list of numbers, enumerate ALL possible ways to pick any two of them "
    "and apply one arithmetic operation (+, -, *, /) to produce a new number.\n"
    "Output each option on its own line in this exact format:\n"
    "  a OP b = result (left: remaining numbers)\n"
    "where 'left' shows the numbers after removing a and b and inserting result.\n"
    "List every distinct (ordered pair, operation) combination — include both "
    "a OP b and b OP a when they differ. Skip any operation that divides by zero.\n"
    "Output only the list, no explanation."
)


def propose_next_states(state: ToTState, client: openai.OpenAI) -> List[ToTState]:
    """
    Thought proposer: prompt GPT-4o to enumerate all candidate next operations.

    For a state with k remaining numbers this generates all ways to combine any
    two of them with +, -, *, /, returning a list of child ToTStates each with
    k-1 remaining numbers and an updated step + expression trace.
    """
    nums_str = " ".join(_fmt_num(n) for n in state.remaining)

    messages = [
        {"role": "system", "content": _SYSTEM_PROPOSER},
        {"role": "user", "content": f"Input: {_FEW_SHOT_INPUT}\nPossible next steps:"},
        {"role": "assistant", "content": _FEW_SHOT_OUTPUT},
        {"role": "user", "content": f"Input: {nums_str}\nPossible next steps:"},
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=600,
        )
        raw = response.choices[0].message.content or ""
    except Exception:
        return []

    parent_exprs = _get_exprs(state)
    rem_fracs = [_to_frac(n) for n in state.remaining]

    children: List[ToTState] = []
    seen: set = set()

    for line in raw.splitlines():
        m = _LINE_RE.search(line.strip())
        if not m:
            continue

        try:
            a_frac = _to_frac(m.group(1))
            op = m.group(2)
            b_frac = _to_frac(m.group(3))
            # Ignore the LLM's claimed result; recompute from scratch.
            if op == "+":
                result_frac = a_frac + b_frac
            elif op == "-":
                result_frac = a_frac - b_frac
            elif op == "*":
                result_frac = a_frac * b_frac
            elif op == "/":
                if b_frac == 0:
                    continue
                result_frac = a_frac / b_frac
            else:
                continue
        except (ValueError, ZeroDivisionError):
            continue

        # Locate a in remaining.
        a_idx = next((i for i, v in enumerate(rem_fracs) if v == a_frac), None)
        if a_idx is None:
            continue

        # Locate b in remaining, skipping the a position.
        b_idx = next(
            (i for i, v in enumerate(rem_fracs) if i != a_idx and v == b_frac),
            None,
        )
        if b_idx is None:
            continue

        # Build the algebraic sub-expression for the merged number.
        new_sub_expr = f"({parent_exprs[a_idx]} {op} {parent_exprs[b_idx]})"

        kept = [i for i in range(len(rem_fracs)) if i != a_idx and i != b_idx]
        new_rem_fracs = [rem_fracs[i] for i in kept] + [result_frac]
        new_exprs = [parent_exprs[i] for i in kept] + [new_sub_expr]

        def _val(f: Fraction):
            return int(f) if f.denominator == 1 else float(f)

        new_remaining = tuple(_val(f) for f in new_rem_fracs)

        key = (new_remaining, tuple(new_exprs))
        if key in seen:
            continue
        seen.add(key)

        rem_display = " ".join(_fmt_num(_val(f)) for f in new_rem_fracs)
        step_str = (
            f"{_fmt_num(_val(a_frac))} {op} {_fmt_num(_val(b_frac))} "
            f"= {_fmt_num(_val(result_frac))} (left: {rem_display})"
        )

        expr_field = new_sub_expr if len(new_remaining) == 1 else "|".join(new_exprs)

        children.append(
            ToTState(
                remaining=new_remaining,
                steps=state.steps + (step_str,),
                expression=expr_field,
            )
        )

    return children


# Few-shot evaluator prompt. Demonstrates the kind of mini-derivation we want
# the model to do before committing to a label, and calibrates each label.
_VALUE_SYSTEM = (
    "You evaluate partial states for the Game of 24. Given remaining numbers "
    "(integers or fractions like a/b), reason briefly about whether 24 is still "
    "reachable using +, -, *, / and parentheses, using each number exactly once. "
    "End your response with a single label on its own line: sure, likely, or impossible."
)

_VALUE_FEWSHOT = [
    # 2 numbers: directly reachable
    ("10 14",
     "10 + 14 = 24\nsure"),
    # 2 numbers: not reachable
    ("3 7",
     "3 + 7 = 10\n7 - 3 = 4\n3 * 7 = 21\n3 / 7 ≈ 0.43\nNone of these are 24.\nimpossible"),
    # 2 numbers: clean fraction case
    ("2/3 36",
     "36 * (2/3) = 24\nsure"),
    # 2 numbers: hopeless fractional case
    ("5/6 16",
     "16 + 5/6 ≈ 16.83\n16 - 5/6 ≈ 15.17\n16 * 5/6 ≈ 13.33\n16 / (5/6) = 19.2\nNone are 24.\nimpossible"),
    # 3 numbers: solvable
    ("4 6 10",
     "(10 - 4) * (6 - ...) — try 4 * 6 = 24, then need to use 10... not direct.\n"
     "10 - 4 = 6, then 6 * 6 = ... can't reuse.\n"
     "(10 + 4 - 6) = 8 — no.\n"
     "Within range, plausible.\nlikely"),
    # 3 numbers: too small
    ("1 2 3",
     "1 + 2 + 3 = 6\n1 * 2 * 3 = 6\n(1 + 2) * 3 = 9\nAll combinations stay small.\nimpossible"),
    # 3 numbers: too big
    ("10 11 12",
     "10 + 11 + 12 = 33\n12 - 11 + 10 = 11\n(12 - 10) * 11 = 22\n(11 - 10) * 12 = 12\n"
     "Cannot bring all three to 24.\nimpossible"),
]


# Zero-shot evaluator system prompt — used for the ablation study to quantify
# how much the few-shot demonstrations matter. Same task, no demos, no chain.
_VALUE_SYSTEM_ZEROSHOT = (
    "You are an evaluator for the Game of 24.\n"
    "Given a multiset of remaining numbers (integers or fractions like a/b), judge "
    "whether they can still reach 24 using only +, -, *, / and parentheses, "
    "using each number exactly once.\n"
    "Respond with exactly one token: sure, likely, or impossible."
)


def _build_value_messages(nums_str: str, mode: str = "fewshot") -> list:
    if mode == "zeroshot":
        return [
            {"role": "system", "content": _VALUE_SYSTEM_ZEROSHOT},
            {"role": "user", "content": f"Remaining numbers: {nums_str}\nJudgement:"},
        ]
    msgs: list = [{"role": "system", "content": _VALUE_SYSTEM}]
    for ex_input, ex_output in _VALUE_FEWSHOT:
        msgs.append({"role": "user", "content": f"Remaining: {ex_input}"})
        msgs.append({"role": "assistant", "content": ex_output})
    msgs.append({"role": "user", "content": f"Remaining: {nums_str}"})
    return msgs


def _parse_judgement(text: str) -> str | None:
    """Extract the final label from the model's response. Look at the last
    non-empty line first, then fall back to keyword search."""
    lines = [ln.strip().lower() for ln in (text or "").splitlines() if ln.strip()]
    last = lines[-1] if lines else ""
    for key in ("sure", "likely", "impossible"):
        if key == last or last.endswith(key) or last.startswith(key):
            return key
    full = " ".join(lines)
    for key in ("impossible", "sure", "likely"):
        if key in full:
            return key
    if "possible" in full or "maybe" in full or "uncertain" in full:
        return "likely"
    if "cannot" in full or "can't" in full or "not possible" in full:
        return "impossible"
    return None


def _get_value_cache(client: openai.OpenAI) -> dict:
    """Per-client cache of (state_key, n_samples) -> score. Avoids re-calling
    the evaluator for states that recur across the beam."""
    cache = getattr(client, "_tot_value_cache", None)
    if cache is None:
        cache = {}
        try:
            setattr(client, "_tot_value_cache", cache)
        except Exception:
            pass
    return cache


def value_state(
    state: ToTState,
    client: openai.OpenAI,
    n_samples: int = 3,
    evaluator_mode: str = "fewshot",
) -> float:
    """
    Prompt the LLM `n_samples` times to judge whether `state.remaining` can
    still reach 24 (sure / likely / impossible). Map each judgement to a number
    (paper: sure=20, likely=1, impossible=0.001) and return the sum.

    Higher score = more promising state. Results are cached per client across
    the experiment, since identical multisets recur across beam expansions.

    `evaluator_mode` is "fewshot" (default, used for headline ToT results) or
    "zeroshot" (ablation only).
    """
    if n_samples <= 0:
        return 0.0

    # Terminal: skip API call when we already know the answer.
    if len(state.remaining) == 1:
        return 20.0 * n_samples if state.remaining[0] == 24 else 0.001 * n_samples

    weights = {"sure": 20.0, "likely": 1.0, "impossible": 0.001}

    nums_str = " ".join(_fmt_num(n) for n in state.remaining)
    cache = _get_value_cache(client)
    cache_key = (evaluator_mode, nums_str, n_samples)
    if cache_key in cache:
        return cache[cache_key]

    messages = _build_value_messages(nums_str, mode=evaluator_mode)
    max_tok = 200 if evaluator_mode == "fewshot" else 5

    score = 0.0
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=max_tok,
            n=n_samples,
        )
        for choice in response.choices:
            content = (choice.message.content or "").strip()
            label = _parse_judgement(content)
            score += weights.get(label or "", weights["impossible"])
    except Exception:
        # Conservative fallback on transient API errors.
        score = weights["impossible"] * n_samples

    cache[cache_key] = float(score)
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
    evaluator_mode: str = "fewshot",
) -> Tuple[str, bool, Dict]:
    """
    Run the ToT BFS for a single Game-of-24 puzzle.

    Args:
        numbers:        4 input integers
        client:         OpenAI client (passed through to proposer/evaluator)
        beam_size:      b, the number of surviving states per step (paper: 5)
        steps:          search depth (paper: 3, reducing 4 nums -> 3 -> 2 -> 1)
        evaluator_mode: "fewshot" (default, used for headline results) or
                        "zeroshot" (ablation only).

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
    trace: Dict = {
        "beam_size": beam_size,
        "steps": steps,
        "evaluator_mode": evaluator_mode,
        "levels": [],
    }

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
        scores = [
            value_state(child, client, evaluator_mode=evaluator_mode)
            for child in proposals
        ]

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
    evaluator_mode: str = "fewshot",
) -> Dict:
    """
    Run ToT-BFS on a list of Game of 24 problems. Result shape matches the
    IO/CoT runners so the summary code can treat all three the same way.

    `evaluator_mode` is "fewshot" (default) or "zeroshot" (ablation).
    Per-problem and total wall-clock time are recorded in the result dict.
    """
    import time

    client = openai.OpenAI(api_key=api_key)

    results: Dict = {
        "method": "ToT-BFS",
        "beam_size": beam_size,
        "steps": steps,
        "evaluator_mode": evaluator_mode,
        "total": len(problems),
        "successes": 0,
        "failures": 0,
        "problems": [],
    }

    run_start = time.time()

    for i, numbers in enumerate(problems):
        print(f"Problem {i+1}/{len(problems)}: {numbers}")
        prob_start = time.time()

        try:
            expression, success, trace = tot_bfs(
                numbers,
                client,
                beam_size=beam_size,
                steps=steps,
                evaluator_mode=evaluator_mode,
            )
            error = None
        except NotImplementedError as e:
            expression, success, trace, error = "", False, {}, str(e)
        except Exception as e:
            expression, success, trace, error = "", False, {}, f"Error: {e}"

        elapsed = time.time() - prob_start

        results["problems"].append(
            {
                "numbers": numbers,
                "expression": expression,
                "success": success,
                "elapsed_s": round(elapsed, 3),
                "trace": trace,
                "error": error,
            }
        )

        if success:
            results["successes"] += 1
            print(f"✓ Success ({elapsed:.1f}s)")
        else:
            results["failures"] += 1
            print(f"✗ Failed{' (' + error + ')' if error else ''} ({elapsed:.1f}s)")

    total_elapsed = time.time() - run_start
    results["total_elapsed_s"] = round(total_elapsed, 3)
    results["avg_elapsed_s"] = (
        round(total_elapsed / len(problems), 3) if problems else 0.0
    )
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
