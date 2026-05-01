"""
CoT (Chain-of-Thought) Prompting for Game of 24.

5-shot prompt matching Yao et al. 2023 (ToT paper, Section 4.1, Baselines).
Each example shows 3 intermediate `a OP b = X (left: ...)` steps before the
final `Answer: <expression> = 24` line, exactly as in the paper's released
prompts (`tree-of-thought-llm/src/tot/prompts/game24.py`). Demo puzzles are
disjoint from our 50-problem test set.
"""

import openai
import json
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv

from validation import validate_24_expression


# 5-shot CoT prompt template, copied from the paper's released code so the
# baseline matches the paper's CoT setup.
COT_PROMPT_TEMPLATE = """Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24
Input: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24
Input: 4 9 10 13
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24
Input: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24
Input: 5 5 5 9
Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24
Input: {input}
"""


def cot_prompt(numbers: List[int], api_key: str) -> Tuple[str, bool]:
    """
    Run CoT prompting (5-shot, paper-faithful): step-by-step reasoning then answer.

    Args:
        numbers: List of 4 integers for the Game of 24
        api_key: OpenAI API key

    Returns:
        Tuple of (model response, success boolean)
    """
    client = openai.OpenAI(api_key=api_key)

    nums_str = " ".join(str(n) for n in numbers)
    prompt = COT_PROMPT_TEMPLATE.format(input=nums_str)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        answer = response.choices[0].message.content.strip()
        result = validate_24_expression(numbers, answer)
        return answer, result.ok

    except Exception as e:
        return f"Error: {str(e)}", False


def run_cot_experiment(problems: List[List[int]], api_key: str) -> Dict:
    """
    Run CoT prompting on a list of Game of 24 problems.
    """
    results = {
        "method": "CoT",
        "total": len(problems),
        "successes": 0,
        "failures": 0,
        "problems": []
    }

    for i, numbers in enumerate(problems):
        print(f"Problem {i+1}/{len(problems)}: {numbers}")

        response, success = cot_prompt(numbers, api_key)

        results["problems"].append({
            "numbers": numbers,
            "response": response,
            "success": success
        })

        if success:
            results["successes"] += 1
            print(f"✓ Success")
        else:
            results["failures"] += 1
            print(f"✗ Failed")

        print()

    results["success_rate"] = results["successes"] / results["total"]

    return results


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in .env file")

    # Note: avoiding the 5 demo puzzles to prevent leakage in the smoke test.
    test_problems = [
        [3, 3, 12, 12],
        [1, 4, 5, 6],
        [2, 3, 5, 12]
    ]

    results = run_cot_experiment(test_problems, api_key)

    print("="*50)
    print(f"CoT Prompting Results (5-shot)")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Successes: {results['successes']}/{results['total']}")
    print("="*50)

    with open("cot_results.json", "w") as f:
        json.dump(results, f, indent=2)
