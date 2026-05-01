"""
IO (Input-Output) Prompting for Game of 24.

5-shot prompt matching Yao et al. 2023 (ToT paper, Section 4.1, Baselines).
The 5 in-context examples are the same ones used in the paper's released
prompts (`tree-of-thought-llm/src/tot/prompts/game24.py`), kept verbatim so
this baseline can be compared apples-to-apples with the paper's IO numbers.
"""

import openai
import json
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv

from validation import validate_24_expression


# 5-shot prompt template, copied from the paper's released code so the baseline
# matches the paper's IO setup. Demo puzzles are disjoint from our test set.
IO_PROMPT_TEMPLATE = """Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Input: 1 4 8 8
Answer: (8 / 4 + 1) * 8 = 24
Input: 5 5 5 9
Answer: 5 + 5 + 5 + 9 = 24
Input: {input}
"""


def io_prompt(numbers: List[int], api_key: str) -> Tuple[str, bool]:
    """
    Run IO prompting (5-shot, paper-faithful): direct question to direct answer.

    Args:
        numbers: List of 4 integers for the Game of 24
        api_key: OpenAI API key

    Returns:
        Tuple of (model response, success boolean)
    """
    client = openai.OpenAI(api_key=api_key)

    nums_str = " ".join(str(n) for n in numbers)
    prompt = IO_PROMPT_TEMPLATE.format(input=nums_str)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )

        answer = response.choices[0].message.content.strip()
        result = validate_24_expression(numbers, answer)
        return answer, result.ok

    except Exception as e:
        return f"Error: {str(e)}", False


def run_io_experiment(problems: List[List[int]], api_key: str) -> Dict:
    """
    Run IO prompting on a list of Game of 24 problems.
    """
    results = {
        "method": "IO",
        "total": len(problems),
        "successes": 0,
        "failures": 0,
        "problems": []
    }

    for i, numbers in enumerate(problems):
        print(f"Problem {i+1}/{len(problems)}: {numbers}")

        response, success = io_prompt(numbers, api_key)

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

    results = run_io_experiment(test_problems, api_key)

    print("="*50)
    print(f"IO Prompting Results (5-shot)")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Successes: {results['successes']}/{results['total']}")
    print("="*50)

    with open("io_results.json", "w") as f:
        json.dump(results, f, indent=2)
