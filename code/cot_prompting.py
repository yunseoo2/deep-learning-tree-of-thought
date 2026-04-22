"""
CoT (Chain-of-Thought) Prompting for Game of 24
Prompt asks model to reason step-by-step before giving final answer.
"""

import openai
import json
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv

from validation import validate_24_expression


def cot_prompt(numbers: List[int], api_key: str) -> Tuple[str, bool]:
    """
    Run CoT prompting: ask model to reason step-by-step.

    Args:
        numbers: List of 4 integers for the Game of 24
        api_key: OpenAI API key

    Returns:
        Tuple of (model response, success boolean)
    """
    client = openai.OpenAI(api_key=api_key)

    prompt = (
        f"Use each of the numbers {numbers[0]}, {numbers[1]}, {numbers[2]}, {numbers[3]} "
        f"exactly once with operations +, -, *, / (and parentheses) to get 24.\n\n"
        f"Think step by step:\n"
        f"1. Consider different ways to combine the numbers\n"
        f"2. Evaluate intermediate results\n"
        f"3. Work towards getting 24\n\n"
        f"End your response with a line of exactly this form:\n"
        f"Final: <expression>\n"
        f"The expression must use only integers, + - * /, and parentheses. No LaTeX, no words."
    )

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

    Args:
        problems: List of problems, each is a list of 4 integers
        api_key: OpenAI API key

    Returns:
        Dictionary with results including success rate and detailed logs
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

    # Test with a few examples
    test_problems = [
        [4, 9, 10, 13],
        [1, 4, 5, 6],
        [2, 3, 5, 12]
    ]

    results = run_cot_experiment(test_problems, api_key)

    print("="*50)
    print(f"CoT Prompting Results")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Successes: {results['successes']}/{results['total']}")
    print("="*50)

    # Save results
    with open("cot_results.json", "w") as f:
        json.dump(results, f, indent=2)
