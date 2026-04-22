"""
IO (Input-Output) Prompting for Game of 24
Single prompt asking for direct answer without intermediate reasoning.
"""

import openai
import json
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv

from validation import validate_24_expression


def io_prompt(numbers: List[int], api_key: str) -> Tuple[str, bool]:
    """
    Run IO prompting: direct question to direct answer.

    Args:
        numbers: List of 4 integers for the Game of 24
        api_key: OpenAI API key

    Returns:
        Tuple of (model response, success boolean)
    """
    client = openai.OpenAI(api_key=api_key)

    prompt = (
        f"Use each of the numbers {numbers[0]}, {numbers[1]}, {numbers[2]}, {numbers[3]} "
        f"exactly once with operations +, -, *, / (and parentheses) to get 24.\n"
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

    Args:
        problems: List of problems, each is a list of 4 integers
        api_key: OpenAI API key

    Returns:
        Dictionary with results including success rate and detailed logs
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

    # Test with a few examples
    test_problems = [
        [4, 9, 10, 13],
        [1, 4, 5, 6],
        [2, 3, 5, 12]
    ]

    results = run_io_experiment(test_problems, api_key)

    print("="*50)
    print(f"IO Prompting Results")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Successes: {results['successes']}/{results['total']}")
    print("="*50)

    # Save results
    with open("io_results.json", "w") as f:
        json.dump(results, f, indent=2)
