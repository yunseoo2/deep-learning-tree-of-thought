"""
IO (Input-Output) Prompting for Game of 24
Single prompt asking for direct answer without intermediate reasoning.
"""

import openai
import json
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv


def verify_expression(numbers: List[int], expression: str) -> bool:
    """
    Verify if the expression is valid and equals 24.

    Args:
        numbers: List of 4 integers to use
        expression: Mathematical expression to verify

    Returns:
        True if expression uses each number exactly once and equals 24
    """
    try:
        # Check if expression evaluates to 24
        result = eval(expression)
        if abs(result - 24) > 0.001:  # Allow small floating point error
            return False

        # Check if each number is used exactly once
        expr_clean = expression.replace('(', '').replace(')', '').replace(' ', '')
        used_numbers = []
        current_num = ""

        for char in expr_clean:
            if char.isdigit():
                current_num += char
            else:
                if current_num:
                    used_numbers.append(int(current_num))
                    current_num = ""
        if current_num:
            used_numbers.append(int(current_num))

        # Sort both lists and compare
        return sorted(used_numbers) == sorted(numbers)

    except:
        return False


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

    prompt = f"""Use each of the numbers {numbers[0]}, {numbers[1]}, {numbers[2]}, {numbers[3]} exactly once with operations +, -, *, / to get 24. Give your answer as a single mathematical expression."""

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

        # Try to extract expression from the answer
        # The model might give explanation, so try to find the expression
        lines = answer.split('\n')
        for line in lines:
            line = line.strip()
            # Skip empty lines or lines that are clearly explanatory
            if not line or line.endswith(':') or line.startswith('Answer'):
                continue
            # Try to verify this line as an expression
            if verify_expression(numbers, line):
                return answer, True

        # If no line worked individually, try the whole answer
        if verify_expression(numbers, answer):
            return answer, True

        return answer, False

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
