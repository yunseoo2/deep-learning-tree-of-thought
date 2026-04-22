"""
CoT (Chain-of-Thought) Prompting for Game of 24
Prompt asks model to reason step-by-step before giving final answer.
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

    prompt = f"""Use each of the numbers {numbers[0]}, {numbers[1]}, {numbers[2]}, {numbers[3]} exactly once with operations +, -, *, / to get 24.

Think step by step:
1. Consider different ways to combine the numbers
2. Evaluate intermediate results
3. Work towards getting 24

Give your final answer as a single mathematical expression at the end."""

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

        # Try to extract expression from the answer
        # Look for the final expression (often at the end or after "Final answer:")
        lines = answer.split('\n')

        # Try lines in reverse order (final answer usually at end)
        for line in reversed(lines):
            line = line.strip()
            # Skip empty lines or section headers
            if not line or line.endswith(':') or line.startswith('Step') or line.startswith('#'):
                continue
            # Try to verify this line as an expression
            if verify_expression(numbers, line):
                return answer, True

        # Also try the whole answer
        if verify_expression(numbers, answer):
            return answer, True

        return answer, False

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
