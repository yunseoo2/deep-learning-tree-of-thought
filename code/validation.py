from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    value: Fraction | None
    error: str | None = None


class _SafeArithmeticEvaluator(ast.NodeVisitor):
    """
    Safely evaluates arithmetic expressions over + - * / and parentheses.

    Notes:
    - Uses Fraction for exactness (no float tolerance issues).
    - Rejects everything except numeric literals and allowed operators.
    """

    def __init__(self) -> None:
        self.numbers: list[Fraction] = []

    def visit_Expression(self, node: ast.Expression) -> Fraction:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> Fraction:
        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right

        raise ValueError(f"disallowed operator: {type(node.op).__name__}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Fraction:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"disallowed unary operator: {type(node.op).__name__}")

    def visit_Constant(self, node: ast.Constant) -> Fraction:
        # Python parses "3" as int, "3.0" as float.
        if isinstance(node.value, bool) or node.value is None:
            raise ValueError("disallowed constant")
        if isinstance(node.value, int):
            val = Fraction(node.value, 1)
        elif isinstance(node.value, float):
            # Keep exact literal meaning; e.g. 0.1 is not representable as Fraction nicely.
            # For our project, decimal literals are not expected; reject to be strict.
            raise ValueError("float literals are not allowed; use integers only")
        else:
            raise ValueError("only integer literals are allowed")

        self.numbers.append(val)
        return val

    # Disallow names, calls, attributes, subscripts, etc.
    def generic_visit(self, node: ast.AST) -> Fraction:  # type: ignore[override]
        raise ValueError(f"disallowed syntax: {type(node).__name__}")


_FRAC_RE = re.compile(r"\\d?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")


def _delatex(s: str) -> str:
    """Rewrite common LaTeX math fragments into plain arithmetic."""
    # \frac{a}{b} -> (a)/(b); apply repeatedly to catch nested uses.
    prev = None
    while prev != s:
        prev = s
        s = _FRAC_RE.sub(r"(\1)/(\2)", s)

    replacements = [
        (r"\\times", "*"),
        (r"\\cdot", "*"),
        (r"\\div", "/"),
        (r"\\left", ""),
        (r"\\right", ""),
        (r"\\[\[\]]", ""),
        (r"\\\(", ""),
        (r"\\\)", ""),
        (r"\$", ""),
    ]
    for pat, repl in replacements:
        s = re.sub(pat, repl, s)

    # Strip stray backslashes and LaTeX brackets.
    s = s.replace("\\[", "").replace("\\]", "")
    s = s.replace("[", "(").replace("]", ")")
    s = s.replace("{", "(").replace("}", ")")
    return s


_EXPR_CHARS_RE = re.compile(r"[0-9+\-*/().\s]+")


def _longest_arith_span(s: str) -> str:
    """Return the longest substring containing only arithmetic characters."""
    best = ""
    for match in _EXPR_CHARS_RE.finditer(s):
        candidate = match.group(0).strip()
        if len(candidate) > len(best):
            best = candidate
    return best


def _extract_expression(text: str) -> str:
    """
    Tries to pull an arithmetic expression out of a model response.

    Heuristics (in order):
    - Prefer content after the last 'Final:' marker.
    - Otherwise, scan lines from bottom and return the longest arithmetic span
      found on any non-empty line.
    - Fallback: the last non-empty line as-is.
    """
    s = (text or "").strip()
    if not s:
        return ""

    s = _delatex(s)

    lowered = s.lower()
    if "final" in lowered:
        idx = lowered.rfind("final")
        tail = s[idx:]
        if ":" in tail:
            after_colon = tail.split(":", 1)[1]
            # Take only the first line after 'Final:' to avoid trailing prose.
            after_colon = after_colon.splitlines()[0].strip() if after_colon else ""
            candidate = _longest_arith_span(after_colon)
            if candidate:
                return candidate
            if after_colon:
                return after_colon

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    for line in reversed(lines):
        candidate = _longest_arith_span(line)
        if candidate and any(c.isdigit() for c in candidate):
            return candidate

    return lines[-1] if lines else s


def validate_24_expression(
    numbers: Iterable[int],
    expression_or_response: str,
    *,
    allow_response_extraction: bool = True,
) -> ValidationResult:
    """
    Validates a candidate solution for the Game of 24.

    Checks:
    - Expression parses using only +, -, *, /, parentheses, and integer literals
    - Uses each provided number exactly once (multiset match)
    - Evaluates exactly to 24 (using Fraction arithmetic)
    """
    nums = list(numbers)
    if len(nums) != 4:
        return ValidationResult(ok=False, value=None, error="expected exactly 4 input numbers")

    expr = _extract_expression(expression_or_response) if allow_response_extraction else (expression_or_response or "").strip()
    if not expr:
        return ValidationResult(ok=False, value=None, error="empty expression")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return ValidationResult(ok=False, value=None, error="could not parse expression")

    evaluator = _SafeArithmeticEvaluator()
    try:
        value = evaluator.visit(tree)
    except ZeroDivisionError:
        return ValidationResult(ok=False, value=None, error="division by zero")
    except Exception as e:
        return ValidationResult(ok=False, value=None, error=str(e))

    # Multiset match on integer literals used.
    used_ints = [int(frac) for frac in evaluator.numbers]
    if any(frac.denominator != 1 for frac in evaluator.numbers):
        return ValidationResult(ok=False, value=value, error="non-integer literal detected")

    if Counter(used_ints) != Counter(nums):
        return ValidationResult(
            ok=False,
            value=value,
            error=f"numbers used {sorted(used_ints)} do not match expected {sorted(nums)}",
        )

    if value != 24:
        return ValidationResult(ok=False, value=value, error=f"evaluates to {value}, not 24")

    return ValidationResult(ok=True, value=value, error=None)

