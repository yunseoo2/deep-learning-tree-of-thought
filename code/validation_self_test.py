from __future__ import annotations

from code.validation import validate_24_expression


def _run(numbers: list[int], s: str) -> None:
    res = validate_24_expression(numbers, s)
    print(f"nums={numbers} expr={s!r} ok={res.ok} value={res.value} err={res.error}")


def main() -> None:
    # Basic success
    _run([1, 3, 4, 6], "6/(1-3/4)")  # 24

    # Extraction from response text
    _run([1, 3, 4, 6], "Reasoning...\nFinal: 6/(1-3/4)\n")

    # Wrong value
    _run([1, 3, 4, 6], "6/(1-3/4)+1")

    # Wrong numbers (extra literal)
    _run([1, 3, 4, 6], "6/(1-3/4)+0")

    # Repeated number mismatch
    _run([1, 1, 4, 6], "6/(1-1/4)")

    # Disallowed syntax
    _run([1, 3, 4, 6], "__import__('os').system('echo nope')")

    # Float literals rejected (strict)
    _run([1, 3, 4, 6], "6/(1-0.75)")

    # Division by zero
    _run([1, 3, 4, 6], "6/(1-1)")


if __name__ == "__main__":
    main()

