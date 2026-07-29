"""Tests for budget_calculator -- same safety contract as the original
calculator in ../tools.py: correct arithmetic, and hard rejection of
anything that isn't plain arithmetic (since that's exactly the boundary
that keeps this tool safe to use instead of a real Python REPL -- see the
module docstring in tools.py for why that substitution was made)."""
from trip_planner.tools import budget_calculator


def _calc(expression: str) -> str:
    return budget_calculator.invoke({"expression": expression})


class TestBudgetCalculatorArithmetic:
    def test_percentage_split(self):
        assert _calc("1500 * 0.35") == "525.0"

    def test_sum_of_categories(self):
        assert _calc("520 + 480 + 300 + 200") == "1500"

    def test_remaining_budget(self):
        assert _calc("1500 - 1420") == "80"

    def test_division(self):
        assert _calc("1500 / 5") == "300.0"


class TestBudgetCalculatorRejectsUnsafeInput:
    def test_rejects_name_lookup(self):
        assert _calc("os.getenv('SECRET')").startswith("Error:")

    def test_rejects_function_call(self):
        assert _calc("__import__('os').system('echo hi')").startswith("Error:")

    def test_rejects_attribute_access(self):
        assert _calc("().__class__").startswith("Error:")

    def test_rejects_string_literal(self):
        assert _calc("'a' + 'b'").startswith("Error:")
