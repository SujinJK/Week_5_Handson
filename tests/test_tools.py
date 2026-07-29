"""Tests for the calculator tool, especially that it rejects anything beyond
plain arithmetic -- it parses expressions with ast, not eval(), specifically
so model-supplied input can never reach arbitrary code execution.

`calculator` is now a LangChain `BaseTool` (built via the `@tool` decorator
in tools.py) rather than a plain function, so it's called with `.invoke({...})`
and a dict of named arguments -- the same shape a model's tool call sends --
instead of a positional string argument like Week 4's version.
"""
import pytest

from tools import calculator


def _calc(expression: str) -> str:
    return calculator.invoke({"expression": expression})


class TestCalculatorArithmetic:
    def test_addition(self):
        assert _calc("2 + 3") == "5"

    def test_operator_precedence(self):
        assert _calc("2 + 3 * 4") == "14"

    def test_parentheses(self):
        assert _calc("(2 + 3) * 4") == "20"

    def test_division(self):
        assert _calc("7 / 2") == "3.5"

    def test_power(self):
        assert _calc("2 ** 10") == "1024"

    def test_modulo(self):
        assert _calc("10 % 3") == "1"

    def test_negative_numbers(self):
        assert _calc("-5 + 3") == "-2"

    def test_decimal_numbers(self):
        assert _calc("(49.99 * 2) - 15") == "84.98"


class TestCalculatorRejectsUnsafeInput:
    def test_rejects_name_lookup(self):
        assert _calc("x + 1").startswith("Error:")

    def test_rejects_function_call(self):
        assert _calc("__import__('os').system('echo hi')").startswith("Error:")

    def test_rejects_attribute_access(self):
        assert _calc("().__class__").startswith("Error:")

    def test_rejects_string_literal(self):
        assert _calc("'a' + 'b'").startswith("Error:")

    def test_rejects_garbage_input(self):
        assert _calc("not even math").startswith("Error:")
