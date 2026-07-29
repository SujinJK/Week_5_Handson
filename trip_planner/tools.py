"""Tools for the trip budget planner agent: research (web_search) + exact
math (budget_calculator) -- the classic "combine fuzzy info with precise
computation" agentic pattern.

A deliberate substitution worth being explicit about: the original idea
for this agent called for a "Python REPL" tool so the model could do
arbitrary budget math. We use a restricted, AST-based calculator instead
(the same design as ../tools.py's `calculator`, copied here rather than
imported, so this agent stays fully self-contained like repo_summarizer).

Why not a real REPL: this agent also has `web_search` bound in the same
tool-calling loop, and web search results are untrusted third-party text
that flows straight into the model's context. A genuine Python REPL tool
(running arbitrary code via exec()) bound alongside that is a real
prompt-injection-to-code-execution risk, not a hypothetical one -- a
sufficiently adversarial search result could try to talk the model into
running harmful code through the REPL, on your actual machine, with your
actual file system. Splitting a fixed budget across categories, computing
percentages, and summing totals never needs a full REPL's power anyway --
the same restricted grammar (numbers, + - * / ** % //, parentheses) this
project has used since Week 4 covers 100% of what a budget breakdown
actually requires, with none of the exec() risk.
"""
import ast
import operator

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a restricted arithmetic AST -- only numbers, the
    operators above, and parentheses are legal. No names, no function calls,
    no attribute access, so nothing the model writes here can ever reach
    arbitrary code execution, unlike a real Python REPL would allow."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression component: {ast.dump(node)}")


@tool
def budget_calculator(expression: str) -> str:
    """Evaluate a plain arithmetic expression -- use this for ALL budget
    math: splitting a total across categories, computing percentages,
    summing a breakdown to check it matches the total budget. Never
    compute or estimate numbers yourself; always call this tool instead.
    Supports + - * / ** % and parentheses. Example: '1500 * 0.35' or
    '520 + 480 + 300 + 200'."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except Exception as exc:
        return f"Error: could not evaluate '{expression}' ({exc})"


# Same client-side web search tool as ../tools.py -- used here for
# researching typical flight/hotel/food price ranges at a destination,
# not for anything Nimbus-specific.
web_search = DuckDuckGoSearchRun()

TOOLS = [web_search, budget_calculator]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
