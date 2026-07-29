"""The agent's three tools, rebuilt as LangChain tools instead of Week 4's
hand-rolled dict-based schemas.

For readers new to this: `@tool` is LangChain's decorator that turns a
plain Python function into a `BaseTool` object -- it reads the function's
type hints and docstring to auto-generate the JSON schema a model needs to
call it, instead of us writing that schema by hand (compare `TOOLS` in
Week 4's tools.py, which spells out `input_schema` manually for every
tool). Binding a list of these to a chat model (`llm.bind_tools([...])`,
see agent.py) is what lets the model request any of them by name.

One deliberate change from Week 4: `web_search` there was Anthropic's
server-side tool (`{"type": "web_search_20260209", ...}`) -- Claude's own
infrastructure ran it, which is what caused most of Week 4's trickiest
bugs (mixed client/server tool blocks, `pause_turn`, the `container_id`
BadRequestError). Here it's `DuckDuckGoSearchRun`, a plain LangChain
community tool that runs like the other two -- we call it, we get a
string back, no special-casing anywhere in the loop. That's a genuine
simplification this rebuild gets from moving off Anthropic's proprietary
server-side tool and onto LangChain's uniform tool interface, not just
from using LangChain in general -- worth knowing the two aren't the same
thing. See the README for the corresponding line-count/complexity diff in
agent.py.
"""
import ast
import operator

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

from rag_tool import search_knowledge_base as _search_knowledge_base

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
    no attribute access -- so the model can never smuggle arbitrary code
    through the expression string the way a bare eval() would allow."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression component: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a plain arithmetic expression. Use this for any math instead of
    computing it yourself -- addition, subtraction, multiplication, division,
    powers, modulo, parentheses. Example: '(49.99 * 2) - 15'."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except Exception as exc:
        return f"Error: could not evaluate '{expression}' ({exc})"


@tool
def search_knowledge_base(query: str) -> str:
    """Search Nimbus Cloud Storage's internal policy documents (employee
    handbook, security policy, refund policy, product FAQ, incident runbook).
    Call this when the question depends on Nimbus-specific policy, pricing,
    or procedure -- do not answer those from general knowledge."""
    return _search_knowledge_base(query)


# A plain client-side LangChain tool -- unlike Week 4's server-side web_search,
# nothing here runs on Anthropic's infrastructure; it's a regular function
# call like the two tools above, so the agent loop treats all three identically.
web_search = DuckDuckGoSearchRun()

TOOLS = [calculator, search_knowledge_base, web_search]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
