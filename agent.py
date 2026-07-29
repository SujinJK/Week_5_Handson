"""Week 4's hand-rolled agent, rebuilt on LangChain's core building blocks.

Same four patterns as Week 4, same corpus, same underlying retriever --
only the scaffolding around them changes, so the two are a fair
line-count/reliability comparison:

  1. PLANNING   -- a plain `llm.invoke()` call, no tools bound.
  2. TOOL USE   -- `llm.bind_tools(TOOLS)` + a loop reading
                   `response.tool_calls` instead of manually inspecting
                   `stop_reason` and raw content blocks.
  3. REFLECTION -- `llm.with_structured_output(Critique)` returns a parsed
                   Pydantic object directly -- no manual `json.loads()`,
                   no hand-written JSON schema.
  4. MULTI-AGENT-- same worker/critic split as Week 4: two system prompts,
                   two distinct jobs.

What's gone, and why (see README for the full comparison): no
server-side-tool detection, no `pause_turn` handling, no `container_id`
recovery workaround -- all of that existed in Week 4 solely because
`web_search` ran on Anthropic's infrastructure. Swapping it for a plain
LangChain community tool (see tools.py) removes that whole category of
complexity; `bind_tools` + `with_structured_output` remove the rest.

Run:
    python ingest.py   # once, to build the vector store (see rag_tool.py)
    python agent.py     # interactive loop
"""
import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import Literal

# Same Windows console fix as Week 4 -- Claude's output can contain Unicode
# characters (e.g. a proper minus sign, U+2212) that Windows' default
# console codepage (cp1252) can't print.
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import ingest
from tools import TOOLS, TOOLS_BY_NAME

load_dotenv()

MODEL = "claude-opus-4-8"
MAX_TOOL_ITERATIONS = 10
MAX_REFLECTION_CYCLES = 2

PLAN_SYSTEM_PROMPT = (
    "You are the planning step of an agent with three tools: `calculator` "
    "(arithmetic), `search_knowledge_base` (searches whatever documents are "
    "currently indexed in the local knowledge base -- contents vary, so try "
    "it for any question that might depend on indexed documents rather than "
    "assuming a fixed topic), and `web_search` (the public web). Given the "
    "user's question, write a short plan -- 2 to 4 bullet points -- naming "
    "which of these tools you expect to need, in what order, and why. Do "
    "not answer the question yet, and do not call any tool. Just the plan."
)

AGENT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to three tools: `calculator`, "
    "`search_knowledge_base` (searches whatever documents are currently "
    "indexed in the local knowledge base), and `web_search` (the public "
    "web). Follow the plan you were given, calling tools as needed -- do "
    "not guess at facts that might be covered by the knowledge base without "
    "checking search_knowledge_base first. When you cite a "
    "search_knowledge_base snippet, reference its number like this: [1]. "
    "Once you are confident in your final answer, state it directly "
    "instead of calling more tools."
)

CRITIC_SYSTEM_PROMPT = (
    "You are a strict reviewer checking another agent's draft answer "
    "before it reaches the user. Given the original question and the "
    "draft answer, check: (1) does it actually answer what was asked, "
    "(2) is every factual or policy claim grounded in a cited source "
    "rather than guessed, (3) is any calculation or lookup missing that "
    "the question requires. Respond with verdict 'approve' if the answer "
    "is good as-is, or 'revise' with a list of specific, actionable issues "
    "if not."
)


class Critique(BaseModel):
    """Structured verdict from the reflection step. Passing this class to
    `with_structured_output()` is what replaces Week 4's hand-written
    `CRITIQUE_SCHEMA` dict and manual `json.loads(response.text)` -- the
    model's output is parsed straight into this object, guaranteed to match
    the shape, or the call raises instead of returning malformed JSON."""

    verdict: Literal["approve", "revise"]
    issues: list[str] = Field(default_factory=list)


_llm = ChatAnthropic(model=MODEL, max_tokens=4096)
_llm_with_tools = _llm.bind_tools(TOOLS)
_critic_llm = ChatAnthropic(model=MODEL, max_tokens=500).with_structured_output(Critique)


def _text(message: AIMessage) -> str:
    """Extract plain text from a response. LangChain's `.content` is usually
    already a string, but can be a list of content blocks (e.g. if the
    model mixes text with other block types) -- concatenate defensively
    rather than risk Week 4's "only read the first block" bug recurring."""
    if isinstance(message.content, str):
        return message.content
    return "".join(block.get("text", "") for block in message.content if isinstance(block, dict) and block.get("type") == "text")


def make_plan(question: str) -> str:
    """Planning step: ask Claude to sketch its approach before it acts."""
    response = _llm.invoke([SystemMessage(PLAN_SYSTEM_PROMPT), HumanMessage(question)])
    return _text(response)


def run_agent_loop(messages: list) -> tuple[str, list]:
    """The act/observe loop: call the model, and while it's asking for a
    tool, run it and feed the result back. `response.tool_calls` is
    LangChain's normalized list of pending calls regardless of provider --
    no `stop_reason` check, no distinguishing block types, because every
    tool here is client-side (see tools.py for why that's true this time)."""
    for _ in range(MAX_TOOL_ITERATIONS):
        response = _llm_with_tools.invoke(messages)
        messages = messages + [response]

        if not response.tool_calls:
            return _text(response), messages

        for call in response.tool_calls:
            tool_fn = TOOLS_BY_NAME[call["name"]]
            result = tool_fn.invoke(call["args"])
            preview = result if len(result) <= 200 else result[:200] + "..."
            print(f"  -> {call['name']}({call['args']})")
            print(f"  <- {preview}")
            messages = messages + [ToolMessage(content=str(result), tool_call_id=call["id"])]

    raise RuntimeError(f"Agent did not finish within {MAX_TOOL_ITERATIONS} tool iterations")


def critique_answer(question: str, answer: str) -> Critique:
    """Reflection step: a second, differently-prompted call reviews the
    draft answer, returning an already-validated Critique object."""
    return _critic_llm.invoke([
        SystemMessage(CRITIC_SYSTEM_PROMPT),
        HumanMessage(f"Question: {question}\n\nDraft answer:\n{answer}"),
    ])


def answer_question(question: str) -> str:
    """Run one full agent turn: plan, act (with tools), reflect, and retry
    at most MAX_REFLECTION_CYCLES times if the critic flags real issues."""
    print("\nPLAN:")
    plan = make_plan(question)
    print(plan)

    messages = [
        SystemMessage(AGENT_SYSTEM_PROMPT),
        HumanMessage(f"Question: {question}\n\nYour plan:\n{plan}"),
    ]

    print("\nACTING:")
    answer, messages = run_agent_loop(messages)

    for cycle in range(1, MAX_REFLECTION_CYCLES + 1):
        print("\nREFLECTION:")
        critique = critique_answer(question, answer)
        print(f"  verdict={critique.verdict}")
        for issue in critique.issues:
            print(f"  - {issue}")

        if critique.verdict == "approve" or cycle == MAX_REFLECTION_CYCLES:
            break

        feedback = "; ".join(critique.issues) or "Double-check completeness and grounding."
        messages = messages + [
            HumanMessage(f"A reviewer flagged issues with your answer: {feedback}\nAddress them and give a final answer.")
        ]
        print("\nACTING (retry):")
        answer, messages = run_agent_loop(messages)

    print(f"\nFINAL ANSWER:\n{answer}\n")
    return answer


def main() -> None:
    """Entry point for `python agent.py` -- the interactive question loop."""
    if not ingest.DB_DIR.exists():
        print("No vector store found. Run `python ingest.py` first.")
        return

    print("Nimbus Agent (LangChain rebuild). Ask a question, or /quit to exit.\n")
    while True:
        question = input("you> ").strip()
        if not question:
            continue
        if question in ("/quit", "/exit"):
            break
        answer_question(question)


if __name__ == "__main__":
    main()
