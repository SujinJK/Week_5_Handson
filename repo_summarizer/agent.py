"""Repo-summarizer agent: point it at a public GitHub repo, and it explores
just enough of it -- choosing which tools to call, and in what order,
based on what each call tells it -- to produce a structured summary.

A second, independent exercise of the same tool-binding + structured-
output skills as ../agent.py (the Nimbus agent), applied to a completely
different tool surface: live external API calls instead of a local RAG
retriever, and no fixed corpus -- the "knowledge base" here is whatever
the model decides to fetch about whichever repo you give it.

Deliberately simpler than the Nimbus agent: just the tool-calling loop
and structured output, no separate planning call and no reflection/
critique step. That's not a downgrade -- it's a focused second look at
the same two building blocks (bind_tools, with_structured_output)
without re-deriving the fuller four-pattern architecture a second time.
See the main README for a side-by-side comparison of the two agents.

Run:
    python -m repo_summarizer.agent octocat/Hello-World
"""
import pathlib
import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import Literal

from repo_summarizer.github_tools import TOOLS, TOOLS_BY_NAME

REPORTS_DIR = pathlib.Path(__file__).parent / "reports"

# Same fix as ../agent.py: a fetched README or file can contain Unicode
# characters (e.g. the arrows in this project's own README, U+2192) that
# Windows' default console codepage (cp1252) can't print -- without this,
# printing a tool-call preview containing one crashes the whole run instead
# of just showing the wrong glyph. Confirmed live: summarizing this very
# repo's own README is exactly what triggered it before this fix.
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

MODEL = "claude-opus-4-8"
MAX_TOOL_ITERATIONS = 8

SYSTEM_PROMPT = (
    "You are a repo-explaining assistant. Given a GitHub repository "
    "(owner/repo), figure out what it does and how healthy it looks by "
    "calling tools -- you decide which ones you need and in what order, "
    "based on what each call tells you. Typical patterns: start with "
    "get_repo_metadata and get_readme for a quick overview; if the README "
    "is missing, thin, or you need the tech stack, use get_repo_structure "
    "to see the top-level layout, then get_file_contents on whatever looks "
    "like a dependency manifest (package.json, requirements.txt, "
    "Cargo.toml, pyproject.toml, go.mod, etc.); use get_commit_history to "
    "judge how actively maintained it is. You don't need to call every "
    "tool -- a repo with a good README may need nothing else. Once you "
    "have enough to answer confidently, stop calling tools."
)


class RepoSummary(BaseModel):
    """The structured verdict this agent produces. Passing this class to
    with_structured_output() is what guarantees the final answer comes
    back as an object matching this exact shape, instead of free text
    that would need to be parsed by hand -- see ../agent.py's Critique
    class for the same pattern applied to a review verdict instead."""

    name: str = Field(description="owner/repo")
    purpose: str = Field(description="One or two sentences: what this project is and does.")
    main_language: str = Field(
        description="A short label only, e.g. 'Python', 'JavaScript', or 'None' -- "
        "not a sentence. Put any nuance (e.g. 'docs-only, no source code') in `summary` instead."
    )
    key_files: list[str] = Field(description="The 1-4 files most worth reading first to understand this repo.")
    health: Literal["active", "stale", "unmaintained"] = Field(
        description="Best-effort read from recent commit activity and open issue count -- not a certainty."
    )
    beginner_friendly: bool = Field(description="Does it have a README/docs clear enough for a newcomer to get started?")
    summary: str = Field(description="A short paragraph tying the above together for a human reader.")


# Two separate runnables, same pattern as the Nimbus agent: one with tools
# bound (for the exploration loop), one wrapped for structured output (for
# the final answer). Neither is piped into the other with LCEL's `|` --
# each is called standalone, in sequence, from explore_repo() below.
_llm_with_tools = ChatAnthropic(model=MODEL, max_tokens=2048).bind_tools(TOOLS)
_structured_llm = ChatAnthropic(model=MODEL, max_tokens=1024).with_structured_output(RepoSummary)


def explore_repo(owner_repo: str) -> RepoSummary:
    """Run the tool-calling loop to gather information about a repo, then
    ask a second, structured-output call to summarize what was found."""
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(f"Explain this GitHub repository: {owner_repo}"),
    ]

    for _ in range(MAX_TOOL_ITERATIONS):
        response = _llm_with_tools.invoke(messages)
        messages = messages + [response]

        if not response.tool_calls:
            break

        for call in response.tool_calls:
            tool_fn = TOOLS_BY_NAME[call["name"]]
            result = tool_fn.invoke(call["args"])
            preview = result if len(result) <= 200 else result[:200] + "..."
            print(f"  -> {call['name']}({call['args']})")
            print(f"  <- {preview}")
            messages = messages + [ToolMessage(content=str(result), tool_call_id=call["id"])]
    else:
        print(f"Warning: stopped after {MAX_TOOL_ITERATIONS} tool calls without settling on a final answer.")

    # A second call, in structured-output mode, over the same conversation
    # the loop above already built up -- reusing everything that was
    # fetched instead of asking the model to re-explain from scratch.
    messages = messages + [HumanMessage("Now summarize everything you found about this repo.")]
    return _structured_llm.invoke(messages)


def main() -> None:
    # Imported here, not at module level, to avoid a circular import --
    # report.py imports RepoSummary from this module, so this module can't
    # import report.py at the top without the two files importing each
    # other before either has finished loading.
    from repo_summarizer.report import render_html

    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        print("Usage: python -m repo_summarizer.agent <owner/repo>")
        print("Example: python -m repo_summarizer.agent octocat/Hello-World")
        return

    owner_repo = sys.argv[1]
    print(f"Exploring {owner_repo}...\n")
    summary = explore_repo(owner_repo)

    print("\nSUMMARY:")
    for field, value in summary.model_dump().items():
        print(f"  {field}: {value}")

    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / f"{owner_repo.replace('/', '_')}.html"
    report_path.write_text(render_html(summary), encoding="utf-8")
    print(f"\nHTML report written to {report_path}")


if __name__ == "__main__":
    main()
