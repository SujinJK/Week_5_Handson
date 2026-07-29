"""Trip budget planner agent: give it a free-text trip request, and it
combines fuzzy web research (typical price ranges) with exact computation
(splitting the budget across categories, checking the totals add up) to
produce a structured travel budget plan.

A third, independent exercise of bind_tools() + with_structured_output(),
alongside the Nimbus agent (../agent.py, all four patterns, local RAG)
and the repo summarizer (../repo_summarizer/agent.py, tool-calling loop +
structured output over a live external API). This one's the classic
"research + compute" agentic pattern: one tool returns fuzzy, unstructured
information (web search results), the other must be trusted for anything
that needs to be exactly right (the arithmetic) -- forcing the model to
route each sub-task to the tool actually suited for it, not blend the two.

Run:
    python -m trip_planner.agent "Tokyo for 5 days with a $1500 budget"
"""
import pathlib
import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import Literal

from trip_planner.tools import TOOLS, TOOLS_BY_NAME

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

MODEL = "claude-opus-4-8"
MAX_TOOL_ITERATIONS = 8
REPORTS_DIR = pathlib.Path(__file__).parent / "reports"

SYSTEM_PROMPT = (
    "You are a trip budget planning assistant. The user will describe a "
    "trip in plain language -- a destination, a number of days, a total "
    "budget, and sometimes an origin/departure city (extract whichever of "
    "these are present, from whatever phrasing they use). Your job: "
    "1) Use `web_search` to research realistic price ranges for flights, "
    "accommodation, food, and local transport/activities at that "
    "destination -- run separate searches per category rather than one "
    "vague query. If an origin city was stated, search for flight prices "
    "specifically from that origin. If it wasn't stated, pick one clearly-"
    "labeled reasonable assumption (e.g. a major hub) for research "
    "purposes, explain that assumption in `notes`, and set the `origin` "
    "field itself to 'Not specified' rather than presenting the assumed "
    "city as if the user had stated it. "
    "2) Use `budget_calculator` for every number you report -- splitting "
    "the total budget across categories, converting a percentage, and "
    "summing the breakdown to check it actually adds up to your total "
    "estimated cost. Never compute or estimate a number yourself; always "
    "call the tool. "
    "3) Decide feasibility by comparing your computed total estimated "
    "cost to the user's stated budget: 'over_budget' if the estimate "
    "exceeds it, 'tight' if it's within about 10% under it, 'comfortable' "
    "otherwise. "
    "Stop calling tools once you have researched enough price ranges and "
    "your budget breakdown has been verified with the calculator to sum "
    "correctly."
)


class TripPlan(BaseModel):
    """The structured plan this agent produces. Same with_structured_output()
    pattern as the other two agents in this project -- see ../agent.py's
    Critique or ../repo_summarizer/agent.py's RepoSummary."""

    origin: str = Field(
        description="The traveler's departure city/country, if the user stated one. "
        "If not stated, use exactly 'Not specified' here -- put any city assumed for "
        "research purposes in `notes` instead, clearly labeled as an assumption."
    )
    destination: str
    days: int = Field(description="Number of days for the trip.")
    total_budget: float = Field(description="The user's stated total budget, in the currency they gave (assume USD if unstated).")
    budget_breakdown: dict[str, float] = Field(
        description="Estimated cost per category, e.g. {'flights': 520.0, 'hotel': 480.0, 'food': 300.0, 'activities': 200.0}. "
        "Must be computed with budget_calculator, and must sum to total_estimated_cost."
    )
    total_estimated_cost: float = Field(description="Sum of budget_breakdown's values -- computed with budget_calculator, not estimated.")
    feasibility: Literal["tight", "comfortable", "over_budget"]
    notes: str = Field(description="Caveats: e.g. that web prices are typical ranges, not live quotes; seasonal variation; unstated origin city.")


_llm_with_tools = ChatAnthropic(model=MODEL, max_tokens=3000).bind_tools(TOOLS)
_structured_llm = ChatAnthropic(model=MODEL, max_tokens=1200).with_structured_output(TripPlan)


def plan_trip(request: str) -> TripPlan:
    """Run the tool-calling loop to research and compute a budget plan,
    then ask a second, structured-output call to produce the final plan."""
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(request),
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

    messages = messages + [HumanMessage("Now give the final structured trip budget plan.")]
    return _structured_llm.invoke(messages)


def main() -> None:
    from trip_planner.report import render_html

    if len(sys.argv) != 2:
        print('Usage: python -m trip_planner.agent "<trip request>"')
        print('Example: python -m trip_planner.agent "Tokyo for 5 days with a $1500 budget"')
        return

    request = sys.argv[1]
    print(f"Planning: {request}...\n")
    plan = plan_trip(request)

    print("\nPLAN:")
    for field, value in plan.model_dump().items():
        print(f"  {field}: {value}")

    REPORTS_DIR.mkdir(exist_ok=True)
    slug = plan.destination.lower().replace(" ", "_").replace(",", "")
    report_path = REPORTS_DIR / f"{slug}.html"
    report_path.write_text(render_html(plan), encoding="utf-8")
    print(f"\nHTML report written to {report_path}")


if __name__ == "__main__":
    main()
