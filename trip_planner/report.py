"""Renders a TripPlan as a self-contained HTML report -- same design
system (color tokens, type pairing) as ../repo_summarizer/report.py, so
the two agents' reports read as one consistent project rather than two
unrelated one-offs, with content specific to a budget plan: a feasibility
pill, a budget-vs-estimate comparison, and each category's cost shown
with a proportional bar rather than just a number.
"""
import html
from datetime import datetime, timezone

from trip_planner.agent import TripPlan

_FEASIBILITY_LABELS = {
    "comfortable": "Comfortable",
    "tight": "Tight",
    "over_budget": "Over budget",
}
# Reuses the same three semantic roles as repo_summarizer's health pills --
# comfortable maps to the "good" role, tight to "caution", over_budget to
# "risk" -- same meaning, different vocabulary for a different domain.
_FEASIBILITY_PILL_CLASS = {
    "comfortable": "pill-active",
    "tight": "pill-stale",
    "over_budget": "pill-unmaintained",
}


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def render_html(plan: TripPlan) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = plan.total_estimated_cost or 1  # guard against a divide-by-zero on an empty breakdown

    rows = []
    for category, amount in plan.budget_breakdown.items():
        pct = max(0.0, min(100.0, (amount / total) * 100))
        rows.append(f"""
        <div class="bar-row">
          <div class="bar-row-label">
            <span>{_esc(category)}</span>
            <span class="mono">${amount:,.0f}</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width: {pct:.1f}%"></div>
          </div>
        </div>""")
    breakdown_html = "".join(rows) or '<p class="muted">(no breakdown available)</p>'

    if plan.origin and plan.origin.strip().lower() != "not specified":
        route_line = f"From {_esc(plan.origin)} &middot; {plan.days} days"
    else:
        route_line = f"{plan.days} days &middot; origin not specified"

    over_under = plan.total_estimated_cost - plan.total_budget
    if over_under > 0:
        diff_line = f'<span class="pill pill-unmaintained">${over_under:,.0f} over</span>'
    else:
        diff_line = f'<span class="pill pill-active">${abs(over_under):,.0f} under</span>'

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(plan.destination)} trip budget</title>
<style>
  :root {{
    --bg: #eff2f5; --surface: #ffffff; --border: #dde3e8; --text: #1a2027; --text-muted: #5b6672;
    --accent: #0e7c6b; --accent-soft: #e3f3f0;
    --health-active-bg: #e6f6ec; --health-active-fg: #1f7a43;
    --health-stale-bg: #fbf0dc; --health-stale-fg: #92620a;
    --health-unmaintained-bg: #fbe7e5; --health-unmaintained-fg: #b23b2e;
    --pill-neutral-bg: #eceff2; --pill-neutral-fg: #4b5563;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #10141a; --surface: #171d24; --border: #2a323c; --text: #e7ebef; --text-muted: #97a2ad;
      --accent: #46d9c0; --accent-soft: rgba(70, 217, 192, 0.12);
      --health-active-bg: rgba(52, 199, 89, 0.15); --health-active-fg: #4ade80;
      --health-stale-bg: rgba(251, 191, 36, 0.15); --health-stale-fg: #fbbf24;
      --health-unmaintained-bg: rgba(248, 113, 113, 0.15); --health-unmaintained-fg: #f87171;
      --pill-neutral-bg: #232a33; --pill-neutral-fg: #b7c0c9;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #10141a; --surface: #171d24; --border: #2a323c; --text: #e7ebef; --text-muted: #97a2ad;
    --accent: #46d9c0; --accent-soft: rgba(70, 217, 192, 0.12);
    --health-active-bg: rgba(52, 199, 89, 0.15); --health-active-fg: #4ade80;
    --health-stale-bg: rgba(251, 191, 36, 0.15); --health-stale-fg: #fbbf24;
    --health-unmaintained-bg: rgba(248, 113, 113, 0.15); --health-unmaintained-fg: #f87171;
    --pill-neutral-bg: #232a33; --pill-neutral-fg: #b7c0c9;
  }}
  :root[data-theme="light"] {{
    --bg: #eff2f5; --surface: #ffffff; --border: #dde3e8; --text: #1a2027; --text-muted: #5b6672;
    --accent: #0e7c6b; --accent-soft: #e3f3f0;
    --health-active-bg: #e6f6ec; --health-active-fg: #1f7a43;
    --health-stale-bg: #fbf0dc; --health-stale-fg: #92620a;
    --health-unmaintained-bg: #fbe7e5; --health-unmaintained-fg: #b23b2e;
    --pill-neutral-bg: #eceff2; --pill-neutral-fg: #4b5563;
  }}

  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: -apple-system, "Segoe UI", system-ui, Roboto, sans-serif;
    line-height: 1.6; padding: 48px 20px 64px;
  }}
  .card {{ max-width: 760px; margin: 0 auto; display: flex; flex-direction: column; gap: 28px; }}
  .mono {{ font-family: ui-monospace, "Cascadia Code", "JetBrains Mono", "SF Mono", Consolas, monospace; }}
  .eyebrow {{ font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); font-weight: 600; }}
  h1 {{ margin: 4px 0 0; font-size: clamp(24px, 4vw, 36px); font-weight: 600; text-wrap: balance; }}
  .pills {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
  .pill {{
    display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px;
    border-radius: 999px; font-size: 13px; font-weight: 600;
  }}
  .pill::before {{ content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; flex-shrink: 0; }}
  .pill-active {{ background: var(--health-active-bg); color: var(--health-active-fg); }}
  .pill-stale {{ background: var(--health-stale-bg); color: var(--health-stale-fg); }}
  .pill-unmaintained {{ background: var(--health-unmaintained-bg); color: var(--health-unmaintained-fg); }}
  .pill-neutral {{ background: var(--pill-neutral-bg); color: var(--pill-neutral-fg); }}

  section {{ display: flex; flex-direction: column; gap: 10px; }}
  .label {{ font-size: 12.5px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--text-muted); font-weight: 600; }}
  p {{ margin: 0; font-size: 16.5px; }}

  .totals-row {{
    display: flex; justify-content: space-between; align-items: baseline;
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px 20px;
  }}
  .totals-row .big {{ font-size: 26px; font-weight: 700; }}
  .totals-row .vs {{ color: var(--text-muted); font-size: 14px; }}

  .bar-row {{ display: flex; flex-direction: column; gap: 5px; }}
  .bar-row-label {{ display: flex; justify-content: space-between; font-size: 14.5px; }}
  .bar-track {{ height: 8px; border-radius: 999px; background: var(--pill-neutral-bg); overflow: hidden; }}
  .bar-fill {{ height: 100%; background: var(--accent); border-radius: 999px; }}

  .notes-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 20px 22px; }}
  .muted {{ color: var(--text-muted); }}

  footer {{
    border-top: 1px solid var(--border); padding-top: 16px; display: flex; flex-wrap: wrap;
    justify-content: space-between; gap: 8px; font-size: 12.5px; color: var(--text-muted);
  }}
  @media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>
</head>
<body>
  <div class="card">
    <div>
      <div class="eyebrow mono">Trip Budget Plan</div>
      <h1>{_esc(plan.destination)}</h1>
      <p class="muted mono" style="font-size: 13.5px; margin-top: 4px;">{route_line}</p>
    </div>

    <div class="pills">
      <span class="pill {_FEASIBILITY_PILL_CLASS.get(plan.feasibility, 'pill-neutral')}">{_esc(_FEASIBILITY_LABELS.get(plan.feasibility, plan.feasibility))}</span>
      {diff_line}
    </div>

    <div class="totals-row mono">
      <div>
        <div class="label">Budget</div>
        <div class="big">${plan.total_budget:,.0f}</div>
      </div>
      <div class="vs">vs.</div>
      <div>
        <div class="label">Estimated cost</div>
        <div class="big">${plan.total_estimated_cost:,.0f}</div>
      </div>
    </div>

    <section>
      <div class="label">Budget breakdown</div>
      {breakdown_html}
    </section>

    <section>
      <div class="label">Notes</div>
      <div class="notes-card">
        <p>{_esc(plan.notes)}</p>
      </div>
    </section>

    <footer class="mono">
      <span>Generated {_esc(generated_at)}</span>
      <span>trip_planner -- Week 5 LangChain project</span>
    </footer>
  </div>
</body>
</html>
"""
