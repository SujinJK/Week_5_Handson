"""Renders a RepoSummary as a self-contained HTML report -- an alternative
to reading the structured fields as plain terminal text. No external
fonts, scripts, or stylesheets: everything needed is inlined, so the
output file works standalone when opened directly in a browser, offline,
with no build step.
"""
import html
from datetime import datetime, timezone

from repo_summarizer.agent import RepoSummary

_HEALTH_LABELS = {
    "active": "Active",
    "stale": "Stale",
    "unmaintained": "Unmaintained",
}


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def render_html(summary: RepoSummary) -> str:
    """Build the report page for one RepoSummary. Called once per run from
    agent.py's main() -- see there for how the output file is named and
    where it's saved."""
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    owner, _, repo = summary.name.partition("/")
    repo_url = f"https://github.com/{summary.name}"

    key_files_html = "\n".join(
        f'            <li>{_esc(f)}</li>' for f in summary.key_files
    ) or '            <li class="muted">(none listed)</li>'

    beginner_label = "Beginner-friendly" if summary.beginner_friendly else "Not beginner-friendly"
    beginner_class = "pill-good" if summary.beginner_friendly else "pill-neutral"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(summary.name)} -- repo summary</title>
<style>
  :root {{
    --bg: #eff2f5;
    --surface: #ffffff;
    --border: #dde3e8;
    --text: #1a2027;
    --text-muted: #5b6672;
    --accent: #0e7c6b;
    --accent-soft: #e3f3f0;
    --health-active-bg: #e6f6ec;    --health-active-fg: #1f7a43;
    --health-stale-bg: #fbf0dc;     --health-stale-fg: #92620a;
    --health-unmaintained-bg: #fbe7e5; --health-unmaintained-fg: #b23b2e;
    --pill-neutral-bg: #eceff2;     --pill-neutral-fg: #4b5563;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #10141a;
      --surface: #171d24;
      --border: #2a323c;
      --text: #e7ebef;
      --text-muted: #97a2ad;
      --accent: #46d9c0;
      --accent-soft: rgba(70, 217, 192, 0.12);
      --health-active-bg: rgba(52, 199, 89, 0.15);       --health-active-fg: #4ade80;
      --health-stale-bg: rgba(251, 191, 36, 0.15);       --health-stale-fg: #fbbf24;
      --health-unmaintained-bg: rgba(248, 113, 113, 0.15); --health-unmaintained-fg: #f87171;
      --pill-neutral-bg: #232a33;                         --pill-neutral-fg: #b7c0c9;
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
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, "Segoe UI", system-ui, Roboto, sans-serif;
    line-height: 1.6;
    padding: 48px 20px 64px;
  }}
  .card {{
    max-width: 760px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 28px;
  }}
  .mono {{
    font-family: ui-monospace, "Cascadia Code", "JetBrains Mono", "SF Mono", Consolas, monospace;
  }}
  .eyebrow {{
    font-size: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 600;
  }}
  h1 {{
    margin: 4px 0 0;
    font-size: clamp(24px, 4vw, 36px);
    font-weight: 600;
    text-wrap: balance;
    word-break: break-word;
  }}
  h1 a {{
    color: inherit;
    text-decoration: none;
    border-bottom: 1px solid var(--border);
    transition: border-color 0.15s ease;
  }}
  h1 a:hover, h1 a:focus-visible {{
    border-color: var(--accent);
  }}
  .pills {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
  }}
  .pill::before {{
    content: "";
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: currentColor;
    flex-shrink: 0;
  }}
  .pill-active {{ background: var(--health-active-bg); color: var(--health-active-fg); }}
  .pill-stale {{ background: var(--health-stale-bg); color: var(--health-stale-fg); }}
  .pill-unmaintained {{ background: var(--health-unmaintained-bg); color: var(--health-unmaintained-fg); }}
  .pill-neutral {{ background: var(--pill-neutral-bg); color: var(--pill-neutral-fg); }}
  .pill-good {{ background: var(--accent-soft); color: var(--accent); }}

  section {{ display: flex; flex-direction: column; gap: 10px; }}
  .label {{
    font-size: 12.5px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-muted);
    font-weight: 600;
  }}
  p {{ margin: 0; font-size: 16.5px; }}
  .purpose {{ font-size: 18px; }}

  ul.files {{
    margin: 0;
    padding: 0;
    list-style: none;
    display: flex;
    flex-direction: column;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }}
  ul.files li {{
    padding: 10px 14px;
    font-size: 14.5px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }}
  ul.files li:last-child {{ border-bottom: none; }}
  ul.files li.muted {{ color: var(--text-muted); font-style: italic; }}

  .summary-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 22px;
  }}

  footer {{
    border-top: 1px solid var(--border);
    padding-top: 16px;
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 8px;
    font-size: 12.5px;
    color: var(--text-muted);
  }}

  @media (prefers-reduced-motion: reduce) {{
    * {{ transition: none !important; }}
  }}
</style>
</head>
<body>
  <div class="card">
    <div>
      <div class="eyebrow mono">Repo Summary</div>
      <h1 class="mono"><a href="{_esc(repo_url)}">{_esc(summary.name)}</a></h1>
    </div>

    <div class="pills">
      <span class="pill pill-{_esc(summary.health)}">{_esc(_HEALTH_LABELS.get(summary.health, summary.health))}</span>
      <span class="pill pill-neutral">{_esc(summary.main_language)}</span>
      <span class="pill {beginner_class}">{_esc(beginner_label)}</span>
    </div>

    <section>
      <div class="label">Purpose</div>
      <p class="purpose">{_esc(summary.purpose)}</p>
    </section>

    <section>
      <div class="label">Key files to read first</div>
      <ul class="files mono">
{key_files_html}
      </ul>
    </section>

    <section>
      <div class="label">Summary</div>
      <div class="summary-card">
        <p>{_esc(summary.summary)}</p>
      </div>
    </section>

    <footer class="mono">
      <span>Generated {_esc(generated_at)}</span>
      <span>repo_summarizer -- Week 5 LangChain project</span>
    </footer>
  </div>
</body>
</html>
"""
