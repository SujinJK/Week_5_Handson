"""GitHub-repo-reading tools for the repo-summarizer agent -- a second,
independent exercise of the same tool-binding + structured-output skills
as ../agent.py (the Nimbus agent), applied to a completely different kind
of tool surface: live calls to a real external API instead of a local RAG
retriever. No local corpus, no vector store -- every tool here fetches
fresh data over the network each time it's called.

Uses GitHub's REST API directly via `requests` -- no GitHub SDK, no
LangChain community GitHub toolkit. That toolkit (`GitHubToolkit` in
`langchain_community`) is built for managing one repo you own through a
GitHub App -- a mismatch for "read and explain any public repo the user
pastes in," which is what this agent actually needs. See the main
README's comparison section for the fuller reasoning.

Unauthenticated requests work for any public repo, capped at 60/hour.
Setting GITHUB_TOKEN in .env (a plain personal access token, `public_repo`
read scope is enough) raises that to 5,000/hour -- see _headers() below.
"""
import base64
import os

import requests
from langchain_core.tools import tool

API_ROOT = "https://api.github.com"


def _headers() -> dict:
    """Auth header if a token is configured, otherwise anonymous (rate-limited
    to 60 requests/hour instead of 5,000 -- fine for occasional testing, but
    easy to hit if you run this agent against several repos back-to-back)."""
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get(path: str, params: dict | None = None) -> requests.Response:
    """One GET call to the GitHub API. Returning the raw response (rather
    than raising) lets each tool below decide how to turn a 404 or a
    rate-limit error into a plain-English message the model can react to,
    instead of the whole agent crashing on an unhandled exception."""
    return requests.get(f"{API_ROOT}{path}", headers=_headers(), params=params, timeout=15)


@tool
def get_repo_metadata(owner: str, repo: str) -> str:
    """Get a GitHub repository's high-level metadata: description, author
    (the owning user/organization), primary language, license, star/fork
    counts, open issue count, when the repo was created, and when it was
    last pushed to. Call this first for a quick overview of any repo."""
    response = _get(f"/repos/{owner}/{repo}")
    if response.status_code == 404:
        return f"Error: repository '{owner}/{repo}' not found (check the owner/repo spelling)."
    if response.status_code != 200:
        return f"Error: GitHub API returned {response.status_code}: {response.text[:200]}"
    data = response.json()
    license_name = (data.get("license") or {}).get("name", "none")
    owner_login = (data.get("owner") or {}).get("login", owner)
    owner_type = (data.get("owner") or {}).get("type", "unknown")
    return (
        f"name: {data['full_name']}\n"
        f"description: {data.get('description') or '(none)'}\n"
        f"author: {owner_login} ({owner_type})\n"
        f"primary_language: {data.get('language') or 'unknown'}\n"
        f"license: {license_name}\n"
        f"stars: {data['stargazers_count']}\n"
        f"forks: {data['forks_count']}\n"
        f"open_issues: {data['open_issues_count']}\n"
        f"default_branch: {data['default_branch']}\n"
        f"repo_created: {data['created_at']}\n"
        f"last_pushed: {data['pushed_at']}\n"
        f"archived: {data['archived']}"
    )


@tool
def get_readme(owner: str, repo: str) -> str:
    """Fetch and decode a GitHub repository's README file (if it has one).
    Call this to understand what the project claims to do, in its own words."""
    response = _get(f"/repos/{owner}/{repo}/readme")
    if response.status_code == 404:
        return "No README found in this repository."
    if response.status_code != 200:
        return f"Error: GitHub API returned {response.status_code}: {response.text[:200]}"
    data = response.json()
    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    # READMEs can be long; truncate rather than flood the model's context
    # window with the whole thing when a summary only needs the gist.
    if len(content) > 6000:
        content = content[:6000] + "\n...[truncated]"
    return content


@tool
def get_repo_structure(owner: str, repo: str, path: str = "") -> str:
    """List the files and folders at a given path in a GitHub repository
    (root if path is empty). Returns each entry's name and whether it's a
    file or a directory. Call this to explore a repo's layout one level at
    a time -- call it again with a subfolder's path to look inside it,
    rather than expecting a full recursive tree back from one call. (A
    real recursive tree for a large repo could be thousands of entries --
    deliberately not offered here, so the model has to explore
    deliberately instead of being handed a wall of irrelevant paths.)"""
    response = _get(f"/repos/{owner}/{repo}/contents/{path}")
    if response.status_code == 404:
        return f"Error: path '{path}' not found in {owner}/{repo}."
    if response.status_code != 200:
        return f"Error: GitHub API returned {response.status_code}: {response.text[:200]}"
    data = response.json()
    # The GitHub "contents" endpoint returns a list for a directory but a
    # single object for a file -- same URL shape, different response shape.
    if isinstance(data, dict):
        return f"'{path}' is a file, not a directory. Use get_file_contents instead."
    lines = [f"{'dir ' if entry['type'] == 'dir' else 'file'}  {entry['name']}" for entry in data]
    return "\n".join(lines) if lines else "(empty directory)"


@tool
def get_file_contents(owner: str, repo: str, path: str) -> str:
    """Fetch a specific file's contents from a GitHub repository, e.g.
    'package.json', 'requirements.txt', or 'src/main.py'. Call this once
    get_repo_structure has told you which file is worth reading."""
    response = _get(f"/repos/{owner}/{repo}/contents/{path}")
    if response.status_code == 404:
        return f"Error: file '{path}' not found in {owner}/{repo}."
    if response.status_code != 200:
        return f"Error: GitHub API returned {response.status_code}: {response.text[:200]}"
    data = response.json()
    if isinstance(data, list):
        return f"'{path}' is a directory, not a file. Use get_repo_structure instead."
    if data.get("encoding") != "base64":
        return f"Error: file '{path}' isn't returned as plain text (encoding: {data.get('encoding')})."
    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    if len(content) > 4000:
        content = content[:4000] + "\n...[truncated]"
    return content


@tool
def get_commit_history(owner: str, repo: str, limit: int = 10) -> str:
    """Get a GitHub repository's most recent commits (message, author, date)
    to gauge how actively it's maintained. Defaults to the 10 most recent."""
    response = _get(f"/repos/{owner}/{repo}/commits", params={"per_page": min(limit, 30)})
    if response.status_code == 404:
        return f"Error: repository '{owner}/{repo}' not found."
    if response.status_code != 200:
        return f"Error: GitHub API returned {response.status_code}: {response.text[:200]}"
    commits = response.json()
    if not commits:
        return "No commits found."
    lines = []
    for commit_entry in commits:
        message = commit_entry["commit"]["message"].splitlines()[0]
        author = commit_entry["commit"]["author"]["name"]
        date = commit_entry["commit"]["author"]["date"]
        lines.append(f"{date}  {author}: {message}")
    return "\n".join(lines)


# Same purpose as the Nimbus agent's TOOLS/TOOLS_BY_NAME in ../tools.py:
# TOOLS is handed to bind_tools(); TOOLS_BY_NAME turns a tool call's name
# (a string) back into the actual function to run.
TOOLS = [get_repo_metadata, get_readme, get_repo_structure, get_file_contents, get_commit_history]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
