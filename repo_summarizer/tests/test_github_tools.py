"""Tests for github_tools.py. Every GitHub API call is mocked (via
unittest.mock.patch on requests.get), so these run instantly, don't need
network access, and never touch the real rate limit -- they check that
each tool parses a realistic response correctly and handles the 404/error
cases sanely, not that GitHub's API itself works.
"""
import base64
from unittest.mock import MagicMock, patch

from repo_summarizer.github_tools import (
    get_commit_history,
    get_file_contents,
    get_readme,
    get_repo_metadata,
    get_repo_structure,
)


def _mock_response(status_code=200, json_data=None, text=""):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data
    response.text = text
    return response


class TestGetRepoMetadata:
    @patch("repo_summarizer.github_tools.requests.get")
    def test_returns_key_fields(self, mock_get):
        mock_get.return_value = _mock_response(200, {
            "full_name": "octocat/Hello-World",
            "description": "My first repository on GitHub!",
            "language": "Python",
            "license": {"name": "MIT License"},
            "stargazers_count": 100,
            "forks_count": 10,
            "open_issues_count": 2,
            "default_branch": "main",
            "pushed_at": "2024-01-01T00:00:00Z",
            "archived": False,
        })
        result = get_repo_metadata.invoke({"owner": "octocat", "repo": "Hello-World"})
        assert "octocat/Hello-World" in result
        assert "Python" in result
        assert "MIT License" in result

    @patch("repo_summarizer.github_tools.requests.get")
    def test_handles_missing_license(self, mock_get):
        mock_get.return_value = _mock_response(200, {
            "full_name": "o/r", "description": None, "language": None, "license": None,
            "stargazers_count": 0, "forks_count": 0, "open_issues_count": 0,
            "default_branch": "main", "pushed_at": "2024-01-01T00:00:00Z", "archived": False,
        })
        result = get_repo_metadata.invoke({"owner": "o", "repo": "r"})
        assert "license: none" in result

    @patch("repo_summarizer.github_tools.requests.get")
    def test_handles_404(self, mock_get):
        mock_get.return_value = _mock_response(404)
        result = get_repo_metadata.invoke({"owner": "nope", "repo": "doesnotexist"})
        assert result.startswith("Error:")


class TestGetReadme:
    @patch("repo_summarizer.github_tools.requests.get")
    def test_decodes_base64_content(self, mock_get):
        raw = "# Hello World\nThis is a test repo."
        encoded = base64.b64encode(raw.encode()).decode()
        mock_get.return_value = _mock_response(200, {"content": encoded, "encoding": "base64"})
        result = get_readme.invoke({"owner": "octocat", "repo": "Hello-World"})
        assert result == raw

    @patch("repo_summarizer.github_tools.requests.get")
    def test_truncates_long_readmes(self, mock_get):
        raw = "x" * 7000
        encoded = base64.b64encode(raw.encode()).decode()
        mock_get.return_value = _mock_response(200, {"content": encoded, "encoding": "base64"})
        result = get_readme.invoke({"owner": "o", "repo": "r"})
        assert len(result) < 7000
        assert result.endswith("[truncated]")

    @patch("repo_summarizer.github_tools.requests.get")
    def test_handles_missing_readme(self, mock_get):
        mock_get.return_value = _mock_response(404)
        result = get_readme.invoke({"owner": "octocat", "repo": "Hello-World"})
        assert "No README" in result


class TestGetRepoStructure:
    @patch("repo_summarizer.github_tools.requests.get")
    def test_lists_files_and_dirs(self, mock_get):
        mock_get.return_value = _mock_response(200, [
            {"name": "src", "type": "dir"},
            {"name": "README.md", "type": "file"},
        ])
        result = get_repo_structure.invoke({"owner": "o", "repo": "r", "path": ""})
        assert "src" in result
        assert "README.md" in result

    @patch("repo_summarizer.github_tools.requests.get")
    def test_rejects_file_path(self, mock_get):
        mock_get.return_value = _mock_response(200, {"name": "README.md", "type": "file"})
        result = get_repo_structure.invoke({"owner": "o", "repo": "r", "path": "README.md"})
        assert "is a file" in result

    @patch("repo_summarizer.github_tools.requests.get")
    def test_handles_empty_directory(self, mock_get):
        mock_get.return_value = _mock_response(200, [])
        result = get_repo_structure.invoke({"owner": "o", "repo": "r", "path": "empty"})
        assert result == "(empty directory)"


class TestGetFileContents:
    @patch("repo_summarizer.github_tools.requests.get")
    def test_decodes_file_content(self, mock_get):
        raw = '{"name": "test-package"}'
        encoded = base64.b64encode(raw.encode()).decode()
        mock_get.return_value = _mock_response(200, {"content": encoded, "encoding": "base64"})
        result = get_file_contents.invoke({"owner": "o", "repo": "r", "path": "package.json"})
        assert result == raw

    @patch("repo_summarizer.github_tools.requests.get")
    def test_rejects_directory_path(self, mock_get):
        mock_get.return_value = _mock_response(200, [{"name": "src", "type": "dir"}])
        result = get_file_contents.invoke({"owner": "o", "repo": "r", "path": "src"})
        assert "is a directory" in result

    @patch("repo_summarizer.github_tools.requests.get")
    def test_handles_404(self, mock_get):
        mock_get.return_value = _mock_response(404)
        result = get_file_contents.invoke({"owner": "o", "repo": "r", "path": "missing.txt"})
        assert result.startswith("Error:")


class TestGetCommitHistory:
    @patch("repo_summarizer.github_tools.requests.get")
    def test_formats_commits(self, mock_get):
        mock_get.return_value = _mock_response(200, [
            {"commit": {"message": "Fix bug\n\nLonger details here", "author": {"name": "Alice", "date": "2024-01-01T00:00:00Z"}}},
        ])
        result = get_commit_history.invoke({"owner": "o", "repo": "r"})
        assert "Fix bug" in result
        assert "Alice" in result
        assert "Longer details here" not in result  # only the message's first line

    @patch("repo_summarizer.github_tools.requests.get")
    def test_handles_no_commits(self, mock_get):
        mock_get.return_value = _mock_response(200, [])
        result = get_commit_history.invoke({"owner": "o", "repo": "r"})
        assert result == "No commits found."
