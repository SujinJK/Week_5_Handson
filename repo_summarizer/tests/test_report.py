"""Tests for report.py's HTML rendering. Checks that real summary content
ends up in the output (escaped where needed) and that both a "good"
health status and a risky one produce valid, non-crashing output --
doesn't check exact visual styling, just that the render is correct and
safe.
"""
from repo_summarizer.agent import RepoSummary
from repo_summarizer.report import render_html


def _sample_summary(**overrides) -> RepoSummary:
    defaults = dict(
        name="octocat/Hello-World",
        purpose="A minimal demo repository used to teach Git and GitHub basics.",
        main_language="None",
        key_files=["README"],
        health="active",
        beginner_friendly=True,
        summary="A tiny, famous placeholder repo with no real source code.",
    )
    defaults.update(overrides)
    return RepoSummary(**defaults)


class TestRenderHtml:
    def test_includes_repo_name_and_link(self):
        html_out = render_html(_sample_summary())
        assert "octocat/Hello-World" in html_out
        assert 'href="https://github.com/octocat/Hello-World"' in html_out

    def test_includes_purpose_and_summary_text(self):
        summary = _sample_summary()
        html_out = render_html(summary)
        assert summary.purpose in html_out
        assert summary.summary in html_out

    def test_includes_key_files(self):
        html_out = render_html(_sample_summary(key_files=["README.md", "src/main.py"]))
        assert "README.md" in html_out
        assert "src/main.py" in html_out

    def test_handles_empty_key_files(self):
        html_out = render_html(_sample_summary(key_files=[]))
        assert "(none listed)" in html_out

    def test_escapes_html_in_free_text_fields(self):
        html_out = render_html(_sample_summary(purpose="Uses <script>alert(1)</script> tags"))
        assert "<script>alert(1)</script>" not in html_out
        assert "&lt;script&gt;" in html_out

    def test_each_health_status_renders_its_own_pill_class(self):
        for status in ("active", "stale", "unmaintained"):
            html_out = render_html(_sample_summary(health=status))
            assert f"pill-{status}" in html_out

    def test_output_is_well_formed_enough_to_be_valid_html(self):
        html_out = render_html(_sample_summary())
        assert html_out.strip().startswith("<!doctype html>")
        assert html_out.strip().endswith("</html>")
