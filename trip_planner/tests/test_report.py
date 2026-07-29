"""Tests for report.py's HTML rendering -- checks real content ends up in
the output (escaped where needed), the feasibility-to-pill-class mapping
is correct for all three states, and the over/under-budget line reflects
the right sign, without checking exact visual styling.
"""
from trip_planner.agent import TripPlan
from trip_planner.report import render_html


def _sample_plan(**overrides) -> TripPlan:
    defaults = dict(
        origin="Los Angeles",
        destination="Tokyo",
        days=5,
        total_budget=1500.0,
        budget_breakdown={"flights": 520.0, "hotel": 480.0, "food": 300.0, "activities": 200.0},
        total_estimated_cost=1500.0,
        feasibility="comfortable",
        notes="Prices are typical ranges, not live quotes; origin city assumed unstated.",
    )
    defaults.update(overrides)
    return TripPlan(**defaults)


class TestRenderHtml:
    def test_includes_destination_and_days(self):
        html_out = render_html(_sample_plan())
        assert "Tokyo" in html_out
        assert "5" in html_out

    def test_shows_origin_when_stated(self):
        html_out = render_html(_sample_plan(origin="Los Angeles"))
        assert "From Los Angeles" in html_out

    def test_shows_not_specified_note_when_origin_unstated(self):
        html_out = render_html(_sample_plan(origin="Not specified"))
        assert "origin not specified" in html_out
        assert "From Not specified" not in html_out

    def test_includes_budget_breakdown_categories_and_amounts(self):
        html_out = render_html(_sample_plan())
        assert "flights" in html_out
        assert "$520" in html_out

    def test_includes_notes_text(self):
        plan = _sample_plan()
        html_out = render_html(plan)
        assert plan.notes in html_out

    def test_escapes_html_in_notes(self):
        html_out = render_html(_sample_plan(notes="Watch for <script>alert(1)</script>"))
        assert "<script>alert(1)</script>" not in html_out
        assert "&lt;script&gt;" in html_out

    def test_each_feasibility_state_renders_its_own_pill_class(self):
        expected = {
            "comfortable": "pill-active",
            "tight": "pill-stale",
            "over_budget": "pill-unmaintained",
        }
        for feasibility, pill_class in expected.items():
            html_out = render_html(_sample_plan(feasibility=feasibility))
            assert pill_class in html_out

    def test_over_budget_shows_over_line(self):
        html_out = render_html(_sample_plan(total_budget=1500.0, total_estimated_cost=1700.0))
        assert "$200 over" in html_out

    def test_under_budget_shows_under_line(self):
        html_out = render_html(_sample_plan(total_budget=1500.0, total_estimated_cost=1300.0))
        assert "$200 under" in html_out

    def test_handles_empty_breakdown_without_crashing(self):
        html_out = render_html(_sample_plan(budget_breakdown={}, total_estimated_cost=0.0))
        assert "no breakdown available" in html_out

    def test_output_is_well_formed_enough_to_be_valid_html(self):
        html_out = render_html(_sample_plan())
        assert html_out.strip().startswith("<!doctype html>")
        assert html_out.strip().endswith("</html>")
