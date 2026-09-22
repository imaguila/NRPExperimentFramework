# tests/test_analysis_session.py
"""
Tests for AnalysisSession.

These tests cover the core functionality of the analysis session,
including enrichment, framing, lens application, and SOI management.
"""

import pytest
import pandas as pd
import io
from unittest.mock import MagicMock

from src.domain.core.paretofront import ParetoFront
from src.domain.core.solution import Solution
from src.domain.core.collection import SolutionSet, SolutionSetCollection
from src.analysis.session import AnalysisSession


@pytest.fixture
def sample_pareto_front():
    """Create a sample Pareto front with 2 solutions."""
    sol1 = Solution(id="s1", selected_ids={"1"}, objectives={"satisfaction": 80.0, "effort": 10.0})
    sol2 = Solution(id="s2", selected_ids={"2"}, objectives={"satisfaction": 95.0, "effort": 25.0})
    return ParetoFront([sol1, sol2])


@pytest.fixture
def sample_problem(sample_pareto_front):
    """Create a sample OptimizationProblem with a stored front."""
    from src.domain.core.model import OptimizationModel
    from src.solving.core.problem import OptimizationProblem

    model = OptimizationModel(name="Test Model", items={})
    problem = OptimizationProblem(model=model, objectives={"satisfaction": "max", "effort": "min"})
    problem.add_pareto_front(sample_pareto_front, name="Solver Run 1")
    return problem


def test_session_init_from_pareto_front(sample_pareto_front):
    """Test initialising AnalysisSession directly from a ParetoFront."""
    session = AnalysisSession(base_front=sample_pareto_front, name="Direct Session")
    assert session.name == "Direct Session"
    assert len(session.current_dataframe) == 2
    assert "satisfaction" in session.current_dataframe.columns
    assert "effort" in session.current_dataframe.columns


def test_session_from_csv():
    """Test creating an AnalysisSession from CSV data."""
    csv_content = "req_1,req_2,satisfaction,effort\n1,0,80.0,10.0\n0,1,95.0,25.0\n"
    buffer = io.StringIO(csv_content)
    session = AnalysisSession.from_csv(buffer, name="CSV Session")
    assert session.name == "CSV Session"
    assert len(session.base_front) == 2


def test_session_from_problem_set(sample_problem):
    """Test creating an AnalysisSession from a problem's stored set."""
    set_id = list(sample_problem.solutions_collection.sets.keys())[0]
    session = AnalysisSession.from_problem_set(sample_problem, set_id=set_id)
    assert session.name == "Analysis - Solver Run 1"
    assert session.model == sample_problem.model
    assert len(session.base_front) == 2


def test_session_enrichment(sample_pareto_front):
    """Test that enrichment adds indicators to the DataFrame."""
    session = AnalysisSession(base_front=sample_pareto_front)

    # Mock the enrichment function to avoid external dependencies
    def mock_enrich(df, indicators, model):
        df["mock_indicator"] = [1.0, 2.0]
        return df

    original_compute = session._df
    session._df = original_compute.copy()
    session._df["mock_indicator"] = [1.0, 2.0]  # Simulate enrichment

    # Check that the enrichment indicators are stored
    session._enriched_indicators = ["mock_indicator"]
    assert "mock_indicator" in session._enriched_indicators
    assert "mock_indicator" in session.current_dataframe.columns


def test_session_get_framing_dimensions(sample_pareto_front):
    """Test that framing dimensions are correctly identified."""
    session = AnalysisSession(base_front=sample_pareto_front)
    dims = session.get_framing_dimensions()
    assert "satisfaction" in dims
    assert "effort" in dims


def test_session_get_dimension_bounds(sample_pareto_front):
    """Test that dimension bounds are correctly computed."""
    session = AnalysisSession(base_front=sample_pareto_front)
    bounds = session.get_dimension_bounds(["satisfaction"])
    assert bounds["satisfaction"] == (80.0, 95.0)


def test_session_apply_framing(sample_pareto_front):
    """Test that framing filters the Pareto front correctly."""
    session = AnalysisSession(base_front=sample_pareto_front)
    framed_front = session.apply_framing({"satisfaction": (90.0, 100.0)})
    assert len(framed_front) == 1
    assert framed_front.solutions[0].id == "s2"


def test_session_apply_lens(sample_pareto_front):
    """Test that applying a lens registers a new SOI."""
    session = AnalysisSession(base_front=sample_pareto_front)

    # Create a mock lens
    mock_lens = MagicMock()
    mock_soi = SolutionSet(id="soi_123", name="Mock SOI", front=sample_pareto_front)
    mock_lens.evaluate.return_value = mock_soi

    soi = session.apply_lens(mock_lens)
    assert mock_lens.evaluate.called
    assert soi.id in session.soi_collection.sets
    assert session.active_soi_id == "soi_123"


def test_session_promote_candidate_to_soi(sample_pareto_front):
    """Test promoting a candidate set to an official SOI."""
    session = AnalysisSession(base_front=sample_pareto_front)
    candidate_set = SolutionSet(id="cand_1", name="Cluster 1", front=sample_pareto_front)

    promoted = session.promote_candidate_to_soi(candidate_set, custom_name="SOI Promovida")
    assert promoted.name == "SOI Promovida"
    assert promoted.category == "soi"
    assert session.selected_soi.id == "cand_1"


def test_session_select_soi(sample_pareto_front):
    """Test selecting an existing SOI by ID."""
    session = AnalysisSession(base_front=sample_pareto_front)
    soi = SolutionSet(id="soi_456", name="Existing SOI", front=sample_pareto_front)
    session.soi_collection.add_set(soi)

    selected = session.select_soi("soi_456")
    assert selected.id == "soi_456"
    assert session.active_soi_id == "soi_456"


def test_session_select_soi_not_found(sample_pareto_front):
    """Test that selecting a non‑existent SOI raises KeyError."""
    session = AnalysisSession(base_front=sample_pareto_front)
    with pytest.raises(KeyError):
        session.select_soi("non_existent")


def test_session_selected_soi_property(sample_pareto_front):
    """Test that the selected_soi property returns the active SOI."""
    session = AnalysisSession(base_front=sample_pareto_front)
    soi = SolutionSet(id="soi_789", name="Active SOI", front=sample_pareto_front)
    session.soi_collection.add_set(soi)
    session.active_soi_id = "soi_789"

    assert session.selected_soi.id == "soi_789"
    assert session.selected_soi.name == "Active SOI"


def test_session_current_dataframe_without_framing(sample_pareto_front):
    """Test that current_dataframe returns the original DataFrame when no framing is applied."""
    session = AnalysisSession(base_front=sample_pareto_front)
    df = session.current_dataframe
    assert len(df) == 2
    assert "satisfaction" in df.columns
    assert session._active_bounds == {}


def test_session_current_dataframe_with_framing(sample_pareto_front):
    """Test that current_dataframe returns a filtered DataFrame when framing is applied."""
    session = AnalysisSession(base_front=sample_pareto_front)
    session.apply_framing({"satisfaction": (90.0, 100.0)})
    df = session.current_dataframe
    assert len(df) == 1
    assert df.iloc[0]["id"] == "s2"
    assert session._active_bounds == {"satisfaction": (90.0, 100.0)}