# tests/test_collection_persistence.py
"""
Tests for the AnalysisSession orchestrator.

These tests verify the core functionality of the analysis session,
including initialisation from various sources, framing, lens application,
and SOI (Set of Interest) management.
"""

import io
from unittest.mock import MagicMock
import pytest

from src.domain.core.paretofront import ParetoFront
from src.domain.core.solution import Solution
from src.domain.core.collection import SolutionSetCollection, SolutionSet
from src.domain.core.model import OptimizationModel
from src.solving.core.problem import OptimizationProblem
from src.analysis.session import AnalysisSession


@pytest.fixture
def sample_pareto_front():
    """Create a sample Pareto front with two solutions for testing."""
    sol1 = Solution(id="s1", selected_ids={"1"}, objectives={"satisfaction": 80.0, "effort": 10.0})
    sol2 = Solution(id="s2", selected_ids={"2"}, objectives={"satisfaction": 95.0, "effort": 25.0})
    return ParetoFront([sol1, sol2])


@pytest.fixture
def sample_problem(sample_pareto_front):
    """Create a sample optimisation problem with a stored Pareto front."""
    model = OptimizationModel(name="Test Model", items={})
    problem = OptimizationProblem(
        model=model,
        objectives={"satisfaction": "max", "effort": "min"}
    )
    problem.add_pareto_front(sample_pareto_front, name="Solver Run 1")
    return problem


def test_session_init_from_pareto_front(sample_pareto_front):
    """Test initialising AnalysisSession directly from a ParetoFront."""
    session = AnalysisSession(base_front=sample_pareto_front, name="Direct Session")
    assert session.name == "Direct Session"
    assert len(session.current_dataframe) == 2
    assert "satisfaction" in session.current_dataframe.columns


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


def test_session_framing_pipeline(sample_pareto_front):
    """
    Test the full framing pipeline: dimension detection, bound computation,
    and application of bounds.
    """
    session = AnalysisSession(base_front=sample_pareto_front)

    # Get framing dimensions
    dims = session.get_framing_dimensions()
    assert "satisfaction" in dims

    # Compute bounds for the satisfaction dimension
    bounds = session.get_dimension_bounds(["satisfaction"])
    assert bounds["satisfaction"] == (80.0, 95.0)

    # Apply framing and verify filtering
    framed_front = session.apply_framing({"satisfaction": (90.0, 100.0)})
    assert len(framed_front) == 1
    assert framed_front.solutions[0].id == "s2"


def test_session_apply_lens_integration(sample_pareto_front):
    """Test that applying a lens registers a new SOI in the collection."""
    session = AnalysisSession(base_front=sample_pareto_front)

    # Create a mock lens that returns a predefined SOI
    mock_lens = MagicMock()
    mock_soi = SolutionSet(id="soi_123", name="Mock SOI", front=sample_pareto_front)
    mock_lens.evaluate.return_value = mock_soi

    soi = session.apply_lens(mock_lens)

    # Verify the lens was called and the SOI was stored
    assert mock_lens.evaluate.called
    assert soi.id in session.soi_collection.sets
    assert session.active_soi_id == "soi_123"


def test_session_promote_candidate(sample_pareto_front):
    """Test promoting a candidate set to an official SOI."""
    session = AnalysisSession(base_front=sample_pareto_front)
    candidate_set = SolutionSet(id="cand_1", name="Cluster 1", front=sample_pareto_front)

    promoted = session.promote_candidate_to_soi(candidate_set, custom_name="SOI Promovida")

    assert promoted.name == "SOI Promovida"
    assert promoted.category == "soi"
    assert session.selected_soi.id == "cand_1"