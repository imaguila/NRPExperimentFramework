# Location: tests/test_paretofront_export.py
"""
Tests for Pareto front export functionality.

These tests verify that a Pareto front can be correctly exported to JSON
and CSV formats, including objective columns and binary decision matrices.
"""

import json
import csv
import pytest
from pathlib import Path

from src.domain.core.paretofront import ParetoFront
from src.domain.core.solution import Solution


@pytest.fixture
def sample_pareto_front():
    """Create a dummy Pareto front with 2 solutions for testing."""
    sol1 = Solution(
        id="sol_1",
        selected_ids={"req_8"},
        objectives={"satisfaction": 1.0, "effort": 1.0, "dissatisfaction": 7.0}
    )
    sol2 = Solution(
        id="sol_2",
        selected_ids={"req_1", "req_3", "req_8"},
        objectives={"satisfaction": 2.5, "effort": 3.0, "dissatisfaction": 4.0}
    )
    return ParetoFront(solutions=[sol1, sol2])


def test_pareto_front_export_json(sample_pareto_front, tmp_path):
    """
    Verify that the Pareto front is correctly exported to JSON.

    The exported file should contain the correct number of solutions,
    and each solution should have the expected fields.
    """
    json_path = tmp_path / "pareto_output.json"

    # 1. Export to JSON in a temporary directory
    sample_pareto_front.export_to_json(json_path)
    assert json_path.exists()

    # 2. Read the file and validate the structure
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_solutions"] == 2
    assert len(data["solutions"]) == 2

    # Verify solution 1
    sol1_data = data["solutions"][0]
    assert sol1_data["solution_id"] == 1
    assert sol1_data["selected_items"] == ["req_8"]
    assert sol1_data["objectives"] == {"satisfaction": 1.0, "effort": 1.0, "dissatisfaction": 7.0}

    # Verify solution 2
    sol2_data = data["solutions"][1]
    assert sol2_data["selected_items"] == ["req_1", "req_3", "req_8"]


def test_pareto_front_export_csv_with_all_items(sample_pareto_front, tmp_path):
    """
    Verify that the CSV export generates the correct objective columns
    and a binary 0/1 matrix for all specified requirements.
    """
    csv_path = tmp_path / "pareto_output.csv"
    all_reqs = [f"req_{i}" for i in range(1, 10)]

    # Export to CSV
    sample_pareto_front.export_to_csv(csv_path, all_item_ids=all_reqs)
    assert csv_path.exists()

    # Read CSV and parse rows
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    # Header: 3 objectives + 9 requirements
    expected_header = ["satisfaction", "effort", "dissatisfaction"] + all_reqs
    assert reader[0] == expected_header

    # Row 1 (sol1: only req_8 selected)
    row1 = reader[1]
    assert row1[:3] == ["1.0", "1.0", "7.0"]
    binary_vector_sol1 = [int(x) for x in row1[3:]]
    assert binary_vector_sol1 == [0, 0, 0, 0, 0, 0, 0, 1, 0]

    # Row 2 (sol2: req_1, req_3, req_8 selected)
    row2 = reader[2]
    binary_vector_sol2 = [int(x) for x in row2[3:]]
    assert binary_vector_sol2 == [1, 0, 1, 0, 0, 0, 0, 1, 0]


def test_pareto_front_export_csv_implicit_items(sample_pareto_front):
    """
    Verify that without `all_item_ids`, the columns are automatically
    deduced from the union of selected items across all solutions.
    """
    csv_str = sample_pareto_front.to_csv()
    lines = csv_str.strip().splitlines()

    header = lines[0].split(",")
    assert header == ["satisfaction", "effort", "dissatisfaction", "req_1", "req_3", "req_8"]