# tests/test_integration_flow.py
"""
Integration test for the full optimisation workflow.

This test verifies the end‑to‑end flow from loading a JSON instance,
through preprocessing, optimisation with a greedy solver, Pareto front
storage, and metrics computation.
"""

import json

from src.domain.core.loader import GenericJSONLoader
from src.domain.core.pipeline import PreprocessingEngine
from src.domain.core.paretofront import ParetoFront
from src.domain.core.solution import Solution
from src.domain.plugins.nrp import NRPPlugin
from src.solving.algorithms.greedy import GreedySolver
from src.solving.core.problem import OptimizationProblem


def test_full_instance_to_front_to_comparison_flow():
    """
    Test the complete workflow from a raw JSON instance to a Pareto front,
    including preprocessing, optimisation, storage, and metrics computation.

    Steps:
        1. Load a raw JSON case using GenericJSONLoader with the NRPPlugin.
        2. Preprocess the model (handling exclusions, generating variants).
        3. Select the first variant and create an optimisation problem.
        4. Run a greedy solver to generate a Pareto front.
        5. Store the front in the problem's collection.
        6. Create a reference front manually and store it.
        7. Verify collection size and metadata.
        8. Restore a front from storage and compute metrics.
    """
    # Define a raw JSON case with 4 requirements, precedences, and exclusions
    raw_case = {
        "name": "Flow Case",
        "requirements": [
            {"id": "R1", "description": "Req 1", "attributes": {"cost": 5, "value": 10}},
            {"id": "R2", "description": "Req 2", "attributes": {"cost": 7, "value": 12}},
            {"id": "R3", "description": "Req 3", "attributes": {"cost": 6, "value": 9}},
            {"id": "R4", "description": "Req 4", "attributes": {"cost": 3, "value": 8}},
        ],
        "relationships": {
            "precedences": [{"from": "R1", "to": "R2"}],
            "exclusions": [{"req1": "R2", "req2": "R4"}],
        },
        "attribute_definitions": {
            "cost": {"type": "scalar", "coupling_rule": "sum", "aggregation_rule": "sum", "evaluation_rule": "sum"},
            "value": {"type": "scalar", "coupling_rule": "sum", "aggregation_rule": "sum", "evaluation_rule": "sum"},
        },
    }

    # 1. Load the model from the JSON string
    base_model = GenericJSONLoader.load(json.dumps(raw_case), plugin=NRPPlugin)

    # 2. Preprocess the model (resolves exclusions, generating multiple variants)
    variants = PreprocessingEngine.process(initial_model=base_model)

    # Expect more than one variant due to the exclusion constraint
    assert len(variants) > 1, "Expected more than one variant due to exclusions"

    # 3. Select the first variant and create an optimisation problem
    selected_variant = variants[0]
    problem = OptimizationProblem(
        model=selected_variant,
        objectives={"cost": "min", "value": "max"},
    )

    # 4. Run the greedy solver to generate a Pareto front
    solver = GreedySolver(weight_samples=5)
    front = solver.solve(problem)
    front.set_objective_directions(problem.get_objective_directions())

    # Verify that the front is non‑empty and directions are correctly set
    assert len(front) > 0
    assert front.objective_directions == {"cost": "min", "value": "max"}

    # 5. Store the generated front in the problem's collection
    saved = problem.add_pareto_front(front, name="Greedy Run", algorithm="GreedySolver")

    # 6. Create a reference front manually and store it
    reference = ParetoFront([
        Solution(id="manual_a", selected_ids={"R1"}, objectives={"cost": 5.0, "value": 10.0}, is_feasible=True),
        Solution(id="manual_b", selected_ids={"R3"}, objectives={"cost": 6.0, "value": 9.0}, is_feasible=True),
    ])
    reference.set_objective_directions(problem.get_objective_directions())
    problem.add_pareto_front(reference, name="Reference Front", algorithm="Manual")

    # 7. Verify that both fronts are stored and metadata is correct
    assert len(problem.solutions_collection) == 2
    assert saved.metadata["problem"]["objectives"] == {"cost": "min", "value": "max"}

    # 8. Restore a front from the stored collection and verify directions
    stored = list(problem.solutions_collection.sets.values())[0]
    restored = ParetoFront.from_dict(stored.to_dict()["front"])
    assert restored.objective_directions == {"cost": "min", "value": "max"}

    # 9. Compute quality metrics and verify they are present
    metrics = restored.compute_metrics(directions=restored.objective_directions)
    assert "Hypervolume" in metrics
    assert "Spacing" in metrics
    assert "Spread" in metrics