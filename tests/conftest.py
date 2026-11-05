# Location: tests/conftest.py
"""
Fixtures y configuraciones globales para la suite de pruebas.
"""

import sys
from pathlib import Path

# Añade la raíz del proyecto al sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.domain.core.item import DecisionItem
from src.domain.core.model import OptimizationModel
from src.domain.core.solution import Solution
from src.domain.core.paretofront import ParetoFront
from src.domain.core.collection import SolutionSet, SolutionSetCollection


@pytest.fixture
def sample_items():
    """Genera un diccionario de items de prueba."""
    return {
        "R1": DecisionItem("R1", "Requisito 1", {"cost": 10.0, "time": 5.0, "satisfaction": 80.0}),
        "R2": DecisionItem("R2", "Requisito 2", {"cost": 20.0, "time": 10.0, "satisfaction": 50.0}),
        "R3": DecisionItem("R3", "Requisito 3", {"cost": 15.0, "time": 8.0, "satisfaction": 60.0}),
        "R4": DecisionItem("R4", "Requisito 4", {"cost": 30.0, "time": 12.0, "satisfaction": 90.0}),
        "R5": DecisionItem("R5", "Requisito 5", {"cost": 25.0, "time": 4.0, "satisfaction": 40.0}),
    }


@pytest.fixture
def base_model(sample_items):
    """Genera un OptimizationModel básico con relaciones diversas."""
    relationships = {
        "precedences": [
            {"req1": "R1", "req2": "R2"},  # R2 depende de R1
            {"req1": "R2", "req2": "R3"},  # R3 depende de R2
        ],
        "couplings": [
            {"req1": "R4", "req2": "R5"}   # R4 y R5 se deben incluir juntos
        ],
        "exclusions": [
            {"req1": "R1", "req2": "R4"}   # R1 y R4 son mutuamente excluyentes
        ],
        "value_dependencies": [
            {
                "reqs": ["R4", "R5"],
                "attribute": "cost",
                "effect": "delta",
                "factor": -5.0  # Sinergia: si van juntos, el costo se reduce en 5
            },
            {
                "reqs": ["R1", "R3"],
                "attribute": "satisfaction",
                "effect": "multiplier",
                "factor": 1.2  # Sinergia: si van juntos, la satisfacción sube 20%
            }
        ]
    }
    return OptimizationModel(
        name="Test_NRP_Model",
        items=sample_items,
        relationships=relationships,
        evaluators={"stakeholders": [{"id": "ClientA", "weight": 1.0}], "developers": [{"id": "DevB", "weight": 1.0}]}
    )


@pytest.fixture
def sample_pareto_front():
    """Genera un ParetoFront con 2 soluciones para pruebas."""
    sol1 = Solution(id="s1", selected_ids={"1"}, objectives={"satisfaction": 80.0, "effort": 10.0})
    sol2 = Solution(id="s2", selected_ids={"2"}, objectives={"satisfaction": 95.0, "effort": 25.0})
    return ParetoFront([sol1, sol2])