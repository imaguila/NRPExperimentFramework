# Location: src/domain/core/generators/precedence_generator.py
"""
Generic Directed Acyclic Graph (DAG) Precedence Generator.
"""

import random
import copy
from typing import List, Dict
from src.domain.core.model import OptimizationModel


def generate_random_dag(item_ids: List[str], density: float = 0.2) -> List[Dict[str, str]]:
    """
    Genera una lista de precedencias válidas (DAG) asegurando que no existan ciclos.
    
    :param item_ids: Lista de IDs de los elementos (ej. ['R1', 'R2', 'R3']).
    :param density: Probabilidad (0.0 a 1.0) de que exista relación entre dos elementos.
    :return: Lista de diccionarios -> [{'req1': 'R1', 'req2': 'R2'}, ...]
    """
    nodes = [str(x) for x in item_ids]
    precedences = []
    n = len(nodes)

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < density:
                # nodes[i] debe realizarse ANTES que nodes[j]
                precedences.append({"req1": nodes[i], "req2": nodes[j]})

    return precedences


def inject_precedences_into_model(model: OptimizationModel, density: float = 0.2) -> OptimizationModel:
    """
    Toma un OptimizationModel, genera precedencias para sus ítems
    y las inyecta en model.relationships["precedences"].
    
    :param model: Modelo original (no se modifica)
    :param density: Densidad de precedencias
    :return: Nueva instancia del modelo con precedencias inyectadas
    """
    # Crear una copia profunda del modelo para no mutar el original
    new_model = copy.deepcopy(model)
    
    item_ids = list(new_model.items.keys())
    new_precedences = generate_random_dag(item_ids, density=density)
    
    if not isinstance(new_model.relationships, dict):
        new_model.relationships = {}
        
    new_model.relationships["precedences"] = new_precedences
    return new_model