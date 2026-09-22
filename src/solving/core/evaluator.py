# Location: src/solving/core/evaluator.py
"""
Candidate Solution Evaluator.

This module provides the SolutionEvaluator class, which evaluates candidate
solutions against a domain model. It computes attribute values (applying
aggregation rules and value dependencies), checks constraint feasibility,
and constructs Solution objects with full objective and constraint data.
"""

import math
from typing import Set, Dict, Any, List, Tuple
from src.domain.core.model import OptimizationModel
from src.domain.core.solution import Solution
from src.solving.core.problem import ThresholdOperator, ConstraintSpec, OptimizationProblem


def _to_dict(val: Any) -> Dict[str, Any]:
    """
    Convert an object to a dictionary using various serialisation strategies.

    Supports objects with `to_dict()`, `dict()`, `model_dump()` (Pydantic),
    or falls back to `__dict__`. Handles None and dict inputs directly.

    Args:
        val: The object to convert.

    Returns:
        A dictionary representation of the object.
    """
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if hasattr(val, "to_dict") and callable(val.to_dict):
        return val.to_dict()
    if hasattr(val, "dict") and callable(val.dict):
        return val.dict()
    if hasattr(val, "model_dump") and callable(val.model_dump):
        return val.model_dump()
    if hasattr(val, "__dict__"):
        return val.__dict__
    return {}


class SolutionEvaluator:
    """
    Generic evaluator for candidate solutions over a domain model.

    This class provides methods to evaluate attributes, apply value
    dependencies (synergies), check constraints, and create Solution objects
    with full objective and constraint data. It is domain-agnostic and works
    with any OptimizationModel.
    """

    def __init__(self, model: OptimizationModel):
        """
        Initialise the evaluator with a domain model.

        Args:
            model: The OptimizationModel instance to evaluate against.
        """
        self.model = model

    def _get_value_dependencies(self) -> List[Any]:
        """
        Retrieve value dependencies from the model.

        Searches for dependencies in `value_dependencies`, `value_deps`, or
        `dependencies` attributes, or within the `relationships` dictionary.

        Returns:
            List of value dependency objects.
        """
        for k in ["value_dependencies", "value_deps", "dependencies"]:
            if hasattr(self.model, k):
                v = getattr(self.model, k)
                if isinstance(v, list):
                    return v

        rel = getattr(self.model, "relationships", None)
        if rel is not None:
            rel_dict = _to_dict(rel)
            for k in ["value_dependencies", "value_deps", "dependencies"]:
                if k in rel_dict and isinstance(rel_dict[k], list):
                    return rel_dict[k]

        return []

    def _extract_item_attribute(self, item: Any, attribute_name: str) -> float:
        """
        Extract a numeric attribute value from an item.

        First checks top-level properties, then the internal 'attributes' dict.

        Args:
            item: The DecisionItem or dict to extract from.
            attribute_name: Name of the attribute to retrieve.

        Returns:
            The attribute value as a float, or 0.0 if not found.
        """
        if item is None:
            return 0.0

        target_key = str(attribute_name).strip().lower()

        # 1. Check direct top-level properties
        item_dict = _to_dict(item)
        if target_key in item_dict and item_dict[target_key] is not None:
            try:
                return float(item_dict[target_key])
            except (ValueError, TypeError):
                pass

        # 2. Check internal attributes dictionary
        attrs = item_dict.get("attributes")
        if attrs and isinstance(attrs, dict):
            attrs_dict = _to_dict(attrs)
            for k, v in attrs_dict.items():
                if str(k).strip().lower() == target_key and v is not None:
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        pass

        return 0.0

    def _parse_dependency_rule(self, dep_obj: Any) -> Tuple[str, Set[str], str, float]:
        """
        Parse a value dependency rule into its components.

        Args:
            dep_obj: A dependency object (dict or other serialisable form).

        Returns:
            A tuple of (attribute_name, set_of_requirements, effect_type, factor_value).
        """
        dep_dict = _to_dict(dep_obj)

        def get_val(keys: List[str]) -> Any:
            for k in keys:
                if isinstance(dep_dict, dict) and k in dep_dict and dep_dict[k] is not None:
                    return dep_dict[k]
            return None

        attr = str(get_val(["attribute", "attr", "target_attribute", "metric"]) or "").strip().lower()
        raw_reqs = get_val(["reqs", "requirements", "items", "targets", "elements"])
        reqs = set()

        if raw_reqs is not None:
            if isinstance(raw_reqs, (list, tuple, set)):
                reqs = {str(x).strip() for x in raw_reqs}
            elif isinstance(raw_reqs, str):
                reqs = {x.strip() for x in raw_reqs.split(",") if x.strip()}

        effect = str(get_val(["effect", "type", "operation", "action"]) or "").strip().lower()

        num_val = 0.0
        for num_key in ["factor", "value", "bonus", "amount", "delta"]:
            v = get_val([num_key])
            if v is not None:
                try:
                    num_val = float(v)
                    break
                except (ValueError, TypeError):
                    continue

        if not effect:
            effect = "multiplier" if get_val(["factor"]) is not None else "delta"

        return attr, reqs, effect, num_val

    def _aggregate_by_rule(self, values: List[float], rule: str) -> float:
        """
        Aggregate a list of numeric values according to the specified rule.

        Supported rules: 'sum', 'max', 'min', 'mean'. Defaults to 'sum'.

        Args:
            values: List of numeric values.
            rule: Aggregation rule string.

        Returns:
            The aggregated value.
        """
        if not values:
            return 0.0

        if rule == "sum":
            return sum(values)
        elif rule == "max":
            return max(values)
        elif rule == "min":
            return min(values)
        elif rule == "mean":
            return sum(values) / len(values)
        else:
            # Fallback to sum
            return sum(values)

    def evaluate_attribute_detailed(self, selected_ids: Set[str], attribute_name: str) -> Dict[str, Any]:
        """
        Evaluate an attribute with detailed breakdown of calculations.

        Computes the base aggregated value, applies value dependencies
        (multipliers and deltas), and returns a detailed report.

        Args:
            selected_ids: Set of selected item IDs.
            attribute_name: Name of the attribute to evaluate.

        Returns:
            Dictionary with base_value, multipliers, deltas, applied_rules, and final_value.
        """
        if not selected_ids:
            return {
                "base_value": 0.0,
                "multipliers": 1.0,
                "deltas": 0.0,
                "applied_rules": [],
                "final_value": 0.0
            }

        selected_set = {str(i).strip() for i in selected_ids}
        target_attr = str(attribute_name).strip().lower()

        # 1. Retrieve the attribute's evaluation rule
        attr_def = self.model.attribute_definitions.get(target_attr)
        eval_rule = attr_def.evaluation_rule if attr_def else "sum"

        # 2. Extract values from selected items
        values = []
        for item_id in selected_set:
            item = self.model.items.get(item_id) or self.model.items.get(
                int(item_id) if item_id.isdigit() else item_id
            )
            val = self._extract_item_attribute(item, attribute_name)
            values.append(val)

        # 3. Aggregate according to the rule
        total_base = self._aggregate_by_rule(values, eval_rule)

        # 4. Apply value dependencies (synergies)
        value_deps = self._get_value_dependencies()
        multipliers = 1.0
        deltas = 0.0
        applied_rules = []

        for dep in value_deps:
            rule_attr, reqs, effect, val = self._parse_dependency_rule(dep)

            if rule_attr != target_attr:
                continue

            if reqs and reqs.issubset(selected_set):
                applied_rules.append(dep)
                if effect in ["multiplier", "mult", "multiply"]:
                    multipliers *= val
                else:
                    deltas += val

        final_value = (total_base * multipliers) + deltas

        return {
            "base_value": total_base,
            "multipliers": multipliers,
            "deltas": deltas,
            "applied_rules": applied_rules,
            "final_value": final_value
        }

    def evaluate_attribute(self, selected_ids: Set[str], attribute_name: str) -> float:
        """
        Evaluate an attribute and return only the final value.

        Args:
            selected_ids: Set of selected item IDs.
            attribute_name: Name of the attribute to evaluate.

        Returns:
            The final evaluated value.
        """
        return self.evaluate_attribute_detailed(selected_ids, attribute_name)["final_value"]

    def check_constraints(self, selected_ids: Set[str], constraints: Dict[str, Any]) -> bool:
        """
        Check whether a set of selected items satisfies all constraints.

        Validates mutual exclusions and threshold constraints.

        Args:
            selected_ids: Set of selected item IDs.
            constraints: Dictionary of constraints (attribute -> ConstraintSpec).

        Returns:
            True if all constraints are satisfied, False otherwise.
        """
        selected_set = {str(i).strip() for i in selected_ids}

        # 1. Validate mutual exclusions from the model
        rel = getattr(self.model, "relationships", {})
        rel_dict = _to_dict(rel) if rel else {}
        exclusions = rel_dict.get("exclusions", []) if isinstance(rel_dict, dict) else []

        for exc in exclusions:
            exc_dict = _to_dict(exc)
            if isinstance(exc, (tuple, list)) and len(exc) >= 2:
                if str(exc[0]).strip() in selected_set and str(exc[1]).strip() in selected_set:
                    return False
            elif isinstance(exc_dict, dict):
                r1 = str(exc_dict.get("r1") or exc_dict.get("from") or "").strip()
                r2 = str(exc_dict.get("r2") or exc_dict.get("to") or "").strip()
                if r1 and r2 and r1 in selected_set and r2 in selected_set:
                    return False

        # 2. Validate threshold constraints
        for attr, spec in constraints.items():
            val = self.evaluate_attribute(selected_ids, attr)

            if hasattr(spec, "operator") and hasattr(spec, "value"):
                op, target = spec.operator, spec.value
            elif isinstance(spec, dict):
                op, target = spec.get("operator", ThresholdOperator.LESS_EQUAL), spec.get("value")
            else:
                op, target = ThresholdOperator.LESS_EQUAL, float(spec)

            if op in [ThresholdOperator.LESS_EQUAL, "<="] and val > float(target):
                return False
            elif op in [ThresholdOperator.GREATER_EQUAL, ">="] and val < float(target):
                return False
            elif op in [ThresholdOperator.EQUAL, "=="] and not math.isclose(val, float(target), abs_tol=1e-5):
                return False
            elif op in [ThresholdOperator.BETWEEN, "between"]:
                min_v, max_v = target
                if val < float(min_v) or val > float(max_v):
                    return False

        return True

    def create_solution(self, sol_id: str, selected_ids: Set[str], problem: OptimizationProblem) -> Solution:
        """
        Create a Solution object by evaluating all objectives and constraints.

        Evaluates all objective attributes, all available model attributes,
        and all constraints. The resulting Solution includes full objective
        and constraint values.

        Args:
            sol_id: Unique identifier for the solution.
            selected_ids: Set of selected item IDs.
            problem: The OptimizationProblem defining objectives and constraints.

        Returns:
            A Solution instance with all evaluated data.
        """
        objs = {}
        # Evaluate objectives
        for obj_name in problem.objectives.keys():
            objs[obj_name] = self.evaluate_attribute(selected_ids, obj_name)

        # Evaluate ALL model attributes (including non-objective ones)
        for attr_name in self.model.get_available_attributes():
            if attr_name not in objs:  # Do not overwrite objectives
                objs[attr_name] = self.evaluate_attribute(selected_ids, attr_name)

        cons = {}
        for con_name in problem.constraints.keys():
            cons[con_name] = self.evaluate_attribute(selected_ids, con_name)

        is_feasible = self.check_constraints(selected_ids, problem.constraints)

        return Solution(
            id=sol_id,
            selected_ids=set(selected_ids),
            objectives=objs,
            constraints=cons,
            is_feasible=is_feasible
        )