"""Correct quality metrics for Pareto fronts.

Calculations use a minimisation-oriented objective space. A reference point
passed to hypervolume is expressed in the original objective orientation and
is transformed internally.
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from src.domain.core.paretofront import ParetoFront


class ParetoMetrics:
    """Quality and descriptive metrics for Pareto fronts."""

    @staticmethod
    def _direction(value):
        raw = getattr(value, "value", value)
        key = str(raw).strip().lower()
        aliases = {
            "min": "min", "minimum": "min", "minimise": "min", "minimize": "min",
            "max": "max", "maximum": "max", "maximise": "max", "maximize": "max",
        }
        if key not in aliases:
            raise ValueError("Unsupported objective direction: %r" % (value,))
        return aliases[key]

    @staticmethod
    def _names(front, objective_names):
        names = (list(objective_names) if objective_names is not None else
                 list(front.solutions[0].objectives.keys()) if front.solutions else [])
        if not names:
            raise ValueError("At least one objective is required.")
        if len(set(names)) != len(names):
            raise ValueError("Objective names must be unique.")
        return names

    @staticmethod
    def _front_directions(front):
        for method_name in ("get_objective_directions", "get_directions"):
            method = getattr(front, method_name, None)
            if callable(method):
                value = method()
                if value:
                    return value
        for attr_name in ("objective_directions", "directions"):
            value = getattr(front, attr_name, None)
            if value:
                return value
        return None

    @classmethod
    def _directions(cls, front, names, directions):
        source = directions or cls._front_directions(front)
        if source is None:
            return {name: "min" for name in names}
        result = {}
        for name in names:
            if name not in source:
                raise ValueError("Missing direction for objective %r." % name)
            result[name] = cls._direction(source[name])
        return result

    @classmethod
    def get_solution_vectors(cls, front, objective_names=None, directions=None,
                             feasible_only=True):
        """Extract finite vectors and negate maximisation objectives."""
        if not getattr(front, "solutions", None):
            return np.empty((0, 0), dtype=float)
        names = cls._names(front, objective_names)
        senses = cls._directions(front, names, directions)
        rows = []
        for solution in front.solutions:
            if feasible_only and getattr(solution, "is_feasible", True) is False:
                continue
            objectives = getattr(solution, "objectives", None)
            if not isinstance(objectives, dict):
                raise ValueError("A solution has no objective mapping.")
            row = []
            for name in names:
                if name not in objectives:
                    raise ValueError("Objective %r is missing from a solution." % name)
                value = float(objectives[name])
                if not np.isfinite(value):
                    raise ValueError("Objective %r is not finite." % name)
                row.append(-value if senses[name] == "max" else value)
            rows.append(row)
        return (np.asarray(rows, dtype=float) if rows else
                np.empty((0, len(names)), dtype=float))

    @staticmethod
    def _non_dominated(points):
        """Remove duplicate and dominated minimisation points."""
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return points
        points = np.unique(points, axis=0)
        keep = np.ones(len(points), dtype=bool)
        for index, point in enumerate(points):
            dominators = np.all(points <= point, axis=1) & np.any(points < point, axis=1)
            if np.any(dominators):
                keep[index] = False
        return points[keep]

    @classmethod
    def _transform_reference(cls, point, names, senses):
        if len(point) != len(names):
            raise ValueError("Reference point dimensionality is invalid.")
        result = []
        for name, raw in zip(names, point):
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError("Reference-point values must be finite.")
            result.append(-value if senses[name] == "max" else value)
        return np.asarray(result, dtype=float)

    @staticmethod
    def _automatic_reference(points):
        """Strictly worse point for one front; not for cross-front comparison."""
        worst = np.max(points, axis=0)
        span = worst - np.min(points, axis=0)
        return worst + np.where(span > 0.0, 0.1 * span, 1.0)

    @classmethod
    def _hv_recursive(cls, points, reference):
        """Exact dimension-sweep hypervolume for minimisation."""
        if points.size == 0:
            return 0.0
        if points.shape[1] == 1:
            return float(max(0.0, reference[0] - np.min(points[:, 0])))
        coordinates = np.unique(points[:, 0])
        coordinates = coordinates[coordinates < reference[0]]
        volume = 0.0
        for index, lower in enumerate(coordinates):
            upper = coordinates[index + 1] if index + 1 < len(coordinates) else reference[0]
            width = upper - lower
            if width > 0.0:
                active = cls._non_dominated(points[points[:, 0] <= lower, 1:])
                volume += width * cls._hv_recursive(active, reference[1:])
        return float(volume)

    @classmethod
    def hypervolume(cls, front, reference_point=None, objective_names=None,
                    directions=None):
        """Compute exact HV. Use one explicit reference for comparable fronts."""
        if not getattr(front, "solutions", None):
            return 0.0
        names = cls._names(front, objective_names)
        senses = cls._directions(front, names, directions)
        points = cls._non_dominated(
            cls.get_solution_vectors(front, names, senses, feasible_only=True)
        )
        if points.size == 0:
            return 0.0
        reference = (cls._automatic_reference(points) if reference_point is None else
                     cls._transform_reference(reference_point, names, senses))
        if np.any(points > reference):
            raise ValueError("Reference point must be no better than every solution.")
        points = points[np.all(points < reference, axis=1)]
        return cls._hv_recursive(points, reference) if points.size else 0.0

    @classmethod
    def spacing(cls, front, objective_names=None, directions=None):
        points = cls.get_solution_vectors(front, objective_names, directions, True)
        points = np.unique(points, axis=0) if points.size else points
        if len(points) < 2:
            return 0.0
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        return float(np.std(np.min(distances, axis=1), ddof=1))

    @classmethod
    def objective_space_extent(cls, front, objective_names=None, directions=None):
        points = cls.get_solution_vectors(front, objective_names, directions, True)
        points = np.unique(points, axis=0) if points.size else points
        if len(points) < 2:
            return 0.0
        return float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)))

    @classmethod
    def spread(cls, front, objective_names=None, directions=None):
        """Compatibility alias. This is extent, not Deb's spread indicator."""
        return cls.objective_space_extent(front, objective_names, directions)

    @classmethod
    def compute_all_metrics(cls, front, reference_point=None,
                            objective_names=None, directions=None):
        if not getattr(front, "solutions", None):
            return {}
        names = cls._names(front, objective_names)
        senses = cls._directions(front, names, directions)
        points = cls.get_solution_vectors(front, names, senses, True)
        distinct = np.unique(points, axis=0) if points.size else points
        extent = cls.objective_space_extent(front, names, senses)
        return {
            "Solutions": int(len(distinct)),
            "Hypervolume": cls.hypervolume(front, reference_point, names, senses),
            "Spacing": cls.spacing(front, names, senses),
            "Objective-space extent": extent,
            "Spread": extent,
        }
