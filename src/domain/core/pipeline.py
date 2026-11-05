# Location: src/domain/core/pipeline.py
"""
Pipeline Preprocessing Engine for Optimization Models.

This module provides a domain-agnostic execution engine for applying
sequential transformation steps (preprocessors) to optimization models.
It supports branching, where a single step can produce multiple model variants.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional, Callable
from src.domain.core.model import OptimizationModel

logger = logging.getLogger(__name__)


class PreprocessingEngine:
    """
    Generic pipeline engine for preprocessing optimization models.

    The engine executes a sequence of transformation steps (preprocessors)
    on a given model. Each step can either modify the model in-place or
    return one or more new model instances (branching). Subsequents steps
    are applied to all resulting branches independently.

    The pipeline steps can be provided explicitly or obtained from the
    model's plugin via the `get_preprocessing_steps()` method.
    """

    @classmethod
    def process(
        cls,
        initial_model: OptimizationModel,
        pipeline_options: Optional[Dict[str, Any]] = None,
        custom_steps: Optional[List[Tuple[str, Any]]] = None
    ) -> List[OptimizationModel]:
        """
        Execute the configured preprocessing pipeline on the initial model.

        The pipeline steps are executed sequentially. Each step is a callable
        (either a function or a module with a `run()` method) that receives
        the current model and returns either a single model or a list of models.
        If a step returns multiple models (branching), subsequent steps are
        applied to each branch independently.

        Args:
            initial_model: The model to preprocess.
            pipeline_options: Optional dictionary of options specific to each step.
                Keys are step names, values are passed as keyword arguments.
            custom_steps: Optional list of (step_name, step_module) tuples.
                If not provided, steps are obtained from the model's plugin.

        Returns:
            A list of model instances after all preprocessing steps have been applied.
            If no steps are defined, the initial model is returned as a single-element list.
        """
        if pipeline_options is None:
            pipeline_options = {}

        # 1. Determine which steps to execute (avoid coupling core with plugins)
        steps = custom_steps
        if steps is None and initial_model.plugin and hasattr(initial_model.plugin, "get_preprocessing_steps"):
            steps = initial_model.plugin.get_preprocessing_steps()

        if not steps:
            logger.info("No preprocessing steps configured. Returning the initial model.")
            return [initial_model]

        current_models: List[OptimizationModel] = [initial_model]

        for step_name, step_module in steps:
            step_opts = pipeline_options.get(step_name, {})
            next_models: List[OptimizationModel] = []

            for model in current_models:
                try:
                    # Support both modules with a .run() method and plain callables
                    if hasattr(step_module, "run") and callable(step_module.run):
                        res = step_module.run(model, **step_opts)
                    elif callable(step_module):
                        res = step_module(model, **step_opts)
                    else:
                        raise ValueError(f"Step '{step_name}' is not a valid executable (no run() method nor callable).")

                    if isinstance(res, list):
                        next_models.extend(res)
                    else:
                        next_models.append(res)
                except Exception as exc:
                    logger.error(f"Error processing step '{step_name}': {exc}")
                    raise exc

            current_models = next_models

        return current_models