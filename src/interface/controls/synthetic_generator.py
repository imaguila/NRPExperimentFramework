# Location: src/interface/controls/synthetic_generator.py
"""
Synthetic Generator Subpanel for Streamlit UI.

This module provides the UI for configuring and generating synthetic
optimisation instances of the active domain (e.g., NRP). Users can
select attributes, distributions, value dependencies, and precedence
settings, and generate a new model on demand.
"""

from typing import Any, Dict, Optional, Tuple
import streamlit as st

from src.domain.core.model import OptimizationModel
from src.domain.plugins.base_domain import DomainPlugin


def render_synthetic_generator_panel(plugin: DomainPlugin) -> Tuple[Optional[OptimizationModel], Dict[str, Any]]:
    """
    Render the subpanel for synthetic case configuration and generation.

    This panel allows the user to:
        - Set the number of items (requirements).
        - Select which attributes to generate and their distributions.
        - Add custom value dependencies (synergies).
        - Enable/disable precedence relationships and adjust density.
        - Generate the synthetic model.

    Args:
        plugin: The active domain plugin (e.g., NRPPlugin).

    Returns:
        A tuple (model, metadata) where:
            - model: The generated OptimizationModel (or None if not generated).
            - metadata: Dictionary with source information.
    """
    attr_schema = plugin.get_attribute_schema()
    available_attrs = list(attr_schema.keys())

    config_limits = plugin.get_generation_config()
    item_limits = config_limits.get("num_requirements", {"min": 5, "max": 300, "default": 20, "step": 5})
    prec_limits = config_limits.get("precedence_density", {"min": 0.05, "max": 0.80, "default": 0.10, "step": 0.05})
    synergy_cfg = config_limits.get("synergies", {})
    dist_options = config_limits.get("supported_distributions", ["uniform", "normal", "fibonacci", "triangular", "constant"])

    # Initialise session state for dependencies
    if "synthetic_val_deps" not in st.session_state:
        st.session_state["synthetic_val_deps"] = []

    # Initialise precedence options
    if "enable_precedences" not in st.session_state:
        st.session_state["enable_precedences"] = prec_limits.get("enabled_by_default", True)

    if "prec_density" not in st.session_state:
        st.session_state["prec_density"] = prec_limits["default"]

    st.caption(f"🎲 **Synthetic {plugin.get_item_label()} Instance Size**")
    num_reqs = st.slider(
        f"Number of {plugin.get_item_label_plural()} (N)",
        min_value=item_limits["min"],
        max_value=item_limits["max"],
        value=item_limits["default"],
        step=item_limits["step"],
        key="synth_num_reqs"  # Fixed key for persistence
    )

    st.caption("📊 **Attribute Selection & Distributions**")
    default_selected = [a for a in list(attr_schema.keys()) if attr_schema[a].get("default_dist") == "fibonacci"] or list(attr_schema.keys())[:2]

    selected_attrs = st.multiselect(
        "Choose Attributes to Generate:",
        options=available_attrs,
        default=default_selected,
        format_func=lambda key: attr_schema[key].get("label", key.capitalize()),
        key="synth_attrs"  # Fixed key for persistence
    )

    attr_configs: Dict[str, Dict[str, Any]] = {}
    if selected_attrs:
        for attr_name in selected_attrs:
            attr_info = attr_schema[attr_name]
            label = attr_info.get("label", attr_name.capitalize())

            with st.expander(f"🔹 Attribute: **{label}**", expanded=False):
                default_dist = attr_info.get("default_dist", "uniform")
                default_idx = dist_options.index(default_dist) if default_dist in dist_options else 0

                dist_type = st.selectbox(
                    f"Distribution for '{attr_name}':",
                    options=dist_options,
                    index=default_idx,
                    key=f"dist_{attr_name}"  # Fixed key per attribute
                )

                config: Dict[str, Any] = {
                    "distribution": dist_type,
                    "is_integer": attr_info.get("is_integer", True),
                    "type": attr_info.get("type", "scalar"),
                    "coupling_rule": attr_info.get("coupling_rule", "sum"),
                    "evaluation_rule": attr_info.get("evaluation_rule", "sum"),
                    "description": attr_info.get("description", "")
                }

                if dist_type == "constant":
                    config["value"] = st.number_input(
                        "Constant Value:",
                        value=10,
                        key=f"const_{attr_name}"
                    )
                elif dist_type == "uniform":
                    c1, c2 = st.columns(2)
                    config["min"] = c1.number_input(
                        "Min Value:",
                        value=1,
                        key=f"umin_{attr_name}"
                    )
                    config["max"] = c2.number_input(
                        "Max Value:",
                        value=100,
                        key=f"umax_{attr_name}"
                    )
                elif dist_type == "normal":
                    c1, c2 = st.columns(2)
                    config["mean"] = c1.number_input(
                        "Mean (μ):",
                        value=50.0,
                        key=f"nmean_{attr_name}"
                    )
                    config["std"] = c2.number_input(
                        "Std Dev (σ):",
                        value=15.0,
                        key=f"nstd_{attr_name}"
                    )
                    config["min"], config["max"] = 1, 200
                elif dist_type == "fibonacci":
                    config["max_fibonacci"] = st.selectbox(
                        "Max Fibonacci:",
                        [8, 13, 21, 34, 55, 89],
                        index=2,
                        key=f"fib_{attr_name}"
                    )
                elif dist_type == "triangular":
                    c1, c2, c3 = st.columns(3)
                    config["min"] = c1.number_input(
                        "Min:",
                        value=1.0,
                        key=f"tmin_{attr_name}"
                    )
                    config["mode"] = c2.number_input(
                        "Mode:",
                        value=50.0,
                        key=f"tmode_{attr_name}"
                    )
                    config["max"] = c3.number_input(
                        "Max:",
                        value=100.0,
                        key=f"tmax_{attr_name}"
                    )

                attr_configs[attr_name] = config

    st.markdown("---")
    st.caption("⚡ **Value Dependencies & Synergies**")
    with st.expander("⚡ Add Custom Value Dependency", expanded=False):
        if selected_attrs and num_reqs > 1:
            item_prefix = plugin.get_item_prefix()
            req_options = [f"{item_prefix}{i}" for i in range(1, num_reqs + 1)]
            c1, c2 = st.columns(2)
            r1 = c1.selectbox(f"{plugin.get_item_label()} 1:", req_options, key="vdep_r1")
            r2 = c2.selectbox(f"{plugin.get_item_label()} 2:", [r for r in req_options if r != r1], key="vdep_r2")

            c3, c4, c5 = st.columns(3)
            v_attr = c3.selectbox("Attribute:", selected_attrs, key="vdep_attr")
            effects = synergy_cfg.get("effects", ["multiplier", "delta"])
            v_effect = c4.selectbox("Effect:", effects, key="vdep_effect")

            def_factor = synergy_cfg.get("default_multiplier", 0.9) if v_effect == "multiplier" else synergy_cfg.get("default_delta", 15.0)
            v_factor = c5.number_input("Factor:", value=float(def_factor), key="vdep_factor")

            v_desc = st.text_input(
                "Description:",
                value=f"{v_attr.capitalize()} {v_effect} factor {v_factor} for {r1} and {r2}",
                key="vdep_desc"
            )

            b_add, b_reset = st.columns(2)
            if b_add.button("➕ Add Dependency", use_container_width=True):
                dep_entry = {
                    "attribute": v_attr,
                    "reqs": [r1, r2],
                    "effect": v_effect,
                    "factor": float(v_factor),
                    "description": v_desc
                }
                st.session_state["synthetic_val_deps"].append(dep_entry)
                st.toast("Synergy added", icon="✅")

                # Regenerate the model automatically
                if selected_attrs:
                    value_dependencies = st.session_state.get("synthetic_val_deps", [])
                    enable_precedences = st.session_state.get("enable_precedences", True)
                    prec_density = st.session_state.get("prec_density", 0.10)

                    synthetic_model = plugin.generate_synthetic_instance(
                        num_items=num_reqs,
                        attribute_configs=attr_configs,
                        value_dependencies=value_dependencies,
                        enable_precedences=enable_precedences,
                        precedence_density=prec_density
                    )
                    st.session_state["generated_model"] = synthetic_model

                st.rerun()

            if b_reset.button("🗑️ Reset All", use_container_width=True):
                st.session_state["synthetic_val_deps"].clear()
                st.toast("Dependencies reset. Regenerating model...", icon="🧹")

                # Regenerate the model without dependencies
                if selected_attrs:
                    enable_precedences = st.session_state.get("enable_precedences", True)
                    prec_density = st.session_state.get("prec_density", 0.10)

                    synthetic_model = plugin.generate_synthetic_instance(
                        num_items=num_reqs,
                        attribute_configs=attr_configs,
                        value_dependencies=[],
                        enable_precedences=enable_precedences,
                        precedence_density=prec_density
                    )
                    st.session_state["generated_model"] = synthetic_model

                st.rerun()

    st.markdown("---")
    st.caption("🔗 **Relationships & Precedence Graph Options**")

    enable_precedences = st.checkbox(
        "Generate Precedence Relationships (DAG)",
        value=st.session_state.get("enable_precedences", True),
        key="enable_prec_synth"
    )
    st.session_state["enable_precedences"] = enable_precedences

    prec_density = 0.0
    if enable_precedences:
        prec_density = st.slider(
            "Precedence Density:",
            min_value=prec_limits["min"],
            max_value=prec_limits["max"],
            value=st.session_state.get("prec_density", prec_limits["default"]),
            step=prec_limits["step"],
            key="synth_prec_density"
        )
        st.session_state["prec_density"] = prec_density

    st.markdown("---")

    # Generate button (only enabled if attributes are selected)
    generate_btn = st.button(
        f"⚡ Generate Synthetic {plugin.get_item_label()} Case",
        type="primary",
        use_container_width=True,
        disabled=not selected_attrs
    )

    if generate_btn and selected_attrs:
        # Clear previous Pareto state
        st.session_state["pareto_df"] = None
        st.session_state["last_pareto_df"] = None
        st.session_state["enriched_df"] = None
        st.session_state["calculated_indicators"] = []

        value_dependencies = st.session_state.get("synthetic_val_deps", [])
        enable_precedences = st.session_state.get("enable_precedences", True)
        prec_density = st.session_state.get("prec_density", 0.10)

        synthetic_model = plugin.generate_synthetic_instance(
            num_items=num_reqs,
            attribute_configs=attr_configs,
            value_dependencies=value_dependencies,
            enable_precedences=enable_precedences,
            precedence_density=prec_density
        )
        st.session_state["generated_model"] = synthetic_model
        st.success(f"Generated: **{synthetic_model.name}**")
        return synthetic_model, {"source": "synthetic_generator"}

    model = st.session_state.get("generated_model")
    if model is not None:
        return model, {"source": "synthetic_generator"}

    return None, {}