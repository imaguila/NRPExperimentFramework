# Location: src/interface/panels/instance_inspector.py
"""
Model Instance Inspector.

This module provides a panel for inspecting the loaded optimisation model,
showing summaries, raw data, active items, and relationship details.
"""

import pandas as pd
import streamlit as st
from typing import Any, Dict, Optional, Set

from src.domain.plugins.base_domain import DomainPlugin
from src.domain.plugins.nrp_utils import extract_pair


def render_instance_inspector(model: Any, plugin: DomainPlugin) -> None:
    """
    Render the model instance inspector panel.

    The panel displays various views of the model:
        - Summary & Metrics: compares active vs original item and relationship counts.
        - Loaded Data (Raw): shows the original JSON data before preprocessing.
        - Valid Case: lists active items with their attributes.
        - Relationships & Value Interactions: displays active precedences and value dependencies.

    Args:
        model: The OptimizationModel to inspect.
        plugin: The active domain plugin (for labels and relationship names).
    """
    with st.expander("🔍 Instance Data Inspector", expanded=True):
        # Plugin labels
        item_label = plugin.get_item_label()
        item_label_plural = plugin.get_item_label_plural()
        rel_labels = plugin.get_relationship_labels()

        views = ["📊 Summary & Metrics"]
        if st.session_state.get("base_model") is not None:
            views.append("📄 Loaded Data (Raw)")
        views.extend(["⚙️ Valid Case", "🔗 Relationships & Value Interactions"])

        view_mode = st.radio("Select Data View:", views, horizontal=True, key="inspector_view_radio")
        st.markdown("---")

        items = getattr(model, "items", {}) or {}
        rels = getattr(model, "relationships", {}) or {}

        if view_mode == "📊 Summary & Metrics":
            # Retrieve data from the original model (base_model) or fall back to the current model
            base_model = st.session_state.get("base_model")
            base_items = getattr(base_model, "items", {}) or {} if base_model else items
            base_rels = getattr(base_model, "relationships", {}) or {} if base_model else rels

            prec_label = rel_labels.get("precedences", "Precedences")
            coup_label = rel_labels.get("couplings", "Couplings")
            excl_label = rel_labels.get("exclusions", "Exclusions")

            # Display six key metrics
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric(f"Active {item_label_plural}", len(items))
            m2.metric(f"Active {prec_label}", len(rels.get("precedences", [])))
            m3.metric(f"Original {item_label_plural}", len(base_items))
            m4.metric(f"Original {prec_label}", len(base_rels.get("precedences", [])))
            m5.metric(f"Original {coup_label}", len(base_rels.get("couplings", [])))
            m6.metric(f"Original {excl_label}", len(base_rels.get("exclusions", [])))

        elif view_mode == "📄 Loaded Data (Raw)":
            base_model = st.session_state.get("base_model")
            if base_model is None:
                st.info("No raw data available. This is a synthetic case.")
            else:
                # Display the base model (before preprocessing)
                base_items = getattr(base_model, "items", {}) or {}
                base_rels = getattr(base_model, "relationships", {}) or {}

                orig_precedences = [extract_pair(p) for p in base_rels.get("precedences", [])]
                orig_couplings = [extract_pair(c) for c in base_rels.get("couplings", [])]
                orig_exclusions = [extract_pair(e) for e in base_rels.get("exclusions", [])]
                orig_val_deps = base_rels.get("value_dependencies", []) or getattr(base_model, "value_dependencies", [])

                flat_rows = []
                for item_id, item in base_items.items():
                    str_id = str(item_id)
                    prereqs = [p for p, s in orig_precedences if s == str_id and p]
                    dependents = [s for p, s in orig_precedences if p == str_id and s]

                    prec_parts = []
                    if prereqs:
                        prec_parts.append(f"Prereqs: [{', '.join(prereqs)}]")
                    if dependents:
                        prec_parts.append(f"Dependents: [{', '.join(dependents)}]")

                    couples = [r2 if r1 == str_id else r1 for r1, r2 in orig_couplings if (r1 == str_id or r2 == str_id) and r1 and r2]
                    excls = [r2 if r1 == str_id else r1 for r1, r2 in orig_exclusions if (r1 == str_id or r2 == str_id) and r1 and r2]

                    row = {
                        "ID": str_id,
                        rel_labels.get("precedences", "Precedences"): " | ".join(prec_parts) if prec_parts else "-",
                        rel_labels.get("couplings", "Couplings"): ", ".join(couples) if couples else "-",
                        rel_labels.get("exclusions", "Exclusions"): ", ".join(excls) if excls else "-",
                    }
                    row.update(getattr(item, "attributes", {}))
                    flat_rows.append(row)

                st.caption(f"📄 **Raw Data (Before Preprocessing) - {plugin.get_display_name()}**")
                st.dataframe(pd.DataFrame(flat_rows), use_container_width=True, hide_index=True)

                if orig_val_deps:
                    st.caption(f"⚡ **Original {rel_labels.get('value_dependencies', 'Value Dependencies')}**")
                    st.dataframe(pd.DataFrame(orig_val_deps), use_container_width=True, hide_index=True)

        elif view_mode == "⚙️ Valid Case":
            rows = []
            for item_id, item in items.items():
                row = {f"{item_label} ID": str(item_id), "Status": "✅ Active"}
                row.update(getattr(item, "attributes", {}))
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        elif view_mode.startswith("🔗"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### ⚡ Active {rel_labels.get('value_dependencies', 'Value Dependencies')}")
                val_deps = rels.get("value_dependencies", []) or getattr(model, "value_dependencies", [])
                if val_deps:
                    st.dataframe(pd.DataFrame(val_deps), use_container_width=True, hide_index=True)
                else:
                    st.info("No active value dependencies.")
            with col2:
                st.markdown(f"#### 📌 Active {rel_labels.get('precedences', 'Precedences')}")
                precs = [extract_pair(p) for p in rels.get("precedences", [])]
                if precs:
                    df_prec = pd.DataFrame([{"Prerequisite": p, "Dependent": s} for p, s in precs])
                    st.dataframe(df_prec, use_container_width=True, hide_index=True)
                else:
                    st.info("No active precedences.")