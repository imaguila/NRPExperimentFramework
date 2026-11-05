
# Location: src/interface/control/enrichment.py
"""
Enrichment Panel UI Component.

Inspects dataset attributes against domain capabilities, shows available
indicators to the user, and triggers calculation on demand via a button.
"""

from typing import List, Any
import streamlit as st
import pandas as pd

from src.analysis.enrichment import get_available_indicators, compute_enrichment_indicators


def render_enrichment_panel(df: pd.DataFrame, model: Any = None) -> List[str]:
    """
    Renderiza el panel de selección e inspección de indicadores de enriquecimiento
    dentro de un expander en la barra lateral.
    
    Retorna la lista de indicadores que han sido efectivamente calculados.
    """
    if df is None or df.empty:
        st.sidebar.caption("ℹ️ Load a dataset to inspect enrichment options.")
        return []

    # 1. Consultar al plugin qué indicadores son viables con los datos actuales
    available_dict = get_available_indicators(df, model=model)

    if not available_dict:
        st.sidebar.caption("ℹ️ No compatible indicators found for current attributes.")
        return []

    # Inicializar claves de estado en session_state
    if "selected_indicators_multiselect" not in st.session_state:
        st.session_state["selected_indicators_multiselect"] = []
    if "calculated_indicators" not in st.session_state:
        st.session_state["calculated_indicators"] = []

    options = list(available_dict.keys())
    selected = []

    # Expander en la barra lateral
    with st.sidebar.expander("🧬 Enrichment", expanded=False):
        # Multiselect con los indicadores viables
        selected = st.multiselect(
            "Available Indicators",
            options=options,
            default=[ind for ind in st.session_state["selected_indicators_multiselect"] if ind in options],
            format_func=lambda x: f"{x} — {available_dict[x]['description']}",
            help="Select indicators supported by current dataset attributes and click 'Calculate'.",
            key="enrichment_multiselect_widget"
        )
        st.session_state["selected_indicators_multiselect"] = selected

        # Resumen visual de atributos requeridos
        if selected:
            req_lines = []
            for ind in selected:
                reqs = available_dict[ind]["required"]
                if reqs:
                    req_lines.append(f"• **{ind}**: requires `{', '.join(reqs)}`")
                else:
                    req_lines.append(f"• **{ind}**: calculated from solution scope")

            for line in req_lines:
                st.markdown(f"<span style='font-size:0.8rem;'>{line}</span>", unsafe_allow_html=True)

        # 2. Botón explícito de cálculo
        btn_calc = st.button("⚡ Calculate Indicators", key="btn_calculate_indicators", use_container_width=True)

        if btn_calc:
            if selected:
                # Calcular los indicadores
                enriched_df = compute_enrichment_indicators(df, selected, model=model)
                st.session_state["enriched_df"] = enriched_df
                st.session_state["calculated_indicators"] = selected
                st.session_state["enrichment_just_calculated"] = True  # <--- FLAG PARA RECARGAR
                st.success(f"Calculated {len(selected)} indicator(s)!")
            else:
                st.info("No indicators selected.")

    # Mantiene activos únicamente los indicadores calculados que continúen seleccionados
    active_calculated = [ind for ind in st.session_state.get("calculated_indicators", []) if ind in selected]

    return active_calculated