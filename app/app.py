from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.pages.analyze import render_analyze_page
from app.pages.download import render_download_page
from app.pages.results import render_results_page
from app.pages.visualization import render_visualization_page

from src.models.dataclasses import PredictionResult


def _inject_styles() -> None:
    css_path = Path(__file__).resolve().parents[1] / "styles.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _render_shell() -> None:
    st.markdown(
        """
<div class="regplex-topbar">
  <div class="regplex-topbar-inner">
    <div class="brand">
      <div class="brand-mark">🧬</div>
      <div>
        <h1>REGPLEX</h1>
        <span>Regulatory Genomics via Perplexity and Motif Exploration</span>
      </div>
    </div>
    <div class="workbench-pill"><span></span> GENOMIC WORKBENCH</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="REGPLEX", page_icon="🧬", layout="wide")
    _inject_styles()
    _render_shell()

    st.markdown(
        """
<div class="hero-center">
  <div class="hero-kicker">SEQUENCE IN · SIGNALS OUT</div>
  <div class="hero-brand">Explore the hidden signals in DNA.</div>
  <div class="hero-subtitle">Find candidate regulatory regions with interpretable perplexity profiles and motif evidence.</div>
  <div class="hero-chips">
    <span class="hero-chip">Perplexity profiling</span>
    <span class="hero-chip">Motif annotation</span>
    <span class="hero-chip">Export-ready results</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    render_analyze_page()

    results = st.session_state.get("analysis_results", [])
    if results:
        selected_sequence_id = st.selectbox(
            "Sequence to explore",
            [result.sequence_id for result in results],
            key="workspace_sequence",
        )
        selected_result = next(
            result for result in results if result.sequence_id == selected_sequence_id
        )
        st.markdown(
            """
<div class="section-header">
  <span class="section-title">Analysis workspace</span>
  <span class="section-subtitle">Review, explore, and export your results without leaving this page</span>
</div>
""",
            unsafe_allow_html=True,
        )
        results_tab, visualization_tab, download_tab = st.tabs(
            ["Results", "Visualize", "Export"]
        )
        with results_tab:
            render_results_page(selected_result)
        with visualization_tab:
            render_visualization_page(selected_result)
        with download_tab:
            render_download_page(selected_result)
    else:
        st.info("Load a DNA sequence and run an analysis to explore results here.")

    with st.expander("How to interpret REGPLEX"):
        st.markdown(
            """
REGPLEX prioritizes **candidate regulatory regions** where local DNA perplexity is persistently lower than the surrounding sequence. Positive perplexity depression (PDS) indicates a less complex local sequence relative to its flanking background; motif hits provide additional evidence, not proof of function.

Predictions are hypothesis-generating and should be validated with biological data. Review the profile and region-level metrics, then export CSV, BED, GFF3, or region FASTA for downstream analysis.
"""
        )


if __name__ == "__main__":
    main()
