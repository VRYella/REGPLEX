from __future__ import annotations

import streamlit as st

from src.models.dataclasses import PredictionResult
from src.output.bed import export_bed
from src.output.csv import export_csv, results_dataframe
from src.output.fasta import export_region_fasta
from src.output.gff import export_gff


def render_download_page(result: PredictionResult) -> None:
    records = st.session_state.get("analysis_records", [])

    record_map = {header: seq for header, seq in records}
    source_sequence = record_map.get(result.sequence_id)
    if source_sequence is None:
        st.error("Source sequence is unavailable for this result. Re-run analysis before downloading FASTA.")
        return

    df = results_dataframe(result)

    st.markdown('<div class="section-subtitle">Export REGPLEX calls with motif annotations for downstream genomics workflows</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button("Download CSV", export_csv(df), file_name="regplex_regions.csv", mime="text/csv", width="stretch")
    with d2:
        st.download_button("Download BED", export_bed(df), file_name="regplex_regions.bed", mime="text/plain", width="stretch")
    with d3:
        st.download_button("Download GFF3", export_gff(df), file_name="regplex_regions.gff3", mime="text/plain", width="stretch")
    with d4:
        st.download_button(
            "Download FASTA",
            export_region_fasta(df, source_sequence),
            file_name="regplex_regions.fasta",
            mime="text/plain",
            width="stretch",
        )

    st.markdown("#### Preview")
    st.dataframe(df, hide_index=True, width="stretch")
