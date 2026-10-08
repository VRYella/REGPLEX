from __future__ import annotations

import pandas as pd


def export_bed(df: pd.DataFrame) -> bytes:
    if df.empty:
        return b""
    bed = df[["Sequence_ID", "Start", "End", "Rank", "Mean_PDS"]].copy()
    bed["End"] = bed["End"].astype(int) + 1
    bed["Name"] = bed.apply(lambda row: f"{row['Sequence_ID']}_region_{int(row['Rank'])}", axis=1)
    bed["Score"] = pd.to_numeric(bed["Mean_PDS"], errors="coerce").fillna(0).clip(0, 1000).round().astype(int)
    bed["Strand"] = "."
    bed = bed[["Sequence_ID", "Start", "End", "Name", "Score", "Strand"]]
    return bed.to_csv(index=False, sep="\t", header=False).encode()
