import pandas as pd

from src.output.bed import export_bed


def test_bed_export_uses_six_columns_and_half_open_coordinates():
    df = pd.DataFrame(
        [
            {
                "Sequence_ID": "chr1",
                "Start": 4,
                "End": 8,
                "Rank": 2,
                "Mean_PDS": 3.6,
            }
        ]
    )

    fields = export_bed(df).decode().strip().split("\t")

    assert fields == ["chr1", "4", "9", "chr1_region_2", "4", "."]
