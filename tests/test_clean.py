import pandas as pd
import sys
from pathlib import Path

# add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline.clean import drop_siblings
from src.pipeline.clean import link_with_g_scores

def test_drop_siblings():
    df = pd.DataFrame({
        "Subject": [1, 2, 3],
        "rel_family_id": ["fam1", "fam1", "fam2"]
    })
    result = drop_siblings(df)

    # One of the "fam1" siblings should be dropped
    assert len(result) == 2
    assert result["rel_family_id"].is_unique


def test_link_with_g_scores():
    # Baseline and followup feature datasets
    df_baseline = pd.DataFrame({
        "src_subject_id": ["S1", "S2", "S3"],
        "feature1": [10, 20, 30],
        "eventname": ["baseline_year_1_arm_1"] * 3
    })

    df_followup = pd.DataFrame({
        "src_subject_id": ["S1", "S2", "S3"],
        "feature1": [40, 50, 60],
        "eventname": ["2_year_follow_up_y_arm_1"] * 3
    })

    # g_factor dataset
    g_factor = pd.DataFrame({
        "subjectkey": ["S1", "S2", "S3", "S4"],   # includes an extra subject S4
        "G_lavaan.baseline": [1.1, 2.2, None, 4.4],  # S3 missing baseline
        "G_lavaan.2Year": [1.5, 2.5, 3.5, None],    # S3 missing baseline, S4 missing followup
        "site_id_l.baseline": ["site1"] * 4,
        "site_id_l.2Year": ["site2"] * 4
    })

    # Run function
    merged_baseline, merged_followup = link_with_g_scores(
        df_baseline, df_followup, g_factor
    )

    # Only S1 and S2 have both baseline + followup g-scores
    assert set(merged_baseline["src_subject_id"]) == {"S1", "S2"}
    assert set(merged_followup["src_subject_id"]) == {"S1", "S2"}

    # Ensure dropped columns are really gone
    assert "G_lavaan.2Year" not in merged_baseline.columns
    assert "G_lavaan.baseline" not in merged_followup.columns
    assert "site_id_l.baseline" not in merged_baseline.columns
    assert "site_id_l.2Year" not in merged_followup.columns
