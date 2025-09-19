import pandas as pd
import pytest
import sys
from pathlib import Path

# add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline.load import filter_subjects_with_two_timepoints


def test_filter_subjects_with_two_timepoints():
    df = pd.DataFrame({
        "Subject": [1, 1, 2],
        "Value": [10, 20, 30]
    })
    result = filter_subjects_with_two_timepoints(df)
    # Subject 1 has two rows, subject 2 only one → only subject 1 stays
    assert set(result["Subject"]) == {1}
