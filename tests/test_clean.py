import pandas as pd
import sys
from pathlib import Path

# add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline.clean import drop_siblings

def test_drop_siblings():
    df = pd.DataFrame({
        "Subject": [1, 2, 3],
        "rel_family_id": ["fam1", "fam1", "fam2"]
    })
    result = drop_siblings(df)

    # One of the "fam1" siblings should be dropped
    assert len(result) == 2
    assert result["rel_family_id"].is_unique
