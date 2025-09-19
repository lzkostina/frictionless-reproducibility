import pandas as pd
import pytest
import sys
from pathlib import Path

# add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.analysis.eda import divide_features, percentage_missing_values

def test_divide_features():
    df = pd.DataFrame({
        "age": [10, 20, 30],           # numeric
        "sex": ["male", "female", "male"],  # categorical
    })
    num, cat = divide_features(df)

    assert "age" in num
    assert "sex" in cat

def test_percentage_missing_values():
    df = pd.DataFrame({
        "age": [10, None, 30],
        "height": [150, 160, None]
    })
    result = percentage_missing_values(df, ["age", "height"])

    assert result["age"] == pytest.approx(1/3 * 100)
    assert result["height"] == pytest.approx(1/3 * 100)
