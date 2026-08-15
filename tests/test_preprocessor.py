import numpy as np
import pandas as pd
import pytest

from defect_eye.data.preprocessor import (
    compute_class_weight_ratio,
    prepare_features_and_target,
    split_and_scale_data,
)


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """Generates synthetic dataset with missing values and imbalance."""
    np.random.seed(42)
    data = {
        "file_path": [f"src/file_{i}.py" for i in range(100)],
        "nloc": np.random.randint(10, 500, size=100),
        "cyclomatic_complexity": np.random.randint(1, 20, size=100),
        "churn": np.random.randint(0, 100, size=100),
        "has_defect": [
            1 if i < 15 else 0 for i in range(100)
        ],  # 15% positive imbalance
    }
    df = pd.DataFrame(data)
    # Inject missing values
    df.loc[5, "nloc"] = np.nan
    return df


def test_prepare_features_and_target(sample_raw_dataframe: pd.DataFrame):
    X, y = prepare_features_and_target(
        sample_raw_dataframe, target_col="has_defect", drop_cols=["file_path"]
    )

    assert "file_path" not in X.columns
    assert "has_defect" not in X.columns
    assert len(X.columns) == 3
    assert len(y) == 100


def test_split_and_scale_data(sample_raw_dataframe: pd.DataFrame):
    X, y = prepare_features_and_target(
        sample_raw_dataframe, target_col="has_defect", drop_cols=["file_path"]
    )
    processed = split_and_scale_data(X, y, test_size=0.2, random_state=42)

    assert len(processed.X_train) == 80
    assert len(processed.X_test) == 20
    # Verify missing value imputation worked (no NaNs remaining)
    assert not processed.X_train.isna().any().any()
    assert not processed.X_test.isna().any().any()


def test_compute_class_weight_ratio():
    y = pd.Series([0, 0, 0, 0, 1])  # 4 negatives, 1 positive -> ratio 4.0
    ratio = compute_class_weight_ratio(y)
    assert ratio == 4.0
