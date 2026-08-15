import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from defect_eye.models.explainability import (
    generate_shap_explainer,
    get_feature_importances,
    plot_shap_summary,
)
from defect_eye.models.train import train_xgboost


@pytest.fixture
def trained_model_and_data():
    """Provides a fitted XGBoost model and a dummy dataset."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "loc": np.random.randint(10, 100, 50),
            "cyclomatic_complexity": np.random.randint(1, 15, 50),
            "churn": np.random.randint(0, 50, 50),
        }
    )
    # Synthetic target logic
    y = ((X["loc"] > 60) | (X["cyclomatic_complexity"] > 8)).astype(int)
    
    model = train_xgboost(X, y)
    return model, X


def test_generate_shap_explainer(trained_model_and_data):
    model, X = trained_model_and_data
    explainer = generate_shap_explainer(model, X)
    
    assert explainer is not None
    # Ensure it's callable for SHAP values
    assert hasattr(explainer, "__call__")


def test_get_feature_importances(trained_model_and_data):
    model, X = trained_model_and_data
    explainer = generate_shap_explainer(model, X)
    
    importances = get_feature_importances(explainer, X)
    
    assert isinstance(importances, dict)
    assert len(importances) == 3
    assert "loc" in importances
    assert "cyclomatic_complexity" in importances


def test_plot_shap_summary(trained_model_and_data, tmp_path: Path):
    model, X = trained_model_and_data
    explainer = generate_shap_explainer(model, X)
    
    save_file = tmp_path / "reports" / "shap_summary.png"
    
    plot_shap_summary(explainer, X, save_path=save_file, show_plot=False)
    
    assert save_file.exists()
    assert os.path.getsize(save_file) > 0
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from defect_eye.models.explainability import (
    generate_shap_explainer,
    get_feature_importances,
    plot_shap_summary,
)
from defect_eye.models.train import train_xgboost


@pytest.fixture
def trained_model_and_data():
    """Provides a fitted XGBoost model and a dummy dataset."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "loc": np.random.randint(10, 100, 50),
            "cyclomatic_complexity": np.random.randint(1, 15, 50),
            "churn": np.random.randint(0, 50, 50),
        }
    )
    # Synthetic target logic
    y = ((X["loc"] > 60) | (X["cyclomatic_complexity"] > 8)).astype(int)
    
    model = train_xgboost(X, y)
    return model, X


def test_generate_shap_explainer(trained_model_and_data):
    model, X = trained_model_and_data
    explainer = generate_shap_explainer(model, X)
    
    assert explainer is not None
    # Ensure it's callable for SHAP values
    assert hasattr(explainer, "__call__")


def test_get_feature_importances(trained_model_and_data):
    model, X = trained_model_and_data
    explainer = generate_shap_explainer(model, X)
    
    importances = get_feature_importances(explainer, X)
    
    assert isinstance(importances, dict)
    assert len(importances) == 3
    assert "loc" in importances
    assert "cyclomatic_complexity" in importances


def test_plot_shap_summary(trained_model_and_data, tmp_path: Path):
    model, X = trained_model_and_data
    explainer = generate_shap_explainer(model, X)
    
    save_file = tmp_path / "reports" / "shap_summary.png"
    
    plot_shap_summary(explainer, X, save_path=save_file, show_plot=False)
    
    assert save_file.exists()
    assert os.path.getsize(save_file) > 0