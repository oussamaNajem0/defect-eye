import numpy as np
import pandas as pd
import pytest
from defect_eye.models.evaluate import evaluate_model
from defect_eye.models.train import train_logistic_regression, train_xgboost


@pytest.fixture
def dummy_training_data():
    """Generate synthetic feature matrix and target vector."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "loc": np.random.randint(10, 100, 100),
            "cyclomatic_complexity": np.random.randint(1, 15, 100),
            "churn": np.random.randint(0, 50, 100),
        }
    )
    # Inject synthetic relationships
    y = ((X["loc"] > 70) & (X["cyclomatic_complexity"] > 10)).astype(int)
    # Ensure at least a few positive cases
    y[:10] = 1

    return X, y


def test_train_logistic_regression(dummy_training_data):
    X, y = dummy_training_data
    model = train_logistic_regression(X, y)

    assert model is not None
    assert hasattr(model, "predict")

    preds = model.predict(X)
    assert len(preds) == len(X)


def test_train_xgboost(dummy_training_data):
    X, y = dummy_training_data
    model = train_xgboost(X, y, scale_pos_weight=2.0)

    assert model is not None
    assert hasattr(model, "predict")

    preds = model.predict(X)
    assert len(preds) == len(X)


def test_evaluate_model(dummy_training_data):
    X, y = dummy_training_data
    model = train_xgboost(X, y)
    metrics = evaluate_model(model, X, y)

    assert isinstance(metrics, dict)
    assert "pr_auc" in metrics
    assert "f2_score" in metrics
    assert "brier_score" in metrics
    assert 0.0 <= metrics["pr_auc"] <= 1.0
