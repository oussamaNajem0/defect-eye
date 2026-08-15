"""Model training module for defect prediction."""

from typing import Any, Dict
import pandas as pd
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> LogisticRegression:
    """Train a baseline Logistic Regression model.

    Args:
        X_train: Scaled training feature matrix.
        y_train: Training target vector.
        random_state: Random seed.

    Returns:
        Fitted LogisticRegression model.
    """
    model = LogisticRegression(
        class_weight="balanced",
        random_state=random_state,
        max_iter=1000,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float = 1.0,
    params: Dict[str, Any] = None,
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier optimized for imbalanced defect data.

    Args:
        X_train: Scaled training feature matrix.
        y_train: Training target vector.
        scale_pos_weight: Weight ratio for the positive class (defects).
        params: Optional dictionary of XGBoost hyperparameters.

    Returns:
        Fitted XGBClassifier model.
    """
    default_params = {
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "random_state": 42,
        "eval_metric": "aucpr",
    }

    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        enable_categorical=False,
        **default_params,
    )
    model.fit(X_train, y_train)
    return model
