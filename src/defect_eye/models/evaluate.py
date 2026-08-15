"""Model evaluation and metrics calculation."""

from typing import Any, Dict
import pandas as pd
from sklearn.metrics import (
    auc,
    brier_score_loss,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Calculate PR-AUC, ROC-AUC, F2-Score, and Brier Score for a fitted model.

    Args:
        model: Fitted scikit-learn or XGBoost model.
        X_test: Scaled testing feature matrix.
        y_test: Testing target vector.

    Returns:
        Dictionary containing calculated evaluation metrics.
    """
    y_pred = model.predict(X_test)

    # Handle probability predictions depending on the model API
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred  # Fallback if probability is unavailable

    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_prob)

    # F2 Score (beta=2 prioritizes recall over precision)
    f2 = fbeta_score(y_test, y_pred, beta=2.0)

    # Brier Score (measures the calibration of predicted probabilities)
    brier = brier_score_loss(y_test, y_prob)

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "f2_score": round(f2, 4),
        "brier_score": round(brier, 4),
    }
