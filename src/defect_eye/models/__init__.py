"""Machine Learning subpackage."""

from defect_eye.models.evaluate import evaluate_model
from defect_eye.models.explainability import (
    generate_shap_explainer,
    get_feature_importances,
    plot_shap_summary,
)
from defect_eye.models.train import train_logistic_regression, train_xgboost

__all__ = [
    "evaluate_model",
    "generate_shap_explainer",
    "get_feature_importances",
    "plot_shap_summary",
    "train_logistic_regression",
    "train_xgboost",
]
