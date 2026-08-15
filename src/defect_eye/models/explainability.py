"""Explainable AI (XAI) module using SHAP for model interpretability."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import matplotlib.pyplot as plt
import pandas as pd
import shap


def generate_shap_explainer(model: Any, X_train: pd.DataFrame) -> shap.Explainer:
    """Create a SHAP TreeExplainer for the given model.

    Args:
        model: Fitted tree-based model (e.g., XGBoost, RandomForest).
        X_train: Training feature matrix used to fit the model.

    Returns:
        Fitted SHAP Explainer object.
    """
    return shap.TreeExplainer(model, X_train, feature_perturbation="tree_path_dependent")


def get_feature_importances(
    explainer: shap.Explainer, X_sample: pd.DataFrame
) -> Dict[str, float]:
    """Calculate the mean absolute SHAP values for a sample of data.

    Args:
        explainer: Fitted SHAP explainer.
        X_sample: Data sample to explain.

    Returns:
        Dictionary mapping feature names to their mean absolute SHAP value.
    """
    shap_values = explainer(X_sample)

    # Calculate mean absolute SHAP values across the sample
    mean_abs_shap = (
        pd.DataFrame(shap_values.values, columns=X_sample.columns).abs().mean()
    )

    # Sort features by importance descending
    importance_dict = mean_abs_shap.sort_values(ascending=False).to_dict()

    return {k: round(v, 4) for k, v in importance_dict.items()}


def plot_shap_summary(
    explainer: shap.Explainer,
    X_sample: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False,
) -> None:
    """Generate and optionally save a SHAP summary plot.

    Args:
        explainer: Fitted SHAP explainer.
        X_sample: Data sample to visualize.
        save_path: Path to save the PNG plot.
        show_plot: Whether to display the plot interactively.
    """
    shap_values = explainer(X_sample)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)

    if save_path:
        path = Path(save_path)
        os.makedirs(path.parent, exist_ok=True)
        plt.savefig(path, bbox_inches="tight", dpi=300)

    if show_plot:
        plt.show()

    plt.close()
