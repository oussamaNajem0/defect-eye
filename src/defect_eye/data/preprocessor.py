"""Data preprocessing and class imbalance handling pipeline."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class ProcessedData:
    """Dataclass holding train/test feature matrices and target vectors."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: List[str]
    scaler: StandardScaler


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str = "has_defect",
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X and target vector y, dropping non-feature columns.

    Args:
        df: Input DataFrame.
        target_col: Name of the target variable column.
        drop_cols: Optional list of additional metadata columns to remove (e.g., 'file_path').

    Returns:
        Tuple of (X, y).
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    cols_to_drop = [target_col]
    if drop_cols:
        cols_to_drop.extend(drop_cols)

    # Retain non-string numeric features
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    X = X.select_dtypes(include=[np.number])
    y = df[target_col].astype(int)

    return X, y


def split_and_scale_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ProcessedData:
    """Perform stratified split and scale numerical features using StandardScaler.

    Scales test data using parameters fit exclusively on training data to prevent leakage.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Fraction of data for testing. Defaults to 0.2.
        random_state: Random seed for reproducibility.

    Returns:
        ProcessedData containing scaled splits and fitted scalar instance.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y if len(y.unique()) > 1 else None,
        random_state=random_state,
    )

    # Impute missing values with column median
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Standardize scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

    return ProcessedData(
        X_train=X_train_df,
        X_test=X_test_df,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(X.columns),
        scaler=scaler,
    )


def compute_class_weight_ratio(y: Union[pd.Series, np.ndarray]) -> float:
    """Calculate ratio of negative to positive samples for XGBoost scale_pos_weight.

    Returns:
        Float weight representing count(negative) / count(positive).
    """
    neg_count = int((y == 0).sum())
    pos_count = int((y == 1).sum())

    if pos_count == 0:
        return 1.0

    return round(neg_count / pos_count, 2)
