"""Dataset construction and loading subpackage."""

from pathlib import Path
from typing import List, Optional, Union
import pandas as pd

from defect_eye.extractor.code_metrics import analyze_file, file_metrics_to_dict
from defect_eye.extractor.git_miner import analyze_file_churn, churn_metrics_to_dict


def build_repository_dataset(
    repo_path: Union[str, Path],
    relative_file_paths: List[str],
    lookback_days: Optional[int] = 90,
) -> pd.DataFrame:
    repo = Path(repo_path)
    records = []

    for rel_path in relative_file_paths:
        full_path = repo / rel_path
        if not full_path.exists():
            continue

        static_metrics = analyze_file(full_path)
        if not static_metrics:
            continue
        static_dict = file_metrics_to_dict(static_metrics)

        churn_metrics = analyze_file_churn(
            repo_path=repo,
            target_file_relative_path=rel_path,
            lookback_days=lookback_days,
        )
        churn_dict = churn_metrics_to_dict(churn_metrics)

        churn_dict.pop("file_path", None)

        merged_record = {**static_dict, **churn_dict}
        records.append(merged_record)

    df = pd.DataFrame(records)
    if not df.empty:
        df["has_defect"] = (df["bug_fix_count"] > 0).astype(int)

    return df


def load_nasa_promise_dataset(filepath_or_url: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(filepath_or_url)
    df.columns = [c.strip().lower() for c in df.columns]

    target_col = None
    for candidate in ["defects", "problems", "has_defect"]:
        if candidate in df.columns:
            target_col = candidate
            break

    if target_col:
        df["has_defect"] = df[target_col].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        ).astype(int)
        if target_col != "has_defect":
            df.drop(columns=[target_col], inplace=True)

    return df