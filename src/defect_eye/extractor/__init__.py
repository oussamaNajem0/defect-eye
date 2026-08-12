"""Metrics extraction subpackage."""

from defect_eye.extractor.code_metrics import analyze_file, file_metrics_to_dict

__all__ = ["analyze_file", "file_metrics_to_dict"]