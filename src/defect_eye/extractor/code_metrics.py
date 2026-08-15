"""Static code metrics extractor using Lizard."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import lizard


@dataclass
class FunctionMetrics:
    name: str
    cyclomatic_complexity: int
    nloc: int
    token_count: int
    parameter_count: int
    start_line: int


@dataclass
class FileMetrics:
    file_path: str
    nloc: int
    total_cyclomatic_complexity: int
    max_cyclomatic_complexity: int
    avg_cyclomatic_complexity: float
    function_count: int
    max_parameter_count: int
    total_tokens: int


def analyze_file(file_path: str | Path) -> Optional[FileMetrics]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found: {path}")

    analysis = lizard.analyze_file(str(path))
    if not analysis:
        return None

    func_list: List[FunctionMetrics] = []
    for func in analysis.function_list:
        func_list.append(
            FunctionMetrics(
                name=func.name,
                cyclomatic_complexity=func.cyclomatic_complexity,
                nloc=func.nloc,
                token_count=func.token_count,
                parameter_count=len(func.parameters),
                start_line=func.start_line,
            )
        )

    function_count = len(func_list)
    complexities = [f.cyclomatic_complexity for f in func_list]
    parameters = [f.parameter_count for f in func_list]
    tokens = [f.token_count for f in func_list]

    max_cc = max(complexities) if complexities else 0
    total_cc = sum(complexities) if complexities else 0
    avg_cc = round(total_cc / function_count, 2) if function_count > 0 else 0.0
    max_params = max(parameters) if parameters else 0
    sum_tokens = sum(tokens) if tokens else 0

    return FileMetrics(
        file_path=str(path),
        nloc=analysis.nloc,
        total_cyclomatic_complexity=total_cc,
        max_cyclomatic_complexity=max_cc,
        avg_cyclomatic_complexity=avg_cc,
        function_count=function_count,
        max_parameter_count=max_params,
        total_tokens=sum_tokens,
    )


def file_metrics_to_dict(metrics: FileMetrics) -> Dict[str, Any]:
    return asdict(metrics)