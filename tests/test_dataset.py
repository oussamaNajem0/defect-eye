import subprocess
from pathlib import Path
import pandas as pd
import pytest
from defect_eye.data.dataset import (
    build_repository_dataset,
    load_nasa_promise_dataset,
)


@pytest.fixture
def mock_repo_with_files(tmp_path: Path) -> Path:
    """Creates a mock git repo with python source files for dataset integration tests."""
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir()

    subprocess.run(["git", "init"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, check=True)

    # File 1: Simple module
    f1 = repo_dir / "calculator.py"
    f1.write_text("def add(a, b):\n    return a + b\n")
    subprocess.run(["git", "add", "calculator.py"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "feat: add calculator"], cwd=repo_dir, check=True)

    # File 2: Complex module with bug fix
    f2 = repo_dir / "service.py"
    f2.write_text(
        "def process(x):\n    if x > 0:\n        return x * 2\n    return 0\n"
    )
    subprocess.run(["git", "add", "service.py"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "fix: resolve bug in service logic"], cwd=repo_dir, check=True)

    return repo_dir


def test_build_repository_dataset(mock_repo_with_files: Path):
    df = build_repository_dataset(
        repo_path=mock_repo_with_files,
        relative_file_paths=["calculator.py", "service.py"],
        lookback_days=None,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "nloc" in df.columns
    assert "total_churn" in df.columns
    assert "has_defect" in df.columns
    assert df["has_defect"].sum() == 1  # Exactly 1 bug fix commit created in fixture


def test_load_nasa_promise_dataset(tmp_path: Path):
    sample_csv = tmp_path / "nasa_sample.csv"
    sample_csv.write_text("loc,v(g),ev(g),iv(g),n,v,l,d,i,b,t,lOCode,lOComment,lOBlank,locCodeAndComment,defects\n1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1,0,0,0,false\n2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2,0,0,0,true\n")

    df = load_nasa_promise_dataset(sample_csv)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "has_defect" in df.columns
    assert list(df["has_defect"]) == [0, 1]