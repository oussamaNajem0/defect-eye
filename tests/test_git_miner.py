import subprocess
from pathlib import Path

import pytest

from defect_eye.extractor.git_miner import analyze_file_churn, churn_metrics_to_dict


@pytest.fixture
def dummy_git_repo(tmp_path: Path) -> Path:
    """Creates a temporary Git repository with commit history for testing."""
    repo_dir = tmp_path / "dummy_repo"
    repo_dir.mkdir()

    # Initialize git repository
    subprocess.run(["git", "init"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo_dir, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo_dir, check=True
    )

    # Initial Commit
    file1 = repo_dir / "app.py"
    file1.write_text("print('hello world')\n")
    subprocess.run(["git", "add", "app.py"], cwd=repo_dir, check=True)
    subprocess.run(
        ["git", "commit", "-m", "feat: initial commit"], cwd=repo_dir, check=True
    )

    # Second Commit (Bug fix)
    file1.write_text("print('hello world')\nprint('bug fix applied')\n")
    subprocess.run(["git", "add", "app.py"], cwd=repo_dir, check=True)
    subprocess.run(
        ["git", "commit", "-m", "fix: resolve bug in app.py"], cwd=repo_dir, check=True
    )

    return repo_dir


def test_analyze_file_churn(dummy_git_repo: Path):
    metrics = analyze_file_churn(
        repo_path=dummy_git_repo,
        target_file_relative_path="app.py",
        lookback_days=None,
    )

    assert metrics.commit_count == 2
    assert metrics.lines_added >= 2
    assert metrics.author_count == 1
    assert metrics.bug_fix_count == 1


def test_churn_metrics_to_dict(dummy_git_repo: Path):
    metrics = analyze_file_churn(
        repo_path=dummy_git_repo,
        target_file_relative_path="app.py",
        lookback_days=None,
    )
    data = churn_metrics_to_dict(metrics)
    assert isinstance(data, dict)
    assert "total_churn" in data
    assert "bug_fix_count" in data
