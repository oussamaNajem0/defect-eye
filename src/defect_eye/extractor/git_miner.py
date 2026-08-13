"""Git history and commit churn extractor using PyDriller."""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, List
from pydriller import Repository


@dataclass
class GitChurnMetrics:
    """Dataclass holding historical git churn metrics for a specific file."""

    file_path: str
    commit_count: int
    lines_added: int
    lines_deleted: int
    total_churn: int
    author_count: int
    bug_fix_count: int


def analyze_file_churn(
    repo_path: str | Path,
    target_file_relative_path: str,
    lookback_days: Optional[int] = 90,
    bug_keywords: Optional[List[str]] = None,
) -> GitChurnMetrics:
    """Mine repository history for churn and defect indicators on a specific file.

    Args:
        repo_path: Local path to the Git repository root.
        target_file_relative_path: Relative file path within the repo (e.g. 'src/main.py').
        lookback_days: Days of commit history to scan. Defaults to 90.
        bug_keywords: Commit message keywords that signal a bug fix.

    Returns:
        GitChurnMetrics object summarizing churn and fix history.
    """
    if bug_keywords is None:
        bug_keywords = ["fix", "bug", "issue", "patch", "error", "defect"]

    since_date = None
    if lookback_days:
        since_date = datetime.now() - timedelta(days=lookback_days)

    commit_count = 0
    lines_added = 0
    lines_deleted = 0
    authors = set()
    bug_fix_count = 0

    target_normalized = str(Path(target_file_relative_path))

    repo = Repository(str(repo_path), since=since_date)
    for commit in repo.traverse_commits():
        msg_lower = commit.msg.lower()
        is_bug_fix = any(kw in msg_lower for kw in bug_keywords)

        for mod in commit.modified_files:
            mod_path = str(Path(mod.new_path or mod.old_path or ""))
            if mod_path == target_normalized:
                commit_count += 1
                lines_added += mod.added_lines
                lines_deleted += mod.deleted_lines
                if commit.author and commit.author.email:
                    authors.add(commit.author.email)
                if is_bug_fix:
                    bug_fix_count += 1

    return GitChurnMetrics(
        file_path=target_normalized,
        commit_count=commit_count,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        total_churn=lines_added + lines_deleted,
        author_count=len(authors),
        bug_fix_count=bug_fix_count,
    )


def churn_metrics_to_dict(metrics: GitChurnMetrics) -> Dict[str, Any]:
    """Convert GitChurnMetrics to dictionary for dataset building."""
    return asdict(metrics)