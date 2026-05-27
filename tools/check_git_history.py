"""
Tool 3: check_git_history

Uses subprocess to query local git history for commits relevant to the
keywords extracted from a parsed stack trace.

Gracefully handles non-git directories and subprocess failures.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from typing import Any

from config import GIT_LOOKBACK_DAYS, GIT_MAX_COMMITS


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _run_git(args: list[str], cwd: str, timeout: int = 15) -> tuple[str, str, int]:
    """Run a git command and return (stdout, stderr, returncode).

    Args:
        args: Command arguments list (without the leading "git").
        cwd: Working directory in which to run the command.
        timeout: Maximum seconds to wait for the process.

    Returns:
        Tuple of (stdout text, stderr text, return code).
    """
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout, result.stderr, result.returncode
    except FileNotFoundError:
        return "", "git executable not found in PATH.", 127
    except subprocess.TimeoutExpired:
        return "", f"git command timed out after {timeout}s.", 1
    except OSError as exc:
        return "", f"OS error running git: {exc}", 1


def _is_git_repo(repo_path: str) -> bool:
    """Return True if ``repo_path`` is inside a git repository.

    Args:
        repo_path: Directory path to check.

    Returns:
        True if the directory is under git version control.
    """
    _, _, code = _run_git(["rev-parse", "--is-inside-work-tree"], cwd=repo_path)
    return code == 0


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_log_line(line: str) -> dict[str, Any] | None:
    """Parse a single ``git log`` output line in the custom format.

    Expected format: ``{hash}|{iso_date}|{author}|{subject}``

    Args:
        line: A single line of git log output.

    Returns:
        Parsed commit dict or None if the line cannot be parsed.
    """
    parts = line.strip().split("|", 3)
    if len(parts) < 4:
        return None
    commit_hash, date, author, subject = parts
    return {
        "hash": commit_hash[:8],
        "full_hash": commit_hash,
        "date": date.strip(),
        "author": author.strip(),
        "message": subject.strip(),
        "changed_files": [],
    }


def _get_changed_files(full_hash: str, repo_path: str) -> list[str]:
    """Retrieve the list of files changed in a specific commit.

    Args:
        full_hash: Full git commit hash.
        repo_path: Repository root path.

    Returns:
        List of changed file paths.
    """
    stdout, _, code = _run_git(
        ["show", "--name-only", "--format=", full_hash],
        cwd=repo_path,
    )
    if code != 0:
        return []
    return [line.strip() for line in stdout.splitlines() if line.strip()]


def _commits_match_keywords(commit: dict[str, Any], keywords: list[str]) -> bool:
    """Check if a commit's message or changed files mention any keyword.

    Args:
        commit: Parsed commit dict (must include ``message`` and ``changed_files``).
        keywords: Case-insensitive keyword list.

    Returns:
        True if any keyword appears in the commit message or changed files.
    """
    haystack = (commit.get("message", "") + " ".join(commit.get("changed_files", []))).lower()
    return any(kw.lower() in haystack for kw in keywords)


def _recent_changed_files(repo_path: str, days: int = 7) -> list[str]:
    """Return file paths changed in the last ``days`` days.

    Args:
        repo_path: Repository root path.
        days: Lookback window in days.

    Returns:
        Sorted, deduplicated list of changed file paths.
    """
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days)).strftime(
        "%Y-%m-%d"
    )
    stdout, _, code = _run_git(
        ["log", f"--since={cutoff}", "--name-only", "--format=", "--diff-filter=AM"],
        cwd=repo_path,
    )
    if code != 0:
        return []
    files = {line.strip() for line in stdout.splitlines() if line.strip()}
    return sorted(files)


def _grep_commits(keyword: str, repo_path: str, max_results: int = 5) -> list[str]:
    """Return commit hashes that match a keyword in their message body.

    Args:
        keyword: Search keyword.
        repo_path: Repository root path.
        max_results: Maximum number of matching commits to return.

    Returns:
        List of short commit hash strings.
    """
    stdout, _, code = _run_git(
        ["log", "--oneline", "--all", f"--grep={keyword}", f"-n{max_results}"],
        cwd=repo_path,
    )
    if code != 0:
        return []
    return [line.split()[0] for line in stdout.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_git_history(
    keywords: list[str],
    repo_path: str = ".",
    max_commits: int = GIT_MAX_COMMITS,
) -> dict[str, Any]:
    """Scan recent git history for commits relevant to the given keywords.

    Steps:
    1. Fetch the last ``max_commits`` commits from the past ``GIT_LOOKBACK_DAYS``.
    2. For each commit, retrieve changed files and filter by keyword match.
    3. Additionally grep the full git log for each keyword.
    4. Collect files changed in the last 7 days.

    Args:
        keywords: Search keywords (usually exception type and function name).
        repo_path: Root of the git repository.  Defaults to CWD.
        max_commits: Maximum number of commits to inspect.

    Returns:
        A dict with keys:

        - ``relevant_commits`` (list[dict]): Matching commits with
          ``hash``, ``date``, ``author``, ``message``, ``changed_files``.
        - ``recent_changes_to_related_files`` (list[str]): Files changed
          in the last 7 days.
        - ``has_git`` (bool): Whether git is available and the directory
          is a repo.
        - ``error`` (str | None): Error message if something went wrong.
    """
    import os

    resolved_path = os.path.abspath(repo_path)

    if not _is_git_repo(resolved_path):
        return {
            "relevant_commits": [],
            "recent_changes_to_related_files": [],
            "has_git": False,
            "error": (
                f"Directory '{resolved_path}' is not inside a git repository, "
                "or git is not installed."
            ),
        }

    # Filter empty keywords
    keywords = [k.strip() for k in keywords if k and k.strip()]

    # Step 1 — fetch recent log
    log_stdout, log_stderr, log_code = _run_git(
        [
            "log",
            "--oneline",
            f"--since={GIT_LOOKBACK_DAYS} days ago",
            f"-n{max_commits}",
            "--format=%H|%ai|%an|%s",
        ],
        cwd=resolved_path,
    )

    if log_code != 0:
        return {
            "relevant_commits": [],
            "recent_changes_to_related_files": [],
            "has_git": True,
            "error": f"git log failed: {log_stderr.strip()}",
        }

    parsed_commits: list[dict[str, Any]] = []
    for raw_line in log_stdout.splitlines():
        commit = _parse_log_line(raw_line)
        if commit is None:
            continue
        # Fetch changed files for each commit
        try:
            commit["changed_files"] = _get_changed_files(
                commit["full_hash"], resolved_path
            )
        except Exception:  # noqa: BLE001
            commit["changed_files"] = []
        parsed_commits.append(commit)

    # Step 2 — filter by keyword relevance (skip if no keywords)
    if keywords:
        relevant = [c for c in parsed_commits if _commits_match_keywords(c, keywords)]
    else:
        relevant = parsed_commits[:GIT_MAX_COMMITS]

    # Step 3 — grep full history for each keyword
    grepped_hashes: set[str] = set()
    for kw in keywords:
        for short_hash in _grep_commits(kw, resolved_path):
            grepped_hashes.add(short_hash)

    # Add grepped commits that aren't already in relevant list
    existing_hashes = {c["hash"] for c in relevant}
    for short_hash in grepped_hashes:
        if short_hash not in existing_hashes:
            # Fetch full info for this commit
            show_stdout, _, show_code = _run_git(
                ["show", "--format=%H|%ai|%an|%s", "--name-only", short_hash],
                cwd=resolved_path,
            )
            if show_code == 0 and show_stdout:
                first_line = show_stdout.splitlines()[0]
                commit = _parse_log_line(first_line)
                if commit:
                    commit["changed_files"] = [
                        ln.strip()
                        for ln in show_stdout.splitlines()[1:]
                        if ln.strip()
                    ]
                    relevant.append(commit)
                    existing_hashes.add(short_hash)

    # Step 4 — recent changed files
    try:
        recent_files = _recent_changed_files(resolved_path, days=7)
    except Exception:  # noqa: BLE001
        recent_files = []

    # Clean up full_hash from output (internal field)
    for c in relevant:
        c.pop("full_hash", None)

    return {
        "relevant_commits": relevant,
        "recent_changes_to_related_files": recent_files,
        "has_git": True,
        "error": None,
    }
