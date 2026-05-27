"""
Tool 2: search_codebase

Pure local filesystem search — no external APIs, no vector databases.

Walks a directory tree, reads source files with known extensions, and ranks
matches by the number of query terms found then by file recency.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from config import (
    SEARCHABLE_EXTENSIONS,
    SEARCH_MAX_MATCHES,
    SEARCH_MAX_SNIPPETS_PER_FILE,
    SEARCH_SNIPPET_CONTEXT_LINES,
    SKIP_DIRS,
)


def _is_text_file(path: str) -> bool:
    """Check whether a file has a searchable extension.

    Args:
        path: Absolute or relative path to the file.

    Returns:
        True if the file extension is in the searchable allow-list.
    """
    _, ext = os.path.splitext(path)
    return ext.lower() in SEARCHABLE_EXTENSIONS


def _read_file_safe(path: str) -> str | None:
    """Read a text file, returning None on any I/O or encoding error.

    Args:
        path: Path to the file to read.

    Returns:
        File contents as a string, or None if the file cannot be read.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def _get_mtime_iso(path: str) -> str:
    """Return the file's modification time as an ISO 8601 string.

    Args:
        path: Path to the file.

    Returns:
        ISO 8601 datetime string (UTC), or an empty string on error.
    """
    try:
        mtime = os.path.getmtime(path)
        return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except OSError:
        return ""


def _extract_snippets(
    lines: list[str],
    match_line_indices: list[int],
    context: int,
    max_snippets: int,
) -> list[str]:
    """Extract context snippets around matched line indices.

    Args:
        lines: All lines of the file (0-indexed).
        match_line_indices: 0-based indices of lines that matched.
        context: Number of context lines to include before and after each match.
        max_snippets: Maximum number of snippets to return.

    Returns:
        List of snippet strings, each a block of consecutive lines.
    """
    snippets: list[str] = []
    used: set[int] = set()

    for idx in match_line_indices[:max_snippets]:
        start = max(0, idx - context)
        end = min(len(lines) - 1, idx + context)
        block_indices = range(start, end + 1)

        # Skip if this block overlaps significantly with a previous snippet
        if any(i in used for i in block_indices):
            continue

        snippet_lines = []
        for i in block_indices:
            prefix = "→ " if i == idx else "  "
            snippet_lines.append(f"{prefix}{i + 1:4d} | {lines[i].rstrip()}")
            used.add(i)

        snippets.append("\n".join(snippet_lines))
        if len(snippets) >= max_snippets:
            break

    return snippets


def _search_file(
    path: str,
    content: str,
    query_terms: list[str],
) -> dict[str, Any] | None:
    """Search a single file for query terms.

    Args:
        path: File path (used in the result).
        content: File content string.
        query_terms: List of search terms (case-insensitive).

    Returns:
        A match dict or None if no terms were found.
    """
    content_lower = content.lower()
    matched_terms: list[str] = []
    match_line_indices: list[int] = []

    for term in query_terms:
        if term.lower() in content_lower:
            matched_terms.append(term)

    if not matched_terms:
        return None

    lines = content.splitlines()
    lines_lower = [ln.lower() for ln in lines]

    for term in matched_terms:
        for idx, line in enumerate(lines_lower):
            if term.lower() in line:
                match_line_indices.append(idx)

    # Deduplicate and sort
    match_line_indices = sorted(set(match_line_indices))

    snippets = _extract_snippets(
        lines,
        match_line_indices,
        context=SEARCH_SNIPPET_CONTEXT_LINES,
        max_snippets=SEARCH_MAX_SNIPPETS_PER_FILE,
    )

    return {
        "file_path": path,
        "matched_terms": matched_terms,
        "snippets": snippets,
        "last_modified": _get_mtime_iso(path),
    }


def search_codebase(
    query_terms: list[str],
    search_path: str = ".",
) -> dict[str, Any]:
    """Search the local filesystem for files containing the given query terms.

    Walks ``search_path`` recursively, skipping hidden and build directories.
    Ranks results by number of matched terms (descending) then by file recency
    (descending).  Returns at most ``SEARCH_MAX_MATCHES`` results.

    Args:
        query_terms: List of strings to search for (case-insensitive).
        search_path: Root directory to search.  Defaults to the current
            working directory.

    Returns:
        A dict with keys:

        - ``matches`` (list[dict]): Ranked list of matching files, each with
          ``file_path``, ``matched_terms``, ``snippets``, ``last_modified``.
        - ``total_files_scanned`` (int): How many files were examined.
        - ``search_path`` (str): The resolved search root used.
    """
    resolved_path = os.path.abspath(search_path)

    if not query_terms:
        return {
            "matches": [],
            "total_files_scanned": 0,
            "search_path": resolved_path,
        }

    # Filter out empty terms
    terms = [t.strip() for t in query_terms if t and t.strip()]
    if not terms:
        return {
            "matches": [],
            "total_files_scanned": 0,
            "search_path": resolved_path,
        }

    raw_matches: list[dict[str, Any]] = []
    total_scanned = 0

    try:
        for dirpath, dirnames, filenames in os.walk(resolved_path, followlinks=False):
            # Prune skipped directories in-place so os.walk won't descend
            dirnames[:] = [
                d for d in dirnames
                if d not in SKIP_DIRS and not d.startswith(".")
            ]

            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                if not _is_text_file(full_path):
                    continue

                total_scanned += 1
                content = _read_file_safe(full_path)
                if content is None:
                    continue

                result = _search_file(full_path, content, terms)
                if result is not None:
                    raw_matches.append(result)

    except OSError as exc:
        # Non-fatal: return what we have so far
        return {
            "matches": raw_matches[:SEARCH_MAX_MATCHES],
            "total_files_scanned": total_scanned,
            "search_path": resolved_path,
            "error": f"Filesystem walk error: {exc}",
        }

    # Sort: most terms matched first, then most recently modified first
    raw_matches.sort(
        key=lambda m: (len(m["matched_terms"]), m["last_modified"]),
        reverse=True,
    )

    return {
        "matches": raw_matches[:SEARCH_MAX_MATCHES],
        "total_files_scanned": total_scanned,
        "search_path": resolved_path,
    }
