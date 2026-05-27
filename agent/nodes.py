"""
LangGraph node functions for the Incident RCA Agent.

Each node receives the full AgentState and returns a *partial* dict that
LangGraph merges into the state.  Nodes are synchronous (no asyncio).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from rich.console import Console

from config import SNIPPET_TRUNCATE_CHARS, TOP_CODEBASE_RESULTS, TOP_GIT_COMMITS
from llm_client import GeminiClient
from prompts import (
    FALLBACK_ERROR_SIGNATURE,
    FALLBACK_POSTMORTEM,
    FALLBACK_ROOT_CAUSE,
    SYSTEM_PROMPT_RCA,
    USER_PROMPT_RCA_TEMPLATE,
)
from schemas import AffectedService, FixStep, RCAOutput
from tools.check_git_history import check_git_history
from tools.parse_stack_trace import parse_stack_trace
from tools.search_codebase import search_codebase

_console = Console(stderr=True)

# Lazily created single GeminiClient instance shared across nodes in one run
_gemini_client: GeminiClient | None = None


def _get_client() -> GeminiClient:
    """Return the singleton GeminiClient, creating it if needed.

    Returns:
        Shared GeminiClient instance.
    """
    global _gemini_client  # noqa: PLW0603
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client


def _make_log_entry(
    tool: str,
    input_summary: str,
    output_summary: str,
    duration_ms: float,
) -> list[dict[str, Any]]:
    """Return a single-item list containing a new tool-call log entry.

    Each node returns only its *new* entry.  The ``operator.add`` reducer
    on ``AgentState.tool_calls_log`` concatenates entries from concurrent
    parallel branches automatically — so nodes must NOT accumulate the full
    log themselves.

    Args:
        tool: Tool name string.
        input_summary: Short description of the input.
        output_summary: Short description of the output.
        duration_ms: Time taken in milliseconds.

    Returns:
        A one-element list containing the new log entry dict.
    """
    entry: dict[str, Any] = {
        "tool": tool,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "duration_ms": round(duration_ms, 2),
    }
    return [entry]


# ---------------------------------------------------------------------------
# Node 1 — Parse input
# ---------------------------------------------------------------------------


def node_parse_input(state: dict[str, Any]) -> dict[str, Any]:
    """Parse the raw stack trace input into structured fields.

    Args:
        state: Current AgentState dict.

    Returns:
        Partial state update with ``parsed_trace``, ``tool_calls_log``,
        and optionally ``error``.
    """
    raw = state.get("raw_input", "")
    _console.print("[bold blue]▶ node_parse_input[/bold blue]")

    t0 = time.perf_counter()
    try:
        result = parse_stack_trace(raw)
        duration_ms = (time.perf_counter() - t0) * 1000

        log = _make_log_entry(
            tool="parse_stack_trace",
            input_summary=f"Input length: {len(raw)} chars",
            output_summary=(
                f"language={result.get('language')}, "
                f"exception={result.get('exception_type')}, "
                f"frames={len(result.get('frames', []))}"
            ),
            duration_ms=duration_ms,
        )

        _console.print(
            f"  [green]✓[/green] Detected [cyan]{result.get('language')}[/cyan] "
            f"— {result.get('exception_type')}: {result.get('exception_message', '')[:80]}"
        )

        return {
            "parsed_trace": result,
            "tool_calls_log": log,
            "error": None,
        }

    except Exception as exc:  # noqa: BLE001
        duration_ms = (time.perf_counter() - t0) * 1000
        error_msg = f"parse_stack_trace failed: {exc}"
        _console.print(f"  [red]✗ {error_msg}[/red]")

        log = _make_log_entry(
            tool="parse_stack_trace",
            input_summary=f"Input length: {len(raw)} chars",
            output_summary=f"ERROR: {error_msg}",
            duration_ms=duration_ms,
        )
        return {
            "parsed_trace": None,
            "tool_calls_log": log,
            "error": error_msg,
        }


# ---------------------------------------------------------------------------
# Node 2 — Search codebase
# ---------------------------------------------------------------------------


def node_search_codebase(state: dict[str, Any]) -> dict[str, Any]:
    """Search the local codebase for symbols from the parsed trace.

    Args:
        state: Current AgentState dict.

    Returns:
        Partial state update with ``codebase_results`` and ``tool_calls_log``.
    """
    _console.print("[bold blue]▶ node_search_codebase[/bold blue]")

    parsed: dict[str, Any] = state.get("parsed_trace") or {}
    root_frame: dict[str, Any] = parsed.get("root_frame") or {}

    exception_type = parsed.get("exception_type", "")
    root_func = root_frame.get("function", "")
    root_file_basename = os.path.basename(root_frame.get("file", ""))

    # Build query terms, filter out empty strings
    query_terms: list[str] = [
        t for t in [exception_type, root_func, root_file_basename] if t
    ]

    search_path = os.getcwd()

    t0 = time.perf_counter()
    try:
        result = search_codebase(query_terms, search_path=search_path)
        duration_ms = (time.perf_counter() - t0) * 1000

        match_count = len(result.get("matches", []))
        scanned = result.get("total_files_scanned", 0)

        log = _make_log_entry(
            tool="search_codebase",
            input_summary=f"terms={query_terms}, path={search_path}",
            output_summary=f"matches={match_count}, scanned={scanned} files",
            duration_ms=duration_ms,
        )
        _console.print(
            f"  [green]✓[/green] Found {match_count} matches "
            f"in {scanned} scanned files"
        )
        return {
            "codebase_results": result,
            "tool_calls_log": log,
        }

    except Exception as exc:  # noqa: BLE001
        duration_ms = (time.perf_counter() - t0) * 1000
        error_msg = f"search_codebase failed: {exc}"
        _console.print(f"  [yellow]⚠ {error_msg}[/yellow]")

        log = _make_log_entry(
            tool="search_codebase",
            input_summary=f"terms={query_terms}",
            output_summary=f"ERROR: {error_msg}",
            duration_ms=duration_ms,
        )
        return {
            "codebase_results": {"matches": [], "total_files_scanned": 0, "error": error_msg},
            "tool_calls_log": log,
        }


# ---------------------------------------------------------------------------
# Node 3 — Check git history
# ---------------------------------------------------------------------------


def node_check_git(state: dict[str, Any]) -> dict[str, Any]:
    """Inspect git history for commits related to the exception.

    Args:
        state: Current AgentState dict.

    Returns:
        Partial state update with ``git_results`` and ``tool_calls_log``.
    """
    _console.print("[bold blue]▶ node_check_git[/bold blue]")

    parsed: dict[str, Any] = state.get("parsed_trace") or {}
    root_frame: dict[str, Any] = parsed.get("root_frame") or {}

    exception_type = parsed.get("exception_type", "")
    root_func = root_frame.get("function", "")

    keywords: list[str] = [k for k in [exception_type, root_func] if k]
    repo_path = os.getcwd()

    t0 = time.perf_counter()
    try:
        result = check_git_history(keywords, repo_path=repo_path)
        duration_ms = (time.perf_counter() - t0) * 1000

        commit_count = len(result.get("relevant_commits", []))
        has_git = result.get("has_git", False)

        log = _make_log_entry(
            tool="check_git_history",
            input_summary=f"keywords={keywords}, repo={repo_path}",
            output_summary=(
                f"has_git={has_git}, relevant_commits={commit_count}"
                + (f", error={result.get('error')}" if result.get("error") else "")
            ),
            duration_ms=duration_ms,
        )
        _console.print(
            f"  [green]✓[/green] has_git={has_git}, "
            f"relevant commits={commit_count}"
        )
        return {
            "git_results": result,
            "tool_calls_log": log,
        }

    except Exception as exc:  # noqa: BLE001
        duration_ms = (time.perf_counter() - t0) * 1000
        error_msg = f"check_git_history failed: {exc}"
        _console.print(f"  [yellow]⚠ {error_msg}[/yellow]")

        log = _make_log_entry(
            tool="check_git_history",
            input_summary=f"keywords={keywords}",
            output_summary=f"ERROR: {error_msg}",
            duration_ms=duration_ms,
        )
        return {
            "git_results": {
                "relevant_commits": [],
                "recent_changes_to_related_files": [],
                "has_git": False,
                "error": error_msg,
            },
            "tool_calls_log": log,
        }


# ---------------------------------------------------------------------------
# Prompt-building helpers
# ---------------------------------------------------------------------------


def _format_codebase_snippets(codebase_results: dict[str, Any] | None) -> str:
    """Format the top codebase matches into a prompt-ready string.

    Args:
        codebase_results: Output dict from ``search_codebase``.

    Returns:
        Human-readable string of file snippets, or a 'no matches' message.
    """
    if not codebase_results:
        return "No codebase search results available."

    matches: list[dict[str, Any]] = codebase_results.get("matches", [])
    if not matches:
        return (
            f"No matches found. "
            f"Files scanned: {codebase_results.get('total_files_scanned', 0)}."
        )

    parts: list[str] = []
    for match in matches[:TOP_CODEBASE_RESULTS]:
        path = match.get("file_path", "unknown")
        terms = ", ".join(match.get("matched_terms", []))
        snippets: list[str] = match.get("snippets", [])
        parts.append(f"### File: {path}\nMatched terms: {terms}")
        for snippet in snippets[:2]:
            truncated = snippet[:SNIPPET_TRUNCATE_CHARS]
            if len(snippet) > SNIPPET_TRUNCATE_CHARS:
                truncated += "…"
            parts.append(f"```\n{truncated}\n```")

    return "\n\n".join(parts)


def _format_git_history(git_results: dict[str, Any] | None) -> str:
    """Format relevant git commits into a prompt-ready string.

    Args:
        git_results: Output dict from ``check_git_history``.

    Returns:
        Human-readable string of commit summaries.
    """
    if not git_results:
        return "No git history available."
    if not git_results.get("has_git"):
        reason = git_results.get("error", "Not a git repository.")
        return f"Git not available: {reason}"

    commits: list[dict[str, Any]] = git_results.get("relevant_commits", [])
    recent_files: list[str] = git_results.get("recent_changes_to_related_files", [])

    parts: list[str] = []
    if commits:
        parts.append(f"**Relevant commits ({len(commits)} found):**")
        for commit in commits[:TOP_GIT_COMMITS]:
            files = ", ".join(commit.get("changed_files", [])[:5])
            parts.append(
                f"- [{commit.get('hash')}] {commit.get('date', '')} "
                f"by {commit.get('author', 'Unknown')}: "
                f"{commit.get('message', '')}\n"
                f"  Changed: {files or 'N/A'}"
            )
    else:
        parts.append("No relevant commits found in the last 30 days.")

    if recent_files:
        parts.append(
            f"\n**Files changed in the last 7 days:**\n"
            + "\n".join(f"- {f}" for f in recent_files[:10])
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Node 4 — LLM Analysis
# ---------------------------------------------------------------------------


def node_analyse(state: dict[str, Any]) -> dict[str, Any]:
    """Invoke the Gemini model to produce a structured RCA result.

    Builds a rich prompt from all gathered context and calls the LLM via
    GeminiClient.generate_structured().

    Args:
        state: Current AgentState dict.

    Returns:
        Partial state update with ``rca_output`` and ``tool_calls_log``.
    """
    _console.print("[bold blue]▶ node_analyse[/bold blue]")

    raw_input: str = state.get("raw_input", "")
    parsed_trace: dict[str, Any] = state.get("parsed_trace") or {}
    codebase_results: dict[str, Any] | None = state.get("codebase_results")
    git_results: dict[str, Any] | None = state.get("git_results")

    # Format context sections
    try:
        parsed_trace_json = json.dumps(parsed_trace, indent=2, default=str)
    except (TypeError, ValueError):
        parsed_trace_json = str(parsed_trace)

    codebase_snippets = _format_codebase_snippets(codebase_results)
    git_history = _format_git_history(git_results)

    user_prompt = USER_PROMPT_RCA_TEMPLATE.format(
        raw_input=raw_input[:3000],  # truncate very long inputs
        parsed_trace_json=parsed_trace_json,
        codebase_snippets=codebase_snippets,
        git_history=git_history,
    )

    t0 = time.perf_counter()
    try:
        client = _get_client()
        rca: RCAOutput = client.generate_structured(  # type: ignore[assignment]
            system_prompt=SYSTEM_PROMPT_RCA,
            user_prompt=user_prompt,
            schema=RCAOutput,
        )
        duration_ms = (time.perf_counter() - t0) * 1000

        log = _make_log_entry(
            tool="gemini_analyse",
            input_summary=f"prompt_chars={len(user_prompt)}",
            output_summary=(
                f"root_cause_category={rca.root_cause_category}, "
                f"confidence={rca.confidence_score:.2f}, "
                f"fix_steps={len(rca.fix_steps)}"
            ),
            duration_ms=duration_ms,
        )

        _console.print(
            f"  [green]✓[/green] RCA complete — "
            f"category={rca.root_cause_category}, "
            f"confidence={rca.confidence_label} ({rca.confidence_score:.2f})"
        )

        return {
            "rca_output": rca,
            "tool_calls_log": log,
        }

    except Exception as exc:  # noqa: BLE001
        duration_ms = (time.perf_counter() - t0) * 1000
        error_msg = f"LLM analysis failed: {exc}"
        _console.print(f"  [red]✗ {error_msg}[/red]")

        log = _make_log_entry(
            tool="gemini_analyse",
            input_summary=f"prompt_chars={len(user_prompt)}",
            output_summary=f"ERROR: {error_msg}",
            duration_ms=duration_ms,
        )
        return {
            "rca_output": None,
            "tool_calls_log": log,
            "error": error_msg,
        }


# ---------------------------------------------------------------------------
# Node 5 — Error handler
# ---------------------------------------------------------------------------


def node_handle_error(state: dict[str, Any]) -> dict[str, Any]:
    """Produce a minimal fallback RCAOutput when the pipeline has failed.

    Args:
        state: Current AgentState dict (may have ``error`` set).

    Returns:
        Partial state update with a fallback ``rca_output``.
    """
    error_msg = state.get("error", "Unknown error.")
    _console.print(f"[bold red]▶ node_handle_error:[/bold red] {error_msg}")

    fallback_rca = RCAOutput(
        root_cause=FALLBACK_ROOT_CAUSE,
        root_cause_category="unknown",
        affected_services=[],
        fix_steps=[
            FixStep(
                step_number=1,
                action="Review the error message in the Debug Info panel",
                rationale="The agent encountered an internal error and could not complete analysis.",
                priority="immediate",
            )
        ],
        confidence_score=0.0,
        confidence_label="low",
        postmortem_draft=FALLBACK_POSTMORTEM,
        similar_known_issues=[],
        error_signature=FALLBACK_ERROR_SIGNATURE,
    )

    return {"rca_output": fallback_rca}
