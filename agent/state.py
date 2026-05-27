"""
AgentState — LangGraph TypedDict for the RCA agent.

Every node reads from and writes to this shared state object.

Important: any field that may be written by multiple concurrent nodes
(i.e. nodes in the parallel fan-out) MUST use an Annotated reducer so
LangGraph knows how to merge the concurrent updates.  Without a reducer,
LangGraph raises InvalidUpdateError when two branches write to the same key.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict

from schemas import RCAOutput


class AgentState(TypedDict, total=False):
    """Shared mutable state passed between LangGraph nodes.

    All fields are optional (``total=False``) so that each node can return
    a partial dict and LangGraph will merge it with the existing state.

    ``tool_calls_log`` uses ``operator.add`` as its reducer so that concurrent
    parallel nodes can each append their own entry without conflict — LangGraph
    concatenates the lists from both branches automatically.
    """

    raw_input: str
    """The original stack trace / log text provided by the user."""

    parsed_trace: dict[str, Any] | None
    """Output of the ``parse_stack_trace`` tool."""

    codebase_results: dict[str, Any] | None
    """Output of the ``search_codebase`` tool."""

    git_results: dict[str, Any] | None
    """Output of the ``check_git_history`` tool."""

    rca_output: RCAOutput | None
    """The fully validated RCA result produced by the analysis node."""

    error: str | None
    """Non-None if a node encountered a fatal error; triggers the error path."""

    start_time: float
    """Unix timestamp (from ``time.perf_counter()``) recorded at agent start."""

    tool_calls_log: Annotated[list[dict[str, Any]], operator.add]
    """Append-only log of every tool invocation with timing metadata.

    The ``operator.add`` reducer tells LangGraph to *concatenate* the lists
    returned by concurrent nodes instead of raising InvalidUpdateError.
    Each node returns only its **new** entries; the reducer merges them.
    """
