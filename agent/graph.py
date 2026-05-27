"""
LangGraph graph definition and compilation for the RCA Agent.

Graph topology:

    START
      │
      ▼
    node_parse_input ─── (error?) ──► node_handle_error ──► END
      │
      ├──► node_search_codebase ──┐
      │                           ├──► node_analyse ──► END
      └──► node_check_git ────────┘

The two middle nodes run in parallel (fan-out / fan-in).  LangGraph natively
supports multiple outgoing edges from one node — both branches are dispatched
concurrently and node_analyse waits for both to finish.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from agent.nodes import (
    node_analyse,
    node_check_git,
    node_handle_error,
    node_parse_input,
    node_search_codebase,
)
from agent.state import AgentState
from config import MODEL_NAME
from schemas import AgentOutput, RCAOutput


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------


def _route_after_parse(state: dict[str, Any]) -> str | list[str]:
    """Decide what to execute after the parse step.

    If an error was recorded, route to the error handler.
    Otherwise fan out to *both* parallel nodes simultaneously by returning
    a list of node names — LangGraph dispatches them concurrently.

    Args:
        state: Current AgentState.

    Returns:
        Either ``"node_handle_error"`` (string) or a list of two node names
        for the parallel fan-out.
    """
    if state.get("error"):
        return "node_handle_error"
    # Return both targets as a list → LangGraph fans out concurrently
    return ["node_search_codebase", "node_check_git"]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_graph() -> Any:
    """Build and compile the LangGraph StateGraph.

    Returns:
        The compiled LangGraph application object.
    """
    builder: StateGraph = StateGraph(AgentState)  # type: ignore[arg-type]

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("node_parse_input", node_parse_input)
    builder.add_node("node_search_codebase", node_search_codebase)
    builder.add_node("node_check_git", node_check_git)
    builder.add_node("node_analyse", node_analyse)
    builder.add_node("node_handle_error", node_handle_error)

    # ── Entry point ───────────────────────────────────────────────────────
    builder.add_edge(START, "node_parse_input")

    # ── Conditional routing: error path OR parallel fan-out ───────────────
    builder.add_conditional_edges(
        "node_parse_input",
        _route_after_parse,
        {
            # Error path → single target
            "node_handle_error": "node_handle_error",
            # Parallel fan-out → both targets
            "node_search_codebase": "node_search_codebase",
            "node_check_git": "node_check_git",
        },
    )

    # ── Fan-in: both parallel nodes converge on node_analyse ─────────────
    builder.add_edge("node_search_codebase", "node_analyse")
    builder.add_edge("node_check_git", "node_analyse")

    # ── Terminal edges ────────────────────────────────────────────────────
    builder.add_edge("node_analyse", END)
    builder.add_edge("node_handle_error", END)

    return builder.compile()


# Compiled graph — module-level singleton
_compiled_graph = _build_graph()


# ---------------------------------------------------------------------------
# Public run_agent() entry point
# ---------------------------------------------------------------------------


def run_agent(raw_input: str) -> AgentOutput:
    """Run the full RCA pipeline for the given stack trace / log input.

    Initialises the AgentState, invokes the compiled graph, and wraps the
    final state into an :class:`~schemas.AgentOutput` instance.

    Args:
        raw_input: The raw error log or stack trace text to analyse.

    Returns:
        A fully populated AgentOutput with the RCA result and metadata.

    Raises:
        RuntimeError: If the graph finishes without producing an rca_output.
    """
    start_time = time.perf_counter()

    initial_state: AgentState = {
        "raw_input": raw_input,
        "parsed_trace": None,
        "codebase_results": None,
        "git_results": None,
        "rca_output": None,
        "error": None,
        "start_time": start_time,
        "tool_calls_log": [],
    }

    final_state: dict[str, Any] = _compiled_graph.invoke(initial_state)

    rca_output: RCAOutput | None = final_state.get("rca_output")
    if rca_output is None:
        raise RuntimeError(
            "Agent graph completed but produced no rca_output. "
            f"Error field: {final_state.get('error')}"
        )

    processing_time = time.perf_counter() - start_time

    tool_outputs: dict[str, Any] = {}
    if final_state.get("parsed_trace") is not None:
        tool_outputs["parse_stack_trace"] = final_state["parsed_trace"]
    if final_state.get("codebase_results") is not None:
        tool_outputs["search_codebase"] = final_state["codebase_results"]
    if final_state.get("git_results") is not None:
        tool_outputs["check_git_history"] = final_state["git_results"]

    return AgentOutput(
        rca=rca_output,
        raw_stack_trace=raw_input,
        tool_outputs=tool_outputs,
        processing_time_seconds=round(processing_time, 3),
        model_used=MODEL_NAME,
    )
