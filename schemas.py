"""
Pydantic v2 schemas for the Incident RCA Agent.

All output models are defined here. Import and use these throughout the
agent pipeline — never define ad-hoc dicts for structured outputs.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class FixStep(BaseModel):
    """A single, actionable remediation step."""

    step_number: int = Field(..., ge=1, description="1-based ordering index.")
    action: str = Field(
        ...,
        min_length=1,
        description="Short imperative sentence describing the action.",
    )
    rationale: str = Field(
        ...,
        min_length=1,
        description="Why this action resolves or mitigates the issue.",
    )
    priority: Literal["immediate", "short_term", "long_term"] = Field(
        ...,
        description="Urgency tier for this fix step.",
    )


class AffectedService(BaseModel):
    """A service impacted by the incident."""

    name: str = Field(..., min_length=1, description="Service or component name.")
    impact: str = Field(
        ...,
        min_length=1,
        description="Brief description of how this service is affected.",
    )


class RCAOutput(BaseModel):
    """Complete root cause analysis result produced by the LLM analysis node."""

    root_cause: str = Field(
        ...,
        min_length=1,
        description="1-3 sentence precise diagnosis naming the exact cause.",
    )
    root_cause_category: Literal[
        "dependency_failure",
        "config_error",
        "code_bug",
        "infrastructure",
        "timeout",
        "auth_failure",
        "unknown",
    ] = Field(..., description="Categorised root cause type.")

    affected_services: list[AffectedService] = Field(
        default_factory=list,
        description="Services impacted by the incident.",
    )
    fix_steps: list[FixStep] = Field(
        ...,
        min_length=1,
        description="Ordered remediation steps, most urgent first.",
    )

    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Analyst confidence in this RCA, from 0.0 to 1.0.",
    )
    confidence_label: Literal["low", "medium", "high"] = Field(
        ...,
        description="Human-readable confidence tier derived from confidence_score.",
    )

    postmortem_draft: str = Field(
        ...,
        min_length=1,
        description="Full markdown postmortem following the required section structure.",
    )
    similar_known_issues: list[str] = Field(
        default_factory=list,
        description="Short descriptions of similar past incidents; empty if none.",
    )
    error_signature: str = Field(
        ...,
        min_length=1,
        description="Short unique fingerprint string for this error class.",
    )

    @field_validator("confidence_label", mode="before")
    @classmethod
    def derive_confidence_label(cls, v: Any, info: Any) -> str:
        """Allow the LLM to supply a label; validate it is consistent.

        If the value is already a valid literal, keep it.  If it is missing
        or empty, derive it from confidence_score when that field is available.
        """
        if v in ("low", "medium", "high"):
            return v
        # Fall back to derivation — will be reconciled by model_validator
        return v  # let model_validator handle it

    @model_validator(mode="after")
    def reconcile_confidence_label(self) -> "RCAOutput":
        """Ensure confidence_label is always consistent with confidence_score."""
        score = self.confidence_score
        if score >= 0.7:
            self.confidence_label = "high"
        elif score >= 0.4:
            self.confidence_label = "medium"
        else:
            self.confidence_label = "low"
        return self

    @model_validator(mode="after")
    def ensure_fix_steps_ordered(self) -> "RCAOutput":
        """Re-number fix steps so step_number is always 1-based sequential."""
        for idx, step in enumerate(self.fix_steps, start=1):
            step.step_number = idx
        return self


class AgentOutput(BaseModel):
    """Top-level output returned by run_agent()."""

    rca: RCAOutput = Field(..., description="The structured RCA result.")
    raw_stack_trace: str = Field(..., description="The original input string.")
    tool_outputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw outputs keyed by tool name.",
    )
    processing_time_seconds: float = Field(
        ...,
        ge=0.0,
        description="Wall-clock time for the full agent run.",
    )
    model_used: str = Field(..., description="Gemini model ID used for analysis.")
