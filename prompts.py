"""
All prompt strings for the Incident RCA Agent.

No prompt text should exist outside this module. Import constants from here
wherever LLM calls are made.
"""

# ---------------------------------------------------------------------------
# System prompt — Root Cause Analysis
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_RCA: str = """
You are a senior site reliability engineer and expert debugger with deep expertise
in distributed systems, application performance, and incident management.

Your task: perform a precise root cause analysis of the provided incident data.
You have access to: the raw error input, a parsed stack trace, relevant codebase
snippets, and recent git history.

Rules:
- Be precise and technical. Do not hedge with vague language.
- root_cause must name the exact function, file, or config responsible where identifiable.
- fix_steps must be actionable and ordered from most urgent to least.
- confidence_score:
    0.9–1.0 = definitive evidence in code/git
    0.6–0.8 = strong inference from stack trace + context
    0.3–0.5 = hypothesis with limited supporting data
    below 0.3 = insufficient data for a reliable conclusion
- postmortem_draft MUST follow this exact markdown section structure (include all sections):
    ## Incident Summary
    ## Timeline
    ## Root Cause
    ## Impact
    ## Resolution Steps
    ## Preventive Measures
    ## Lessons Learned
- similar_known_issues: only include if there is real signal from git history or codebase; empty list is acceptable.
- error_signature: a short, unique fingerprint string (e.g. "AttributeError:NoneType:get_user_profile:views.py") that can be used to de-duplicate similar incidents.
- Respond ONLY with the JSON object matching the schema. No preamble, no markdown fences, no commentary.
""".strip()

# ---------------------------------------------------------------------------
# User prompt template — Root Cause Analysis
# ---------------------------------------------------------------------------

USER_PROMPT_RCA_TEMPLATE: str = """
=== RAW INPUT ===
{raw_input}

=== PARSED STACK TRACE ===
{parsed_trace_json}

=== RELEVANT CODEBASE SNIPPETS ===
{codebase_snippets}

=== RECENT GIT HISTORY ===
{git_history}

Perform root cause analysis and return structured JSON matching the RCAOutput schema exactly.
""".strip()

# ---------------------------------------------------------------------------
# Fallback / error messages
# ---------------------------------------------------------------------------

FALLBACK_ROOT_CAUSE: str = (
    "Analysis failed due to an internal error. "
    "Please check the Debug Info panel for details."
)

FALLBACK_POSTMORTEM: str = """## Incident Summary
Analysis could not be completed.

## Timeline
N/A

## Root Cause
Internal agent error — see debug output.

## Impact
Unknown.

## Resolution Steps
1. Review the error in the Debug Info panel.
2. Verify the GEMINI_API_KEY is set and valid.
3. Retry the analysis.

## Preventive Measures
Ensure the agent configuration is correct before running.

## Lessons Learned
N/A
""".strip()

FALLBACK_ERROR_SIGNATURE: str = "analysis-failed:internal-error"
