"""
Centralised configuration for the Incident RCA Agent.

All tuneable constants live here. Import from this module — never hardcode
values in other source files.
"""

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------

MODEL_NAME: str = "gemini-2.5-flash"
"""Gemini model identifier passed to the generativeai SDK.

Available flash models confirmed on this account:
  - gemini-3.1-flash-lite-preview   ← currently selected
  - gemini-3.5-flash
  - gemini-3-flash-preview
  - gemini-2.5-flash
  - gemini-2.5-flash-lite
  - gemini-2.0-flash
  - gemini-2.0-flash-lite
"""

TEMPERATURE: float = 0.2
"""Low temperature keeps the output deterministic and factual."""

MAX_OUTPUT_TOKENS: int = 8192
"""Maximum number of tokens the model may generate per response."""

MAX_RETRIES: int = 3
"""Number of retry attempts on transient API errors.

On 429 responses the retry waits the duration advertised in the error
``retryDelay`` field rather than using a fixed exponential back-off.
"""

# ---------------------------------------------------------------------------
# Tool settings
# ---------------------------------------------------------------------------

SEARCH_PATH: str = "."
"""Default filesystem root for the codebase search tool."""

GIT_LOOKBACK_DAYS: int = 30
"""How many days back git history is scanned for relevant commits."""

GIT_MAX_COMMITS: int = 50
"""Upper bound on commits fetched from git log."""

SEARCH_MAX_MATCHES: int = 10
"""Maximum total file matches returned by the codebase search tool."""

SEARCH_SNIPPET_CONTEXT_LINES: int = 3
"""Lines of context shown around each match in search snippets."""

SEARCH_MAX_SNIPPETS_PER_FILE: int = 3
"""Maximum number of match snippets extracted from a single file."""

SNIPPET_TRUNCATE_CHARS: int = 300
"""Characters at which each codebase snippet is truncated in the analysis prompt."""

TOP_CODEBASE_RESULTS: int = 5
"""How many codebase matches are injected into the analysis prompt."""

TOP_GIT_COMMITS: int = 5
"""How many git commits are injected into the analysis prompt."""

# ---------------------------------------------------------------------------
# File extension allow-list for codebase search
# ---------------------------------------------------------------------------

SEARCHABLE_EXTENSIONS: tuple[str, ...] = (
    ".py", ".js", ".ts", ".java", ".go", ".rb",
    ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".env.example",
)

# ---------------------------------------------------------------------------
# Directories to skip during codebase search
# ---------------------------------------------------------------------------

SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".tox",
})
