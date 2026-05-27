"""
Tool 1: parse_stack_trace

Pure regex / string-based parser — no LLM calls, no external I/O.

Detects the programming language automatically and extracts structured
information from stack trace text in Python, Java, Node.js, Go, Ruby,
and generic unknown formats.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


# ---------------------------------------------------------------------------
# Language detection patterns
# ---------------------------------------------------------------------------

_PYTHON_MARKER = re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE)
_JAVA_MARKER = re.compile(
    r"(Exception in thread|Caused by:|at\s+[\w$.]+\([\w.]+:\d+\))", re.IGNORECASE
)
_NODE_MARKER = re.compile(
    r"(UnhandledPromiseRejection|Error:.*\n\s+at\s|at\s+\S+\s+\(.*:\d+:\d+\))",
    re.IGNORECASE,
)
_GO_MARKER = re.compile(r"goroutine\s+\d+|panic:", re.IGNORECASE)
_RUBY_MARKER = re.compile(r"\.rb:\d+:in\s+`", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Python parsing patterns
# ---------------------------------------------------------------------------

_PY_FRAME = re.compile(
    r'File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)\s*\n\s*(?P<code>.+)?',
)
_PY_EXCEPTION = re.compile(r"^(?P<type>[\w.]+(?:Error|Exception|Warning|Interrupt)): (?P<msg>.+)$", re.MULTILINE)

# ---------------------------------------------------------------------------
# Java parsing patterns
# ---------------------------------------------------------------------------

_JAVA_EXCEPTION_LINE = re.compile(
    r"^(?:Exception in thread \"[^\"]*\"\s+)?(?P<type>[\w$.]+(?:Exception|Error|Throwable)): (?P<msg>.+)$",
    re.MULTILINE,
)
_JAVA_CAUSED_BY = re.compile(
    r"^Caused by:\s+(?P<type>[\w$.]+(?:Exception|Error|Throwable)): (?P<msg>.+)$",
    re.MULTILINE,
)
_JAVA_FRAME = re.compile(
    r"^\s+at\s+(?P<class>[\w$.]+)\.(?P<method>[\w$<>]+)\((?P<file>[^:)]+)(?::(?P<line>\d+))?\)$",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Node.js parsing patterns
# ---------------------------------------------------------------------------

_NODE_ERROR_LINE = re.compile(
    r"^(?:(?:Unhandled(?:Rejection|PromiseRejectionWarning)|Error):\s*)?(?P<type>[\w]+(?:Error|Exception|Rejection)?):\s*(?P<msg>.+)$",
    re.MULTILINE,
)
_NODE_FRAME = re.compile(
    r"^\s+at\s+(?P<func>[^\(]+?)\s+\((?P<file>[^:)]+):(?P<line>\d+):\d+\)$",
    re.MULTILINE,
)
_NODE_FRAME_ANON = re.compile(
    r"^\s+at\s+(?P<file>[^:]+):(?P<line>\d+):\d+$",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Go parsing patterns
# ---------------------------------------------------------------------------

_GO_PANIC = re.compile(r"^panic:\s+(?P<msg>.+)$", re.MULTILINE)
_GO_FRAME = re.compile(
    r"^(?P<func>[\w./]+)\(.*\)\n\s+(?P<file>[^:]+):(?P<line>\d+)",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Ruby parsing patterns
# ---------------------------------------------------------------------------

_RUBY_ERROR = re.compile(r"^(?P<file>[^:]+):(?P<line>\d+):in\s+`(?P<func>[^']+)': (?P<msg>.+) \((?P<type>\w+)\)$", re.MULTILINE)
_RUBY_FRAME = re.compile(r"^\s+from (?P<file>[^:]+):(?P<line>\d+):in\s+`(?P<func>[^']+)'$", re.MULTILINE)

# ---------------------------------------------------------------------------
# Generic file:line pattern (fallback)
# ---------------------------------------------------------------------------

_GENERIC_FILE_LINE = re.compile(
    r'(?P<file>[\w./\\-]+\.\w+)[:\(](?P<line>\d+)',
)

# ---------------------------------------------------------------------------
# Directories / modules to classify as stdlib / vendor (for root frame heuristic)
# ---------------------------------------------------------------------------

_STDLIB_PREFIXES: tuple[str, ...] = (
    "/usr/lib", "/usr/local/lib", "site-packages", "dist-packages",
    "node_modules", "/lib/node", "<anonymous>", "<internal",
    "java.lang", "java.util", "java.io", "sun.", "com.sun.",
    "org.springframework.boot", "org.apache.catalina",
    "jdk.", "java.base",
)


def _is_vendor_frame(file_path: str, func: str) -> bool:
    """Return True if a frame looks like stdlib or third-party code.

    Args:
        file_path: File path string from the frame.
        func: Function name from the frame.

    Returns:
        True if the frame should be skipped when selecting the root frame.
    """
    combined = f"{file_path} {func}".lower()
    return any(prefix.lower() in combined for prefix in _STDLIB_PREFIXES)


def _compute_error_hash(exception_type: str, root_file: str, root_func: str) -> str:
    """Compute a stable MD5 fingerprint for an error class.

    Args:
        exception_type: The exception type name.
        root_file: File path of the root frame.
        root_func: Function name of the root frame.

    Returns:
        8-character hex MD5 digest.
    """
    raw = f"{exception_type}:{root_file}:{root_func}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Language-specific parsers
# ---------------------------------------------------------------------------


def _parse_python(text: str) -> dict[str, Any]:
    """Parse a Python traceback string.

    Args:
        text: Raw stack trace text.

    Returns:
        Partial result dict with language-specific fields.
    """
    frames: list[dict[str, Any]] = []
    for m in _PY_FRAME.finditer(text):
        frames.append({
            "file": m.group("file"),
            "line": int(m.group("line")),
            "function": m.group("func"),
            "code_snippet": (m.group("code") or "").strip(),
        })

    exception_type = "UnknownException"
    exception_message = ""
    exc_match = _PY_EXCEPTION.search(text)
    if exc_match:
        exception_type = exc_match.group("type")
        exception_message = exc_match.group("msg")

    return {
        "language": "python",
        "exception_type": exception_type,
        "exception_message": exception_message,
        "frames": frames,
    }


def _parse_java(text: str) -> dict[str, Any]:
    """Parse a Java / JVM stack trace.

    Args:
        text: Raw stack trace text.

    Returns:
        Partial result dict with language-specific fields.
    """
    frames: list[dict[str, Any]] = []
    for m in _JAVA_FRAME.finditer(text):
        frames.append({
            "file": f"{m.group('class').replace('.', '/')}.java",
            "line": int(m.group("line")) if m.group("line") else 0,
            "function": f"{m.group('class')}.{m.group('method')}",
            "code_snippet": "",
        })

    exception_type = "UnknownException"
    exception_message = ""

    # Prefer "Caused by" as the deepest root cause
    caused = list(_JAVA_CAUSED_BY.finditer(text))
    if caused:
        last = caused[-1]
        exception_type = last.group("type").split(".")[-1]
        exception_message = last.group("msg")
    else:
        main_match = _JAVA_EXCEPTION_LINE.search(text)
        if main_match:
            exception_type = main_match.group("type").split(".")[-1]
            exception_message = main_match.group("msg")

    return {
        "language": "java",
        "exception_type": exception_type,
        "exception_message": exception_message,
        "frames": frames,
    }


def _parse_node(text: str) -> dict[str, Any]:
    """Parse a Node.js / JavaScript stack trace.

    Args:
        text: Raw stack trace text.

    Returns:
        Partial result dict with language-specific fields.
    """
    frames: list[dict[str, Any]] = []
    for m in _NODE_FRAME.finditer(text):
        frames.append({
            "file": m.group("file"),
            "line": int(m.group("line")),
            "function": m.group("func").strip(),
            "code_snippet": "",
        })
    for m in _NODE_FRAME_ANON.finditer(text):
        frames.append({
            "file": m.group("file"),
            "line": int(m.group("line")),
            "function": "<anonymous>",
            "code_snippet": "",
        })

    exception_type = "Error"
    exception_message = ""

    # Detect UnhandledPromiseRejection
    unhandled = re.search(
        r"UnhandledPromiseRejectionWarning:\s*([\w]+(?:Error|Exception)?):\s*(.+)",
        text,
        re.IGNORECASE,
    )
    if unhandled:
        exception_type = unhandled.group(1)
        exception_message = unhandled.group(2)
    else:
        err_match = re.search(r"^([\w]+(?:Error|Exception)):\s*(.+)$", text, re.MULTILINE)
        if err_match:
            exception_type = err_match.group(1)
            exception_message = err_match.group(2)

    return {
        "language": "nodejs",
        "exception_type": exception_type,
        "exception_message": exception_message,
        "frames": frames,
    }


def _parse_go(text: str) -> dict[str, Any]:
    """Parse a Go panic / runtime stack trace.

    Args:
        text: Raw stack trace text.

    Returns:
        Partial result dict with language-specific fields.
    """
    frames: list[dict[str, Any]] = []
    for m in _GO_FRAME.finditer(text):
        frames.append({
            "file": m.group("file"),
            "line": int(m.group("line")),
            "function": m.group("func"),
            "code_snippet": "",
        })

    exception_type = "panic"
    exception_message = ""
    panic_match = _GO_PANIC.search(text)
    if panic_match:
        exception_message = panic_match.group("msg")

    return {
        "language": "go",
        "exception_type": exception_type,
        "exception_message": exception_message,
        "frames": frames,
    }


def _parse_ruby(text: str) -> dict[str, Any]:
    """Parse a Ruby exception / backtrace.

    Args:
        text: Raw stack trace text.

    Returns:
        Partial result dict with language-specific fields.
    """
    frames: list[dict[str, Any]] = []
    exception_type = "RuntimeError"
    exception_message = ""

    first = _RUBY_ERROR.search(text)
    if first:
        exception_type = first.group("type")
        exception_message = first.group("msg")
        frames.append({
            "file": first.group("file"),
            "line": int(first.group("line")),
            "function": first.group("func"),
            "code_snippet": "",
        })

    for m in _RUBY_FRAME.finditer(text):
        frames.append({
            "file": m.group("file"),
            "line": int(m.group("line")),
            "function": m.group("func"),
            "code_snippet": "",
        })

    return {
        "language": "ruby",
        "exception_type": exception_type,
        "exception_message": exception_message,
        "frames": frames,
    }


def _parse_generic(text: str) -> dict[str, Any]:
    """Fallback parser for unknown formats.

    Extracts any file:line references it can find.

    Args:
        text: Raw stack trace text.

    Returns:
        Partial result dict with language-specific fields.
    """
    frames: list[dict[str, Any]] = []
    seen: set[str] = set()
    for m in _GENERIC_FILE_LINE.finditer(text):
        key = f"{m.group('file')}:{m.group('line')}"
        if key not in seen:
            seen.add(key)
            frames.append({
                "file": m.group("file"),
                "line": int(m.group("line")),
                "function": "unknown",
                "code_snippet": "",
            })

    # Try to grab an error message from the first line that looks like one
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    exception_type = "UnknownError"
    exception_message = first_line[:200]

    return {
        "language": "generic",
        "exception_type": exception_type,
        "exception_message": exception_message,
        "frames": frames,
    }


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def _detect_language(text: str) -> str:
    """Auto-detect the programming language of a stack trace.

    Args:
        text: Raw stack trace text.

    Returns:
        One of: "python", "java", "nodejs", "go", "ruby", "generic".
    """
    if _PYTHON_MARKER.search(text):
        return "python"
    if _GO_MARKER.search(text):
        return "go"
    if _RUBY_MARKER.search(text):
        return "ruby"
    if _NODE_MARKER.search(text):
        return "nodejs"
    if _JAVA_MARKER.search(text):
        return "java"
    return "generic"


# ---------------------------------------------------------------------------
# Root frame selection
# ---------------------------------------------------------------------------


def _select_root_frame(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the innermost application frame (skip stdlib/vendor frames).

    Iterates the frame list in reverse (innermost first) and returns the
    first frame that does not look like stdlib or vendor code.

    Args:
        frames: Ordered list of parsed frames (outermost first).

    Returns:
        The selected root frame dict, or an empty dict if frames is empty.
    """
    if not frames:
        return {}
    for frame in reversed(frames):
        if not _is_vendor_frame(frame.get("file", ""), frame.get("function", "")):
            return frame
    # If all frames look like vendor code, just return the innermost
    return frames[-1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_stack_trace(raw_input: str) -> dict[str, Any]:
    """Parse a stack trace string into structured fields.

    Supports Python, Java, Node.js, Go, Ruby, and a generic fallback.
    No LLM calls are made — purely regex / string parsing.

    Args:
        raw_input: The raw log / stack trace text to analyse.

    Returns:
        A dict with the following keys:

        - ``language`` (str): Detected language.
        - ``exception_type`` (str): Exception class name.
        - ``exception_message`` (str): Exception message text.
        - ``frames`` (list[dict]): All parsed frames, each with keys
          ``file``, ``line``, ``function``, ``code_snippet``.
        - ``root_frame`` (dict): The innermost non-vendor frame.
        - ``error_hash`` (str): 8-char MD5 fingerprint of the error class.
    """
    if not raw_input or not raw_input.strip():
        return {
            "language": "generic",
            "exception_type": "UnknownError",
            "exception_message": "Empty input provided.",
            "frames": [],
            "root_frame": {},
            "error_hash": _compute_error_hash("UnknownError", "", ""),
        }

    language = _detect_language(raw_input)

    parsers = {
        "python": _parse_python,
        "java": _parse_java,
        "nodejs": _parse_node,
        "go": _parse_go,
        "ruby": _parse_ruby,
        "generic": _parse_generic,
    }

    try:
        partial = parsers[language](raw_input)
    except Exception as exc:  # noqa: BLE001
        # If the language-specific parser fails, fall back to generic
        partial = _parse_generic(raw_input)
        partial["parse_warning"] = f"Language parser '{language}' raised: {exc}"

    frames = partial.get("frames", [])
    root_frame = _select_root_frame(frames)
    error_hash = _compute_error_hash(
        partial.get("exception_type", "UnknownError"),
        root_frame.get("file", ""),
        root_frame.get("function", ""),
    )

    return {
        "language": partial.get("language", language),
        "exception_type": partial.get("exception_type", "UnknownError"),
        "exception_message": partial.get("exception_message", ""),
        "frames": frames,
        "root_frame": root_frame,
        "error_hash": error_hash,
    }
