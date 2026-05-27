"""
GeminiClient — a thin, robust wrapper around google-genai (the modern SDK).

All LLM calls in the agent go through this class.  It handles:
  - API key configuration
  - Structured JSON output via response_schema
  - Exponential back-off retries on transient errors
  - Rich logging of every call (model, latency, token estimate)
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Type

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel
from rich.console import Console
from rich.markup import escape

from config import MAX_OUTPUT_TOKENS, MAX_RETRIES, MODEL_NAME, TEMPERATURE

load_dotenv()

_console = Console(stderr=True)


def _get_api_key() -> str:
    """Read GEMINI_API_KEY from the environment.

    Raises:
        EnvironmentError: If the key is absent or empty.
    """
    import os

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file or export it before running."
        )
    return key


def _extract_text(response: Any) -> str:
    """Extract the raw text from a google-genai response object.

    Tries multiple paths to accommodate different response shapes.

    Args:
        response: The raw response returned by ``client.models.generate_content``.

    Returns:
        The text content of the first candidate part.

    Raises:
        ValueError: If no text can be extracted from the response.
    """
    # Path 1 — modern SDK: response.text (shorthand for first part text)
    try:
        text = response.text
        if text is not None:
            return str(text)
    except (AttributeError, ValueError):
        pass

    # Path 2 — traverse candidates → content → parts
    try:
        part = response.candidates[0].content.parts[0]
        text = part.text
        if text is not None:
            return str(text)
    except (AttributeError, IndexError, TypeError):
        pass

    raise ValueError(
        "Could not extract text from Gemini response. "
        f"Response object type: {type(response)!r}. "
        "Check that the model returned content and that response_schema is compatible."
    )


def _extract_retry_delay(exc: Exception) -> float | None:
    """Parse the recommended retry delay (seconds) from a 429 API error.

    The Gemini API embeds a ``retryDelay`` field like ``"45s"`` inside the
    error detail when it returns RESOURCE_EXHAUSTED.  Honouring it avoids
    hammering the API with retries that will all fail anyway.

    Args:
        exc: The exception raised by the Gemini SDK.

    Returns:
        Delay in seconds, or None if it cannot be parsed.
    """
    text = str(exc)
    # Pattern: 'retryDelay': '45s'  or  "retryDelay": "45.5s"
    match = re.search(r"['\"]retryDelay['\"]:?\s*['\"]([\\d.]+)s", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    # Also try plain  "Please retry in 45.1s"
    match2 = re.search(r"retry in ([\d.]+)s", text, re.IGNORECASE)
    if match2:
        try:
            return float(match2.group(1))
        except ValueError:
            pass
    return None


class GeminiClient:
    """Wrapper around google-genai for the RCA Agent.

    Usage::

        client = GeminiClient()
        raw_json = client.generate(system_prompt, user_prompt, MySchema)
        result   = client.generate_structured(system_prompt, user_prompt, MySchema)
    """

    def __init__(self) -> None:
        """Initialise the client by reading the API key and configuring genai."""
        api_key = _get_api_key()
        self._client = genai.Client(api_key=api_key)
        self._model_name: str = MODEL_NAME
        _console.print(
            f"[bold green]GeminiClient[/bold green] initialised "
            f"(model=[cyan]{escape(MODEL_NAME)}[/cyan])"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[BaseModel] | None = None,
    ) -> str:
        """Call the Gemini API and return the raw text response.

        If ``response_schema`` is provided the call uses Gemini's structured
        output mode (``response_mime_type="application/json"``).

        Retries up to ``MAX_RETRIES`` times with exponential back-off on any
        ``Exception``.

        Args:
            system_prompt: The instruction context sent as the system turn.
            user_prompt: The user-facing content for this specific request.
            response_schema: Optional Pydantic model class used to constrain
                the JSON output shape.

        Returns:
            The raw string content returned by the model.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self._call_api(
                    system_prompt, user_prompt, response_schema, attempt
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                # Respect the API's own recommended retry delay when available
                api_delay = _extract_retry_delay(exc)
                if api_delay is not None:
                    wait = min(api_delay, 60.0)  # cap at 60s per attempt
                    _console.print(
                        f"[yellow]Attempt {attempt}/{MAX_RETRIES} failed (429 quota). "
                        f"API requests waiting {wait:.0f}s before retry…[/yellow]"
                    )
                else:
                    wait = 2 ** (attempt - 1)  # 1 s, 2 s, 4 s
                    _console.print(
                        f"[yellow]Attempt {attempt}/{MAX_RETRIES} failed: "
                        f"{escape(str(exc)[:120])}. Retrying in {wait}s…[/yellow]"
                    )
                if attempt < MAX_RETRIES:
                    time.sleep(wait)

        raise RuntimeError(
            f"Gemini API call failed after {MAX_RETRIES} attempts. "
            f"Last error: {last_exc}"
        ) from last_exc

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[BaseModel],
    ) -> BaseModel:
        """Call the Gemini API and return a validated Pydantic model instance.

        Calls :meth:`generate` with ``response_schema=schema``, then parses
        and validates the returned JSON string.

        Args:
            system_prompt: The instruction context sent as the system turn.
            user_prompt: The user-facing content for this specific request.
            schema: The Pydantic model class to parse the response into.

        Returns:
            A validated instance of ``schema``.

        Raises:
            ValueError: If the response cannot be parsed as valid JSON or
                fails Pydantic validation.
        """
        raw_text = self.generate(system_prompt, user_prompt, response_schema=schema)
        return self._parse_and_validate(raw_text, schema)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[BaseModel] | None,
        attempt: int,
    ) -> str:
        """Execute a single Gemini API call and return the extracted text.

        Args:
            system_prompt: System-level instructions.
            user_prompt: User-level request.
            response_schema: Optional Pydantic schema for structured output.
            attempt: Current attempt number (used for logging).

        Returns:
            Extracted text from the model response.
        """
        # Build generation config
        config_kwargs: dict[str, Any] = {
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "system_instruction": system_prompt,
        }
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        generation_config = genai_types.GenerateContentConfig(**config_kwargs)

        token_estimate = (len(system_prompt) + len(user_prompt)) // 4
        _console.print(
            f"[dim]→ Calling {escape(self._model_name)} "
            f"(attempt {attempt}, ~{token_estimate} tokens)[/dim]"
        )

        t0 = time.perf_counter()
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=user_prompt,
            config=generation_config,
        )
        latency = time.perf_counter() - t0

        text = _extract_text(response)
        _console.print(
            f"[dim]← Response received in {latency:.2f}s "
            f"({len(text)} chars)[/dim]"
        )
        return text

    @staticmethod
    def _parse_and_validate(raw_text: str, schema: Type[BaseModel]) -> BaseModel:
        """Parse JSON text and validate it against a Pydantic schema.

        Strips Markdown code fences if the model included them despite
        instructions to the contrary.

        Args:
            raw_text: The raw string returned by the model.
            schema: The Pydantic model class to validate against.

        Returns:
            A validated instance of ``schema``.

        Raises:
            ValueError: If JSON parsing or Pydantic validation fails.
        """
        # Strip markdown fences defensively
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            inner_lines = lines[1:] if lines[0].startswith("```") else lines
            if inner_lines and inner_lines[-1].strip() == "```":
                inner_lines = inner_lines[:-1]
            cleaned = "\n".join(inner_lines).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM response is not valid JSON. "
                f"Parse error: {exc}. "
                f"Raw text (first 500 chars): {raw_text[:500]!r}"
            ) from exc

        try:
            return schema.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"LLM JSON failed Pydantic validation for schema "
                f"'{schema.__name__}'. Error: {exc}. "
                f"Data keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}"
            ) from exc
