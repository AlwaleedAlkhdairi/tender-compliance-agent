"""Gemini implementation of the agent primitives (free-tier friendly).

Mirrors the Anthropic implementation in src/llm.py: the same bounded ReAct
tool loop and validated structured calls, so agents and the workflow are
provider-agnostic. Selected when LLM_PROVIDER=gemini (or a GEMINI_API_KEY
is configured and no Anthropic key is).

Free-tier realities handled here: retry with backoff on 429/500/503 (the
newest flash models see "high demand" spikes) and an automatic fallback
model for a persistently overloaded primary.
"""

import json
import logging
import os
import time
from typing import Type

from google import genai

# The SDK logs a recommendation about automatic function calling whenever raw
# function declarations are used with generate_content; we run a manual loop
# deliberately, so silence the noise.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
from google.genai import errors as gerrors
from google.genai import types

from src import config
from src.graph.state import make_event
from src.llm import (
    BUDGET_EXHAUSTED_NOTE,
    BUDGET_NUDGE,
    M,
    AgentError,
    AgentRunResult,
    EventFn,
    ToolCallRecord,
    _emit_tool_events,
)
from src.tools.registry import REGISTRY, execute_tool

_client: genai.Client | None = None

RETRYABLE_CODES = {429, 500, 503}
MAX_ATTEMPTS_PER_MODEL = 4
MAX_BACKOFF_SECONDS = 70

# Models whose free-tier DAILY quota is spent (no point retrying today).
_exhausted_models: set[str] = set()


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise AgentError(
                "No Gemini API key configured. Copy .env.example to .env and set "
                "GEMINI_API_KEY, then restart the app."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def to_gemini_schema(schema: dict, defs: dict | None = None) -> dict:
    """Convert a Pydantic JSON schema to what Gemini function declarations
    accept: $refs inlined, unsupported keys removed."""
    if defs is None:
        defs = schema.get("$defs", {})
    if "$ref" in schema:
        return to_gemini_schema(defs[schema["$ref"].split("/")[-1]], defs)
    out = {}
    for key, val in schema.items():
        if key in ("$defs", "title", "additionalProperties", "default"):
            continue
        if key == "properties":
            out[key] = {k: to_gemini_schema(v, defs) for k, v in val.items()}
        elif key == "items":
            out[key] = to_gemini_schema(val, defs)
        elif key == "anyOf":
            out[key] = [to_gemini_schema(v, defs) for v in val]
        else:
            out[key] = val
    return out


def _tool_declarations(tool_names: list[str]) -> list[types.Tool]:
    declarations = [
        types.FunctionDeclaration(
            name=REGISTRY[name].name,
            description=REGISTRY[name].description,
            parameters_json_schema=to_gemini_schema(
                REGISTRY[name].input_model.model_json_schema()
            ),
        )
        for name in tool_names
    ]
    return [types.Tool(function_declarations=declarations)]


def _retry_delay(exc: gerrors.APIError, attempt: int) -> float:
    """Honor the server's RetryInfo delay when present; else back off.

    Free-tier 429s reset on a per-minute window, so short sleeps only burn
    attempts — the server-suggested delay (e.g. '41s') is authoritative.
    """
    try:
        for detail in (exc.details.get("error", {}) or {}).get("details", []):
            if str(detail.get("@type", "")).endswith("RetryInfo"):
                seconds = float(str(detail.get("retryDelay", "0s")).rstrip("s"))
                return min(max(seconds + 1, 5.0), MAX_BACKOFF_SECONDS)
    except Exception:
        pass
    return min(10.0 * (attempt + 1), MAX_BACKOFF_SECONDS)


def _daily_quota_hit(exc: gerrors.APIError) -> bool:
    """True when the 429 is the per-day free-tier cap (retrying is futile)."""
    try:
        for detail in (exc.details.get("error", {}) or {}).get("details", []):
            if str(detail.get("@type", "")).endswith("QuotaFailure"):
                for violation in detail.get("violations", []):
                    if "PerDay" in str(violation.get("quotaId", "")):
                        return True
    except Exception:
        pass
    return False


def _generate(contents, gen_config) -> types.GenerateContentResponse:
    """generate_content with backoff and a fallback model for overload.

    A model whose daily free-tier quota is spent is skipped for the rest of
    the process so every later call fails over immediately.
    """
    client = get_client()
    last_error: Exception | None = None
    for model in (config.GEMINI_MODEL, config.GEMINI_FALLBACK_MODEL):
        if model in _exhausted_models:
            continue
        for attempt in range(MAX_ATTEMPTS_PER_MODEL):
            try:
                return client.models.generate_content(
                    model=model, contents=contents, config=gen_config
                )
            except gerrors.APIError as exc:
                last_error = exc
                if exc.code not in RETRYABLE_CODES:
                    raise AgentError(f"Gemini API error ({exc.code}): {exc.message}") from exc
                if exc.code == 429 and _daily_quota_hit(exc):
                    _exhausted_models.add(model)
                    break  # next model — this one is done for the day
                time.sleep(_retry_delay(exc, attempt))
    raise AgentError(
        "The Gemini API is overloaded or rate-limited right now "
        f"(last error: {getattr(last_error, 'code', last_error)}). "
        "If this is the free tier's daily cap, switch GEMINI_MODEL in .env "
        "(e.g. gemini-3.5-flash-lite) or try again tomorrow."
    ) from last_error


def _check_blocked(response) -> None:
    feedback = getattr(response, "prompt_feedback", None)
    if feedback and getattr(feedback, "block_reason", None):
        raise AgentError(
            f"The model declined to process this request ({feedback.block_reason}). "
            "Rephrase and try again."
        )


def _normalize_contents(messages: list | str) -> list[types.Content]:
    """Accept plain strings, {'role','content'} dicts and native Content."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    contents = []
    for message in messages:
        if isinstance(message, types.Content):
            contents.append(message)
            continue
        role = "model" if message["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=str(message["content"]))]))
    return contents


def run_tool_loop(
    agent: str,
    system: str,
    user_prompt: str,
    tool_names: list[str],
    emit: EventFn,
    max_turns: int = config.MAX_AGENT_TURNS,
) -> AgentRunResult:
    """Gemini version of the bounded ReAct loop (same contract as llm.py)."""
    gen_config = types.GenerateContentConfig(
        system_instruction=system,
        tools=_tool_declarations(tool_names),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        max_output_tokens=config.MAX_TOKENS,
    )
    contents: list = [types.Content(role="user", parts=[types.Part(text=user_prompt)])]
    run = AgentRunResult(final_text="", messages=contents)

    for turn in range(max_turns):
        response = _generate(contents, gen_config)
        _check_blocked(response)
        candidate = response.candidates[0] if response.candidates else None
        parts = list(candidate.content.parts or []) if candidate and candidate.content else []

        for part in parts:
            if part.text and part.text.strip():
                emit(make_event(agent, "reasoning", part.text.strip()[:300]))

        calls = [p.function_call for p in parts if p.function_call]
        if not calls:
            run.final_text = "\n".join(p.text for p in parts if p.text).strip()
            if candidate and candidate.content:
                contents.append(candidate.content)
            return run

        contents.append(candidate.content)
        result_parts = []
        for call in calls:
            tool_input = dict(call.args or {})
            result_json, is_error = execute_tool(call.name, tool_input)
            _emit_tool_events(agent, call.name, tool_input, result_json, is_error, emit)
            run.tool_calls.append(
                ToolCallRecord(
                    name=call.name,
                    input=tool_input,
                    result=None if is_error else json.loads(result_json),
                    is_error=is_error,
                )
            )
            payload = {"error" if is_error else "result": json.loads(result_json)}
            result_parts.append(
                types.Part.from_function_response(name=call.name, response=payload)
            )
        contents.append(types.Content(role="user", parts=result_parts))
        if turn == max_turns - 2:
            contents.append(
                types.Content(role="user", parts=[types.Part(text=BUDGET_NUDGE)])
            )

    # Safety bound reached: stop researching and work with what was gathered.
    emit(make_event(agent, "reasoning", BUDGET_EXHAUSTED_NOTE))
    run.final_text = BUDGET_EXHAUSTED_NOTE
    return run


def _flatten_for_structured(contents: list[types.Content]) -> list[types.Content]:
    """Render tool-call/tool-result parts as text so the structured-output
    request needs no tool declarations (Gemini does not combine function
    calling with response_schema)."""
    flattened = []
    for content in contents:
        texts = []
        for part in content.parts or []:
            if part.text:
                texts.append(part.text)
            elif part.function_call:
                texts.append(
                    f"[tool call: {part.function_call.name}"
                    f"({json.dumps(dict(part.function_call.args or {}), default=str)[:2000]})]"
                )
            elif part.function_response:
                texts.append(
                    f"[tool result: {json.dumps(dict(part.function_response.response or {}), default=str)[:6000]}]"
                )
        if texts:
            flattened.append(
                types.Content(role=content.role, parts=[types.Part(text="\n".join(texts))])
            )
    return flattened


def structured_call(
    agent: str,
    system: str,
    messages: list | str,
    output_model: Type[M],
    tool_names: list[str] | None = None,  # noqa: ARG001 - transcripts are flattened instead
) -> M:
    """One call that must return a validated instance of `output_model`."""
    contents = _flatten_for_structured(_normalize_contents(messages))
    gen_config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=output_model,
        max_output_tokens=config.MAX_TOKENS,
    )
    for attempt in range(2):
        response = _generate(contents, gen_config)
        _check_blocked(response)
        if response.parsed is not None:
            return response.parsed
    raise AgentError(f"The {agent} returned no structured output. Try again.")
