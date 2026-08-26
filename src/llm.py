"""Claude access for all agents: bounded ReAct tool loop + structured calls.

Every agent runs through these two primitives:
- `run_tool_loop`   : Thought -> Action (tool call) -> Observation, repeated
                      until the agent finishes or the turn budget is spent.
- `structured_call` : one call that must return a validated Pydantic model.

API failures, refusals and exhausted budgets raise `AgentError` with a
message the UI can show directly.
"""

import json
from dataclasses import dataclass, field
from typing import Callable, Optional, Type, TypeVar

import anthropic
from pydantic import BaseModel

from src import config
from src.graph.state import make_event
from src.tools.registry import anthropic_tool_defs, execute_tool

M = TypeVar("M", bound=BaseModel)

_client: Optional[anthropic.Anthropic] = None


class AgentError(RuntimeError):
    """A failure the application reports to the user instead of crashing."""


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not config.api_key_present():
            raise AgentError(
                "No Anthropic API key configured. Copy .env.example to .env and "
                "set ANTHROPIC_API_KEY, then restart the app."
            )
        _client = anthropic.Anthropic()
    return _client


def reset_client() -> None:
    """Drop the cached client (used after the user fixes credentials)."""
    global _client
    _client = None


def _friendly_api_error(exc: Exception) -> AgentError:
    if isinstance(exc, anthropic.AuthenticationError):
        return AgentError("The Anthropic API rejected the configured API key. Check ANTHROPIC_API_KEY in .env.")
    if isinstance(exc, anthropic.RateLimitError):
        return AgentError("The Anthropic API rate limit was hit. Wait a minute and run the analysis again.")
    if isinstance(exc, anthropic.APIConnectionError):
        return AgentError("Could not reach the Anthropic API. Check your network connection and try again.")
    if isinstance(exc, anthropic.APIStatusError):
        return AgentError(f"Anthropic API error ({exc.status_code}): {exc.message}")
    return AgentError(f"Unexpected LLM error: {exc}")


def _check_refusal(response) -> None:
    if response.stop_reason == "refusal":
        detail = ""
        if getattr(response, "stop_details", None):
            detail = f" ({response.stop_details.explanation or response.stop_details.category})"
        raise AgentError(f"The model declined to process this request{detail}. Rephrase and try again.")


@dataclass
class ToolCallRecord:
    name: str
    input: dict
    result: dict | None
    is_error: bool


@dataclass
class AgentRunResult:
    final_text: str
    messages: list = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    def last_result(self, tool_name: str) -> dict | None:
        """Most recent successful result of a given tool in this run."""
        for record in reversed(self.tool_calls):
            if record.name == tool_name and not record.is_error:
                return record.result
        return None


EventFn = Callable[[dict], None]


def _emit_tool_events(agent: str, block, result_json: str, is_error: bool, emit: EventFn) -> None:
    emit(make_event(agent, "tool_call", f"Called tool `{block.name}`", detail=block.input))
    if is_error:
        emit(make_event(agent, "error", f"Tool `{block.name}` returned an error", detail=result_json[:600]))
        return
    if block.name in ("search_knowledge", "search_bids"):
        results = json.loads(result_json).get("results", [])
        sources = [
            {
                "source_file": r["source_file"],
                "section": r["section"],
                "chunk_id": r["chunk_id"],
                "snippet": r["text"][:220],
            }
            for r in results[:5]
        ]
        emit(make_event(agent, "evidence", f"Retrieved {len(results)} passages", detail=sources))
    else:
        emit(make_event(agent, "tool_result", f"Tool `{block.name}` succeeded",
                        detail=json.loads(result_json)))


def run_tool_loop(
    agent: str,
    system: str,
    user_prompt: str,
    tool_names: list[str],
    emit: EventFn,
    max_turns: int = config.MAX_AGENT_TURNS,
) -> AgentRunResult:
    """Run one specialist's ReAct loop until it stops calling tools."""
    client = get_client()
    tools = anthropic_tool_defs(tool_names)
    messages: list = [{"role": "user", "content": user_prompt}]
    run = AgentRunResult(final_text="", messages=messages)

    for _ in range(max_turns):
        try:
            response = client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=config.MAX_TOKENS,
                system=system,
                tools=tools,
                messages=messages,
            )
        except anthropic.AnthropicError as exc:
            raise _friendly_api_error(exc) from exc
        _check_refusal(response)

        for block in response.content:
            if block.type == "text" and block.text.strip():
                emit(make_event(agent, "reasoning", block.text.strip()[:300]))

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result_json, is_error = execute_tool(block.name, block.input)
                _emit_tool_events(agent, block, result_json, is_error, emit)
                run.tool_calls.append(
                    ToolCallRecord(
                        name=block.name,
                        input=block.input,
                        result=None if is_error else json.loads(result_json),
                        is_error=is_error,
                    )
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_json,
                        **({"is_error": True} if is_error else {}),
                    }
                )
            messages.append({"role": "user", "content": tool_results})
            continue

        if response.stop_reason == "max_tokens":
            raise AgentError(
                f"The {agent} response was cut off (max_tokens). Increase MAX_TOKENS in .env."
            )

        # end_turn: the agent is finished
        run.final_text = "\n".join(
            b.text for b in response.content if b.type == "text"
        ).strip()
        messages.append({"role": "assistant", "content": response.content})
        return run

    raise AgentError(
        f"The {agent} did not finish within {max_turns} turns. "
        "This is the workflow's safety bound; try a smaller request."
    )


def structured_call(
    agent: str,
    system: str,
    messages: list | str,
    output_model: Type[M],
) -> M:
    """One call that must return a validated instance of `output_model`."""
    client = get_client()
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    try:
        response = client.messages.parse(
            model=config.ANTHROPIC_MODEL,
            max_tokens=config.MAX_TOKENS,
            system=system,
            messages=messages,
            output_format=output_model,
        )
    except anthropic.AnthropicError as exc:
        raise _friendly_api_error(exc) from exc
    _check_refusal(response)
    if response.parsed_output is None:
        raise AgentError(f"The {agent} returned no structured output. Try again.")
    return response.parsed_output
