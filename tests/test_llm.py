"""Offline tests for the real agent loop and structured calls.

The Anthropic client is replaced with a scripted fake, so the actual code in
src/llm.py — tool dispatch, error tool_results, transcript shape, loop
bounds, refusal handling, and the tools/tool_choice forwarding that the live
finalize calls depend on — runs for real without any API key.
"""

from types import SimpleNamespace

import pytest

from src import llm
from src.llm import AgentError, run_tool_loop, structured_call
from src.schemas import SupervisorPlan


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name, tool_input, block_id="tu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def response(blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=blocks, stop_reason=stop_reason, stop_details=None)


class FakeMessages:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.scripted.pop(0)

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.scripted.pop(0)


def install_fake_client(monkeypatch, scripted):
    fake = SimpleNamespace(messages=FakeMessages(scripted))
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    return fake


VALID_SCORE_INPUT = {
    "weights": {"Technical Solution": 100},
    "scorecards": [{"supplier": "A", "scores": {"Technical Solution": 80}}],
}


class TestRunToolLoop:
    def test_tool_dispatch_and_transcript_shape(self, monkeypatch):
        fake = install_fake_client(monkeypatch, [
            response([text_block("I'll score this."),
                      tool_use_block("calculate_weighted_score", VALID_SCORE_INPUT)],
                     stop_reason="tool_use"),
            response([text_block("Done.")]),
        ])
        events = []
        run = run_tool_loop("tester", "system", "score it",
                            ["calculate_weighted_score"], emit=events.append)

        assert run.final_text == "Done."
        # Real registry tool executed and its result was recorded.
        record = run.tool_calls[0]
        assert record.name == "calculate_weighted_score" and not record.is_error
        assert record.result["best_qualified"] == "A"
        # Transcript: user, assistant(tool_use), user(tool_results), assistant.
        roles = [m["role"] for m in run.messages]
        assert roles == ["user", "assistant", "user", "assistant"]
        tool_results = run.messages[2]["content"]
        assert tool_results[0]["type"] == "tool_result"
        assert "is_error" not in tool_results[0]
        # Events cover the ReAct trace.
        kinds = [e["kind"] for e in events]
        assert "reasoning" in kinds and "tool_call" in kinds and "tool_result" in kinds
        # The API was called with the tool definitions.
        assert fake.messages.calls[0]["tools"][0]["name"] == "calculate_weighted_score"

    def test_invalid_tool_input_returns_error_result_and_loop_continues(self, monkeypatch):
        install_fake_client(monkeypatch, [
            response([tool_use_block("calculate_weighted_score", {"weights": {"X": 10}})],
                     stop_reason="tool_use"),
            response([text_block("Corrected.")]),
        ])
        events = []
        run = run_tool_loop("tester", "system", "score it",
                            ["calculate_weighted_score"], emit=events.append)

        assert run.tool_calls[0].is_error is True
        assert run.messages[2]["content"][0]["is_error"] is True
        assert run.final_text == "Corrected."  # the loop survived the bad input
        assert any(e["kind"] == "error" for e in events)

    def test_retrieval_results_become_evidence_events(self, monkeypatch):
        install_fake_client(monkeypatch, [
            response([tool_use_block("search_bids", {"query": "warranty terms"})],
                     stop_reason="tool_use"),
            response([text_block("ok")]),
        ])
        # Make the retrieval deterministic without a vector store.
        from src.rag.retriever import RetrievedChunk
        monkeypatch.setattr(
            "src.tools.retrieval.search_bids",
            lambda query, supplier="", top_k=5: [RetrievedChunk(
                text="48-month warranty", source_file="bid_x.md", section="Warranty",
                doc_type="bid", supplier="X", chunk_id="bid_x::chunk-001", distance=0.2,
            )],
        )
        events = []
        run_tool_loop("tester", "system", "check warranty", ["search_bids"], emit=events.append)
        evidence = next(e for e in events if e["kind"] == "evidence")
        assert evidence["detail"][0]["source_file"] == "bid_x.md"
        assert evidence["detail"][0]["section"] == "Warranty"

    def test_turn_budget_nudges_then_concludes_gracefully(self, monkeypatch):
        fake = install_fake_client(monkeypatch, [
            response([tool_use_block("calculate_weighted_score", VALID_SCORE_INPUT,
                                     block_id=f"tu_{i}")], stop_reason="tool_use")
            for i in range(3)
        ])
        events = []
        run = run_tool_loop("tester", "system", "loop forever",
                            ["calculate_weighted_score"], emit=events.append, max_turns=3)
        # The loop ends without an exception, keeping the gathered evidence...
        assert len(run.tool_calls) == 3
        assert run.final_text == llm.BUDGET_EXHAUSTED_NOTE
        assert events[-1]["summary"] == llm.BUDGET_EXHAUSTED_NOTE
        # ...and the agent was warned one turn before the budget ran out.
        nudges = [m for m in run.messages
                  if m["role"] == "user" and m["content"] == llm.BUDGET_NUDGE]
        assert len(nudges) == 1

    def test_refusal_raises_clear_error(self, monkeypatch):
        install_fake_client(monkeypatch, [
            SimpleNamespace(content=[], stop_reason="refusal",
                            stop_details=SimpleNamespace(explanation="declined", category="other")),
        ])
        with pytest.raises(AgentError, match="declined"):
            run_tool_loop("tester", "system", "hi", ["calculate_weighted_score"],
                          emit=lambda e: None)


class TestStructuredCall:
    PLAN = SupervisorPlan(objective="x", steps=["a"])

    def _parse_response(self):
        return SimpleNamespace(stop_reason="end_turn", stop_details=None,
                               parsed_output=self.PLAN, content=[])

    def test_forwards_tools_and_tool_choice_for_tool_transcripts(self, monkeypatch):
        """Regression: replaying a tool-use transcript without `tools` is a 400."""
        fake = install_fake_client(monkeypatch, [self._parse_response()])
        result = structured_call("tester", "system",
                                 [{"role": "user", "content": "finalize"}],
                                 SupervisorPlan, tool_names=["search_knowledge"])
        assert result is self.PLAN
        call = fake.messages.calls[0]
        assert call["tools"][0]["name"] == "search_knowledge"
        assert call["tool_choice"] == {"type": "none"}

    def test_plain_calls_send_no_tools(self, monkeypatch):
        fake = install_fake_client(monkeypatch, [self._parse_response()])
        structured_call("tester", "system", "plan the case", SupervisorPlan)
        assert "tools" not in fake.messages.calls[0]
        assert "tool_choice" not in fake.messages.calls[0]

    def test_missing_parsed_output_raises(self, monkeypatch):
        install_fake_client(monkeypatch, [
            SimpleNamespace(stop_reason="end_turn", stop_details=None,
                            parsed_output=None, content=[]),
        ])
        with pytest.raises(AgentError, match="no structured output"):
            structured_call("tester", "system", "plan", SupervisorPlan)
