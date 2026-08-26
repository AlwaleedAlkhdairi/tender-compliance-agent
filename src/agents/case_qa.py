"""Case Q&A assistant for follow-up questions about a finished analysis.

CONTRACT
  Responsibility : Answer follow-up questions about the completed case.
  Input          : The case's short-term memory + the user's question.
  Permitted      : `search_knowledge` / `search_bids` retrieval only. May
                   not re-run the analysis or alter any recorded finding.
  Output         : A grounded text answer, appended to the conversation
                   memory.
  Completion     : The answer cites memory or retrieved sources, or states
                   plainly that the information is not available.

Grounded in two places: the short-term case memory (facts + conversation of
the current case) and the long-term vector store (it selects search_knowledge
for tender/guidance questions, search_bids for bid-content questions, or no
tool at all when memory already answers the question).
"""

from src.llm import EventFn, run_tool_loop
from src.memory.store import CaseMemory

SYSTEM = """You are the evaluation team's case assistant. A tender analysis has been
completed and the procurement officer is asking follow-up questions about it.

Ground every answer:
- The case memory below contains the finished analysis (rankings, gaps, decisions);
  answer from it directly when it suffices.
- Use search_bids when asked what a supplier's bid actually says.
- Use search_knowledge when asked about the tender's requirements, the scoring
  methodology or the procurement guidance.
Cite the source file when you quote a document. If neither memory nor the documents
answer the question, say so plainly - never invent facts or figures.

Keep answers short and businesslike.

{context}"""


def answer(memory: CaseMemory, question: str, emit: EventFn) -> str:
    memory.add_user_turn(question)
    result = run_tool_loop(
        agent="case_qa",
        system=SYSTEM.format(context=memory.render_context()),
        user_prompt=question,
        tool_names=["search_knowledge", "search_bids"],
        emit=emit,
        max_turns=6,
    )
    reply = result.final_text or "I could not produce an answer for that question."
    memory.add_assistant_turn(reply)
    return reply
