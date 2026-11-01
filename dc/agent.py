from dataclasses import dataclass, field

import httpx

from dc.tools import TOOLS, pick_tool

ANSWER_SYSTEM = (
    "You are QuickKart's customer support assistant. Answer ONLY from the context given. "
    "If the context does not contain the answer, reply exactly: I don't know. "
    "Reply in the customer's language style (English or Hinglish). Be brief."
)
VERIFY_SYSTEM = (
    "You check support answers. Given a context, a question and an answer, reply with exactly one word: "
    "SUPPORTED if every claim in the answer is stated in the context, otherwise UNSUPPORTED."
)


class BudgetExceeded(Exception):
    pass


class Budget:
    """A hard cap on model calls per question, enforced in code before each call is made."""

    def __init__(self, max_calls):
        self.max_calls = max_calls
        self.used = 0

    def spend(self):
        if self.used >= self.max_calls:
            raise BudgetExceeded(f"limit of {self.max_calls} model calls reached")
        self.used += 1


@dataclass
class AgentResult:
    answer: str
    doc_ids: list
    verified: bool
    tool_used: str | None
    llm_calls: int
    degraded: bool
    features: dict = field(default_factory=dict)


class Agent:
    """Gather (retrieve + maybe a tool) -> act (draft an answer) -> verify -> retry once with more context."""

    def __init__(self, retriever, llm, max_llm_calls=4):
        self.retriever = retriever
        self.llm = llm
        self.max_llm_calls = max_llm_calls
        self.page_text = dict(retriever.pages)

    def _context(self, hits, tool_note):
        parts = [f"[{doc_id}]\n{self.page_text[doc_id]}" for doc_id, _ in hits]
        if tool_note:
            parts.append(f"[tool result]\n{tool_note}")
        return "\n\n".join(parts)

    def answer(self, query):
        budget = Budget(self.max_llm_calls)
        features = self.retriever.raw_features(query)

        tool_used = None
        tool_note = None
        picked = pick_tool(query)
        if picked is not None:
            tool_used, order_id = picked
            tool_note = TOOLS[tool_used](order_id)

        k = 3
        hits = self.retriever.search(query, k=k)
        draft = ""
        verified = False
        try:
            for _ in range(2):
                hits = self.retriever.search(query, k=k)
                context = self._context(hits, tool_note)
                budget.spend()
                draft = self.llm.complete(ANSWER_SYSTEM, f"Context:\n{context}\n\nQuestion: {query}")
                budget.spend()
                verdict = self.llm.complete(
                    VERIFY_SYSTEM, f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer: {draft}"
                )
                verified = verdict.strip().upper().startswith("SUPPORTED")
                if verified:
                    break
                k = 5
        except (BudgetExceeded, LookupError, httpx.HTTPError):
            if not draft:
                # Graceful degradation: the model is unavailable, so fall back to retrieval only.
                return AgentResult(
                    answer=self.page_text[hits[0][0]],
                    doc_ids=[doc_id for doc_id, _ in hits],
                    verified=False,
                    tool_used=tool_used,
                    llm_calls=budget.used,
                    degraded=True,
                    features=features,
                )
        return AgentResult(
            answer=draft,
            doc_ids=[doc_id for doc_id, _ in hits],
            verified=verified,
            tool_used=tool_used,
            llm_calls=budget.used,
            degraded=False,
            features=features,
        )
