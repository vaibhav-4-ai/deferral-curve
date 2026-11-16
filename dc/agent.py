from dataclasses import dataclass

import httpx

from dc.tools import TOOLS, fresh_orders, pick_tool

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
    """Everything the agent hands back for one question."""

    answer: str
    doc_ids: list  # the help pages it read, best first
    verified: bool  # did the checking step say SUPPORTED?
    tool_used: str | None  # name of the order tool that ran, or None
    llm_calls: int
    degraded: bool  # True when the model was unreachable and the answer is just the best help page
    features: dict  # the retriever's match scores, used later as confidence signals
    error: str = ""


class Agent:
    """Gather (retrieve + maybe a tool) -> act (draft an answer) -> verify -> retry once with more context."""

    def __init__(self, retriever, llm, max_llm_calls=4):
        self.retriever = retriever
        self.llm = llm
        self.max_llm_calls = max_llm_calls
        self.page_text = dict(retriever.pages)

    def _context(self, hits, tool_note):
        # The text the model is allowed to answer from: each help page under its name, then the tool's result.
        parts = []
        for doc_id, _score in hits:
            parts.append(f"[{doc_id}]\n{self.page_text[doc_id]}")
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
            tool = TOOLS[tool_used]
            tool_note = tool(fresh_orders(), order_id)  # a fresh copy, so a cancel can't leak into the next question

        hits = []
        draft = ""
        verified = False
        degraded = False
        error = ""
        try:
            for k in (3, 5):  # 3 pages of context on the first try, 5 on the one retry
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
        except (BudgetExceeded, LookupError, httpx.HTTPError) as exc:
            # Never fail silently: keep the reason, so the eval run can print it.
            error = f"{type(exc).__name__}: {exc}"
            if not draft:
                # Graceful degradation: the model is unavailable, so fall back to the best help page.
                degraded = True
                draft = self.page_text[hits[0][0]]

        doc_ids = []
        for doc_id, _score in hits:
            doc_ids.append(doc_id)
        return AgentResult(
            answer=draft,
            doc_ids=doc_ids,
            verified=verified,
            tool_used=tool_used,
            llm_calls=budget.used,
            degraded=degraded,
            features=features,
            error=error,
        )
