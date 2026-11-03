import pytest

from dc.agent import Agent
from dc.corpus import load_corpus
from dc.llm import CachedLLM
from dc.retrieve import HybridRetriever
from dc.tools import pick_tool


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever(load_corpus())


def make_llm(tmp_path, verdict):
    seen = []

    def backend(system, user):
        seen.append(user)
        if system.startswith("You check"):
            return verdict
        return "Refunds take 5 to 7 business days."

    return CachedLLM(backend, cache_path=tmp_path / "cache.json"), seen


def test_pick_tool_reads_intent_and_order_number():
    assert pick_tool("mera order #4821 kahan hai?") == ("order_status", "4821")
    assert pick_tool("order #4821 ka refund status kya hai?") == ("refund_status", "4821")
    assert pick_tool("order #4821 cancel kar do please") == ("cancel_order", "4821")
    assert pick_tool("mera refund kab aayega") is None


def test_tool_result_reaches_the_model(retriever, tmp_path):
    llm, seen = make_llm(tmp_path, "SUPPORTED")
    result = Agent(retriever, llm).answer("mera order #4821 kahan hai?")
    assert result.tool_used == "order_status"
    assert "shipped" in seen[0]
    assert result.verified is True
    assert result.llm_calls == 2


def test_cancelling_in_one_question_does_not_leak_into_the_next(retriever, tmp_path):
    llm, seen = make_llm(tmp_path, "SUPPORTED")
    agent = Agent(retriever, llm)
    agent.answer("please cancel order 5530")
    agent.answer("mera order #5530 ka status kya hai?")
    assert "has been cancelled" in seen[0]
    assert "processing" in seen[2]  # second question's draft prompt still sees the original order


def test_failed_verification_retries_once_then_stops(retriever, tmp_path):
    llm, _ = make_llm(tmp_path, "UNSUPPORTED")
    result = Agent(retriever, llm).answer("mera refund kab aayega")
    assert result.verified is False
    assert result.llm_calls == 4


def test_budget_cap_is_enforced(retriever, tmp_path):
    llm, _ = make_llm(tmp_path, "UNSUPPORTED")
    result = Agent(retriever, llm, max_llm_calls=3).answer("mera refund kab aayega")
    assert result.llm_calls == 3


def test_falls_back_to_retrieval_when_the_model_is_unavailable(retriever, tmp_path):
    llm = CachedLLM(None, cache_path=tmp_path / "empty.json")
    result = Agent(retriever, llm).answer("mera refund kab aayega")
    assert result.degraded is True
    assert result.answer != ""
    assert len(result.doc_ids) > 0
