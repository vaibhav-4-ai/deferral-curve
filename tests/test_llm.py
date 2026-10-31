import pytest

from dc.llm import CachedLLM


def test_second_identical_question_uses_the_cache(tmp_path):
    calls = []

    def fake_backend(system, user):
        calls.append(user)
        return "hello"

    llm = CachedLLM(fake_backend, cache_path=tmp_path / "cache.json")
    assert llm.complete("be brief", "hi") == "hello"
    assert llm.complete("be brief", "hi") == "hello"
    assert len(calls) == 1


def test_cache_survives_a_restart(tmp_path):
    path = tmp_path / "cache.json"
    CachedLLM(lambda system, user: "saved", cache_path=path).complete("s", "u")
    replay = CachedLLM(None, cache_path=path)
    assert replay.complete("s", "u") == "saved"


def test_unknown_question_without_a_key_raises(tmp_path):
    replay = CachedLLM(None, cache_path=tmp_path / "cache.json")
    with pytest.raises(LookupError):
        replay.complete("s", "never asked")
