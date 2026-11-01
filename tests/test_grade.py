from dataclasses import dataclass

from dc.grade import facts_present, is_correct


@dataclass
class FakeResult:
    answer: str = ""
    tool_used: str | None = None
    degraded: bool = False


FACTS = [["5 to 7 business days", "5-7 business days"]]


def test_any_accepted_phrasing_counts():
    assert facts_present("Refund aayega in 5-7 business days.", FACTS)
    assert facts_present("It takes 5 to 7 business days", FACTS)


def test_missing_fact_fails():
    assert not facts_present("Refund jaldi aayega.", FACTS)


def test_every_fact_group_is_required():
    two_facts = [["upi"], ["2 to 3 business days"]]
    assert not facts_present("UPI refunds are fast", two_facts)
    assert facts_present("UPI refunds take 2 to 3 business days", two_facts)


def test_unanswerable_questions_are_never_correct():
    item = {"difficulty": "unanswerable", "requires_tool": False, "key_facts": []}
    assert not is_correct(item, FakeResult(answer="Sure, here is a gift card."))


def test_tool_question_is_correct_when_a_tool_ran():
    item = {"difficulty": "easy", "requires_tool": True, "key_facts": []}
    assert is_correct(item, FakeResult(tool_used="order_status"))
    assert not is_correct(item, FakeResult(tool_used=None))


def test_degraded_answers_are_never_correct():
    item = {"difficulty": "easy", "requires_tool": False, "key_facts": FACTS}
    assert not is_correct(item, FakeResult(answer="5 to 7 business days", degraded=True))
