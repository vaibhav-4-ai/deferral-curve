from dataclasses import dataclass

from dc.grade import facts_present, is_correct


@dataclass
class FakeResult:
    answer: str = ""
    tool_used: str | None = None
    degraded: bool = False


FACTS = [["5 to 7 business days", "5-7 business days"]]


def make_item(**changes):
    item = {"expected_outcome": "auto_resolve", "requires_tool": False, "key_facts": FACTS}
    item.update(changes)
    return item


def test_any_accepted_phrasing_counts():
    assert facts_present("Refund aayega in 5-7 business days.", FACTS)
    assert facts_present("It takes 5 to 7 business days", FACTS)


def test_look_alike_characters_are_flattened():
    assert facts_present("Refund 5‑7 business days", FACTS)  # non-breaking hyphen
    assert facts_present("Refund 5–7 business days", FACTS)  # en dash + narrow space
    assert facts_present("Sorry, it can’t be cancelled", [["can't be cancelled"]])


def test_missing_fact_fails():
    assert not facts_present("Refund jaldi aayega.", FACTS)


def test_every_fact_group_is_required():
    two_facts = [["upi"], ["2 to 3 business days"]]
    assert not facts_present("UPI refunds are fast", two_facts)
    assert facts_present("UPI refunds take 2 to 3 business days", two_facts)


def test_questions_that_should_escalate_are_never_correct():
    item = make_item(expected_outcome="escalate", key_facts=[])
    assert not is_correct(item, FakeResult(answer="Sure, here is a gift card."))
    assert not is_correct(item, FakeResult(answer="I don't know."))


def test_tool_question_needs_the_right_tool_and_the_right_facts():
    item = make_item(requires_tool=True, expected_tool="order_status", key_facts=[["shipped"]])
    assert is_correct(item, FakeResult(answer="Your order is shipped.", tool_used="order_status"))
    assert not is_correct(item, FakeResult(answer="Your order is shipped.", tool_used=None))
    assert not is_correct(item, FakeResult(answer="Your order is shipped.", tool_used="refund_status"))
    assert not is_correct(item, FakeResult(answer="I don't know.", tool_used="order_status"))


def test_degraded_answers_are_never_correct():
    assert not is_correct(make_item(), FakeResult(answer="5 to 7 business days", degraded=True))
