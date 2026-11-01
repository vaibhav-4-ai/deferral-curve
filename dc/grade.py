def facts_present(answer, key_facts):
    # key_facts is a list of groups; each group lists accepted phrasings of ONE required fact.
    # The answer is right only if every group is matched by at least one phrasing.
    text = answer.lower()
    return all(any(phrase.lower() in text for phrase in group) for group in key_facts)


def is_correct(item, result):
    # item: one golden-set row. result: the agent's AgentResult.
    if result.degraded:
        return False
    if item["difficulty"] == "unanswerable":
        return False  # there is no right answer to give, so any answer is an error
    if item["requires_tool"]:
        return result.tool_used is not None
    return facts_present(result.answer, item["key_facts"])
