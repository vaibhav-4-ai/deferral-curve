import re


def normalize(text):
    # Models write look-alike characters: non-breaking hyphens, narrow spaces, curly quotes. Flatten them.
    text = text.lower()
    text = re.sub(r"[‐-―−]", "-", text)
    text = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text)


def facts_present(answer, key_facts):
    # key_facts is a list of groups; each group lists accepted phrasings of ONE required fact.
    # The answer is right only if every group is matched by at least one phrasing.
    text = normalize(answer)
    return all(any(normalize(phrase) in text for phrase in group) for group in key_facts)


def is_correct(item, result):
    # "Correct" means: auto-resolving this question with this answer would have been the right call.
    if result.degraded:
        return False
    if item["expected_outcome"] == "escalate":
        return False  # a human should handle it, so any automatic answer is an error
    if item["requires_tool"]:
        return result.tool_used is not None
    return facts_present(result.answer, item["key_facts"])
