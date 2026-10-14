import pytest
from assistant.routing_rules import apply_rules

@pytest.mark.parametrize("text,intent,has,expect", [
    ("write a short report on these papers", "chat", True, ("create_report", True)),
    ("turn these 5 papers into a review", "unclear", True, ("create_report", True)),
    ("summarize paper 2", "chat", True, ("chat", False)),
    ("write a report on inflation", "create_report", True, ("create_report", False)),
    ("write a report on these papers", "chat", False, ("chat", False)),
])
def test_rules(text, intent, has, expect):
    got = apply_rules(text, intent, has)
    assert got[:2] == expect