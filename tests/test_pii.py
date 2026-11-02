import pytest

from guardrails.pii import check_pii

POSITIVE = {
    "mobile_number": ["call me on 9876543210", "my number is +91 9123456789", "reach 91-8123456780 please"],
    "pan": ["PAN is ABCDE1234F", "ok BXZPK9087L here", "use QWERT0001Z for the form"],
    "aadhaar": ["aadhaar 1234 5678 9012", "id 234567890123", "number 2345 6789 0123 ok"],
    "email": ["mail me at a.b@example.com", "x_y+z@mail.co.in", "write to someone@site.org"],
    "card_number": ["card 4111 1111 1111 1111", "4012-8888-8888-1881 expires soon", "5500000000000004"],
    "account_number": ["account 123456789012345", "a/c 98765432101234", "acct 1234567890123456"],
}

NEGATIVE = [
    "I moved in 2023",
    "I hold about ₹50,00,000 in savings",
    "I spent 182 days in India last year",
    "Moved in March 2024 and stayed 45 days",
    "What forms apply to my 401(k)?",
]


@pytest.mark.parametrize("category,texts", POSITIVE.items())
def test_positive_cases(category, texts):
    for t in texts:
        res = check_pii(t)
        assert res.blocked, t
        assert category in res.categories, (t, res.categories)


@pytest.mark.parametrize("text", NEGATIVE)
def test_negative_cases_do_not_trigger(text):
    assert not check_pii(text).blocked


def test_result_never_contains_the_value():
    res = check_pii("PAN ABCDE1234F and mail a.b@example.com")
    blob = res.message + str(res.categories)
    assert "ABCDE1234F" not in blob and "example.com" not in blob


def test_card_with_bad_luhn_is_not_a_card():
    assert "card_number" not in check_pii("4111 1111 1111 1112").categories


def test_clean_text_has_empty_message():
    assert check_pii("hello").message == ""
