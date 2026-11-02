"""PII guard: runs first, before any model sees the text.

It returns only the categories found, never the matched values, so the
result is safe to log and safe to show the user.
"""
import re
from dataclasses import dataclass, field

import config

MOBILE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
AADHAAR = re.compile(r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)")
ACCOUNT = re.compile(r"(?<!\d)\d{9,18}(?!\d)")

MESSAGES = {
    "mobile_number": "a phone number",
    "pan": "a PAN number",
    "aadhaar": "an Aadhaar-style 12-digit number",
    "email": "an email address",
    "card_number": "a payment card number",
    "account_number": "a long number that looks like an account number",
}


@dataclass
class PIIResult:
    blocked: bool
    categories: list[str] = field(default_factory=list)  # never the values

    @property
    def message(self) -> str:
        """User-facing text. Names the type of detail, never echoes it."""
        if not self.blocked:
            return ""
        kinds = ", ".join(MESSAGES[c] for c in self.categories)
        return (f"Looks like your message has {kinds} in it. To keep your details private, "
                "please send it again without personal identifiers. Just the context is enough.")


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def check_pii(text: str) -> PIIResult:
    found: list[str] = []

    if EMAIL.search(text):
        found.append("email")
    if PAN.search(text):
        found.append("pan")

    # Cards: needs a valid Luhn checksum to cut false positives.
    for m in CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if config.CARD_MIN_DIGITS <= len(digits) <= config.CARD_MAX_DIGITS and _luhn_ok(digits):
            found.append("card_number")
            break

    if MOBILE.search(text):
        found.append("mobile_number")
    if AADHAAR.search(text):
        found.append("aadhaar")
    # Generic long digit run. Skipped when a more specific number type
    # already matched, so a phone number is not also reported as an account.
    specific_number = {"mobile_number", "aadhaar", "card_number"} & set(found)
    if not specific_number:
        for m in ACCOUNT.finditer(text):
            if config.ACCOUNT_MIN_DIGITS <= len(m.group()) <= config.ACCOUNT_MAX_DIGITS:
                found.append("account_number")
                break

    cats = list(dict.fromkeys(found))
    return PIIResult(blocked=bool(cats), categories=cats)
