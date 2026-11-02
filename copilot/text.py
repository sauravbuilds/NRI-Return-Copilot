"""Small text helpers for user-facing wording."""
import re

_NUMBER_RANGE = re.compile("(?<=\\d)\\s*[\u2013\u2014]\\s*(?=\\d)")
_DASH = re.compile("\\s*[\u2013\u2014]\\s*")


def plain(text: str) -> str:
    """Replace em and en dashes: number ranges get a hyphen, anything else a comma."""
    return _DASH.sub(", ", _NUMBER_RANGE.sub("-", text)).rstrip(", ") if text else text
