from datetime import date

import pytest

from copilot.tools import count_days_by_year, days_in_india_calculator, scrub


def test_single_day_is_one():
    assert count_days_by_year([(date(2024, 5, 5), date(2024, 5, 5))]) == {2024: 1}


def test_year_boundary_split_and_end_counted():
    # 30 Dec to 2 Jan inclusive = 2 days in 2023, 2 days in 2024
    assert count_days_by_year([(date(2023, 12, 30), date(2024, 1, 2))]) == {2023: 2, 2024: 2}


def test_leap_year_full_year():
    assert count_days_by_year([(date(2024, 1, 1), date(2024, 12, 31))]) == {2024: 366}
    assert count_days_by_year([(date(2023, 1, 1), date(2023, 12, 31))]) == {2023: 365}


def test_overlapping_ranges_not_double_counted():
    r = [(date(2024, 3, 1), date(2024, 3, 10)), (date(2024, 3, 5), date(2024, 3, 15))]
    assert count_days_by_year(r) == {2024: 15}


def test_end_before_start_rejected():
    with pytest.raises(ValueError):
        count_days_by_year([(date(2024, 3, 2), date(2024, 3, 1))])


def test_tool_wrapper_states_end_counted_and_no_status():
    out = days_in_india_calculator.invoke({"date_ranges": [["2024-01-01", "2024-01-10"]]})
    assert out["days_per_year"] == {2024: 10} and out["end_date_counted"] is True
    assert "residential status" in out["note"]


@pytest.mark.parametrize("raw", ["thanks u/some_user for this", "ask @rahul99 about it", "r/IndiaTax said so"])
def test_scrubber_removes_handles(raw):
    cleaned = scrub(raw)
    assert "u/" not in cleaned and "@" not in cleaned and "r/" not in cleaned


def test_scrubber_removes_email_and_long_numbers_but_keeps_plain_text():
    cleaned = scrub("mail a.b@x.com or 123456789012; NRO accounts are fine")
    assert "x.com" not in cleaned and "123456789012" not in cleaned
    assert "NRO accounts are fine" in cleaned
