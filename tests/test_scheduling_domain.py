"""Business effects, independently specified for the displayed demonstration week."""

from datetime import date
import pytest


def session():
    from speaker_lab.scheduling.domain import BookingStore, SchedulingSession

    return SchedulingSession(BookingStore(), today=date(2026, 10, 20))


def test_correction_is_the_only_booking_written_after_explicit_confirmation():
    s = session()
    s.hear("Book Thursday at three pm")
    old = s.version
    s.hear("Actually Friday at four thirty")
    assert s.proposal["start"] == "2026-10-23T16:30:00-05:00"
    assert s.version > old
    assert s.store.bookings == {}
    s.hear("uh huh")
    assert s.store.bookings == {}
    s.hear("yes confirm")
    assert len(s.store.bookings) == 1
    assert s.booking["start"] == "2026-10-23T16:30:00-05:00"
    assert s.status == "booked"


def test_reschedule_updates_one_record_and_releases_old_slot():
    s = session()
    s.hear("Book Thursday at three pm")
    s.hear("yes confirm")
    original = s.booking["id"]
    s.hear("Move it to Friday at four thirty")
    assert s.booking["start"] == "2026-10-22T15:00:00-05:00"
    s.hear("yes confirm")
    assert s.booking["id"] == original
    assert len(s.store.bookings) == 1
    assert s.booking["start"] == "2026-10-23T16:30:00-05:00"
    assert s.store.available("2026-10-22T15:00:00-05:00")


def test_stale_query_and_confirmation_cannot_write_old_intent():
    s = session()
    old = s.begin_proposal("2026-10-22T15:00:00-05:00")
    new = s.begin_proposal("2026-10-23T16:30:00-05:00")
    assert not s.finish_query(old, available=True)
    assert s.status == "checking"
    assert s.finish_query(new, available=True)
    with pytest.raises(ValueError, match="version"):
        s.confirm(old)
    assert not s.store.bookings
    s.confirm(new)
    assert s.booking["start"] == "2026-10-23T16:30:00-05:00"


def test_duplicate_confirmation_does_not_duplicate_booking():
    s = session()
    s.hear("Book Thursday at three pm")
    v = s.version
    s.confirm(v)
    s.confirm(v)
    s.hear("yes confirm")
    assert len(s.store.bookings) == 1


def test_withdrawal_invalidates_in_flight_query_and_confirmation():
    s = session()
    v = s.begin_proposal("2026-10-22T15:00:00-05:00")
    s.hear("never mind")
    assert not s.finish_query(v, available=True)
    s.hear("yes confirm")
    assert not s.store.bookings
    assert s.proposal is None


def test_unavailable_slot_cannot_be_confirmed():
    s = session()
    s.store.blocked.add("2026-10-23T16:30:00-05:00")
    s.hear("Book Friday at four thirty")
    assert s.status == "unavailable"
    s.hear("yes confirm")
    assert not s.store.bookings


@pytest.mark.parametrize(
    "text", ["yes", "uh huh", "maybe", "no", "Friday", "at four thirty"]
)
def test_incomplete_or_ambiguous_input_never_creates_booking(text):
    s = session()
    s.hear(text)
    assert not s.store.bookings


def test_reject_cancels_proposal_but_keeps_existing_booking():
    s = session()
    s.hear("Book Thursday at three pm")
    s.hear("yes confirm")
    s.hear("Move it to Friday at four thirty")
    s.hear("no")
    assert s.booking["start"] == "2026-10-22T15:00:00-05:00"
    assert s.proposal is None


def test_closed_hours_and_weekends_rejected():
    s = session()
    s.hear("Book Saturday at three pm")
    assert s.proposal is None
    s.hear("Book Friday at eight pm")
    assert s.proposal is None


def test_time_only_correction_reuses_current_date_but_does_not_commit():
    s = session()
    s.hear("Book Thursday at three pm")
    s.hear("Actually four thirty pm")
    assert s.proposal["start"] == "2026-10-22T16:30:00-05:00"
    assert not s.store.bookings


def test_prompt_contains_current_tool_facts_and_never_authorizes_unconfirmed_write():
    s = session()
    s.hear("Book Thursday at three pm")
    prompt = s.prompt()
    assert "2026-10-22" in prompt
    assert "15:00" in prompt
    assert "NOT BOOKED" in prompt
    s.hear("yes confirm")
    assert s.booking["id"] in s.prompt()
    assert "TOOL RECEIPT" in s.prompt()


def test_partial_correction_immediately_invalidates_pending_confirmation():
    s = session()
    s.hear("Book Thursday at three pm")
    old = s.version
    s.interrupt_correction()
    with pytest.raises(ValueError, match="version"):
        s.confirm(old)
    s.hear("yes confirm")
    assert not s.store.bookings
    s.hear("Actually Friday at four thirty")
    s.hear("yes confirm")
    assert s.booking["start"] == "2026-10-23T16:30:00-05:00"


def test_receipt_readback_uses_saved_booking_not_unconfirmed_new_proposal():
    s = session()
    assert s.receipt_text() is None
    s.hear("Book Thursday at three pm")
    s.hear("yes confirm")
    s.hear("Actually Friday at four thirty")
    text = s.receipt_text()
    assert "Thursday" in text and "3:00 PM" in text and "Friday" not in text
    s.hear("yes confirm")
    text = s.receipt_text()
    assert "Friday" in text and "4:30 PM" in text and "Thursday" not in text
    assert "B K 0 0 0 1" in text


def test_native_tool_context_uses_current_proposal_and_saved_receipt():
    from speaker_lab.scheduling.domain import BookingStore, SchedulingSession
    from datetime import date

    s = SchedulingSession(BookingStore(), today=date(2026, 10, 20))
    s.hear("Book Friday at four thirty")
    assert "already in progress" in s.tool_context("proposal")
    assert "Friday October 23 at 4:30 PM" in s.tool_context("proposal")
    assert "not saved" in s.tool_context("proposal")
    s.hear("Yes confirm")
    s.hear("Move it to Thursday at two pm")
    assert "Friday October 23 at 4:30 PM" in s.tool_context("receipt")
    assert "Thursday October 22 at 2:00 PM" in s.tool_context("proposal")
    assert "B K number 1" in s.tool_context("receipt")
    assert "BK-0001" not in s.tool_context("receipt")


def test_native_receipt_opening_spells_saved_reference_and_leaves_time_to_model():
    from speaker_lab.scheduling.domain import BookingStore, SchedulingSession
    from datetime import date

    s = SchedulingSession(BookingStore(), today=date(2026, 10, 20))
    s.hear("Book Friday at four thirty")
    s.hear("Yes confirm")
    assert s.tool_opening("receipt") == "Your booking reference is B K number 1 confirmed for"
    assert s.tool_opening("proposal") == "The available time is"
    s.booking = {**s.booking, "id": "BK-0012"}
    assert s.tool_opening("receipt") == "Your booking reference is B K number 12 confirmed for"
    assert "B K number 12" in s.tool_context("receipt")


def test_controlled_native_speech_derives_words_from_latest_tool_facts():
    from speaker_lab.scheduling.domain import BookingStore, SchedulingSession
    from datetime import date

    s = SchedulingSession(BookingStore(), today=date(2026, 10, 20))
    s.hear("Book Friday at four thirty")
    s.hear("Yes confirm")
    speech = s.tool_speech("receipt")
    assert "Friday October twenty third at four thirty P M" in speech
    assert "B K number 1" in speech
    s.hear("Move it to Thursday at two pm")
    assert "Thursday October twenty second at two P M" in s.tool_speech("proposal")
    assert "Friday" in s.tool_speech("receipt")


def test_both_tool_contexts_share_the_fact_free_cache_header():
    from speaker_lab.scheduling.native_control import TOOL_CONTEXT_PREFIX, TOOL_PERSONA
    s = session()
    s.hear("Book Tuesday at eleven am")
    proposal = TOOL_PERSONA + s.tool_context("proposal")
    s.hear("yes confirm")
    receipt = TOOL_PERSONA + s.tool_context("receipt")
    assert proposal.startswith(TOOL_CONTEXT_PREFIX)
    assert receipt.startswith(TOOL_CONTEXT_PREFIX)
    assert "Tuesday" in proposal[len(TOOL_CONTEXT_PREFIX):]
    assert "number 1" in receipt[len(TOOL_CONTEXT_PREFIX):]
