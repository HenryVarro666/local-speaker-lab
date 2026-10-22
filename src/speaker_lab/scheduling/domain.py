"""Small, explicit booking domain, independent of voice/model dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
import re
from uuid import uuid4
from zoneinfo import ZoneInfo

ZONE = ZoneInfo("America/Chicago")
DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
NUMBERS = dict(
    zip(
        (
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
        ),
        range(1, 13),
    )
)


def parse_request(text: str, today: date, current: str | None = None) -> str | None:
    """Parse the displayed week's English weekday/time, never arbitrary free text."""
    text = re.sub(r"[^a-z0-9: ]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    day = next(
        (i for i, name in enumerate(DAYS) if re.search(rf"\b{name}\b", text)), None
    )
    if day is not None:
        target = today + timedelta(days=(day - today.weekday()) % 7)
    elif current and re.search(r"\b(actually|instead|change|move|make|no)\b", text):
        target = datetime.fromisoformat(current).date()
    else:
        return None
    # Only a time following 'at', or an explicit correction, can become a slot.
    match = re.search(r"\b(?:at|actually|instead|make it|change to)\s+(.+)", text)
    tail = match.group(1) if match else text
    digits = re.search(r"\b(\d{1,2})(?::(\d{2}))?\b", tail)
    words = re.search(
        r"\b(" + "|".join(NUMBERS) + r")\b(?:\s+(thirty|fifteen|forty five|o clock))?",
        tail,
    )
    if digits:
        hour, minute = int(digits[1]), int(digits[2] or 0)
    elif words:
        hour, minute = (
            NUMBERS[words[1]],
            {"thirty": 30, "fifteen": 15, "forty five": 45}.get(words[2], 0),
        )
    else:
        return None
    if re.search(r"\b(pm|p m|afternoon|evening)\b", text) or (
        1 <= hour <= 5 and not re.search(r"\b(am|a m|morning)\b", text)
    ):
        if hour < 12:
            hour += 12
    elif hour == 12 and re.search(r"\b(am|a m)\b", text):
        hour = 0
    if target.weekday() > 4 or not 9 <= hour < 17 or minute not in (0, 30):
        return None
    return datetime.combine(target, time(hour, minute), ZONE).isoformat()


@dataclass
class BookingStore:
    """In-memory tools. Changes are atomic in the gateway's one event loop."""

    bookings: dict = field(default_factory=dict)
    blocked: set = field(default_factory=set)
    operations: dict = field(default_factory=dict)

    def available(self, start: str, excluding: str | None = None) -> bool:
        return start not in self.blocked and all(
            b["start"] != start or key == excluding for key, b in self.bookings.items()
        )

    def commit(self, operation: str, start: str, booking_id: str | None = None) -> dict:
        if operation in self.operations:
            return dict(self.operations[operation])
        if not self.available(start, booking_id):
            raise ValueError("Slot is no longer available")
        key = booking_id or f"BK-{len(self.bookings) + 1:04d}"
        receipt = {
            "id": key,
            "start": start,
            "service": "Consultation",
            "timezone": "America/Chicago",
            "simulated": True,
        }
        self.bookings[key] = receipt
        self.operations[operation] = dict(receipt)
        return dict(receipt)


class SchedulingSession:
    def __init__(self, store: BookingStore, today: date | None = None):
        self.id = uuid4().hex
        self.store = store
        self.today = today or datetime.now(ZONE).date()
        self.version = 0
        self.proposal = None
        self.booking = None
        self.status = "collecting"
        self.history = []
        self.events = []
        self.notice = "Tell me a weekday and time, for example Thursday at three pm."
        self._confirmed = set()

    def emit(self, kind: str, **data):
        event = {"kind": kind, "version": self.version, **data}
        self.events.append(event)
        return event

    def snapshot(self) -> dict:
        return {
            "session_id": self.id,
            "version": self.version,
            "status": self.status,
            "proposal": self.proposal,
            "booking": self.booking,
            "notice": self.notice,
            "today": str(self.today),
            "timezone": "America/Chicago",
            "simulated": True,
        }

    def begin_proposal(self, start: str) -> int:
        parsed = datetime.fromisoformat(start)
        if (
            parsed.tzinfo is None
            or parsed.weekday() > 4
            or not 9 <= parsed.hour < 17
            or parsed.minute not in (0, 30)
        ):
            raise ValueError("Unsupported slot")
        self.version += 1
        self.proposal = {
            "start": start,
            "version": self.version,
            "operation": "reschedule" if self.booking else "create",
        }
        self.status = "checking"
        self.notice = "Checking the proposed time; no change has been saved."
        self.emit("query_started", start=start, request_id=f"{self.id}:{self.version}")
        return self.version

    def finish_query(self, version: int, available: bool) -> bool:
        if version != self.version or self.proposal is None:
            self.emit("stale_query_ignored", request_version=version)
            return False
        self.status = "awaiting_confirmation" if available else "unavailable"
        self.notice = (
            "Please say yes confirm to save this exact proposal, or give a correction."
            if available
            else "That time is unavailable. Please choose another time."
        )
        self.emit("query_finished", available=available, start=self.proposal["start"])
        return True

    def confirm(self, version: int) -> dict | None:
        if version != self.version:
            raise ValueError("Confirmation version is stale")
        if version in self._confirmed:
            return self.booking
        if self.status != "awaiting_confirmation" or self.proposal is None:
            return None
        try:
            receipt = self.store.commit(
                f"{self.id}:{version}",
                self.proposal["start"],
                self.booking["id"] if self.booking else None,
            )
        except ValueError:
            self.status = "unavailable"
            self.notice = (
                "That slot was taken before confirmation. Please choose another."
            )
            self.emit("commit_rejected", reason="unavailable")
            return None
        self.booking = receipt
        self._confirmed.add(version)
        self.status = "booked"
        self.notice = f"Confirmed: {receipt['start']}, reference {receipt['id']}."
        self.emit("tool_receipt", receipt=receipt)
        return receipt

    def withdraw(self):
        self.version += 1
        self.proposal = None
        self.status = "booked" if self.booking else "collecting"
        self.notice = (
            "Pending change withdrawn. Existing booking is unchanged."
            if self.booking
            else "No appointment has been booked."
        )
        self.emit("proposal_withdrawn")

    def interrupt_correction(self):
        """Revoke the old confirmation as soon as a correction is detected."""
        if self.proposal:
            self.version += 1
            self.proposal = {**self.proposal, "version": self.version}
            self.status = "clarifying"
            self.notice = "Listening to your correction. The previous proposal cannot be confirmed."
            self.emit("correction_started")

    def hear(self, text: str, *, defer_query: bool = False) -> dict:
        cleaned = re.sub(r"[^a-z0-9 ]", " ", text.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        self.history.append(text[:300])
        self.history = self.history[-8:]
        self.emit("user_final", text=text)
        current = (self.proposal or self.booking or {}).get("start")
        start = parse_request(text, self.today, current)
        if start:
            version = self.begin_proposal(start)
            if not defer_query:
                self.finish_query(
                    version,
                    self.store.available(
                        start, self.booking["id"] if self.booking else None
                    ),
                )
        elif cleaned in (
            "no",
            "no thanks",
            "never mind",
            "nevermind",
            "stop",
            "cancel that",
            "do not book",
            "don t book",
        ):
            self.withdraw()
        elif cleaned in (
            "yes confirm",
            "yes please confirm",
            "confirm",
            "confirm booking",
            "confirm appointment",
            "yes book it",
            "yes please book it",
        ):
            self.confirm(self.version)
        elif cleaned in ("uh huh", "mhm", "hmm", "okay", "ok", "yes", "right"):
            self.emit("backchannel", text=text)
        else:
            self.notice = "Please give a weekday and time, or say yes confirm for the displayed proposal."
        return self.snapshot()

    def prompt(self) -> str:
        def describe(start):
            d = datetime.fromisoformat(start)
            return d.strftime("%A %Y-%m-%d at %H:%M (%I:%M %p)")

        confirmed = (
            f"TOOL RECEIPT: {self.booking['id']}, confirmed {describe(self.booking['start'])}."
            if self.booking
            else "NOT BOOKED: no appointment is saved."
        )
        proposal = (
            f"Requested time: {describe(self.proposal['start'])}. "
            if self.proposal
            else ""
        )
        instructions = {
            "collecting": "Ask for a weekday and time.",
            "awaiting_confirmation": "Requested slot is available. Ask the user to say yes confirm. This pending change is not saved.",
            "unavailable": "Requested slot is unavailable. Ask for another time.",
            "booked": "Tell the user their exact confirmed time and booking reference from the TOOL RECEIPT.",
            "clarifying": "Listen to the correction; the previous proposal cannot be confirmed.",
        }.get(self.status, "Wait for the availability result.")
        return (
            "You work for Lakeside Appointment Desk and your name is Alex. Information: "
            "Speak concise English. Yield to corrections. Weekdays 9am-5pm, half-hour slots, America/Chicago. "
            f"Today {self.today}. {proposal}{confirmed} {instructions} "
            "Only the external tool can save changes. Never invent a successful booking. "
            "Answer the user directly without repeating your greeting."
        )

    def receipt_text(self) -> str | None:
        if not self.booking:
            return None
        d = datetime.fromisoformat(self.booking["start"])
        clock = d.strftime("%I:%M %p").lstrip("0")
        reference = " ".join(self.booking["id"].replace("-", ""))
        return f"Your simulated appointment is confirmed for {d.strftime('%A, %B')} {d.day}, at {clock}, Central Time. Your booking reference is {reference}."

    def tool_context(self, purpose):
        fact = self.proposal if purpose == "proposal" else self.booking
        if not fact:
            raise ValueError("No current tool fact")
        d = datetime.fromisoformat(fact["start"])
        when = f"{d.strftime('%A %B')} {d.day} at {d.strftime('%I:%M %p').lstrip('0')} Central"
        if purpose == "proposal":
            content = f"{when} is available, not saved. Tell the user the available time and ask for confirmation."
        else:
            content = f"appointment with reference {self.spoken_reference()} has been saved successfully for {when}. Tell the user their confirmed time and reference now."
        return (
            "The conversation is already in progress. Tool result: "
            + content
            + " Answer directly without a greeting."
        )

    def tool_opening(self, purpose):
        if purpose == "proposal":
            return "The available time is"
        return "Your booking reference is " + self.spoken_reference() + " confirmed for"

    def spoken_reference(self):
        if not self.booking:
            raise ValueError("No saved booking reference")
        prefix, counter = self.booking["id"].rsplit("-", 1)
        # BK number 1 is the spoken alias of the canonical displayed BK-0001.
        # Repeated leading zero words were ambiguous in waveform verification.
        return " ".join(prefix) + " number " + str(int(counter))

    def tool_speech(self, purpose):
        fact = self.proposal if purpose == "proposal" else self.booking
        if not fact:
            raise ValueError("No tool fact to speak")
        d = datetime.fromisoformat(fact["start"])
        ordinals = (
            "",
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
            "eleventh",
            "twelfth",
            "thirteenth",
            "fourteenth",
            "fifteenth",
            "sixteenth",
            "seventeenth",
            "eighteenth",
            "nineteenth",
        )
        day = (
            ordinals[d.day]
            if d.day < 20
            else (
                {20: "twentieth", 30: "thirtieth"}.get(d.day)
                or ("twenty " if d.day < 30 else "thirty ") + ordinals[d.day % 10]
            )
        )
        hour = next(
            word for word, number in NUMBERS.items() if number == (d.hour % 12 or 12)
        )
        clock = (
            hour
            + (" thirty" if d.minute else "")
            + (" P M" if d.hour >= 12 else " A M")
        )
        when = f"{d.strftime('%A %B')} {day} at {clock}"
        if purpose == "proposal":
            return when + " is available. Say yes confirm to save."
        return self.tool_opening("receipt") + " " + when + "."

    def proposal_text(self) -> str | None:
        if self.status != "awaiting_confirmation" or not self.proposal:
            return None
        d = datetime.fromisoformat(self.proposal["start"])
        clock = d.strftime("%I:%M %p").lstrip("0")
        return f"The available time is {d.strftime('%A, %B')} {d.day}, at {clock}, Central Time. Say yes confirm to save this proposal, or tell me a correction."
