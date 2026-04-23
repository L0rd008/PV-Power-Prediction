"""Date/time utilities shared across the service."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def local_yesterday(timezone: str) -> tuple[datetime, datetime]:
    """Return start/end datetimes for yesterday in the given local timezone."""
    tz = ZoneInfo(timezone)
    now = datetime.now(tz)
    yesterday = (now - timedelta(days=1)).date()
    start = datetime(yesterday.year, yesterday.month, yesterday.day, tzinfo=tz)
    end = start + timedelta(days=1) - timedelta(seconds=1)
    return start, end


def to_epoch_ms(dt: datetime) -> int:
    """Convert a timezone-aware datetime to epoch milliseconds."""
    return int(dt.timestamp() * 1000)


def from_epoch_ms(ts_ms: int) -> datetime:
    """Convert epoch milliseconds to a UTC datetime."""
    from datetime import timezone
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
