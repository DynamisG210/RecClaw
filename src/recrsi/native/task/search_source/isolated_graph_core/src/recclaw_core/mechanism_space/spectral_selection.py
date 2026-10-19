"""Column selection shared by the declarative compiler and train-basis fit."""

from typing import Any


def spectral_index_ranges(rank: int, ranges: Any) -> tuple[tuple[int, int], ...]:
    """Validate ascending disjoint half-open intervals selecting exactly rank."""
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError("spectral rank must be a positive integer")
    if not isinstance(ranges, (list, tuple)) or not ranges:
        raise ValueError("spectral_index_ranges must contain half-open intervals")
    result: list[tuple[int, int]] = []
    end = 0
    count = 0
    for interval in ranges:
        if (
            not isinstance(interval, (list, tuple))
            or len(interval) != 2
            or any(not isinstance(x, int) or isinstance(x, bool) for x in interval)
        ):
            raise ValueError("each spectral interval must contain two integers")
        start, stop = interval
        if start < end or stop <= start:
            raise ValueError("spectral intervals must be ascending, nonnegative and disjoint")
        if result and start == end:
            raise ValueError("adjacent spectral intervals must be merged in the draft")
        result.append((start, stop))
        count += stop - start
        end = stop
    if count != rank:
        raise ValueError("spectral interval lengths must sum to rank")
    return tuple(result)
