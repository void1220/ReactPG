"""OpenExp dataset loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def iter_openexp_records(
    path: str | Path,
    *,
    limit: int | None = None,
    offset: int = 0,
    chunk_size: int = 1024 * 1024,
) -> Iterator[dict[str, Any]]:
    """Stream records from the OpenExp JSON array.

    OpenExp is stored as one large JSON list. Loading it fully is convenient but
    wasteful for quick graph-building and visualization runs, so this iterator
    decodes one object at a time from a growing buffer.
    """

    if offset < 0:
        raise ValueError("offset must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    path = Path(path)
    decoder = json.JSONDecoder()
    yielded = 0
    seen = 0
    buffer = ""

    with path.open("r", encoding="utf-8") as handle:
        first = _read_next_nonspace(handle)
        if first != "[":
            raise ValueError(f"Expected JSON array in {path}, found {first!r}")

        while True:
            buffer = _skip_delimiters(buffer)
            while not buffer:
                chunk = handle.read(chunk_size)
                if not chunk:
                    return
                buffer += chunk
                buffer = _skip_delimiters(buffer)

            if buffer.startswith("]"):
                return

            while True:
                try:
                    record, end = decoder.raw_decode(buffer)
                    buffer = buffer[end:]
                    break
                except json.JSONDecodeError:
                    chunk = handle.read(chunk_size)
                    if not chunk:
                        raise
                    buffer += chunk

            if seen >= offset:
                yield record
                yielded += 1
                if limit is not None and yielded >= limit:
                    return
            seen += 1


def _read_next_nonspace(handle) -> str:
    while True:
        char = handle.read(1)
        if not char:
            return ""
        if not char.isspace():
            return char


def _skip_delimiters(buffer: str) -> str:
    buffer = buffer.lstrip()
    while buffer.startswith(","):
        buffer = buffer[1:].lstrip()
    return buffer
