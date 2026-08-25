#!/usr/bin/env python3
"""Fast regressions for deterministic benchmark-freeze comparison logic."""

from __future__ import annotations

import tempfile
from pathlib import Path

from benchmark_freeze import compare_records, file_records, records_digest


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "a.cpp"
        second = root / "b.py"
        first.write_text("one\n")
        second.write_text("two\n")
        frozen = file_records([first, second], root)
        require(records_digest(frozen) == records_digest(reversed(frozen)), "digest order")
        require(not any(compare_records(frozen, file_records([first, second], root)).values()), "clean tree")

        first.write_text("changed\n")
        modified = compare_records(frozen, file_records([first, second], root))
        require(modified["modified"] == ["a.cpp"], "modified file not detected")

        third = root / "c.yaml"
        third.write_text("new: true\n")
        changed_set = compare_records(frozen, file_records([first, third], root))
        require(changed_set["missing"] == ["b.py"], "missing file not detected")
        require(changed_set["added"] == ["c.yaml"], "added file not detected")

    print("benchmark freeze regression passed (3/3)")


if __name__ == "__main__":
    main()
