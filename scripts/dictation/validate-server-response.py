#!/usr/bin/env python3
"""Validate that a whisper-server response contains non-empty transcript text."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    try:
        if len(sys.argv) == 2:
            payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        elif len(sys.argv) == 1:
            payload = json.load(sys.stdin)
        else:
            return 2
        text = payload.get("text")
    except (AttributeError, json.JSONDecodeError, OSError):
        return 1
    return 0 if isinstance(text, str) and text.strip() else 1


if __name__ == "__main__":
    raise SystemExit(main())
