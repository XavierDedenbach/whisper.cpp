"""Build Whisper initial prompt from config and vocabulary.txt."""

from __future__ import annotations

import os
from pathlib import Path


def load_config() -> dict[str, str]:
    cfg: dict[str, str] = {}
    script_dir = Path(__file__).resolve().parent
    for path in (
        script_dir / "config.env",
        Path.home() / ".config/whisper-dictation/config.env",
    ):
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            val = val.strip().strip('"').strip("'")
            val = os.path.expandvars(val)
            cfg[key.strip()] = val
    return cfg


def load_vocabulary(path: Path) -> list[str]:
    """Parse vocabulary.txt; supports # comments and 'alias / alias' lines."""
    if not path.is_file():
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for part in line.split("/"):
            term = part.strip()
            key = term.lower()
            if term and key not in seen:
                seen.add(key)
                terms.append(term)
    return terms


def default_vocabulary_path(cfg: dict[str, str]) -> Path:
    raw = cfg.get(
        "WHISPER_VOCABULARY_FILE",
        "~/.config/whisper-dictation/vocabulary.txt",
    ).strip()
    return Path(os.path.expanduser(raw))


def build_whisper_prompt(cfg: dict[str, str] | None = None) -> str:
    """Compose initial prompt from prefix + vocabulary terms."""
    cfg = cfg or load_config()
    explicit = cfg.get("WHISPER_PROMPT", "").strip()
    prefix = cfg.get("WHISPER_PROMPT_PREFIX", "Technical dictation.").strip()
    terms = load_vocabulary(default_vocabulary_path(cfg))
    if terms:
        return f"{prefix} Terms: {', '.join(terms)}."
    if explicit:
        return explicit
    return prefix or "Technical dictation."


def main() -> None:
    print(build_whisper_prompt())


if __name__ == "__main__":
    main()
