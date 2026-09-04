"""Build Whisper initial prompt from config and vocabulary.txt."""

from __future__ import annotations

import os
import re
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


def _split_aliases(raw: str) -> list[str]:
    parts: list[str] = []
    seen: set[str] = set()
    for part in raw.split("/"):
        term = part.strip()
        key = term.lower()
        if term and key not in seen:
            seen.add(key)
            parts.append(term)
    return parts


def parse_vocabulary(text: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (prompt terms, replacements of (mishearing, canonical))."""
    terms: list[str] = []
    replacements: list[tuple[str, str]] = []
    seen_terms: set[str] = set()
    seen_src: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=>" in line:
            left, _, right = line.partition("=>")
            canonicals = _split_aliases(left)
            if not canonicals:
                continue
            canonical = canonicals[0]
            key = canonical.lower()
            if key not in seen_terms:
                seen_terms.add(key)
                terms.append(canonical)
            for src in _split_aliases(right):
                src_key = src.lower()
                if src == canonical or src_key in seen_src:
                    continue
                seen_src.add(src_key)
                replacements.append((src, canonical))
            continue
        for term in _split_aliases(line):
            key = term.lower()
            if key not in seen_terms:
                seen_terms.add(key)
                terms.append(term)
    return terms, replacements


def load_vocabulary(path: Path) -> tuple[list[str], list[tuple[str, str]]]:
    """Parse one vocabulary file. Missing files yield empty results."""
    if not path.is_file():
        return [], []
    return parse_vocabulary(path.read_text())


def _merge_vocab(
    parts: list[tuple[list[str], list[tuple[str, str]]]],
) -> tuple[list[str], list[tuple[str, str]]]:
    terms: list[str] = []
    replacements: list[tuple[str, str]] = []
    seen_terms: set[str] = set()
    seen_src: set[str] = set()
    for part_terms, part_repls in parts:
        for term in part_terms:
            key = term.lower()
            if key not in seen_terms:
                seen_terms.add(key)
                terms.append(term)
        for src, dest in part_repls:
            key = src.lower()
            if key not in seen_src:
                seen_src.add(key)
                replacements.append((src, dest))
    return terms, replacements


def repo_vocabulary_path() -> Path:
    return Path(__file__).resolve().parent / "vocabulary.txt"


def user_vocabulary_path(cfg: dict[str, str]) -> Path:
    raw = cfg.get(
        "WHISPER_VOCABULARY_FILE",
        "~/.config/whisper-dictation/vocabulary.txt",
    ).strip()
    return Path(os.path.expanduser(raw))


def load_all_vocabulary(
    cfg: dict[str, str] | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Load committed repo terms, then optional user overlay."""
    cfg = cfg or load_config()
    repo = repo_vocabulary_path()
    user = user_vocabulary_path(cfg)
    parts = [load_vocabulary(repo)]
    if user.resolve() != repo.resolve():
        parts.append(load_vocabulary(user))
    return _merge_vocab(parts)


def default_vocabulary_path(cfg: dict[str, str]) -> Path:
    """Preferred existing vocabulary file (repo first, then user overlay)."""
    repo = repo_vocabulary_path()
    if repo.is_file():
        return repo
    return user_vocabulary_path(cfg)


def apply_replacements(text: str, replacements: list[tuple[str, str]]) -> str:
    """Rewrite mishearings to canonical terms (longest match first)."""
    if not text or not replacements:
        return text
    ordered = sorted(replacements, key=lambda item: len(item[0]), reverse=True)
    result = text
    for src, dest in ordered:
        pattern = re.compile(r"(?<!\w)" + re.escape(src) + r"(?!\w)", re.IGNORECASE)
        result = pattern.sub(dest, result)
    return result


def _as_prompt_prefix(raw: str) -> str:
    """Keep the prefix unfinished; a trailing period can bias punctuation output."""
    prefix = raw.strip()
    if prefix.endswith("."):
        return prefix[:-1].rstrip() + ":"
    return prefix


def build_whisper_prompt(cfg: dict[str, str] | None = None) -> str:
    """Compose initial prompt from prefix + vocabulary terms."""
    cfg = cfg or load_config()
    explicit = cfg.get("WHISPER_PROMPT", "").strip()
    prefix = _as_prompt_prefix(cfg.get("WHISPER_PROMPT_PREFIX", "Technical dictation:"))
    terms, _replacements = load_all_vocabulary(cfg)
    if terms:
        return f"{prefix} {', '.join(terms)}"
    if explicit:
        return explicit
    return prefix or "Technical dictation:"


def main() -> None:
    print(build_whisper_prompt())


if __name__ == "__main__":
    main()
