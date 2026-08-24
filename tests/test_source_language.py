from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

# Curated French tokens/phrases that have appeared in active Python source during
# repository cleanup. Patterns are word-bounded where necessary so English
# identifiers such as ``ambiguous_fake`` are not false positives.
FORBIDDEN = [
    r"\bcolumns?\s+manquantes?\b",
    r"\bmanquants?\b",
    r"\binsuffisant\w*\b",
    r"\bpointent\b",
    r"\bont\b",
    r"\bdoublons?\b",
    r"\bcomplets?\b",
    r"\bcanonique\b",
    r"\bincorrecte\b",
    r"\bbande\b",
    r"\bnominale\b",
    r"\boui\b",
    r"\bfinaux\b",
    r"\bfinale\b",
    r"\bexistent\b",
    r"\bambigu\b",
    r"\brequis(?:e|es|s)?\b",
    r"\bintrouvable\b",
    r"\brelancez\b",
    r"\br[ée]sultat\s+final\b",
    r"\br[ée]sum[ée]\s+json\b",
    r"\btermin[ée](?:e|es|s)?\b",
    r"\bsorties?\b",
    r"\bsauvegard[ée](?:e|es|s)?\b",
    r"\binattendu(?:e|es|s)?\b",
    r"\babsent(?:e|es|s)?\b",
    r"\bancien(?:ne|nes|s)?\b",
    r"\bd[ée]j[àa]\b",
]

PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in FORBIDDEN]


def _python_sources() -> list[Path]:
    files = [path for path in ROOT.rglob("*.py") if "__pycache__" not in path.parts]
    return sorted(path for path in files if path.resolve() != SELF)


def test_no_french_or_mixed_french_english_in_python_sources() -> None:
    hits: list[str] = []
    for path in _python_sources():
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = path.relative_to(ROOT)
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in PATTERNS:
                match = pattern.search(line)
                if match:
                    hits.append(
                        f"{rel}:{lineno}: {match.group(0)!r} :: {line.strip()}"
                    )
                    break

    assert not hits, "French/mixed-language remnants found:\n" + "\n".join(hits)
