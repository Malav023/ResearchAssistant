from __future__ import annotations

import hashlib
import re
import unicodedata

# Keep '+' and '#': "c++" / "c#" / "c" must NOT collapse to the same key.
_NON_KEY = re.compile(r"[^a-z0-9+#]+")


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return _NON_KEY.sub(" ", s).strip()


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:20]


def extraction_key(query: str, intent: str, extractor_version: str) -> str:
    # intent + model version in the key: same text under another intent or a v2 model must miss.
    return f"extract:{extractor_version}:{intent}:{_h(normalize_text(query))}"


def sources_key(topic: str, *, sources: tuple[str, ...] = ("papers",), limit: int = 5) -> str:
    return f"sources:v1:{'+'.join(sorted(sources))}:{limit}:{_h(normalize_text(topic))}"


def session_key(session_id: str, part: str) -> str:
    return f"session:{session_id}:{part}"
