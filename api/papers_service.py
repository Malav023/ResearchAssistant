"""OpenAlex (+ Semantic Scholar abstract fallback) -> ra_core.Paper."""
from __future__ import annotations

import dataclasses
import re

import httpx

from ra_core.rag.types import Paper
import asyncio, logging
log = logging.getLogger("papers")

OPENALEX_BASE = "https://api.openalex.org"
S2_BASE = "https://api.semanticscholar.org/graph/v1"
_SELECT = ("id,doi,title,authorships,publication_year,cited_by_count,concepts,abstract_inverted_index")
_TAGS = re.compile(r"<[^>]+>")


def reconstruct_abstract(inv: dict | None) -> str:
    """OpenAlex ships abstracts as {word: [positions]}."""
    if not inv:
        return ""
    words: dict[int, str] = {}
    for w, positions in inv.items():
        for p in positions:
            words[p] = w
    return " ".join(words[i] for i in sorted(words))


def clean_doi(doi: str | None) -> str:
    return (doi or "").replace("https://doi.org/", "").strip()


def to_paper(p: dict) -> Paper:
    authorships = p.get("authorships") or []
    first = (authorships[0].get("author") or {}).get("display_name", "") if authorships else ""
    concepts = p.get("concepts") or []
    doi = clean_doi(p.get("doi"))
    return Paper(
        id=p.get("id", ""),
        title=_TAGS.sub("", p.get("title") or "Untitled").strip(),
        abstract=_TAGS.sub("", reconstruct_abstract(p.get("abstract_inverted_index"))).strip(),
        url=(f"https://doi.org/{doi}" if doi else p.get("id", "")),
        year=p.get("publication_year"),
        first_author=first,
        citations=p.get("cited_by_count") or 0,
        top_concept=(concepts[0].get("display_name", "") if concepts else ""),
        doi=doi,
    )


async def fetch_openalex(query: str, api_key: str = "", *, per_page: int = 15,
                         client: httpx.AsyncClient | None = None) -> list[dict]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    params = {"search": query, "per_page": per_page, "select": _SELECT}
    own = client is None
    client = client or httpx.AsyncClient(timeout=15)
    try:
        r = await client.get(f"{OPENALEX_BASE}/works", params=params, headers=headers)
        r.raise_for_status()
        return r.json().get("results", [])
    finally:
        if own:
            await client.aclose()


async def fetch_s2_abstracts(dois, api_key="", *, client=None):
    """Best effort: one abstract per DOI ('' when unknown). Never raises, but logs why."""
    if not dois:
        return []
    headers = {"x-api-key": api_key} if api_key else {}
    own = client is None
    client = client or httpx.AsyncClient(timeout=15)
    try:
        for attempt in (1, 2):                      # one retry on 429
            r = await client.post(f"{S2_BASE}/paper/batch", params={"fields": "paperId,abstract"},
                                  json={"ids": [f"DOI:{d}" for d in dois]}, headers=headers)
            if r.status_code == 429 and attempt == 1:
                await asyncio.sleep(1.5)
                continue
            break
        if r.status_code != 200:
            log.warning("S2 fallback HTTP %s: %s", r.status_code, r.text[:200])
            return [""] * len(dois)
        return [((x or {}).get("abstract") or "") for x in r.json()]
    except Exception as e:
        log.warning("S2 fallback failed: %r", e)
        return [""] * len(dois)
    finally:
        if own:
            await client.aclose()


async def search_papers(query, *, limit=5, openalex_key="", s2_key="", client=None):
    raw = await fetch_openalex(query, openalex_key, client=client)
    ranked = sorted(raw, key=lambda p: p.get("cited_by_count") or 0, reverse=True)
    pool = [to_paper(p) for p in ranked]                       # all 15, not just the top 5

    missing = [i for i, p in enumerate(pool) if not p.abstract and p.doi]
    if missing:
        found = await fetch_s2_abstracts([pool[i].doi for i in missing], s2_key, client=client)
        for i, ab in zip(missing, found):
            if ab:
                pool[i] = dataclasses.replace(pool[i], abstract=ab)

    # Prefer papers that have an abstract; backfill with abstract-less ones only if needed.
    chosen = ([p for p in pool if p.abstract] + [p for p in pool if not p.abstract])[:limit]
    chosen.sort(key=lambda p: p.citations, reverse=True)      # keep the citation ordering
    return chosen