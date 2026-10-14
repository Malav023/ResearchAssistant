import httpx
from fastapi import APIRouter, HTTPException, Query

from config import get_config
from papers_service import search_papers

router = APIRouter(prefix="/api", tags=["papers"])


@router.get("/papers")
async def get_top_papers(q: str = Query(..., min_length=2)):
    """Lightweight top-5 list (unchanged contract, plus `abstract`). The assistant pipeline uses the same service."""
    cfg = get_config()
    try:
        papers = await search_papers(q, limit=5, openalex_key=cfg.openalex_key, s2_key=cfg.s2_key)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"OpenAlex fetch failed: {e}")
    return {"query": q, "papers": [
        {"id": p.id, "title": p.title, "year": p.year, "citations": p.citations,
         "first_author": p.first_author, "top_concept": p.top_concept, "doi": p.doi, "abstract": p.abstract}
        for p in papers
    ]}
