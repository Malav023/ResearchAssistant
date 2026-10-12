import os
from fastapi import APIRouter, Query, HTTPException

from graph_builder import fetch_openalex_papers

router = APIRouter(prefix="/api", tags=["papers"])

OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")


def _clean_doi(doi: str) -> str:
    if not doi:
        return ""
    return doi.replace("https://doi.org/", "")


@router.get("/papers")
async def get_top_papers(q: str = Query(..., min_length=2)):
    """
    Lightweight endpoint for the results view: top 5 papers by citation count.
    Separate from /api/graph so the initial search stays fast; the full
    citation graph is only fetched on demand when the user opens that modal.
    """
    try:
        papers = await fetch_openalex_papers(q, api_key=OPENALEX_API_KEY)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAlex fetch failed: {e}")

    ranked = sorted(papers, key=lambda p: p.get("cited_by_count", 0), reverse=True)[:5]

    out = []
    for p in ranked:
        authorships = p.get("authorships", [])
        first_author = ""
        if authorships:
            first_author = authorships[0].get("author", {}).get("display_name", "")
        concepts = p.get("concepts", [])
        top_concept = concepts[0].get("display_name", "") if concepts else ""

        out.append({
            "id": p.get("id", ""),
            "title": p.get("title") or "Untitled",
            "year": p.get("publication_year"),
            "citations": p.get("cited_by_count", 0),
            "first_author": first_author,
            "top_concept": top_concept,
            "doi": _clean_doi(p.get("doi", "")),
        })

    return {"query": q, "papers": out}
