import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from graph_builder import fetch_openalex_papers, fetch_s2_paper_details, build_graph
from auth_routes import router as auth_router
from papers_routes import router as papers_router
from sessions_routes import router as sessions_router

load_dotenv()

OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")
S2_API_KEY = os.getenv("S2_API_KEY", "")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

app = FastAPI(title="Research Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,  # required so the auth cookie is sent/received
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(papers_router)
app.include_router(sessions_router)


@app.get("/api/graph")
async def get_citation_graph(q: str = Query(..., min_length=2, description="Research topic or paper title")):
    """
    Full citation graph: only called when the user opens the Citation Graph
    modal (kept separate from /api/papers so the initial search stays fast).
    """
    try:
        oa_papers = await fetch_openalex_papers(q, api_key=OPENALEX_API_KEY)

        if not oa_papers:
            return {"nodes": [], "edges": [], "query": q}

        dois = [
            f"DOI:{p['doi'].replace('https://doi.org/', '')}"
            for p in oa_papers
            if p.get("doi")
        ]

        s2_data = []
        if dois:
            try:
                s2_data = await fetch_s2_paper_details(dois[:20], api_key=S2_API_KEY)
            except Exception as e:
                print(f"S2 fetch failed (non-fatal): {e}")

        graph = build_graph(oa_papers, s2_data)
        graph["query"] = q
        return graph

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
