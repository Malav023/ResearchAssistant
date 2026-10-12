import httpx
import asyncio
from typing import Optional

OPENALEX_BASE = "https://api.openalex.org"
S2_BASE = "https://api.semanticscholar.org/graph/v1"

# Put your keys in .env and load them in main.py, then pass here
async def fetch_openalex_papers(query: str, api_key: Optional[str] = None) -> list[dict]:
    """Search OpenAlex for papers matching the query."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    params = {
        "search": query,
        "per_page": 15,
        "select": "id,doi,title,authorships,publication_year,cited_by_count,concepts,referenced_works,related_works"
    }

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{OPENALEX_BASE}/works", params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])


async def fetch_s2_paper_details(paper_ids: list[str], api_key: Optional[str] = None) -> list[dict]:
    """
    Fetch Semantic Scholar details for a list of paper IDs.
    paper_ids should be S2 paper IDs or DOIs like 'DOI:10.xxxx/xxxx'
    """
    if not paper_ids:
        return []

    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    fields = "paperId,title,year,citationCount,references,citations,embedding"
    results = []

    async with httpx.AsyncClient(timeout=20) as client:
        # S2 batch endpoint supports up to 500 IDs
        payload = {"ids": paper_ids[:50]}  # keep it small for demo
        resp = await client.post(
            f"{S2_BASE}/paper/batch",
            json=payload,
            params={"fields": fields},
            headers=headers
        )
        if resp.status_code == 200:
            results = resp.json()

    return results


def build_graph(openalex_papers: list[dict], s2_details: list[dict]) -> dict:
    """
    Combine OpenAlex + S2 data into a graph structure: { nodes, edges }

    Node schema:
        id, label, year, citations, abstract_snippet, doi, source

    Edge schema:
        source, target, type  (type: "citation" | "related")
    """
    nodes = []
    edges = []
    node_ids = set()

    # --- Build nodes from OpenAlex results ---
    doi_to_node_id = {}
    openalex_ids = set()

    for i, paper in enumerate(openalex_papers):
        node_id = paper.get("id", f"oa_{i}")  # OpenAlex URL as ID
        doi = paper.get("doi", "")
        title = paper.get("title", "Untitled")
        year = paper.get("publication_year")
        citations = paper.get("cited_by_count", 0)

        # Extract first author
        authorships = paper.get("authorships", [])
        first_author = ""
        if authorships:
            author_obj = authorships[0].get("author", {})
            first_author = author_obj.get("display_name", "")

        # Extract top concept/topic
        concepts = paper.get("concepts", [])
        top_concept = concepts[0].get("display_name", "") if concepts else ""

        nodes.append({
            "id": node_id,
            "label": title,
            "year": year,
            "citations": citations,
            "doi": doi,
            "first_author": first_author,
            "top_concept": top_concept,
            "source": "openalex",
            "referenced_works": paper.get("referenced_works", []),
            "related_works": paper.get("related_works", []),
        })

        node_ids.add(node_id)
        openalex_ids.add(node_id)
        if doi:
            doi_to_node_id[doi] = node_id

    # --- Build citation edges from OpenAlex reference lists ---
    for paper_node in nodes:
        source_id = paper_node["id"]

        for ref_id in paper_node.get("referenced_works", []):
            if ref_id in node_ids and ref_id != source_id:
                edges.append({
                    "source": source_id,
                    "target": ref_id,
                    "type": "citation"
                })

        for rel_id in paper_node.get("related_works", []):
            if rel_id in node_ids and rel_id != source_id:
                edges.append({
                    "source": source_id,
                    "target": rel_id,
                    "type": "related"
                })

    # --- Enrich nodes with S2 data if available ---
    s2_by_title = {}
    for s2 in s2_details:
        if s2 and s2.get("title"):
            s2_by_title[s2["title"].lower().strip()] = s2

    for node in nodes:
        s2_match = s2_by_title.get(node["label"].lower().strip())
        if s2_match:
            # Use S2 citation count if higher resolution
            node["s2_id"] = s2_match.get("paperId", "")
            # Add cross-paper citation edges from S2
            for cited in s2_match.get("references", [])[:10]:  # limit for perf
                cited_title = (cited.get("title") or "").lower().strip()
                cited_node = s2_by_title.get(cited_title)
                if cited_node:
                    # find matching node id
                    for n in nodes:
                        if n["label"].lower().strip() == cited_title:
                            edge = {"source": node["id"], "target": n["id"], "type": "citation"}
                            if edge not in edges:
                                edges.append(edge)

    # Clean up internal fields not needed by frontend
    for node in nodes:
        node.pop("referenced_works", None)
        node.pop("related_works", None)

    # Deduplicate edges
    seen_edges = set()
    unique_edges = []
    for e in edges:
        key = (e["source"], e["target"], e["type"])
        rev_key = (e["target"], e["source"], e["type"])
        if key not in seen_edges and rev_key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(e)

    return {"nodes": nodes, "edges": unique_edges}
