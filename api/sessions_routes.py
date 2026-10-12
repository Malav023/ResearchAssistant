from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from bson import ObjectId

from db import sessions_collection
from auth_routes import get_current_user
from models import SaveSessionRequest

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
async def list_sessions(user=Depends(get_current_user)):
    cursor = sessions_collection().find({"user_id": str(user["_id"])}).sort("created_at", -1).limit(50)
    items = []
    async for doc in cursor:
        items.append({
            "id": str(doc["_id"]),
            "query": doc["query"],
            "created_at": doc["created_at"].isoformat(),
        })
    return {"sessions": items}


@router.post("")
async def save_session(body: SaveSessionRequest, user=Depends(get_current_user)):
    doc = {
        "user_id": str(user["_id"]),
        "query": body.query,
        "paper_ids": body.paper_ids,
        "created_at": datetime.now(timezone.utc),
    }
    result = await sessions_collection().insert_one(doc)
    return {"id": str(result.inserted_id)}
