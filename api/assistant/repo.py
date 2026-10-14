from __future__ import annotations

from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId


class SessionRepo:
    """Mongo = source of truth for sessions, paper snapshots and messages (Redis is only a hot cache).
    Sessions live in the existing `search_sessions` collection; `query` stays the sidebar title."""

    def __init__(self, db):
        self.db = db
        self.sessions = db["search_sessions"]
        self.messages = db["chat_messages"]

    async def ensure_indexes(self) -> None:
        await self.sessions.create_index([("user_id", 1), ("created_at", -1)])
        await self.messages.create_index([("session_id", 1), ("ts", 1)])

    @staticmethod
    def _oid(sid: str) -> ObjectId | None:
        try:
            return ObjectId(sid)
        except (InvalidId, TypeError):
            return None

    async def create_session(self, user_id: str, title: str) -> str:
        now = datetime.now(timezone.utc)
        r = await self.sessions.insert_one({"user_id": user_id, "query": title, "paper_ids": [], "papers": [],
                                            "created_at": now, "updated_at": now})
        return str(r.inserted_id)

    async def get_session(self, user_id: str, sid: str) -> dict | None:
        oid = self._oid(sid)
        if oid is None:
            return None
        return await self.sessions.find_one({"_id": oid, "user_id": user_id})   # ownership enforced here

    async def save_papers(self, sid: str, papers: list[dict]) -> None:
        await self.sessions.update_one({"_id": ObjectId(sid)}, {"$set": {
            "papers": papers, "paper_ids": [p["id"] for p in papers], "updated_at": datetime.now(timezone.utc)}})

    async def touch(self, sid: str, intent: str) -> None:
        await self.sessions.update_one({"_id": ObjectId(sid)}, {"$set": {
            "last_intent": intent, "updated_at": datetime.now(timezone.utc)}})

    async def add_messages(self, sid: str, user_id: str, msgs: list[dict]) -> None:
        if msgs:
            await self.messages.insert_many([{**m, "session_id": sid, "user_id": user_id} for m in msgs])

    async def get_messages(self, sid: str, limit: int = 100, *, last: bool = False) -> list[dict]:
        cur = self.messages.find({"session_id": sid}).sort("ts", -1 if last else 1).limit(limit)
        docs = [d async for d in cur]
        return docs[::-1] if last else docs

    async def delete_if_empty(self, sid: str) -> None:
        if await self.messages.count_documents({"session_id": sid}) == 0:
            await self.sessions.delete_one({"_id": ObjectId(sid)})
