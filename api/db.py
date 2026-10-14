from motor.motor_asyncio import AsyncIOMotorClient

from config import get_config

_cfg = get_config()
_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(_cfg.mongo_uri)
    return _client


def get_db():
    return get_client()[_cfg.mongo_db]


def users_collection():
    return get_db()["users"]


def sessions_collection():
    return get_db()["search_sessions"]
