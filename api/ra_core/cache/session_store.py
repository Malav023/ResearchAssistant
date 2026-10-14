from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from ..rag.embedder import Embedder
from ..rag.index import HybridIndex, IndexStale
from ..rag.types import Paper, papers_fingerprint
from ..settings import Settings
from .keys import session_key
from .redis_cache import RedisCache


@dataclass
class _Local:
    papers: list[Paper] | None = None
    index: HybridIndex | None = None
    history: list[dict] = field(default_factory=list)
    expires: float = 0.0
    last_touch: float = 0.0


class SessionStore:
    """Session state = fetched papers, RAG index, chat history.

    Unlike the extraction/source caches this is NOT optional data (chat breaks without it),
    so Redis is backed by a small in-process LRU that also acts as L1 (saves Redis commands).
    Single-replica assumption for the L1 (matches the single-Space deployment). If you scale to
    multiple replicas without sticky sessions, set local_sessions_max=0 to go Redis-only.
    """

    def __init__(self, cache: RedisCache, settings: Settings):
        self.cache, self.s = cache, settings
        self._local: OrderedDict[str, _Local] = OrderedDict()
        self._build_locks: dict[str, asyncio.Lock] = {}

    # ---------- local LRU ----------
    def _get_local(self, sid: str) -> _Local | None:
        ent = self._local.get(sid)
        if ent and ent.expires < time.monotonic():
            self._local.pop(sid, None)
            return None
        if ent:
            self._local.move_to_end(sid)
            ent.expires = time.monotonic() + self.s.ttl_session_s  # sliding
        return ent

    def _put_local(self, sid: str, ent: _Local) -> None:
        if self.s.local_sessions_max <= 0:
            return
        now = time.monotonic()
        ent.last_touch = now
        ent.expires = now + self.s.ttl_session_s
        self._local[sid] = ent
        self._local.move_to_end(sid)
        while len(self._local) > self.s.local_sessions_max:
            self._local.popitem(last=False)

    def _keys(self, sid: str) -> list[str]:
        return [session_key(sid, p) for p in ("papers", "rag", "history")]

    async def _maybe_touch(self, sid: str, ent: _Local | None) -> None:
        """Sliding Redis expiry without paying a command on every turn."""
        now = time.monotonic()
        if ent is None or now - ent.last_touch > self.s.ttl_session_s / 4:
            if ent:
                ent.last_touch = now
            await self.cache.touch(self._keys(sid), self.s.ttl_session_s)

    # ---------- papers ----------
    async def save_papers(self, sid: str, papers: list[Paper]) -> None:
        """A new fetch REPLACES papers + index. Chat history is kept (decision: easy to change)."""
        old = self._get_local(sid)
        ent = _Local(papers=list(papers), history=old.history if old else [])
        self._put_local(sid, ent)
        await self.cache.set_json(session_key(sid, "papers"), [p.to_dict() for p in papers],
                                  self.s.ttl_session_s)
        await self.cache.delete(session_key(sid, "rag"))
        await self._maybe_touch(sid, None)

    async def get_papers(self, sid: str) -> list[Paper] | None:
        ent = self._get_local(sid)
        if ent and ent.papers is not None:
            await self._maybe_touch(sid, ent)
            return ent.papers
        raw = await self.cache.get_json(session_key(sid, "papers"))
        if raw is None:
            return None  # expired / never fetched -> router should ask the student to fetch again
        papers = [Paper.from_dict(d) for d in raw]
        ent = ent or _Local()
        ent.papers = papers
        self._put_local(sid, ent)
        await self._maybe_touch(sid, ent)
        return papers

    # ---------- RAG index ----------
    async def get_index(self, sid: str, embedder: Embedder) -> HybridIndex | None:
        papers = await self.get_papers(sid)
        if not papers:
            return None
        fp = papers_fingerprint(papers)
        ent = self._get_local(sid)
        if ent and ent.index and ent.index.fingerprint == fp:
            return ent.index

        lock = self._build_locks.setdefault(sid, asyncio.Lock())
        try:
            async with lock:
                ent = self._get_local(sid)
                if ent and ent.index and ent.index.fingerprint == fp:
                    return ent.index  # built by a concurrent request while we waited
                idx = None
                blob = await self.cache.get_bytes(session_key(sid, "rag"))
                if blob:
                    try:
                        idx = HybridIndex.from_bytes(blob, embedder, expected_fingerprint=fp)
                    except IndexStale:
                        idx = None
                if idx is None:
                    # CPU-bound (embeds ~20 chunks): keep it off the event loop.
                    idx = await asyncio.to_thread(HybridIndex.build, papers, embedder, fingerprint=fp)
                    await self.cache.set_bytes(session_key(sid, "rag"), idx.to_bytes(),
                                               self.s.ttl_session_s)
                if ent is None:
                    ent = _Local(papers=papers)
                    self._put_local(sid, ent)
                ent.index = idx
                return idx
        finally:
            if not lock.locked():
                self._build_locks.pop(sid, None)

    # ---------- chat history ----------
    async def append_turn(self, sid: str, role: str, content: str) -> None:
        msg = {"role": role, "content": content}
        ent = self._get_local(sid)
        if ent is None:
            ent = _Local()
            self._put_local(sid, ent)
        ent.history.append(msg)
        del ent.history[: -self.s.history_max_msgs]
        await self.cache.list_push(session_key(sid, "history"), json.dumps(msg, ensure_ascii=False),
                                   maxlen=self.s.history_max_msgs, ttl=self.s.ttl_session_s)

    async def get_history(self, sid: str, last_n: int | None = None) -> list[dict]:
        raw = await self.cache.list_get(session_key(sid, "history")) if self.cache.enabled else None
        if raw is not None:
            hist = [json.loads(r) for r in raw]
        else:  # Redis down/disabled -> local copy
            ent = self._get_local(sid)
            hist = list(ent.history) if ent else []
        return hist[-last_n:] if last_n else hist
