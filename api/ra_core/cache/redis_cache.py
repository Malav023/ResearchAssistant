from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from typing import Any, Awaitable, Callable, Iterable

from redis.exceptions import RedisError

log = logging.getLogger(__name__)
_FAIL = (RedisError, OSError, asyncio.TimeoutError)


class RedisCache:
    """Async Redis wrapper that FAILS OPEN.

    - Any Redis error/timeout -> returns the 'miss' default, never raises into the request path.
    - Circuit breaker: after N consecutive failures, skip Redis for a cooldown so a dead cache
      doesn't add its timeout to every request.
    - url="" (or no client) -> disabled; every call is a miss.
    """

    def __init__(self, url: str = "", *, timeout_s: float = 0.3, breaker_threshold: int = 3,
                 breaker_cooldown_s: float = 30.0, client: Any = None):
        self._client = client
        if client is None and url:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(
                url, socket_timeout=timeout_s, socket_connect_timeout=timeout_s,
                health_check_interval=30, max_connections=10,  # free tiers cap connections
            )
        self.timeout_s = timeout_s
        self._thr, self._cool = breaker_threshold, breaker_cooldown_s
        self._fails, self._open_until = 0, 0.0
        self._inflight: dict[str, asyncio.Future] = {}
        self.stats: Counter = Counter()  # hit / miss / set / error / skipped -> feeds Phase 10 profiling

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def _run(self, fn: Callable[[], Awaitable[Any]], default: Any = None) -> Any:
        if not self.enabled or time.monotonic() < self._open_until:
            self.stats["skipped"] += 1
            return default
        try:
            out = await asyncio.wait_for(fn(), timeout=self.timeout_s * 3)
            self._fails = 0
            return out
        except _FAIL as e:
            self.stats["error"] += 1
            self._fails += 1
            if self._fails >= self._thr:
                self._open_until = time.monotonic() + self._cool
                log.warning("Redis circuit open for %.0fs (%s)", self._cool, e)
            return default

    async def ping(self) -> bool:
        return bool(await self._run(lambda: self._client.ping(), False))

    # ---------- primitives ----------
    async def get_bytes(self, key: str) -> bytes | None:
        v = await self._run(lambda: self._client.get(key))
        self.stats["hit" if v is not None else "miss"] += 1
        return v

    async def set_bytes(self, key: str, value: bytes, ttl: int) -> bool:
        ok = await self._run(lambda: self._client.set(key, value, ex=ttl), False)
        self.stats["set"] += bool(ok)
        return bool(ok)

    async def get_json(self, key: str) -> Any | None:
        raw = await self.get_bytes(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None  # corrupt entry == miss

    async def set_json(self, key: str, value: Any, ttl: int) -> bool:
        return await self.set_bytes(key, json.dumps(value, ensure_ascii=False).encode(), ttl)

    async def delete(self, *keys: str) -> None:
        if keys:
            await self._run(lambda: self._client.delete(*keys))

    async def touch(self, keys: Iterable[str], ttl: int) -> None:
        keys = list(keys)

        async def _go():
            pipe = self._client.pipeline(transaction=False)
            for k in keys:
                pipe.expire(k, ttl)
            return await pipe.execute()
        await self._run(_go)

    async def list_push(self, key: str, value: str, *, maxlen: int, ttl: int) -> bool:
        async def _go():
            pipe = self._client.pipeline(transaction=False)
            pipe.rpush(key, value).ltrim(key, -maxlen, -1).expire(key, ttl)
            return await pipe.execute()
        return await self._run(_go, None) is not None

    async def list_get(self, key: str) -> list[str] | None:
        """None = Redis unavailable (caller should fall back); [] = genuinely empty."""
        raw = await self._run(lambda: self._client.lrange(key, 0, -1), None)
        if raw is None:
            return None
        return [r.decode() if isinstance(r, bytes) else r for r in raw]

    # ---------- cache-aside with single-flight ----------
    async def get_or_compute(self, key: str, ttl: int, compute: Callable[[], Awaitable[Any]]
                             ) -> tuple[Any, bool]:
        """Returns (value, from_cache). Concurrent identical misses in this process share ONE
        compute() call (deadline-day stampede protection). None results are not cached."""
        cached = await self.get_json(key)
        if cached is not None:
            return cached, True

        pending = self._inflight.get(key)
        if pending is not None:
            try:
                return await asyncio.shield(pending), False
            except asyncio.CancelledError:
                if pending.cancelled():           # leader was cancelled, not us
                    return await compute(), False
                raise

        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._inflight[key] = fut
        try:
            val = await compute()
            if val is not None:
                await self.set_json(key, val, ttl)
            fut.set_result(val)
            return val, False
        except asyncio.CancelledError:
            fut.cancel()
            raise
        except Exception as e:
            fut.set_exception(e)
            fut.exception()  # mark retrieved; avoids "never retrieved" warnings
            raise
        finally:
            self._inflight.pop(key, None)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
