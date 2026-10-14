"""In-process runtime (needs requirements-ml.txt on THIS machine). Same Engine as the Colab server."""
from __future__ import annotations

import asyncio

from .runtime import ClassifyResult, ExtractResult, MLError


class LocalMLRuntime:
    name = "local"

    def __init__(self, engine=None):
        from .engine import Engine
        self.engine = engine or Engine()
        self._lock = asyncio.Lock()

    async def _ensure(self) -> None:
        if not self.engine.ready:
            async with self._lock:
                if not self.engine.ready:
                    try:
                        await asyncio.to_thread(self.engine.load)
                    except Exception as e:
                        raise MLError("ML_NOT_READY", f"local models failed to load: {e}", retryable=False) from e

    async def classify(self, text: str) -> ClassifyResult:
        await self._ensure()
        label, conf, probs = await asyncio.to_thread(self.engine.classify, text)
        return ClassifyResult(label, conf, probs, version=self.engine.CLF_VERSION)

    async def extract(self, text: str, intent: str) -> ExtractResult:
        await self._ensure()
        topic, lp = await asyncio.to_thread(self.engine.extract_topic, text)
        return ExtractResult(topic=topic, avg_logprob=lp, version=self.engine.EXT_VERSION)

    async def health(self) -> dict:
        return {"ok": self.engine.ready, "backend": "local", **self.engine.info}

    async def warmup(self) -> None:
        await self._ensure()
        await asyncio.to_thread(self.engine.warmup)

    async def aclose(self) -> None:
        return None
