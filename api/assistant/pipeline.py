"""classify -> (clarify | extract -> fetch_sources | create_report | chat) -> persist -> respond."""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from llm import prompts
from llm.client import LLMError
from ml.runtime import INTENTS, ClassifyResult, MLError
from ml.stub import heuristic_classify, heuristic_topic
from ra_core.cache import extraction_key, sources_key
from ra_core.rag import Hit, Paper, render_context
from ra_core.rag.chunking import chunk_paper
from ra_core.rag.paper_ref import extract_paper_refs

from .errors import AppError
from .schemas import AssistantMessage, Citation, MessageRequest, MessageResponse, PaperView, Suggestion
from .services import Services
from .tracing import Trace
from .routing_rules import apply_rules

log = logging.getLogger("assistant")

_EXISTING = re.compile(r"\b(these|those|the|my|above|current|fetched|same)\s+(papers|articles|sources)\b", re.I)
NO_ABSTRACTS = ("The papers I found don't include abstracts I can read, so I can't answer questions about "
                "them yet. Try searching again with different keywords.")


@dataclass
class Ctx:
    svc: Services
    uid: str
    sid: str
    text: str
    trace: Trace
    intent: str = "unclear"
    confidence: float | None = None
    topic: str | None = None
    brief: str | None = None
    use_session_papers: bool = False

    @property
    def rkey(self) -> str:                      # user-namespaced: one user can never address another's session
        return f"{self.uid}:{self.sid}"


# ------------------------------------------------------------------ helpers
def _views(papers: list[Paper]) -> list[PaperView]:
    return [PaperView(n=i, id=p.id, title=p.title, abstract=p.abstract, year=p.year, citations=p.citations,
                      first_author=p.first_author, top_concept=p.top_concept, doi=p.doi, url=p.url)
            for i, p in enumerate(papers, 1)]


def _cits(nums: list[int], papers: list[Paper]) -> list[Citation]:
    return [Citation(paper=n, title=papers[n - 1].title) for n in nums if 1 <= n <= len(papers)]


def _all_hits(papers: list[Paper]) -> list[Hit]:
    return [Hit(c, 0.0, None, None) for i, p in enumerate(papers, 1) for c in chunk_paper(p, i)]


def _valid_topic(t: str) -> bool:
    return 2 <= len(t) <= 200 and "{" not in t and "\n" not in t


async def _resolve_session(svc: Services, uid: str, sid_in: str | None, text: str) -> tuple[str, bool]:
    if sid_in:
        doc = await svc.repo.get_session(uid, sid_in)
        if not doc:
            raise AppError("SESSION_NOT_FOUND", "That conversation no longer exists.", stage="session", status=404)
        return str(doc["_id"]), False
    return await svc.repo.create_session(uid, text[:80]), True


# ------------------------------------------------------------------ stages
async def _classify(ctx: Ctx) -> ClassifyResult:
    svc, tr = ctx.svc, ctx.trace
    try:
        with tr.stage("classifier"):
            res = await asyncio.wait_for(svc.ml.classify(ctx.text), timeout=svc.cfg.classifier_timeout_s)
        if res.label not in INTENTS:
            raise MLError("ML_BAD_RESPONSE", f"unknown label {res.label!r}")
    except Exception as e:
        log.warning("classifier failed, using keyword fallback: %s: %s", type(e).__name__, e)
        tr.degrade("classifier")
        tr.put("classifier", error=f"{type(e).__name__}: {e}")
        res = heuristic_classify(ctx.text)
    gated = res.label != "unclear" and res.confidence < svc.cfg.intent_min_conf
    tr.put("classifier", probs=res.probs, top=res.label, p_top=res.confidence, margin=res.margin, gated=gated,
           version=res.version, synthetic=res.synthetic, backend=svc.ml.name)
    ctx.confidence = res.confidence
    ctx.intent = "unclear" if gated else res.label
    tr.set(intent=ctx.intent)
    return res


async def _extract(ctx: Ctx) -> None:
    svc, tr, cfg = ctx.svc, ctx.trace, ctx.svc.cfg
    mode = "topic_only"
    key = extraction_key(ctx.text, ctx.intent, f"{cfg.ra.extractor_version}-{mode}")

    async def compute():
        r = await asyncio.wait_for(svc.ml.extract(ctx.text, ctx.intent), timeout=cfg.extractor_timeout_s)
        topic = (r.topic or "").strip()
        if not _valid_topic(topic):
            raise ValueError(f"invalid topic {topic!r}")
        return {"topic": topic, "brief": r.brief, "avg_logprob": r.avg_logprob, "version": r.version}

    with tr.stage("extractor"):
        try:
            val, hit = await svc.cache.get_or_compute(key, cfg.ra.ttl_extract_s, compute)
            ctx.topic, ctx.brief = val["topic"], val.get("brief")
            tr.put("extractor", ran=True, mode=mode, cache_hit=hit, fallback=False,
                   avg_logprob=val.get("avg_logprob"), version=val.get("version"))
        except Exception as e:
            log.warning("extractor failed, using raw query: %s: %s", type(e).__name__, e)
            tr.degrade("extractor")
            ctx.topic, ctx.brief = heuristic_topic(ctx.text), None
            tr.put("extractor", ran=True, mode=mode, cache_hit=False, fallback=True,
                   error=f"{type(e).__name__}: {e}")


async def _sources(ctx: Ctx) -> list[Paper]:
    svc, tr, cfg = ctx.svc, ctx.trace, ctx.svc.cfg
    topic = ctx.topic or heuristic_topic(ctx.text)
    key = sources_key(topic, limit=cfg.sources_limit)

    async def compute():
        papers = await svc.fetch_papers(topic, cfg.sources_limit)
        return [p.to_dict() for p in papers] or None        # never cache an empty result

    with tr.stage("sources"):
        try:
            raw, hit = await svc.cache.get_or_compute(key, cfg.ra.ttl_sources_s, compute)
        except Exception as e:
            tr.put("sources", error=f"{type(e).__name__}: {e}", topic=topic)
            raise AppError("SOURCES_UPSTREAM", "I couldn't reach the paper index (OpenAlex). Please try again in a moment.",
                           stage="sources", status=502, retryable=True) from e
    papers = [Paper.from_dict(d) for d in (raw or [])]
    tr.put("sources", cache_hit=hit, n=len(papers), topic=topic,
           n_with_abstract=sum(bool(p.abstract) for p in papers))
    return papers


async def _save_papers(ctx: Ctx, papers: list[Paper]) -> None:
    try:
        await ctx.svc.sessions.save_papers(ctx.rkey, papers)
        await ctx.svc.repo.save_papers(ctx.sid, [p.to_dict() for p in papers])
    except Exception:
        log.exception("could not persist papers")
        ctx.trace.degrade("persistence")


async def _session_papers(ctx: Ctx) -> list[Paper] | None:
    papers = await ctx.svc.sessions.get_papers(ctx.rkey)
    if papers:
        return papers
    doc = await ctx.svc.repo.get_session(ctx.uid, ctx.sid)          # rehydrate after Redis expiry / restart
    if doc and doc.get("papers"):
        papers = [Paper.from_dict(d) for d in doc["papers"]]
        await ctx.svc.sessions.save_papers(ctx.rkey, papers)
        return papers
    return None


async def _history(ctx: Ctx) -> list[dict]:
    n = ctx.svc.cfg.history_turns
    h = await ctx.svc.sessions.get_history(ctx.rkey, last_n=n)
    if not h:
        h = [{"role": m["role"], "content": m["content"]} for m in await ctx.svc.repo.get_messages(ctx.sid, n, last=True)]
    return [{"role": m["role"], "content": m["content"][:1500]} for m in h if m.get("role") in ("user", "assistant")]


async def _retrieve(ctx: Ctx, papers: list[Paper], base: list[Hit]) -> list[Hit]:
    svc, tr, cfg = ctx.svc, ctx.trace, ctx.svc.cfg
    mode, hits = cfg.rag_mode, base
    with tr.stage("retrieval"):
        if mode == "hybrid":
            try:
                idx = await svc.sessions.get_index(ctx.rkey, svc.embedder)
                if idx is None:
                    raise RuntimeError("no index")
                hits = await asyncio.to_thread(idx.search, ctx.text, k=cfg.ra.rag_top_k, extra_query=ctx.topic)
            except Exception as e:
                log.warning("hybrid retrieval failed, stuffing all chunks: %s: %s", type(e).__name__, e)
                tr.degrade("embedder")
                tr.put("retrieval", error=f"{type(e).__name__}: {e}")
                mode, hits = "stuff", base
    tr.put("retrieval", mode=mode, refs_detected=sorted(extract_paper_refs(ctx.text, len(papers))), n_hits=len(hits),
           top=[{"paper": h.chunk.paper_idx, "section": h.chunk.section, "score": round(h.score, 5),
                 "dense_rank": h.dense_rank, "sparse_rank": h.sparse_rank} for h in hits[:5]])
    return hits


async def _llm(ctx: Ctx, kind: str, messages: list[dict]):
    svc, tr, cfg = ctx.svc, ctx.trace, ctx.svc.cfg
    model = cfg.llm_chat_model if kind == "chat" else cfg.llm_report_model
    max_tokens = cfg.llm_max_tokens_chat if kind == "chat" else cfg.llm_max_tokens_report
    try:
        with tr.stage("llm"):
            res = await svc.llm.chat(model=model, messages=messages, max_tokens=max_tokens,
                                     fallback_model=cfg.llm_fallback_model)
    except LLMError as e:
        tr.put("llm", model=model, error=e.code)
        raise AppError(e.code, e.message, stage="llm", status=e.status, retryable=e.retryable) from e
    tr.put("llm", model=res.model, prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
           finish_reason=res.finish_reason, retries=res.retries, used_fallback=res.used_fallback)
    if res.used_fallback:
        tr.degrade("llm_fallback")
    return res


def _check_citations(ctx: Ctx, text: str, n: int) -> tuple[str, list[int]]:
    body, cited, invalid = prompts.validate_citations(text, n)
    total = len(cited) + len(invalid)
    ctx.trace.put("llm", citations_found=cited, citations_invalid=invalid,
                  citation_valid=round(len(cited) / total, 2) if total else None)
    if invalid:
        ctx.trace.degrade("citations")
    return body, cited


async def _finish(ctx: Ctx, action: str, content: str, *, papers: list[Paper] | None = None,
                  citations: list[Citation] | None = None, suggestions: list[dict] | None = None) -> MessageResponse:
    svc, tr = ctx.svc, ctx.trace
    cits = citations or []
    tr.set(action=action, topic=ctx.topic)
    now = datetime.now(timezone.utc)
    try:
        await svc.repo.add_messages(ctx.sid, ctx.uid, [
            {"role": "user", "content": ctx.text, "trace_id": tr.trace_id, "ts": now},
            {"role": "assistant", "content": content, "intent": ctx.intent, "action": action,
             "citations": [c.model_dump() for c in cits], "trace_id": tr.trace_id,
             "ts": now + timedelta(milliseconds=1)},
        ])
        await svc.repo.touch(ctx.sid, ctx.intent)
    except Exception:
        log.exception("could not persist messages")
        tr.degrade("persistence")
    await svc.sessions.append_turn(ctx.rkey, "user", ctx.text)           # fail-open
    await svc.sessions.append_turn(ctx.rkey, "assistant", content)
    return MessageResponse(
        session_id=ctx.sid, trace_id=tr.trace_id, intent=ctx.intent, confidence=ctx.confidence, action=action,
        topic=ctx.topic, brief=ctx.brief, message=AssistantMessage(content=content, citations=cits),
        papers=_views(papers) if papers is not None else None,
        suggestions=[Suggestion(**s) for s in suggestions] if suggestions else None,
        degraded=list(tr.degraded),
    )


# ------------------------------------------------------------------ flows
async def _fetch_flow(ctx: Ctx) -> MessageResponse:
    papers = await _sources(ctx)
    topic = ctx.topic or ctx.text
    if not papers:
        return await _finish(ctx, "papers", f'I couldn\'t find papers for "{topic}". Try different keywords.', papers=[])
    await _save_papers(ctx, papers)
    return await _finish(
        ctx, "papers",
        f'Here are the top {len(papers)} papers for "{topic}". Ask me about any of them, for example '
        f'"what does paper 2 say about the method?", or ask for a short report.',
        papers=papers)


async def _chat_flow(ctx: Ctx) -> MessageResponse:
    papers = await _session_papers(ctx)
    if not papers:
        return await _finish(ctx, "needs_papers", prompts.NEEDS_PAPERS)
    base = _all_hits(papers)
    if not base:
        return await _finish(ctx, "answer", NO_ABSTRACTS, papers=papers)

    hits = await _retrieve(ctx, papers, base)
    refs = extract_paper_refs(ctx.text, len(papers))
    have = {h.chunk.paper_idx for h in hits}
    hits = hits + [h for h in base if h.chunk.paper_idx in refs and h.chunk.paper_idx not in have]
    context = prompts.ensure_ref_context(render_context(hits), papers, sorted(refs))
    msgs = prompts.chat_messages(await _history(ctx), context, ctx.text, papers)

    res = await _llm(ctx, "chat", msgs)
    body, cited = _check_citations(ctx, res.text, len(papers))
    return await _finish(ctx, "answer", body, citations=_cits(cited, papers))


async def _report_flow(ctx: Ctx) -> MessageResponse:
    use_existing = ctx.use_session_papers or bool(extract_paper_refs(ctx.text, 8)) or bool(_EXISTING.search(ctx.text))
    papers = await _session_papers(ctx) if use_existing else None
    if not papers:
        papers = await _sources(ctx)
        if not papers:
            return await _finish(ctx, "papers", f'I couldn\'t find sources for "{ctx.topic or ctx.text}", '
                                                 f'so I can\'t write a report. Try different keywords.', papers=[])
        await _save_papers(ctx, papers)
    base = _all_hits(papers)
    if not base:
        return await _finish(ctx, "papers", NO_ABSTRACTS, papers=papers)
    msgs = prompts.report_messages(ctx.text, ctx.topic or ctx.text, render_context(base), papers)
    try:
        res = await _llm(ctx, "report", msgs)
    except AppError as e:                         # we still have useful sources: show them instead of failing
        ctx.trace.degrade("llm")
        ctx.trace.fail(e)
        return await _finish(ctx, "papers", f"I found {len(papers)} sources but couldn't draft the report right now "
                                            f"({e.message}) Try again in a moment.", papers=papers)
    body, cited = _check_citations(ctx, res.text, len(papers))
    body = body.rstrip() + "\n\n" + prompts.references_md(papers)
    return await _finish(ctx, "report", body, papers=papers, citations=_cits(cited, papers))


# ------------------------------------------------------------------ entry point
async def handle_message(svc: Services, user: dict, req: MessageRequest, trace: Trace) -> MessageResponse:
    cfg, uid, text = svc.cfg, str(user["_id"]), req.message.strip()
    if not text:
        raise AppError("VALIDATION", "Message is empty.", stage="validate", status=422)
    if not svc.limiter.allow(uid):
        raise AppError("RATE_LIMITED", "You're sending messages too quickly. Please wait a few seconds.",
                       stage="ratelimit", status=429, retryable=True)
    if cfg.trace_store_text:
        trace.set(text=text[:500])

    with trace.stage("session"):
        sid, is_new = await _resolve_session(svc, uid, req.session_id, text)
    trace.set(session_id=sid)
    ctx = Ctx(svc, uid, sid, text, trace)
    try:
        await _classify(ctx)

        has_papers = bool(await _session_papers(ctx))       # cheap: Redis, else Mongo
        ctx.intent, ctx.use_session_papers, rule = apply_rules(text, ctx.intent, has_papers)
        if rule:
            trace.put("router", override=rule)
            trace.set(intent=ctx.intent)

        if ctx.intent == "unclear":
            return await _finish(ctx, "clarify", prompts.CLARIFY, suggestions=prompts.SUGGESTIONS)
        if ctx.intent == "chat" and not cfg.extract_on_chat:
            trace.put("extractor", ran=False, reason="skipped_for_chat")
        elif ctx.use_session_papers:
            trace.put("extractor", ran=False, reason="session_papers")
        else:
            await _extract(ctx)
        if ctx.intent == "fetch_sources":
            return await _fetch_flow(ctx)
        if ctx.intent == "create_report":
            return await _report_flow(ctx)
        return await _chat_flow(ctx)
    except BaseException:
        if is_new:                                  # don't leave an empty conversation in the sidebar
            try:
                await svc.repo.delete_if_empty(sid)
            except Exception:
                log.exception("cleanup of empty session failed")
        raise
