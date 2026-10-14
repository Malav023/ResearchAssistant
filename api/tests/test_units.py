import httpx
import pytest

import config
from llm import prompts
from llm.client import LLMClient, LLMError
from ml import engine as eng
from ml.remote import RemoteEmbedder, RemoteMLRuntime
from ml.runtime import MLError
from ml.stub import heuristic_classify, heuristic_topic
from papers_service import reconstruct_abstract, search_papers, to_paper


# ---------------- config ----------------
def test_placeholder_keys_count_as_unset(monkeypatch):
    for v in ("ignoreWritten", "ignore", "change-this-to-x", "", "  "):
        monkeypatch.setenv("OPENALEX_API_KEY", v)
        assert config._key("OPENALEX_API_KEY") == ""
    monkeypatch.setenv("OPENALEX_API_KEY", "real-key-123")
    assert config._key("OPENALEX_API_KEY") == "real-key-123"


def test_config_defaults():
    c = config.get_config()
    assert c.ml_backend in ("stub", "remote", "local") and c.intent_min_conf == 0.6 and c.extract_on_chat is False


# ---------------- stub heuristics (queries from research_pipeline.py TEST_QUERIES) ----------------
@pytest.mark.parametrize("q,label", [
    ("pull me recent papers on gut microbiome and depression", "fetch_sources"),
    ("what did paper 4 say about sampling", "chat"),
    ("write a report on inflation", "create_report"),
    ("find newer work on federated learning privacy", "fetch_sources"),
    ("which paper had the largest sample size?", "chat"),
    ("i need a full review of neural plasticity for my seminar", "create_report"),
    ("how did the authors in paper 2 handle missing data", "chat"),
    ("Sociolinguistics", "unclear"),
])
def test_stub_labels(q, label):
    r = heuristic_classify(q)
    assert r.label == label and r.synthetic and abs(sum(r.probs.values()) - 1) < 0.01


def test_stub_topics():
    assert heuristic_topic("pull me recent papers on gut microbiome and depression") == "gut microbiome and depression"
    assert heuristic_topic("write a report on inflation") == "inflation"
    assert heuristic_topic("find newer work on federated learning privacy") == "federated learning privacy"


# ---------------- engine prompt must match training/Colab cells ----------------
def test_engine_prompt_matches_colab_script():
    p = eng.build_prompt("hello") + eng.PREFILL
    assert p.startswith("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n")
    assert "Request: hello<|im_end|>\n<|im_start|>assistant\n" + eng.PREFILL == p[p.index("Request: hello"):]
    assert len(eng.build_prompt("x" * 5000)) < 900            # input truncated: stays inside n_ctx=512


def test_engine_repo_names_follow_hf_user():
    e = eng.Engine({"HF_USER": "alice"})
    assert e.clf_repo == "alice/research-intent-clf-v1" and e.gguf_repo == "alice/research-extractor-qwen1.5b-gguf-v1"
    assert eng.Engine({"HF_CLASSIFIER_REPO": "x/y"}).clf_repo == "x/y"


# ---------------- Colab model server <-> RemoteMLRuntime contract ----------------
async def test_model_server_contract(monkeypatch):
    import ml_service
    monkeypatch.setattr(ml_service, "TOKEN", "secret")
    monkeypatch.setattr(ml_service.engine, "ready", True)
    monkeypatch.setattr(ml_service.engine, "classify", lambda t: ("chat", 0.91, {"chat": 0.91, "unclear": 0.09}))
    monkeypatch.setattr(ml_service.engine, "extract_topic", lambda t: ("federated learning", -0.3))
    tr = httpx.ASGITransport(app=ml_service.app)
    rt = RemoteMLRuntime("http://ml", "secret", transport=tr)
    c = await rt.classify("hi")
    assert (c.label, c.confidence, c.probs["unclear"]) == ("chat", 0.91, 0.09) and abs(c.margin - 0.82) < 1e-6
    x = await rt.extract("hi", "fetch_sources")
    assert x.topic == "federated learning" and x.avg_logprob == -0.3
    assert (await rt.health())["reachable"] is True
    bad = RemoteMLRuntime("http://ml", "wrong", transport=tr)
    with pytest.raises(MLError) as ei:
        await bad.classify("hi")
    assert ei.value.code == "ML_AUTH"


async def test_model_server_not_ready(monkeypatch):
    import ml_service
    monkeypatch.setattr(ml_service, "TOKEN", "t")
    monkeypatch.setattr(ml_service.engine, "ready", False)
    rt = RemoteMLRuntime("http://ml", "t", transport=httpx.ASGITransport(app=ml_service.app))
    with pytest.raises(MLError) as ei:
        await rt.classify("hi")
    assert ei.value.code == "ML_UPSTREAM"


@pytest.mark.parametrize("exc,code", [(httpx.ReadTimeout("t"), "ML_TIMEOUT"), (httpx.ConnectError("c"), "ML_UNREACHABLE")])
async def test_remote_error_mapping(exc, code):
    def boom(request):
        raise exc
    rt = RemoteMLRuntime("http://ml", "t", transport=httpx.MockTransport(boom))
    with pytest.raises(MLError) as ei:
        await rt.classify("x")
    assert ei.value.code == code
    assert (await rt.health())["reachable"] is False


def test_remote_embedder_checks_model_and_batches():
    seen = []

    def handler(request):
        import json
        body = json.loads(request.content)
        seen.append((body["kind"], len(body["texts"])))
        return httpx.Response(200, json={"model": "m", "vectors": [[1.0, 0.0]] * len(body["texts"])})

    emb = RemoteEmbedder("http://ml", "t", "m", transport=httpx.MockTransport(handler), batch_size=2)
    assert emb.encode_passages(["a", "b", "c"]).shape == (3, 2) and seen == [("passage", 2), ("passage", 1)]
    assert emb.encode_query("q").shape == (2,)
    wrong = RemoteEmbedder("http://ml", "t", "other", transport=httpx.MockTransport(handler))
    with pytest.raises(MLError) as ei:
        wrong.encode_query("q")
    assert ei.value.code == "EMBED_MODEL_MISMATCH"


# ---------------- LLM client ----------------
def _client(responses, **kw):
    it = iter(responses)
    calls = []

    def handler(request):
        import json
        calls.append(json.loads(request.content)["model"])
        r = next(it)
        return r(request) if callable(r) else r

    async def nosleep(_):
        return None

    return LLMClient(base_url="http://llm", api_key="k", transport=httpx.MockTransport(handler), sleep=nosleep, **kw), calls


def _ok(text="hi", model_usage=True):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                                     "usage": {"prompt_tokens": 7, "completion_tokens": 3}})


async def test_llm_success_and_think_stripping():
    c, calls = _client([_ok("<think>hmm</think>Answer [Paper 1]")])
    r = await c.chat(model="m", messages=[{"role": "user", "content": "x"}], max_tokens=10)
    assert r.text == "Answer [Paper 1]" and r.prompt_tokens == 7 and r.retries == 0 and calls == ["m"]


async def test_llm_retries_then_succeeds_and_then_falls_back():
    c, calls = _client([httpx.Response(429, headers={"retry-after": "1"}), _ok("second try")])
    assert (await c.chat(model="m", messages=[], max_tokens=5)).retries == 1 and calls == ["m", "m"]
    c, calls = _client([httpx.Response(500), httpx.Response(503), _ok("from fallback")])
    r = await c.chat(model="m", messages=[], max_tokens=5, fallback_model="fb")
    assert r.used_fallback and r.model == "fb" and calls == ["m", "m", "fb"]


async def test_llm_error_mapping():
    c, _ = _client([httpx.Response(429), httpx.Response(429)])
    with pytest.raises(LLMError) as e:
        await c.chat(model="m", messages=[], max_tokens=5)
    assert e.value.code == "LLM_RATE_LIMITED" and e.value.status == 429 and e.value.retryable
    c, calls = _client([httpx.Response(401)])
    with pytest.raises(LLMError) as e:
        await c.chat(model="m", messages=[], max_tokens=5, fallback_model="fb")
    assert e.value.code == "LLM_UNAVAILABLE" and not e.value.retryable and calls == ["m"]   # fatal: no retry
    c, calls = _client([httpx.Response(404), _ok("fb ok")])
    assert (await c.chat(model="bad", messages=[], max_tokens=5, fallback_model="fb")).model == "fb"
    assert calls == ["bad", "fb"]                                                           # 404 skips same-model retry

    def timeout(request):
        raise httpx.ReadTimeout("slow")
    c, _ = _client([timeout, timeout])
    with pytest.raises(LLMError) as e:
        await c.chat(model="m", messages=[], max_tokens=5)
    assert e.value.code == "LLM_TIMEOUT" and e.value.status == 504


async def test_llm_not_configured_and_malformed():
    c = LLMClient(base_url="http://llm", api_key="")
    with pytest.raises(LLMError) as e:
        await c.chat(model="m", messages=[], max_tokens=5)
    assert e.value.status == 503
    c, _ = _client([httpx.Response(200, json={"oops": 1}), httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})])
    with pytest.raises(LLMError):
        await c.chat(model="m", messages=[], max_tokens=5)


# ---------------- papers service ----------------
def test_reconstruct_abstract_and_to_paper():
    assert reconstruct_abstract({"world": [1], "hello": [0, 2]}) == "hello world hello"
    assert reconstruct_abstract(None) == ""
    p = to_paper({"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/x", "title": "A <i>Great</i> Paper",
                  "authorships": [{"author": {"display_name": "Jane Doe"}}], "publication_year": 2023,
                  "cited_by_count": 9, "concepts": [{"display_name": "NLP"}],
                  "abstract_inverted_index": {"Short": [0], "abstract.": [1]}})
    assert (p.title, p.abstract, p.doi, p.first_author, p.top_concept, p.citations) == \
           ("A Great Paper", "Short abstract.", "10.1/x", "Jane Doe", "NLP", 9)


async def test_search_papers_ranks_by_citations_and_uses_s2_fallback():
    works = [{"id": f"W{i}", "doi": f"https://doi.org/10.1/{i}", "title": f"T{i}", "cited_by_count": i,
              "abstract_inverted_index": ({"has": [0], "abstract": [1]} if i % 2 else None)} for i in range(1, 9)]

    def handler(request):
        if "openalex" in str(request.url):
            assert "abstract_inverted_index" in request.url.params["select"]
            return httpx.Response(200, json={"results": works})
        import json
        ids = json.loads(request.content)["ids"]
        return httpx.Response(200, json=[{"paperId": "p", "abstract": "from s2"} for _ in ids])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        ps = await search_papers("x", limit=5, client=c)
    assert [p.title for p in ps] == ["T8", "T7", "T6", "T5", "T4"]
    assert ps[0].abstract == "from s2" and ps[1].abstract == "has abstract"


# ---------------- prompts ----------------
def test_citation_validation():
    out, cited, bad = prompts.validate_citations("A [Paper 1]. B [Paper 2, Paper 9]. C [Paper 8]. D [see Paper]", 3)
    assert cited == [1, 2] and bad == [8, 9] and "[Paper 8]" not in out and "[Paper 2]" in out and "[see Paper]" in out


def test_references_escape_markdown():
    from ra_core.rag import Paper
    md = prompts.references_md([Paper("1", "A *bold* [title]", "x", year=2020, first_author="Jo", doi="10.1/z")])
    assert "\\*bold\\*" in md and "https://doi.org/10.1/z" in md and "Jo et al. (2020)" in md
