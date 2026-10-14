# Running the backend (M0–M4)

## 0. Apply the update
1. `git commit` (or copy) your current `project/` first.
2. Extract `api_m0_m4.zip` into `project/` so you get `project/api/...`. It **replaces**: `main.py`, `auth_utils.py`,
   `auth_routes.py`, `db.py`, `papers_routes.py`, `requirements*.txt`, `.env.example`. It **adds**: `config.py`, `ra_core/`,
   `ml/`, `llm/`, `assistant/`, `papers_service.py`, `ml_service.py`, `tests/`, `scripts/`, `pytest.ini`.
   It does **not** touch `graph_builder.py`, `models.py`, `sessions_routes.py` (frontend keeps working).
3. Delete the old `project/ra/` folder (everything in it now lives in `api/ra_core` + `api/tests`).
4. In `api/.env`: set **empty** `OPENALEX_API_KEY=` / `S2_API_KEY=` (the placeholder text `ignoreWritten` was being sent as a real
   key once `.env` is loaded correctly), **rotate `JWT_SECRET`**, then add the new variables from `.env.example`.

## 1. Install + test (no Mongo / Redis / network / models needed)
```powershell
cd project\api
python -m venv .venv ; .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q          # expect: 75 passed
```
`requirements-ml.txt` is NOT needed on this machine when the models run on Colab.

## 2. Start the models on Colab
Upload `ml_server_bundle.zip` to Drive (`research_assistant-1/`), open `research_assistant_model_server.ipynb`, edit
`HF_USER` in cell 3, run cells 1-6. Paste the three printed lines (`ML_BACKEND`, `ML_REMOTE_URL`, `ML_SERVICE_TOKEN`) into `api/.env`.
The quick-tunnel URL changes every Colab run, so update `ML_REMOTE_URL` and restart the API each time.

## 3. Chat/report LLM
```
LLM_BASE_URL=https://api.groq.com/openai/v1     # any OpenAI-compatible endpoint
LLM_API_KEY=...
LLM_CHAT_MODEL=<model id from the provider's CURRENT model list>
LLM_REPORT_MODEL=                               # empty = same as chat
LLM_FALLBACK_MODEL=                             # optional, used after retries fail
```

## 4. Run + smoke test
```powershell
uvicorn main:app --reload --port 8000           # fine to use --reload: no models live in this process
python scripts\smoke.py --register              # needs Mongo running; prints intent, confidence, timings
```
Set `DEBUG=true` to get the per-request trace in every response and the `/api/debug/traces` endpoints.
No Redis? Leave `REDIS_URL` empty: everything works, just uncached.

## 5. Debugging cheatsheet
| Symptom | Look at |
|---|---|
| Every reply has `degraded:["classifier"]` | `/api/health/deep` → `ml`. Colab stopped, URL changed, or token mismatch (`ML_AUTH`) |
| `degraded:["extractor"]` | trace `extractor.error`; `EXTRACTOR_TIMEOUT_S` too low for a cold Colab CPU |
| `degraded:["embedder"]` | `EMBED_MODEL` differs between `.env` and the notebook (`EMBED_MODEL_MISMATCH`) |
| Intent looks wrong | `GET /api/debug/traces?max_conf=0.7` and `POST /api/assistant/feedback {trace_id, correct_intent}` (training data for v2) |
| `LLM_UNAVAILABLE` 503 | `LLM_API_KEY` / `LLM_CHAT_MODEL` not set |
| `LLM_RATE_LIMITED` | provider free-tier limit; set `LLM_FALLBACK_MODEL` |
| Answers say "abstracts only" | OpenAlex returned no abstract; trace `sources.n_with_abstract` |
| Papers numbered differently than in the UI | UI must show `papers[].n`; `n` is what "paper 2" resolves to |

Every response carries `X-Trace-Id`; the same id is in the JSON logs and in Mongo `assistant_traces` (30-day TTL).
