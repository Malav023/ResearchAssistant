"""Quick manual check of the whole backend (no frontend needed).

    python scripts/smoke.py --register                      # creates smoke@example.com, runs a 3-message session
    python scripts/smoke.py --email a@b.c --password ...     # existing account
    python scripts/smoke.py --say "find papers on x" --say "what does paper 1 say?"

Set DEBUG=true in api/.env to also see per-stage timings and cache hits.
"""
import argparse
import json
import sys

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://localhost:8000")
ap.add_argument("--email", default="smoke@example.com")
ap.add_argument("--password", default="smoke-test-pw-123")
ap.add_argument("--register", action="store_true")
ap.add_argument("--say", action="append", help="message to send (repeatable)")
a = ap.parse_args()
msgs = a.say or ["find the latest papers on federated learning privacy",
                 "what does the second paper say about the method?",
                 "write a short report on these papers"]

with httpx.Client(base_url=a.base, timeout=120) as c:
    deep = c.get("/api/health/deep").json()
    print("health:", json.dumps({k: deep.get(k) for k in ("ok", "mongo", "redis")}),
          "| ml:", deep["ml"].get("backend"), "ok=" + str(deep["ml"].get("ok")),
          "| embedder:", deep["embedder"]["backend"], "| llm configured:", deep["llm"]["configured"])
    if a.register:
        r = c.post("/api/auth/register", json={"name": "Smoke", "email": a.email, "password": a.password})
        if r.status_code == 409:
            r = c.post("/api/auth/login", json={"email": a.email, "password": a.password})
    else:
        r = c.post("/api/auth/login", json={"email": a.email, "password": a.password})
    if r.status_code != 200:
        sys.exit(f"auth failed: {r.status_code} {r.text}")
    sid = None
    for m in msgs:
        r = c.post("/api/assistant/message", json={"message": m, **({"session_id": sid} if sid else {})})
        print(f"\n> {m}\n  HTTP {r.status_code}  trace={r.headers.get('x-trace-id')}")
        d = r.json()
        if r.status_code != 200:
            e = d.get("error", d)
            print(f"  ERROR {e.get('code')} @ {e.get('stage')}: {e.get('message')} (retryable={e.get('retryable')})")
            continue
        sid = d["session_id"]
        conf = f"{d['confidence']:.2f}" if d.get("confidence") is not None else "-"
        print(f"  intent={d['intent']} conf={conf} action={d['action']} topic={d.get('topic')!r} degraded={d['degraded']}")
        if d.get("papers"):
            for p in d["papers"]:
                print(f"    #{p['n']} {p['title'][:70]} ({p['year']})  abstract={'yes' if p['abstract'] else 'NO'}")
        print("  reply:", d["message"]["content"][:400].replace("\n", " "))
        if d.get("debug"):
            dbg = d["debug"]
            print("  stages_ms:", dbg["stages_ms"], "| total_ms:", dbg["total_ms"])
            print("  classifier:", {k: dbg.get("classifier", {}).get(k) for k in ("top", "p_top", "margin", "gated", "backend")})
            print("  extractor:", dbg.get("extractor"), "| sources:", dbg.get("sources"))
