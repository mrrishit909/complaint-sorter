"""A small OpenAI client (standard library only) that keeps a running cost tally (copied from the sec-10k-rag project).

The key is read from .env (OPENAI_API_KEY=...), which git ignores and nothing ever prints.
Prices are $ per 1M tokens from developers.openai.com/api/docs/pricing, checked 2026-10-01.
"""
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
CHAT_MODEL = "gpt-5.4-mini"
PRICES = {"gpt-5.4-mini": {"in": 0.75, "out": 4.50}}
CTX = ssl.create_default_context(cafile="/etc/ssl/cert.pem" if Path("/etc/ssl/cert.pem").exists() else None)
spent = {"usd": 0.0, "in_tokens": 0, "out_tokens": 0, "calls": 0}


def _key():
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    for line in (HERE / ".env").read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("OPENAI_API_KEY missing: put it in .env")


def _post(path, body, tries=4):
    req = urllib.request.Request(f"https://api.openai.com/v1/{path}", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"})
    for attempt in range(tries):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=180, context=CTX).read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < tries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            raise SystemExit(f"OpenAI API error {e.code}: {e.read()[:300].decode(errors='replace')}")


def _bill(model, usage):
    i, o = usage.get("prompt_tokens", usage.get("input_tokens", 0)), usage.get("completion_tokens", usage.get("output_tokens", 0))
    spent["in_tokens"] += i; spent["out_tokens"] += o; spent["calls"] += 1
    spent["usd"] += (i * PRICES[model]["in"] + o * PRICES[model]["out"]) / 1e6


def chat_json(system, user, schema, name="answer"):
    """One chat call that must return JSON matching `schema` (OpenAI structured outputs)."""
    r = _post("chat/completions", {
        "model": CHAT_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
        "reasoning_effort": "low",
    })
    _bill(CHAT_MODEL, r["usage"])
    return json.loads(r["choices"][0]["message"]["content"])
