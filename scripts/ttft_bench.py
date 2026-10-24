#!/usr/bin/env python3
"""Streamed TTFT + prefill bench against llama-server /v1/chat/completions.

Times from request send (not first SSE chunk), unique nonce per request so
the prompt cache never reuses a prefix. Reports llama-server's own
timings.prompt_per_second as the prefill figure.

usage: ttft_bench.py [URL] [LEN [LEN...]]   # lengths in approximate tokens
"""
import json, sys, time, urllib.request, random, string

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
LENS = [int(a) for a in sys.argv[2:]] or [512, 2048, 8192]
WORDS = ["alpha", "bridge", "copper", "delta", "ember", "forest", "glacier", "harbor", "island", "juniper"]

def ttft(n):
    nonce = "".join(random.choices(string.ascii_letters, k=12))
    body = json.dumps({
        "messages": [{"role": "user", "content":
            f"[nonce {nonce}] Ignore the following text, it is filler. Text: " +
            " ".join(WORDS[i % len(WORDS)] for i in range(n)) +
            " Now summarize what I asked in one short sentence: say OK."}],
        "max_tokens": 64, "temperature": 0, "stream": True,
        "cache_prompt": False, "timings_per_token": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time(); first = None; last = None; toks = 0; timings = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: "): continue
            d = line[6:]
            if d == "[DONE]": break
            j = json.loads(d)
            if "timings" in j: timings = j["timings"]
            c = j["choices"][0].get("delta", {}).get("content", "") if j.get("choices") else ""
            if c:
                toks += 1
                if first is None: first = time.time()
                last = time.time()
    if not first or toks <= 1:
        print(f"{n}toks ERR"); return
    pp = timings.get("prompt_per_second") if timings else None
    tg = timings.get("predicted_per_second") if timings else None
    print(f"{n}toks ttft={first-t0:.2f}s prefill={pp and round(pp)}t/s "
          f"decode={tg and round(tg,1)}t/s")

for n in LENS:
    ttft(n)
