#!/usr/bin/env python3
"""Decode-speed bench against a running llama-server (/completion timings).

usage: bolt_bench.py URL LABEL [N_PREDICT]
Fixed prompts, temp 0, one warmup, prints JSON lines + a summary row.
"""
import json, sys, time, urllib.request

URL, LABEL = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
PROMPTS = {
    "code": "Write a Python function that parses an ISO-8601 date string and returns a datetime, with error handling and docstring.",
    "math": "Solve step by step: a train travels 180 km in 2.5 hours, then 120 km in 1.5 hours. What is its average speed?",
    "prose": "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "extract": "Extract every person, date and amount as JSON from: 'On March 3, 2026 Alice Chen paid Bob Ortiz $1,250 for the design work; on April 9 Bob refunded $300.'",
}

def chat(content):
    body = json.dumps({
        "messages": [{"role": "user", "content": content}],
        "max_tokens": N, "temperature": 0, "seed": 1,
        "chat_template_kwargs": {"enable_thinking": False},
        "timings_per_token": False,
    }).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return d, time.time() - t0

chat("Say hi.")  # warmup
rows = []
for name, p in PROMPTS.items():
    d, wall = chat(p)
    t = d.get("timings", {})
    row = {"label": LABEL, "prompt": name, "n": t.get("predicted_n"),
           "tg_tps": round(t.get("predicted_per_second", 0), 2),
           "pp_tps": round(t.get("prompt_per_second", 0), 1),
           "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"), "wall_s": round(wall, 1),
           "head": d["choices"][0]["message"].get("content", "")[:60].replace("\n", " ")}
    rows.append(row)
    print(json.dumps(row), flush=True)
avg = sum(r["tg_tps"] for r in rows) / len(rows)
print(f"SUMMARY {LABEL} mean_tg_tps={avg:.2f}")
