#!/usr/bin/env python3
"""Real-usage speed: server-default sampling (no temperature/top_p sent, so the
GGUF defaults apply: temp 1.0, top-p 0.95, top-k 64, random seed), thinking ON
(llama-server default), streamed. Reports TTFT, decode tok/s and draft acceptance
per request from the server's own timings.

usage: real_bench.py LABEL [URL] [REPS]
"""
import json, sys, time, urllib.request

LABEL = sys.argv[1]
URL = (sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000") + "/v1/chat/completions"
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
PROMPTS = {
    "chat": "I'm planning a weekend trip to Montreal in winter. What should I pack and do?",
    "code": "Write a Python function that parses an ISO-8601 date string and returns a datetime, with error handling and docstring.",
    "math": "Solve step by step: a train travels 180 km in 2.5 hours, then 120 km in 1.5 hours. What is its average speed?",
    "extract": "Extract every person, date and amount as JSON from: 'On March 3, 2026 Alice Chen paid Bob Ortiz $1,250 for the design work; on April 9 Bob refunded $300.'",
    "prose": "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
}

def run(prompt):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 2048, "stream": True}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time(); first = None; first_answer = None; timings = None; finish = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            j = json.loads(line[6:])
            if "timings" in j: timings = j["timings"]
            delta = j["choices"][0].get("delta", {}) if j.get("choices") else {}
            if first is None and (delta.get("content") or delta.get("reasoning_content")):
                first = time.time()
            if first_answer is None and delta.get("content"):
                first_answer = time.time()
            if j.get("choices") and j["choices"][0].get("finish_reason"):
                finish = j["choices"][0]["finish_reason"]
    return {"ttft_s": round(first - t0, 3),
            "time_to_answer_s": round(first_answer - t0, 3) if first_answer else None,
            "prompt_tokens": timings.get("prompt_n"), "finish": finish, "decode_tps": round(timings["predicted_per_second"], 2),
            "tokens": timings["predicted_n"], "draft_n": timings.get("draft_n"),
            "draft_acc": timings.get("draft_n_accepted")}

run("Say hi.")  # warmup
rows = []
for name, p in PROMPTS.items():
    for rep in range(REPS):
        r = run(p); r.update({"label": LABEL, "prompt": name, "rep": rep})
        rows.append(r); print(json.dumps(r), flush=True)
tot_tok = sum(r["tokens"] for r in rows)
tot_time = sum(r["tokens"] / r["decode_tps"] for r in rows)
med = lambda xs: sorted(xs)[len(xs) // 2]
ans = [r["time_to_answer_s"] for r in rows if r["time_to_answer_s"] is not None]
print(f"SUMMARY {LABEL} token-weighted decode={tot_tok / tot_time:.2f} tok/s "
      f"median_ttft={med([r['ttft_s'] for r in rows]):.2f}s "
      f"median_time_to_answer={med(ans):.2f}s n={len(rows)} truncated={sum(r['finish'] == 'length' for r in rows)}")
