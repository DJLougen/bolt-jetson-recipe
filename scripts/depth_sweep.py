#!/usr/bin/env python3
"""Prompt-depth sweep against the RUNNING bolt-llm server (no restarts).

Corpus is tokenized ONCE via the server's /tokenize; each request chunk is
built with /detokenize over exact token slices — no char-per-token guessing.
Task: "Explain each section of the text above in detail" (long-form, stops
naturally well past 256 tokens; NO ignore_eos — greedy loops inflate decode).
Records TTFT (request-send → first content chunk), prefill and decode from
the final chunk's `timings`; x-axis is the server's timings.prompt_n.

usage: depth_sweep.py --out results/depth_dflash.json --label dflash-q8d
                      [--depths 256,1024,2048,4096,8192,14336] [--reps 3]
                      [--max-tokens 768]
"""
import argparse, json, time, urllib.request, random, string, os, glob

def load_corpus():
    # Public text only (llama.cpp docs + sources) — nothing private goes into results.
    texts = []
    for pat in ("/home/djl/llama.cpp-bolt/docs/*.md", "/home/djl/llama.cpp-bolt/README.md",
                "/home/djl/llama.cpp-bolt/src/*.cpp"):
        for p in sorted(glob.glob(pat)):
            try:
                texts.append(open(p, encoding="utf-8", errors="ignore").read())
            except OSError:
                pass
    return "\n".join(texts)

TASK = ("Go through the text above section by section and explain each section "
        "in detail in your own words, in at least 400 words total.")

CORPUS = load_corpus()

def api(port, path, payload):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

def build_tokens(port, depth, rng):
    toks = api(port, "/tokenize", {"content": CORPUS, "add_special": False})["tokens"]
    assert len(toks) >= depth + 64, f"corpus too small: {len(toks)} tokens < {depth}"
    start = rng.randrange(0, len(toks) - depth - 64)
    chunk = api(port, "/detokenize", {"tokens": toks[start:start + depth]})["content"]
    nonce = "".join(rng.choices(string.ascii_letters, k=10))
    return f"[nonce {nonce}]\n\nText:\n{chunk}\n\n{TASK}"

def stream_request(port, depth, rng, max_tokens):
    body = json.dumps({
        "messages": [{"role": "user", "content": build_tokens(port, depth, rng)}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
        "cache_prompt": False, "timings_per_token": True,
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; toks = 0; timings = None; err = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
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
    except Exception as e:
        err = repr(e)[:200]
    if err or not first or toks <= 1:
        return {"requested_depth": depth, "error": err or "no content", "timings": timings}
    return {"requested_depth": depth,
            "prompt_tokens": timings and timings.get("prompt_n"),
            "ttft_s": round(first - t0, 3),
            "prefill_tps": timings and timings.get("prompt_per_second"),
            "prefill_ms": timings and timings.get("prompt_ms"),
            "decode_tps": timings and timings.get("predicted_per_second"),
            "draft_acc": timings and timings.get("draft_n_accepted"),
            "draft_n": timings and timings.get("draft_n"),
            "tokens_out": toks}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--depths", default="256,1024,2048,4096,8192,14336")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--out", default="results/depth_sweep.json")
    ap.add_argument("--label", default="dflash-q8d")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = []
    for depth in [int(d) for d in args.depths.split(",")]:
        for rep in range(args.reps):
            r = stream_request(args.port, depth, rng, args.max_tokens)
            r.update({"rep": rep, "label": args.label})
            rows.append(r)
            print(json.dumps(r), flush=True)
            time.sleep(1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"label": args.label, "seed": args.seed, "rows": rows}, f, indent=1)
    print("WROTE", args.out)

if __name__ == "__main__":
    main()
