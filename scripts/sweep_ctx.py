#!/usr/bin/env python3
"""Context-length sweep for the Nemotron Bolt E4B + DFlash Jetson recipe.

For each context length: launches llama-server (via scripts/bolt-serve.sh env
vars), measures decode (bolt_bench-style /completion timings), prefill and
TTFT (ttft_bench-style streamed, nonce'd prompts at n = 512/2048/8192), and
resident memory. Writes results/sweep_<label>.json.

Run ON the Jetson:  python3 sweep_ctx.py --label q8d
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request, random, string

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

WORDS = ["alpha", "bridge", "copper", "delta", "ember", "forest", "glacier",
         "harbor", "island", "juniper"]
PROMPTS = {
    "code": "Write a Python function that parses an ISO-8601 date string and returns a datetime, with error handling and docstring.",
    "math": "Solve step by step: a train travels 180 km in 2.5 hours, then 120 km in 1.5 hours. What is its average speed?",
    "prose": "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "extract": "Extract every person, date and amount as JSON from: 'On March 3, 2026 Alice Chen paid Bob Ortiz $1,250 for the design work; on April 9 Bob refunded $300.'",
}

def decode_bench(url, label, n_predict=256):
    rows = []
    for name, p in PROMPTS.items():
        body = json.dumps({"messages": [{"role": "user", "content": p}],
                           "max_tokens": n_predict, "temperature": 0, "seed": 1,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
        t = d.get("timings", {})
        rows.append({"prompt": name, "tg_tps": round(t.get("predicted_per_second", 0), 2),
                     "pp_tps": round(t.get("prompt_per_second", 0), 1),
                     "draft_acc": t.get("draft_n_accepted"), "draft_n": t.get("draft_n")})
    rows and rows.append({"prompt": "mean", "tg_tps": round(sum(r["tg_tps"] for r in rows[:-0]) / len(rows), 2)})
    return rows

def stream_ttft(url, n_tokens):
    nonce = "".join(random.choices(string.ascii_letters, k=12))
    body = json.dumps({
        "messages": [{"role": "user", "content":
            f"[nonce {nonce}] Ignore the following text, it is filler. Text: " +
            " ".join(WORDS[i % len(WORDS)] for i in range(n_tokens)) +
            " Now summarize what I asked in one short sentence: say OK."}],
        "max_tokens": 64, "temperature": 0, "stream": True,
        "cache_prompt": False, "timings_per_token": True,
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; toks = 0; timings = None
    with urllib.request.urlopen(req, timeout=900) as r:
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
    if not first or toks <= 1: return None
    return {"n_prompt": n_tokens, "ttft_s": round(first - t0, 2),
            "prefill_tps": timings and round(timings.get("prompt_per_second", 0)),
            "decode_tps_stream": timings and round(timings.get("predicted_per_second", 0), 1)}

def mem_avail_mib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) // 1024

def wait_healthy(url, log_path, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url + "/health", timeout=2)
            return True
        except Exception:
            if os.path.exists(log_path):
                with open(log_path) as f:
                    if "exiting" in f.read(): return False
            time.sleep(2)
    return False

def run_config(ctx, draft, port, label, max_prompt):
    lens = [n for n in (512, 2048, 8192) if n <= max_prompt]
    skipped = [n for n in (512, 2048, 8192) if n > max_prompt]
    if skipped:
        print(f"  (skipping prompt lengths {skipped}: would overflow ctx {ctx})", flush=True)
    env = dict(os.environ, BOLT_CTX=str(ctx), BOLT_DRAFT=draft,
               BOLT_HOST="127.0.0.1", BOLT_PORT=str(port))
    log = f"/tmp/sweep_{label}_c{ctx}.log"
    proc = subprocess.Popen(["bash", os.path.join(HERE, "bolt-serve.sh"), "dflash"],
                            env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                            start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    if not wait_healthy(url, log):
        os.killpg(proc.pid, signal.SIGKILL); return {"ctx": ctx, "error": "failed to start"}
    time.sleep(2)
    out = {"ctx": ctx,
           "decode": decode_bench(url, label),
           "ttft": [t for t in (stream_ttft(url, n) for n in lens) if t],
           "mem_avail_mib": mem_avail_mib()}
    proc.kill(); time.sleep(4)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="q8d")
    ap.add_argument("--draft", default="Nemotron-3.5-Bolt-E4B-DFlash-Q8_0.gguf")
    ap.add_argument("--ctxs", default="8192,16384,32768")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--model-dir", default=os.path.expanduser("~/models/bolt"))
    args = ap.parse_args()
    os.chdir(args.model_dir)
    os.environ.setdefault("GGML_CUDA_NO_PINNED", "1")
    subprocess.run(["sudo", "-n", "systemctl", "stop", "bolt-llm.service"], check=False)
    time.sleep(3)
    try:
        results = []
        for ctx in [int(c) for c in args.ctxs.split(",")]:
            print(f"== ctx {ctx}", flush=True)
            r = run_config(ctx, args.draft, args.port, args.label, max_prompt=ctx - 512)
            results.append(r); print(json.dumps(r), flush=True)
    finally:
        subprocess.run(["sudo", "-n", "systemctl", "start", "bolt-llm.service"], check=False)
        print("service restored", flush=True)
    os.makedirs("results", exist_ok=True)
    path = f"results/sweep_{args.label}.json"
    with open(path, "w") as f:
        json.dump({"label": args.label, "draft": args.draft, "results": results}, f, indent=1)
    print("WROTE", path)

if __name__ == "__main__":
    main()
