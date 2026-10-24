#!/usr/bin/env bash
# Hill-climb bolt-llm serving configs on Jetson.
# Stops bolt-llm.service first, restores it at the end.
set -u
cd ~/models/bolt
PY=~/sglang-env/bin/python

sudo -n systemctl stop bolt-llm.service
pkill -x llama-server 2>/dev/null; sleep 3

ttft() { # $1 n_prompt_tokens ; streams, times from request send, unique nonce
  $PY - "$1" <<'EOF'
import sys, json, time, urllib.request, random, string
n = int(sys.argv[1])
words = ["alpha","bridge","copper","delta","ember","forest","glacier","harbor","island","juniper"]
nonce = "".join(random.choices(string.ascii_letters, k=12))
body = json.dumps({
    "messages": [{"role": "user", "content":
        f"[nonce {nonce}] Ignore the following text, it is filler. Text: " +
        " ".join(words[i % len(words)] for i in range(n)) +
        " Now summarize what I asked in one short sentence: say OK."}],
    "max_tokens": 64, "temperature": 0, "stream": True,
    "cache_prompt": False, "timings_per_token": True,
    "chat_template_kwargs": {"enable_thinking": False},
}).encode()
req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", body,
                             {"Content-Type": "application/json"})
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
if first and toks > 1:
    pp = timings.get("prompt_per_second") if timings else None
    print(f"{n}toks ttft={first-t0:.2f}s tg={(toks-1)/(last-first):.1f}t/s prefill={pp and round(pp)}t/s")
else:
    print(f"{n}toks ERR")
EOF
}

run_config() { # $1 name; $2.. = env assignments (VAR=x); EXTRA_ARGS env carries server flags
  local name=$1; shift
  env "$@" EXTRA_ARGS="$EXTRA_ARGS" setsid nohup ./bolt-serve.sh dflash $EXTRA_ARGS > srv-hc-$name.log 2>&1 < /dev/null &
  for i in $(seq 1 120); do curl -sf localhost:8000/health >/dev/null && break; grep -q "exiting" srv-hc-$name.log && break; sleep 2; done
  curl -sf localhost:8000/health >/dev/null || { echo "CONFIG $name FAILED_TO_START"; pkill -x llama-server; sleep 4; return; }
  echo "== CONFIG $name  env:$* extra:$EXTRA_ARGS"
  timeout 600 $PY bolt_bench.py http://127.0.0.1:8000 hc-$name 256 | grep -E "tg_tps|SUMMARY"
  for n in 512 2048 8192; do printf "TTFT "; ttft $n; done
  echo "MEM $name $(free -m | awk '/Mem:/{print "used="$3" avail="$7}')"
  pkill -x llama-server; sleep 4
}

EXTRA_ARGS="" run_config A-nmax7-c16k       BOLT_CTX=16384 BOLT_NMAX=7
EXTRA_ARGS="" run_config B-nmax4-c16k       BOLT_CTX=16384 BOLT_NMAX=4
EXTRA_ARGS="" run_config C-nmax2-c16k       BOLT_CTX=16384 BOLT_NMAX=2
EXTRA_ARGS="-ctk q8_0 -ctv q8_0" run_config D-nmax7-kvq8-c16k BOLT_CTX=16384 BOLT_NMAX=7
EXTRA_ARGS="-b 2048 -ub 2048" run_config E-nmax7-b2048-c16k   BOLT_CTX=16384 BOLT_NMAX=7
EXTRA_ARGS="-ctk q8_0 -ctv q8_0" run_config F-nmax7-kvq8-c32k BOLT_CTX=32768 BOLT_NMAX=7

sudo -n systemctl start bolt-llm.service
for i in $(seq 1 60); do curl -sf localhost:8000/health >/dev/null && break; sleep 2; done
systemctl is-active bolt-llm
echo HILLCLIMB_DONE
