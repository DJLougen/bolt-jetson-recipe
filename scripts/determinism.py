#!/usr/bin/env python3
"""Repeatability check: 4 identical temp-0 requests per prompt, with the prompt
cache on (server default) and off. usage: determinism.py LABEL [URL]"""
import json, sys, hashlib, urllib.request
URL = (sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000") + "/v1/chat/completions"
PROMPTS = {
 "prose": "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
 "code": "Write a Python function that parses an ISO-8601 date string and returns a datetime, with error handling and docstring.",
}
def run(p, **kw):
    body = {"messages":[{"role":"user","content":p}], "max_tokens":200, "temperature":0, "seed":1,
            "chat_template_kwargs":{"enable_thinking":False}}
    body.update(kw)
    r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type":"application/json"}), timeout=600)
    return json.load(r)["choices"][0]["message"]["content"]
label = sys.argv[1]
for name, p in PROMPTS.items():
    for mode, kw in (("cache_on", {}), ("cache_off", {"cache_prompt": False})):
        outs = [run(p, **kw) for _ in range(4)]
        hs = [hashlib.sha1(o.encode()).hexdigest()[:8] for o in outs]
        print(json.dumps({"arm": label, "prompt": name, "mode": mode, "hashes": hs,
                          "distinct": len(set(hs)), "first": outs[0][:70].replace("\n"," ")}), flush=True)
        for o in outs: print("TEXT", label, name, mode, json.dumps(o))
