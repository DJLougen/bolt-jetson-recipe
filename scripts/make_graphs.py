#!/usr/bin/env python3
"""Generate README charts from results/depth_*.json + results/decode_by_task.json.

matplotlib, Okabe-Ito palette, value labels at bar tips, single-run bars hatched.
Asserts the schema of every input file so silent data changes fail loudly.
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)

C_PLAIN, C_DFLASH, C_MEAN = "#0072B2", "#E69F00", "#009E73"

def load(name, expect_label=None, require_draft=False):
    with open(os.path.join(RES, name)) as f:
        d = json.load(f)
    assert "rows" in d and d["rows"], f"{name}: no rows"
    if expect_label:
        assert d.get("label") == expect_label, f"{name}: label {d.get('label')} != {expect_label}"
    ok = [r for r in d["rows"] if not r.get("error")]
    if require_draft:
        ok = [r for r in ok if r.get("draft_n") is not None]
    # exactly 3 reps per requested depth
    counts = {}
    for r in ok: counts[r["requested_depth"]] = counts.get(r["requested_depth"], 0) + 1
    assert all(v == 3 for v in counts.values()), f"{name}: reps per depth != 3: {counts}"
    return ok

def mean(xs): xs = list(xs); return sum(xs) / len(xs)

def grouped_by_depth(rows, key):
    # group by requested_depth; x position = mean server-measured prompt_tokens
    out = {}
    for r in rows:
        assert r.get(key) is not None, f"missing {key} in {r}"
        out.setdefault(r["requested_depth"], []).append((r["prompt_tokens"], r[key]))
    return {d: (mean([p for p, _ in v]), [x for _, x in v]) for d, v in sorted(out.items())}

# ---- chart 1: decode by task (single-run bars from bench JSON) -------------
bench = json.load(open(os.path.join(RES, "decode_by_task.json")))
assert {"plain", "dflash"} <= bench.keys(), bench.keys()
tasks = bench["plain"]["rows"]
assert [t["prompt"] for t in tasks] == [t["prompt"] for t in bench["dflash"]["rows"]]
names = [t["prompt"] for t in tasks] + ["mean"]
plain_v = [t["tg_tps"] for t in tasks] + [mean(t["tg_tps"] for t in tasks)]
dflash_v = [t["tg_tps"] for t in bench["dflash"]["rows"]] + [mean(t["tg_tps"] for t in bench["dflash"]["rows"])]

fig, ax = plt.subplots(figsize=(8, 4.4))
y = range(len(names))
ax.barh([v + 0.19 for v in y], plain_v, height=0.36, color=C_PLAIN, label="plain (no drafter)", hatch="//", edgecolor="white")
ax.barh([v - 0.19 for v in y], dflash_v, height=0.36, color=C_DFLASH, label="+ DFlash (Q8_0 drafter)")
for v, yp in zip(plain_v, y): ax.text(v + 0.6, yp + 0.19, f"{v:.1f}", va="center", fontsize=9, color=C_PLAIN)
for v, yp in zip(dflash_v, y): ax.text(v + 0.6, yp - 0.19, f"{v:.1f}", va="center", fontsize=9, color="#8a5a00")
ax.set_yticks(list(y), names); ax.set_xlabel("decode tok/s (temp 0, single run)")
ax.set_xlim(0, max(dflash_v) * 1.18)
ax.set_title("Nemotron-3.5-Bolt-E4B-4GB on Jetson Orin Nano 8GB — decode speed", pad=30)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.invert_yaxis()
fig.text(0.99, 0.01, "llama.cpp b11840 · ctx 16384 · /v1/chat/completions timings", ha="right", fontsize=7, color="#666")
fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "decode_by_task.png"), dpi=160); plt.close(fig)

# ---- charts 2-4: depth curves (dflash vs plain, reps as dots) -------------
dflash = load("depth_dflash.json", "dflash-q8d", require_draft=True)
plain = load("depth_plain.json", "plain")

def curve(ax, data_p, data_d, ylabel, title):
    # data_*: {requested_depth: (mean_prompt_tokens, [values])}
    for data, color, label in ((data_p, C_PLAIN, "plain"), (data_d, C_DFLASH, "+ DFlash")):
        xs = sorted(data)
        xm = [data[x][0] for x in xs]
        means = [mean(data[x][1]) for x in xs]
        for x in xs:
            px, vals = data[x]
            ax.plot([px] * len(vals), vals, "o", color=color, alpha=0.35, ms=4)
        ax.plot(xm, means, "-o", color=color, label=label, ms=5)
    ax.set_xlabel("prompt tokens (server-measured mean)"); ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ticks = sorted({round(data[x][0]) for data in (data_p, data_d) for x in data})
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v//1024}k" if v >= 1024 else v for v in ticks])

fig, ax = plt.subplots(figsize=(8, 4.2))
curve(ax, grouped_by_depth(plain, "decode_tps"), grouped_by_depth(dflash, "decode_tps"),
      "decode tok/s (mean of 3 reps, dots = reps)",
      "Decode vs prompt depth — 400-word explain task, temp 0")
fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "decode_vs_depth.png"), dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.2))
curve(ax, grouped_by_depth(plain, "prefill_tps"), grouped_by_depth(dflash, "prefill_tps"),
      "prefill tok/s (mean of 3 reps, dots = reps)", "Prefill vs prompt depth")
ax.set_ylim(0, None)
fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "prefill_vs_depth.png"), dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.2))
curve(ax, grouped_by_depth(plain, "ttft_s"), grouped_by_depth(dflash, "ttft_s"),
      "TTFT seconds (mean of 3 reps, dots = reps)", "TTFT vs prompt depth (streamed, nonce'd prompts)")
ax.set_ylim(0, None)
fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "prefill_ttft_vs_depth.png"), dpi=160); plt.close(fig)

print("charts written to", ASSETS)
