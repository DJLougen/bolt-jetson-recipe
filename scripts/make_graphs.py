#!/usr/bin/env python3
"""Generate README charts (dark HUD theme matching the MLX model card).

Inputs (results/):
  depth_dflash.json   depth sweep, production DFlash server   (label dflash-q8d)
  depth_plain.json    depth sweep, no drafter                 (label plain)
  decode_by_task.json short-prompt decode per task, both arms
  lc_<arm>.json       optional long-context tuning arms (4k/8k/14k)

Outputs (assets/):
  overview.png        prefill | decode | TTFT vs prompt depth (one figure)
  decode_by_task.png  short-prompt decode per task
Prints the per-depth means used in the README so text never drifts from charts.
"""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, ASSETS = os.path.join(ROOT, "results"), os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)

# MLX-card palette
BG, PANEL, GRID = "#04070d", "#0a1520", "#16303f"
CY, LIME, WHITE, GREY, DIM = "#19e6ff", "#c8f531", "#e8f6ff", "#9fb2c4", "#5c6f82"
MONO = "DejaVu Sans Mono"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": GREY, "xtick.color": GREY, "ytick.color": GREY,
    "text.color": WHITE, "font.family": MONO, "font.size": 11,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})

def mean(xs): xs = list(xs); return sum(xs) / len(xs)

def load_depth(name, label, reps, require_draft):
    d = json.load(open(os.path.join(RES, name)))
    assert d.get("label") == label, f"{name}: label {d.get('label')} != {label}"
    rows = [r for r in d["rows"] if not r.get("error")]
    dropped = len(d["rows"]) - len(rows)
    if require_draft:
        rows = [r for r in rows if r.get("draft_n") is not None]
    by = {}
    for r in rows:
        by.setdefault(r["requested_depth"], []).append(r)
    bad = {k: len(v) for k, v in by.items() if len(v) != reps}
    assert not bad, f"{name}: reps per depth != {reps}: {bad}"
    if dropped:
        print(f"  note: {name} dropped {dropped} errored rows")
    return by

def series(by, key):
    xs = sorted(by)
    return ([mean(r["prompt_tokens"] for r in by[x]) for x in xs],
            [mean(r[key] for r in by[x]) for x in xs],
            [[r[key] for r in by[x]] for x in xs])

def style(ax, title, ylabel):
    ax.set_title(title, color=CY, fontsize=13, fontweight="bold", loc="left", pad=10)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("prompt tokens")
    for s in ax.spines.values(): s.set_color(GRID)
    ax.set_ylim(0, None)

def ticks(ax, xs):
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{round(v/1024)}k" if v >= 1000 else f"{v/1024:.1f}k" for v in xs])

# ---------------- overview: prefill | decode | TTFT ----------------
arms = [("plain", load_depth("depth_plain.json", "plain", 3, False), CY, "plain"),
        ("dflash", load_depth("depth_dflash.json", "dflash-q8d", 3, True), LIME, "+ DFlash")]

fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
panels = [("prefill_tps", "PREFILL", "tok/s"),
          ("decode_tps", "DECODE", "tok/s (long-form task)"),
          ("ttft_s", "TTFT", "seconds")]
print("per-depth means (README source):")
for ax, (key, title, ylab) in zip(axes, panels):
    for name, by, color, label in arms:
        x, m, reps = series(by, key)
        for xi, rv in zip(x, reps):
            ax.scatter([xi] * len(rv), rv, color=color, alpha=0.35, s=18, zorder=3)
        ax.plot(x, m, "-o", color=color, lw=2.4, ms=6, label=label, zorder=4)
        ax.annotate(f"{m[-1]:.1f}" if key != "prefill_tps" else f"{m[-1]:.0f}",
                    (x[-1], m[-1]), textcoords="offset points", xytext=(6, 6),
                    color=color, fontsize=10, fontweight="bold")
        print(f"  {title:7s} {label:9s} " + "  ".join(f"{xi:5.0f}:{mi:6.2f}" for xi, mi in zip(x, m)))
    style(ax, title, ylab)
    ticks(ax, series(arms[0][1], key)[0])
axes[0].legend(loc="lower right", frameon=False, labelcolor=WHITE)
fig.suptitle("NEMOTRON-3.5 BOLT-E4B // JETSON ORIN NANO 8GB — prompt-depth sweep",
             color=WHITE, fontsize=15, fontweight="bold", x=0.01, ha="left", y=0.99)
fig.text(0.99, 0.01, "llama.cpp b11840 · ctx 16384 · Q8_0 DFlash drafter n-max 7 · temp 0 · 3 reps (dots) · cold cache",
         ha="right", color=DIM, fontsize=9)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig(os.path.join(ASSETS, "overview_v2.png"), dpi=150)
plt.close(fig)

# ---------------- decode by task (short prompts) ----------------
bench = json.load(open(os.path.join(RES, "decode_by_task.json")))
tasks = [t["prompt"] for t in bench["plain"]["rows"]]
assert tasks == [t["prompt"] for t in bench["dflash"]["rows"]]
pv = [t["tg_tps"] for t in bench["plain"]["rows"]]
dv = [t["tg_tps"] for t in bench["dflash"]["rows"]]
names, pv, dv = tasks + ["mean"], pv + [mean(pv)], dv + [mean(dv)]
fig, ax = plt.subplots(figsize=(10, 4.8))
y = list(range(len(names)))
ax.barh([v + 0.2 for v in y], pv, height=0.38, color=CY, alpha=0.55, label="plain")
ax.barh([v - 0.2 for v in y], dv, height=0.38, color=LIME, label="+ DFlash")
for v, yy in zip(pv, y): ax.text(v + 0.8, yy + 0.2, f"{v:.1f}", va="center", color=CY, fontsize=10)
for v, yy, p in zip(dv, y, pv):
    ax.text(v + 0.8, yy - 0.2, f"{v:.1f}  ({v/p:.1f}×)", va="center", color=LIME, fontsize=10, fontweight="bold")
ax.set_yticks(y, names); ax.invert_yaxis()
ax.set_xlim(0, max(dv) * 1.3); ax.set_xlabel("decode tok/s (short prompts, temp 0, single run)")
ax.grid(axis="y", visible=False)
ax.set_title("DECODE BY TASK", color=CY, fontsize=13, fontweight="bold", loc="left", pad=28)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, labelcolor=WHITE)
for s in ax.spines.values(): s.set_color(GRID)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "decode_by_task_v2.png"), dpi=150)
plt.close(fig)

# ---------------- optional: long-context tuning arms ----------------
lc = sorted(glob.glob(os.path.join(RES, "lc_*.json")))
if lc:
    rows = []
    for p in lc:
        d = json.load(open(p))
        ok = [r for r in d["rows"] if not r.get("error")]
        by = {}
        for r in ok: by.setdefault(r["requested_depth"], []).append(r)
        rows.append((d["label"], by))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    depths = sorted({k for _, by in rows for k in by})
    width = 0.8 / len(rows)
    pal = [LIME, CY, "#ff9f1c", "#b388ff", "#ff5d8f", WHITE]
    print("long-context arms:")
    for ax, (key, title, ylab) in zip(axes, panels):
        for i, (label, by) in enumerate(rows):
            vals = [mean(r[key] for r in by[d]) if d in by else 0 for d in depths]
            ax.bar([j + i * width - 0.4 + width / 2 for j in range(len(depths))], vals,
                   width=width, color=pal[i % len(pal)], label=label)
            print(f"  {title:7s} {label:8s} " + "  ".join(f"{d}:{v:6.2f}" for d, v in zip(depths, vals)))
        ax.set_xticks(range(len(depths)), [f"{d//1024}k" for d in depths])
        style(ax, title, ylab); ax.set_xlabel("requested prompt depth")
    axes[0].legend(frameon=False, labelcolor=WHITE, fontsize=9)
    fig.suptitle("LONG-CONTEXT TUNING ARMS", color=WHITE, fontsize=15, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(ASSETS, "longctx_arms_v2.png"), dpi=150)
    plt.close(fig)

print("charts written to", ASSETS)
