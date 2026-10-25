#!/usr/bin/env python3
"""Banner in the MLX-card style. Stat chips are computed from results/real_*.jsonl
(server-default sampling, thinking on), so they can't drift from the charts."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, OUT = os.path.join(ROOT, "results"), os.path.join(ROOT, "assets", "banner_v2.png")

def load(f): return [json.loads(l) for l in open(os.path.join(RES, f)) if l.startswith("{")]
def decode(rows): return sum(r["tokens"] for r in rows) / sum(r["tokens"] / r["decode_tps"] for r in rows)
def answer(rows): return sum(r["time_to_answer_s"] for r in rows) / len(rows)
P, D = load("real_plain.jsonl"), load("real_dflash.jsonl")
dp, dd, ap, ad = decode(P), decode(D), answer(P), answer(D)
print(f"chips: decode {dp:.1f} -> {dd:.1f} ({dd/dp:.1f}x), answer {ap:.1f}s -> {ad:.1f}s")

fig = plt.figure(figsize=(24.0, 12.6), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 2400); ax.set_ylim(0, 1260); ax.axis("off")
ax.add_patch(plt.Rectangle((0, 0), 2400, 1260, color="#04070d"))
rng = np.random.default_rng(7)
for _ in range(90):
    ax.scatter(rng.uniform(0, 2400), rng.uniform(0, 1260), s=rng.uniform(1, 10),
               color="white", alpha=rng.uniform(0.15, 0.7), zorder=1)
for i in range(14):
    t = i / 13
    ax.plot([0, 2400], [40 + 300 * t**1.6] * 2, color="#19e6ff", alpha=0.05 + 0.05 * t, lw=0.8, zorder=1)
for x in range(-6, 25):
    ax.plot([x * 120, (x - 12) * 120 + 1200], [40, 340], color="#19e6ff", alpha=0.05, lw=0.8, zorder=1)

CY, LIME, WHITE, GREY = "#19e6ff", "#c8f531", "#e8f6ff", "#9fb2c4"
mono = {"family": "monospace"}

# planet: one sphere with a radial shade, ring split behind/in front
px, py, pr = 1870, 800, 175
ax.add_patch(Circle((px, py), pr * 1.75, color=CY, alpha=0.05, zorder=2))
ax.add_patch(Circle((px, py), pr * 1.35, fill=False, edgecolor=CY, lw=1.4, alpha=0.5, zorder=2))
ax.add_patch(Circle((px, py), pr * 1.62, fill=False, edgecolor=CY, lw=1.0, alpha=0.25, zorder=2))
th = np.linspace(0, 2 * np.pi, 400)
rx, ry = pr * 1.6, pr * 0.4
back = np.sin(th) > 0
ax.plot(np.where(back, px + rx * np.cos(th), np.nan), py + ry * np.sin(th), color=WHITE, alpha=0.5, lw=2, zorder=2.5)
for k in range(40, 0, -1):  # radial gradient sphere
    f = k / 40
    c = np.array([0.10, 0.22, 0.33]) * f + np.array([0.35, 0.78, 0.95]) * (1 - f)
    ax.add_patch(Circle((px - pr * 0.3 * (1 - f), py + pr * 0.3 * (1 - f)), pr * f, color=c, zorder=3))
ax.plot(np.where(~back, px + rx * np.cos(th), np.nan), py + ry * np.sin(th), color=WHITE, alpha=0.9, lw=2.4, zorder=4)
d = 44
ax.add_patch(Polygon([(px, py+d), (px+d*1.05, py), (px, py-d), (px-d*1.05, py)], closed=True,
                     fill=False, edgecolor=LIME, lw=3.5, zorder=6))
ax.add_patch(Polygon([(px, py+d*0.45), (px+d*0.48, py), (px, py-d*0.45), (px-d*0.48, py)], closed=True, color=LIME, zorder=6))
for sx, sy, s_ in ((1510, 1080, 90), (1400, 1150, 50), (2080, 1060, 60), (2255, 990, 40)):
    ax.plot([sx - s_, sx + s_], [sy, sy], color=WHITE, lw=1.6, alpha=0.8, zorder=2)
    ax.plot([sx, sx], [sy - s_, sy + s_], color=WHITE, lw=1.6, alpha=0.8, zorder=2)
for x0 in range(8):
    ax.add_patch(Polygon([(2210 + x0*26, 1255), (2232 + x0*26, 1255), (2212 + x0*26, 1228), (2190 + x0*26, 1228)],
                         color=LIME, alpha=0.9, zorder=6))

ax.text(150, 1092, "NMT-3.5/BLT-E4B  //  LLAMA.CPP-DFLASH  //  JETSON", color=CY, fontsize=21, fontweight="bold", **mono)
ax.text(144, 955, "NEMOTRON-3.5", color=WHITE, fontsize=105, fontweight="bold")
ax.text(144, 800, "BOLT-E4B // JETSON", color=LIME, fontsize=66, fontweight="bold")
ax.text(150, 716, f"{dd/dp:.1f}× real-use decode on an 8 GB Orin Nano with DFlash speculative decoding",
        color=GREY, fontsize=25)

def chip(x, y, w, h, label, value):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,rounding_size=6", fc="#0a1520", ec=CY, lw=1.6, zorder=6))
    for cx, cy_, dx, dy in ((x-7, y+h+7, 1, -1), (x+w+7, y+h+7, -1, -1), (x-7, y-7, 1, 1), (x+w+7, y-7, -1, 1)):
        ax.plot([cx, cx + 26*dx], [cy_, cy_], color=CY, lw=2, zorder=6)
        ax.plot([cx, cx], [cy_, cy_ + 20*dy], color=CY, lw=2, zorder=6)
    ax.text(x + 18, y + h - 34, label, color="#7f93a8", fontsize=17, **mono, zorder=7)
    ax.text(x + 18, y + 20, value, color=CY, fontsize=30, fontweight="bold", **mono, zorder=7)

chip(150, 380, 400, 160, "REAL-USE DECODE", f"{dp:.1f} → {dd:.1f} t/s")
chip(590, 380, 330, 160, "SPEEDUP", f"{dd/dp:.1f}×")
chip(150, 180, 400, 160, "TIME TO ANSWER", f"{ap:.1f} → {ad:.1f} s")
chip(590, 180, 330, 160, "FULL CONTEXT", "64K")
ax.text(150, 60, "DJLougen  //  GESTALTLABS", color=GREY, fontsize=19, **mono)
ax.text(2250, 60, "temp 1.0 · thinking on · server defaults", color="#5c6f82", fontsize=16, ha="right", **mono)
fig.savefig(OUT, dpi=100)
print("wrote", OUT)
