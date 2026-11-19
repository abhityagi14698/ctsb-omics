#!/usr/bin/env python3
"""
Supplementary Fig. S3 -- pH dependence of the eight compared proteases
(data: pH_optimum_table_FINAL.csv) shown on one pH axis together with the
model's pH-dependent activation rate of pro-CTSB, k_pH(h) / alpha_h.

Run:  python3 make_ph_figure.py      (writes FigS3_protease_pH.pdf / .png)
"""
import time
T0 = time.time()
print("Building Fig. S3 ...")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- data (from pH_optimum_table_FINAL.csv)
ACID, NEUTRAL = "#eb6834", "#2a78d6"     # validated categorical pair (dataviz palette slots 2 and 1)
ROWS = [  # label, optimum range (lo, hi), point values, activation group, activation route (short)
    ("CTSD",    (3.5, 5.0), [],         "acid",    "autocatalytic"),
    ("CTSL",    (4.2, 4.8), [],         "acid",    "autocatalytic"),
    ("CTSK",    (4.0, 7.0), [],         "acid",    "autocatalytic"),
    ("CTSB",    (5.4, 6.2), [5.4, 6.2], "acid",    "autocatalytic"),
    ("MT1-MMP", (7.0, 7.4), [],         "neutral", "furin"),
    ("MMP-2",   (7.5, 7.6), [],         "neutral", "MT1-MMP"),
    ("MMP-9",   (7.8, 7.8), [7.8],      "neutral", "plasmin, MMP-3"),
    ("uPA",     (7.5, 9.5), [],         "neutral", "plasmin; CTSB"),
]
CTSB_HALFMAX = (4.6, 7.4)   # >=50% of maximal activity, DPCP 4.6-5.8 and endopeptidase 5.0-7.4 (Yoon et al. 2022)
BANDS = [  # (lo, hi, label, grey level)
    (4.5, 5.0, "lysosome", "0.86"),
    (6.5, 7.0, "tumour\npHe", "0.80"),
    (7.2, 7.4, "normal\ntissue", "0.90"),
]
H_OPT, SIGMA_H = 5.5, 0.5   # model parameters (Table 3 of the paper)

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.6, "font.family": "DejaVu Sans"})
fig, (axm, ax) = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True,
                              gridspec_kw=dict(height_ratios=[1, 2.7], hspace=0.22))
XMIN, XMAX = 2.5, 10.0

for a in (axm, ax):
    for lo, hi, _, g in BANDS:
        a.axvspan(lo, hi, color=g, lw=0, zorder=0)
    a.axvline(H_OPT, color="0.35", lw=1, ls=(0, (4, 3)), zorder=1)
    a.spines[["top", "right"]].set_visible(False)

# (a) model activation curve
h = np.linspace(XMIN, XMAX, 600)
k = np.exp(-(h - H_OPT) ** 2 / (2 * SIGMA_H ** 2))
axm.fill_between(h, k, color=ACID, alpha=0.18, lw=0, zorder=2)
axm.plot(h, k, color=ACID, lw=2, zorder=3)
axm.set_ylim(0, 1.15); axm.set_yticks([0, 0.5, 1])
axm.set_ylabel("$k_{pH}(h)/\\alpha_h$")
axm.set_title("(a) Model: pH-dependent activation rate of pro-CTSB", loc="left", fontsize=9, fontweight="bold")
axm.text(6.35, 0.80, f"$h_{{opt}}$ = {H_OPT}\n$\\sigma_h$ = {SIGMA_H}", fontsize=8, color="0.25", va="center")

# (b) literature pH optima
y = np.arange(len(ROWS))[::-1]
for yi, (name, (lo, hi), pts, grp, route) in zip(y, ROWS):
    col = ACID if grp == "acid" else NEUTRAL
    if name == "CTSB":
        ax.plot(CTSB_HALFMAX, [yi, yi], color=col, lw=9, alpha=0.25, solid_capstyle="round", zorder=2)
    if hi - lo > 0.05:
        ax.plot([lo, hi], [yi, yi], color=col, lw=5, solid_capstyle="round", zorder=3)
    for p in (pts or ([lo] if hi - lo <= 0.05 else [])):
        ax.plot(p, yi, "o", ms=7, color=col, mec="white", mew=1.2, zorder=4)
    ax.text(XMAX + 0.12, yi, route, va="center", ha="left", fontsize=7.5, color="0.30", clip_on=False)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in ROWS])
for t in ax.get_yticklabels():
    if t.get_text() == "CTSB":
        t.set_fontweight("bold")
ax.axhline(3.5, color="0.6", lw=0.6, ls=":", zorder=1)          # separates the two activation groups
ax.set_ylim(-0.75, len(ROWS) - 0.4)
ax.set_xlim(XMIN, XMAX)
ax.set_xlabel("pH")
ax.tick_params(axis="y", length=0)
ax.set_title("(b) Reported pH optimum of the human enzymes", loc="left", fontsize=9, fontweight="bold")
for lo, hi, lab, _ in BANDS:
    ax.text((lo + hi) / 2, -0.62, lab.replace("\n", " ") if lab == "lysosome" else lab, ha="center", va="bottom",
            fontsize=7, color="0.25", linespacing=0.9, zorder=5)
ax.text(XMAX + 0.12, len(ROWS) - 0.55, "zymogen\nactivation", ha="left", va="bottom",
        fontsize=7.5, color="0.30", fontweight="bold", clip_on=False, linespacing=0.9)

handles = [
    Line2D([0], [0], color=ACID, lw=5, label="activated autocatalytically at acidic pH"),
    Line2D([0], [0], color=NEUTRAL, lw=5, label="activated by other proteases at neutral pH"),
    Line2D([0], [0], color=ACID, lw=9, alpha=0.25, label="CTSB: ≥50% of maximal activity"),
    Line2D([0], [0], color="0.35", lw=1, ls=(0, (4, 3)), label="model $h_{opt}$"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=7.5,
          frameon=False, handlelength=2.2, columnspacing=1.6)

for ext in ("pdf", "png"):
    fig.savefig(f"FigS3_protease_pH.{ext}", dpi=300, bbox_inches="tight")
print(f"Saved FigS3_protease_pH.pdf/.png   (total time {time.time() - T0:.1f} s)")
