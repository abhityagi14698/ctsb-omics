# -*- coding: utf-8 -*-
"""
===============================================================================
 ENZYME vs ACIDITY-GENE COMPARISON  (TIMER3 Gene_Correlation exports)
 Builds: ranking table + enzyme-by-marker table + enzyme-by-cancer table
         + one 3-panel figure (PNG 300 dpi + vector PDF)
===============================================================================

 INPUT
 -----
   One CSV per enzyme, downloaded from TIMER3 -> Exploration -> Gene_Correlation
   (Purity Adjustment ON), named:   genecorr_table_<GENE>.csv
   Columns expected:  cancer, gene, rho, p, adj.p
       cancer : e.g. "BRCA (n=1086)"
       gene   : the acidity marker (CA9, CA12, SLC16A3, ...)
       rho    : purity-adjusted partial Spearman correlation
       adj.p  : FDR-adjusted p-value (as given by TIMER3)

 USAGE (Anaconda Prompt)
 -----------------------
   python make_enzyme_acidity_figure.py --data-dir path\\to\\csv_folder
   python make_enzyme_acidity_figure.py --data-dir . --out-dir results_enzyme

 All analysis choices live in the CONTROL PANEL below.
===============================================================================
"""

import os
import sys
import glob
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
#  ####################  CONTROL PANEL  #####################################
# =============================================================================

# ---- (1) Which enzyme files to use -----------------------------------------
#      None = every genecorr_table_<GENE>.csv found in --data-dir.
#      Or give an explicit list, e.g. ["CTSB", "CTSD", "CTSL", "CTSK",
#                                      "MMP2", "MMP9", "MMP14", "PLAU", "PLAUR"]
ENZYMES = None
SKIP_GENES = ["PAUPAR"]          # wrong gene picked by mistake -> always dropped

# ---- (2) Gene roles (only used for labelling) ------------------------------
NON_PROTEASE = {"PLAUR": "uPA receptor"}   # marked with a dagger in the figure

# ---- (3) Which cancer types to keep -----------------------------------------
DROP_SUBTYPES   = True           # drop BRCA-Basal, HNSC-HPV+, SKCM-Primary ...
                                 # (they repeat their parent cancer type)
EXCLUDE_CANCERS = ["LAML",       # blood cancer, no data (NA) in TIMER3
                   "DLBC",       # lymphoma: no solid-tissue ECM invasion
                   "THYM"]       # thymoma: lymphocyte-rich tissue

# ---- (4) Significance thresholds --------------------------------------------
FDR_THRESHOLD = 0.05             # adj.p below this = significant
STRONG_RHO    = 0.30             # "strong" = significant AND rho >= this

# ---- (5) Acidity markers, in the order shown in panel (b) -------------------
MARKERS = [("SLC16A3", "MCT4\n(SLC16A3)"),
           ("LDHA",    "LDHA"),
           ("SLC2A1",  "GLUT1\n(SLC2A1)"),
           ("SLC9A1",  "NHE1\n(SLC9A1)"),
           ("CA12",    "CA12"),
           ("SLC16A1", "MCT1\n(SLC16A1)"),
           ("CA9",     "CA9")]

# ---- (6) Figure settings -----------------------------------------------------
HIGHLIGHT  = "CTSB"              # bar coloured in panel (a)
HIGHLIGHT_COLOR = "#c2410c"
OTHER_COLOR     = "#9ca3af"
CMAP       = "RdBu_r"            # diverging: blue (negative) - white (0) - red (+)
VMAX       = 0.45                # colour scale runs from -VMAX to +VMAX
RANK_BY    = "median_rho"        # how enzymes are ordered top->bottom
DPI        = 300
FIG_NAME   = "enzyme_acidity_comparison"
# =============================================================================
#  ####################  END OF CONTROL PANEL  ##############################
# =============================================================================


def safe_write_csv(df, path):
    """Write a CSV and force it to disk (flush + fsync), crash-safe."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        df.to_csv(fh)
        fh.flush()
        os.fsync(fh.fileno())


def phase_done(name, t_phase):
    print(f"  [done] {name:<32s} {time.time() - t_phase:6.2f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", help="folder with genecorr_table_*.csv")
    ap.add_argument("--out-dir", default="results_enzyme_acidity")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t_all = time.time()
    print("=" * 70)
    print(" Enzyme vs acidity-gene comparison")
    print("=" * 70)

    # ------------------------------------------------------------------ load
    t = time.time()
    files = sorted(glob.glob(os.path.join(args.data_dir, "genecorr_table_*.csv")))
    if not files:
        sys.exit(f"[FATAL] no genecorr_table_*.csv found in {args.data_dir}")
    frames = []
    for k, f in enumerate(files, 1):
        gene = os.path.basename(f)[len("genecorr_table_"):-4]
        status = "loaded"
        if gene in SKIP_GENES or (ENZYMES is not None and gene not in ENZYMES):
            status = "skipped"
        else:
            d = pd.read_csv(f, na_values=["NA"])
            d["enzyme"] = gene
            frames.append(d)
        pct = 100 * k / len(files)
        print(f"\r  loading [{'#' * int(pct // 5):<20s}] {pct:5.1f}%  "
              f"{k}/{len(files)}  {gene:<8s} {status:<8s} "
              f"elapsed {time.time() - t:5.1f}s", end="", flush=True)
    print()
    d = pd.concat(frames, ignore_index=True)
    phase_done(f"load ({len(frames)} enzymes)", t)

    # ---------------------------------------------------------------- filter
    t = time.time()
    n0 = len(d)
    d = d.dropna(subset=["rho", "adj.p"])
    d["cancer_type"] = d["cancer"].str.split(" ").str[0]
    if DROP_SUBTYPES:
        d = d[~d["cancer_type"].str.contains("-")]
    d = d[~d["cancer_type"].isin(EXCLUDE_CANCERS)]
    n_ct = d["cancer_type"].nunique()
    print(f"  rows kept {len(d)}/{n0}; cancer types kept = {n_ct}")
    phase_done("filter", t)

    # ------------------------------------------------------------- summarise
    t = time.time()
    sig = d["adj.p"] < FDR_THRESHOLD
    d["sig_pos"] = sig & (d["rho"] > 0)
    d["sig_neg"] = sig & (d["rho"] < 0)
    d["strong"] = sig & (d["rho"] >= STRONG_RHO)
    s = d.groupby("enzyme").agg(n_pairs=("rho", "size"),
                                sig_pos=("sig_pos", "sum"),
                                sig_neg=("sig_neg", "sum"),
                                strong=("strong", "sum"),
                                median_rho=("rho", "median"),
                                mean_rho=("rho", "mean"))
    s["pct_sig_pos"] = 100 * s["sig_pos"] / s["n_pairs"]
    s["pct_strong"] = 100 * s["strong"] / s["n_pairs"]

    pc = d.groupby(["enzyme", "cancer_type"])["rho"].mean().unstack()
    ranks = pc.rank(axis=0, ascending=False)          # rank within each cancer
    s["mean_rank_across_cancers"] = ranks.mean(axis=1)
    s["n_cancers_ranked_first"] = (ranks == 1).sum(axis=1)

    order = s.sort_values(RANK_BY, ascending=False).index.tolist()
    s = s.loc[order]
    mk_codes = [m for m, _ in MARKERS if m in d["gene"].unique()]
    mk = d.pivot_table(index="enzyme", columns="gene", values="rho",
                       aggfunc="median").loc[order, mk_codes]
    pc = pc.loc[order, pc.mean().sort_values(ascending=False).index]
    phase_done("summarise", t)

    # ------------------------------------------------------------- save CSVs
    t = time.time()
    safe_write_csv(s.round(4), os.path.join(args.out_dir, "enzyme_ranking_summary.csv"))
    safe_write_csv(mk.round(4), os.path.join(args.out_dir, "enzyme_by_marker_median_rho.csv"))
    safe_write_csv(pc.round(4), os.path.join(args.out_dir, "enzyme_by_cancer_mean_rho.csv"))
    phase_done("save CSVs", t)

    # ------------------------------------------------------------------ plot
    t = time.time()
    label = {e: (f"{e}†" if e in NON_PROTEASE else e) for e in order}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig = plt.figure(figsize=(14, 8.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1, 1.05],
                          hspace=0.45, wspace=0.3)

    # (a) ranking bars
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(order))[::-1]
    cols = [HIGHLIGHT_COLOR if e == HIGHLIGHT else OTHER_COLOR for e in order]
    ax.barh(y, s["pct_sig_pos"], color=cols, height=0.62, alpha=0.45)
    ax.barh(y, s["pct_strong"], color=cols, height=0.62)
    for yi, e in zip(y, order):
        ax.text(s.loc[e, "pct_sig_pos"] + 1, yi,
                f"{s.loc[e, 'pct_sig_pos']:.0f}%  (median ρ {s.loc[e, 'median_rho']:.2f})",
                va="center", fontsize=8, color="#374151")
    ax.set_yticks(y, [label[e] for e in order])
    ax.set_xlim(0, 100)
    n_pairs = int(s["n_pairs"].iloc[0])
    ax.set_xlabel(f"% of enzyme–marker–cancer pairs (n = {n_pairs} each)")
    ax.set_title(f"(a) Positive association with acidity genes\n"
                 f"light = FDR<{FDR_THRESHOLD} & ρ>0;  dark = FDR<{FDR_THRESHOLD} & ρ≥{STRONG_RHO}",
                 loc="left", fontsize=10)
    ax.grid(axis="x", alpha=0.25)

    # (b) enzyme x marker
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(mk.values, cmap=CMAP, vmin=-VMAX, vmax=VMAX, aspect="auto")
    lab = dict(MARKERS)
    ax2.set_xticks(range(len(mk_codes)), [lab[m] for m in mk_codes], fontsize=8)
    ax2.set_yticks(range(len(order)), [label[e] for e in order])
    for i in range(mk.shape[0]):
        for j in range(mk.shape[1]):
            ax2.text(j, i, f"{mk.values[i, j]:.2f}", ha="center", va="center",
                     fontsize=7.5,
                     color="white" if abs(mk.values[i, j]) > 0.7 * VMAX else "#111827")
    ax2.set_title(f"(b) Median purity-adjusted ρ across {n_ct} cancer types",
                  loc="left", fontsize=10)
    plt.colorbar(im, ax=ax2, fraction=0.04, pad=0.02, label="partial Spearman ρ")

    # (c) enzyme x cancer
    ax3 = fig.add_subplot(gs[1, :])
    im3 = ax3.imshow(pc.values, cmap=CMAP, vmin=-VMAX, vmax=VMAX, aspect="auto")
    ax3.set_xticks(range(pc.shape[1]), pc.columns, rotation=60, fontsize=8)
    ax3.set_yticks(range(len(order)), [label[e] for e in order])
    ax3.set_title(f"(c) Mean ρ over the {len(mk_codes)} acidity genes, per cancer type "
                  f"(purity-adjusted; excluded: {', '.join(EXCLUDE_CANCERS)})",
                  loc="left", fontsize=10)
    plt.colorbar(im3, ax=ax3, fraction=0.02, pad=0.01, label="mean ρ")
    if any(e in NON_PROTEASE for e in order):
        note = "; ".join(f"† {e} = {r}, not a protease"
                         for e, r in NON_PROTEASE.items() if e in order)
        fig.text(0.01, 0.005, note, fontsize=8, color="#4b5563")

    fig.savefig(os.path.join(args.out_dir, FIG_NAME + ".png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(os.path.join(args.out_dir, FIG_NAME + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    phase_done("figure (PNG + PDF)", t)

    # --------------------------------------------------------------- report
    print("\n  Ranking (top -> bottom):")
    print(s[["pct_sig_pos", "pct_strong", "median_rho",
             "mean_rank_across_cancers", "n_cancers_ranked_first"]].round(2).to_string())
    tot = time.time() - t_all
    print(f"\n  Outputs written to ./{args.out_dir}/")
    print(f"  TOTAL WALL TIME: {tot:.2f} s ({tot / 60:.2f} min)")


if __name__ == "__main__":
    main()
