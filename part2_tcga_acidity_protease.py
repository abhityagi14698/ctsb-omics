# -*- coding: utf-8 -*-
"""
===============================================================================
 PART 2 - MAIN EXPERIMENT
 Which tumour protease is most consistently associated with an acidic
 (glycolytic / acid-exporting) tumour programme?   TCGA Pan-Cancer, RNA-seq
===============================================================================

 WHAT THIS SCRIPT DOES
 ---------------------
   1. Reads the TCGA phenotype file  -> cancer type of every sample
   2. Streams the big expression file -> keeps only the genes we need and
      only PRIMARY TUMOUR samples (barcode type "01")
   3. Drops cancer types listed in EXCLUDE_CANCERS and those with n < MIN_N
   4. Builds one ACIDITY SCORE per patient:
        z-score each acidity gene WITHIN its cancer type, then average them
   5. For every cancer type x protease:
        Spearman rho (protease vs acidity score)            -> "raw"
        partial Spearman rho, controlling tumour purity     -> "purity-adjusted"
        (only if a purity file is given)
   6. Benjamini-Hochberg FDR across all tests of each analysis
   7. Sensitivity checks (same analysis, different acidity score):
        - score without CA9 (hypoxia-dominated marker)
        - any extra gene list placed in data/signatures/*.txt
          (e.g. MSigDB HALLMARK_GLYCOLYSIS, HALLMARK_HYPOXIA)
   8. Optional: agreement with the Part-1 TIMER3 exports (--timer-dir)
   9. Writes CSV tables + one 3-panel figure (PNG 300 dpi + vector PDF)

 INPUT FILES (put them in data/)
 -------------------------------
   EXPRESSION : EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz
                (UCSC Xena, TCGA Pan-Cancer (PANCAN) hub; log2(norm_value+1))
   PHENOTYPE  : TCGA_phenotype_denseDataOnlyDownload.tsv.gz
                (sample, sample_type, _primary_disease)
   PURITY     : TCGA_mastercalls.abs_tables_JSedit.fixed.txt   (OPTIONAL)
                (ABSOLUTE purity, GDC PanCanAtlas publication page)

 USAGE (Anaconda Prompt, inside the bioinfo_ctsb folder)
 -------------------------------------------------------
   python scripts\\part2_tcga_acidity_protease.py
   python scripts\\part2_tcga_acidity_protease.py --timer-dir ..\\part1_csv
   python scripts\\part2_tcga_acidity_protease.py --no-purity
===============================================================================
"""

import os
import sys
import gzip
import glob
import time
import argparse

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
#  ####################  CONTROL PANEL  #####################################
# =============================================================================

# ---- (1) File names inside --data-dir ---------------------------------------
EXPR_FILE   = "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"
PHENO_FILE  = "TCGA_phenotype_denseDataOnlyDownload.tsv.gz"
PURITY_FILE = "TCGA_mastercalls.abs_tables_JSedit.fixed.txt"   # optional
SIGNATURE_DIR = "signatures"      # optional: MSigDB .grp or .txt, one gene per line

# ---- (2) Genes ----------------------------------------------------------------
PROTEASES = ["CTSB", "CTSD", "CTSL", "CTSK",    # cathepsins (acid-active)
             "MMP2", "MMP9", "MMP14",           # matrix metalloproteinases
             "PLAU"]                            # uPA
ACIDITY_GENES = ["CA9", "CA12", "SLC16A3", "SLC16A1",
                 "LDHA", "SLC2A1", "SLC9A1"]
SENSITIVITY_DROP = ["CA9"]        # score variant without these genes
# Old gene symbols used in the 2016 PanCanAtlas matrix -> current symbol.
# (CTSL is stored as "CTSL1" there; without this, CTSL is silently missing.)
GENE_ALIASES = {"CTSL1": "CTSL", "CTSL2": "CTSV"}

# ---- (3) Samples and cancer types --------------------------------------------
SAMPLE_TYPE_CODE = "01"           # 01 = primary solid tumour
EXCLUDE_CANCERS  = ["LAML", "DLBC", "THYM"]   # blood / lymphoid, no ECM invasion
MIN_N            = 30             # drop cancer types with fewer tumours

# ---- (4) Statistics ------------------------------------------------------------
FDR_THRESHOLD = 0.05
STRONG_RHO    = 0.30
PRIMARY       = "purity_adjusted"  # main analysis; falls back to "raw"
                                   # automatically if no purity file

# ---- (5) Figure ----------------------------------------------------------------
HIGHLIGHT = "CTSB"
HIGHLIGHT_COLOR, OTHER_COLOR = "#c2410c", "#9ca3af"
CMAP, VMAX, DPI = "RdBu_r", 0.6, 300

# ---- (5b) Extra stand-alone figures (alternative views of the robustness
#      panel). The main 3-panel figure is always made. Remove a name from
#      this list to skip that figure.
#        "dotplot"      : median rho per protease, one marker per variant
#        "bump"         : rank of each protease across variants (rank chart)
#        "grouped_bar"  : median rho, bars grouped by protease
#        "distribution" : rho in every cancer type, box + points per protease
EXTRA_FIGURES = ["dotplot", "bump", "grouped_bar", "distribution"]
VARIANT_COLORS = ["#111827", "#0072B2", "#009E73", "#E69F00", "#CC79A7",
                  "#56B4E9", "#D55E00"]          # Okabe-Ito, colour-blind safe

# ---- (6) TCGA full disease name -> abbreviation ------------------------------
DISEASE_ABBR = {
    "adrenocortical cancer": "ACC", "adrenocortical carcinoma": "ACC",
    "bladder urothelial carcinoma": "BLCA",
    "breast invasive carcinoma": "BRCA",
    "cervical & endocervical cancer": "CESC",
    "cervical squamous cell carcinoma and endocervical adenocarcinoma": "CESC",
    "cholangiocarcinoma": "CHOL",
    "colon adenocarcinoma": "COAD",
    "diffuse large b-cell lymphoma": "DLBC",
    "lymphoid neoplasm diffuse large b-cell lymphoma": "DLBC",
    "esophageal carcinoma": "ESCA",
    "glioblastoma multiforme": "GBM",
    "head & neck squamous cell carcinoma": "HNSC",
    "head and neck squamous cell carcinoma": "HNSC",
    "kidney chromophobe": "KICH",
    "kidney clear cell carcinoma": "KIRC",
    "kidney renal clear cell carcinoma": "KIRC",
    "kidney papillary cell carcinoma": "KIRP",
    "kidney renal papillary cell carcinoma": "KIRP",
    "acute myeloid leukemia": "LAML",
    "brain lower grade glioma": "LGG",
    "liver hepatocellular carcinoma": "LIHC",
    "lung adenocarcinoma": "LUAD",
    "lung squamous cell carcinoma": "LUSC",
    "mesothelioma": "MESO",
    "ovarian serous cystadenocarcinoma": "OV",
    "pancreatic adenocarcinoma": "PAAD",
    "pheochromocytoma & paraganglioma": "PCPG",
    "pheochromocytoma and paraganglioma": "PCPG",
    "prostate adenocarcinoma": "PRAD",
    "rectum adenocarcinoma": "READ",
    "sarcoma": "SARC",
    "skin cutaneous melanoma": "SKCM",
    "stomach adenocarcinoma": "STAD",
    "testicular germ cell tumor": "TGCT",
    "testicular germ cell tumors": "TGCT",
    "thyroid carcinoma": "THCA",
    "thymoma": "THYM",
    "uterine corpus endometrioid carcinoma": "UCEC",
    "uterine corpus endometrial carcinoma": "UCEC",
    "uterine carcinosarcoma": "UCS",
    "uveal melanoma": "UVM",
}
# =============================================================================
#  ####################  END OF CONTROL PANEL  ##############################
# =============================================================================

T_ALL = time.time()


def fmt_t(sec):
    sec = max(0.0, sec)
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def bar(frac, width=24):
    n = int(width * min(max(frac, 0.0), 1.0))
    return "#" * n + "-" * (width - n)


def phase(name):
    """Context-style phase timer: call once to start, returned fn to finish."""
    print(f"\n[{fmt_t(time.time() - T_ALL)}] >>> {name}")
    t0 = time.time()

    def done(extra=""):
        print(f"[{fmt_t(time.time() - T_ALL)}] <<< {name} finished in "
              f"{time.time() - t0:.1f} s {extra}")
    return done


def safe_write_csv(df, path, index=True):
    """CSV written and forced to disk (flush + fsync): crash-safe."""
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        df.to_csv(fh, index=index)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (NaN-safe)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = ok.sum()
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.minimum(ranked, 1.0)
    out[ok] = res
    return out


def spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan, np.nan, int(ok.sum())
    r, p = stats.spearmanr(x[ok], y[ok])
    return float(r), float(p), int(ok.sum())


def partial_spearman(x, y, z):
    """Spearman correlation of x and y controlling for z
       (rank-transform, linear residuals on rank(z), Pearson of residuals)."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    n = int(ok.sum())
    if n < 10:
        return np.nan, np.nan, n
    rx, ry, rz = (stats.rankdata(v[ok]) for v in (x, y, z))
    A = np.column_stack([np.ones(n), rz])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    if ex.std() == 0 or ey.std() == 0:
        return np.nan, np.nan, n
    r = float(np.corrcoef(ex, ey)[0, 1])
    df = n - 3
    t = r * np.sqrt(df / max(1e-12, 1 - r * r))
    p = float(2 * stats.t.sf(abs(t), df))
    return r, p, n


# =============================================================================
#  LOADERS
# =============================================================================
def open_text(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def load_phenotype(path):
    ph = pd.read_csv(path, sep="\t", compression="infer", dtype=str)
    id_col = ph.columns[0]
    abbr_col = next((c for c in ph.columns if "abbreviation" in c.lower()), None)
    dis_col = next((c for c in ph.columns if "primary_disease" in c.lower()
                    or c.lower() in ("primary disease", "cancer type")), None)
    ph["sid"] = ph[id_col].str[:15]
    if abbr_col is not None:
        ph["cancer"] = ph[abbr_col].str.strip().str.upper()
    elif dis_col is not None:
        names = ph[dis_col].str.strip().str.lower()
        ph["cancer"] = names.map(DISEASE_ABBR)
        unknown = sorted(names[ph["cancer"].isna()].dropna().unique())
        if unknown:
            print(f"  [warn] unmapped disease names (kept as-is): {unknown}")
        ph["cancer"] = ph["cancer"].fillna(ph[dis_col].str.strip())
    else:
        sys.exit(f"[FATAL] no disease column found in {path}: {list(ph.columns)}")
    ph = ph.drop_duplicates("sid").set_index("sid")
    return ph["cancer"]


def load_purity(path):
    pu = pd.read_csv(path, sep="\t", dtype=str)
    pcol = next((c for c in pu.columns if c.lower() == "purity"), None)
    scol = next((c for c in pu.columns
                 if pu[c].astype(str).str.startswith("TCGA-").mean() > 0.9), None)
    if pcol is None or scol is None:
        print(f"  [warn] purity file not understood ({list(pu.columns)}); skipped")
        return None
    pu["sid"] = pu[scol].str[:15]
    pu["purity"] = pd.to_numeric(pu[pcol], errors="coerce")
    return pu.dropna(subset=["purity"]).groupby("sid")["purity"].mean()


def stream_expression(path, wanted_genes, keep_code):
    """Stream the large genes x samples matrix, keeping only wanted genes and
       samples whose barcode type equals keep_code. Live progress with ETA."""
    size = os.path.getsize(path)
    raw = open(path, "rb")
    fh = gzip.GzipFile(fileobj=raw) if path.endswith(".gz") else raw
    header = fh.readline().decode().rstrip("\n").split("\t")
    samples = [s[:15] for s in header[1:]]
    keep_idx = [i for i, s in enumerate(samples) if s[13:15] == keep_code]
    print(f"  samples in file: {len(samples):,};  type '{keep_code}': {len(keep_idx):,}")
    wanted = set(wanted_genes)
    alias_hits = {}
    rows, names = [], []
    t0, n_lines, last = time.time(), 0, 0.0
    for line in fh:
        n_lines += 1
        line = line.decode()
        tab = line.find("\t")
        gene = line[:tab].split("|")[0]
        if gene in GENE_ALIASES and GENE_ALIASES[gene] in wanted:
            alias_hits[gene] = GENE_ALIASES[gene]
            gene = GENE_ALIASES[gene]
        if gene in wanted and gene not in names:
            vals = np.array(line.rstrip("\n").split("\t")[1:], dtype=object)[keep_idx]
            rows.append(pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy())
            names.append(gene)
        now = time.time()
        if now - last > 0.5:
            frac = raw.tell() / size
            el = now - t0
            eta = el * (1 - frac) / frac if frac > 1e-6 else float("nan")
            sys.stdout.write(f"\r  reading [{bar(frac)}] {100*frac:5.1f}%  "
                             f"lines {n_lines:,}  genes found {len(names)}/{len(wanted)}"
                             f"  elapsed {fmt_t(el)}  ETA {fmt_t(eta) if eta == eta else '--:--'} ")
            sys.stdout.flush()
            last = now
    sys.stdout.write(f"\r  reading [{bar(1)}] 100.0%  lines {n_lines:,}  "
                     f"genes found {len(names)}/{len(wanted)}  "
                     f"elapsed {fmt_t(time.time() - t0)}  ETA 00:00   \n")
    fh.close()
    raw.close()
    for old, new in alias_hits.items():
        print(f"  [alias] '{old}' in file used as '{new}'")
    kept_samples = [samples[i] for i in keep_idx]
    df = pd.DataFrame(np.vstack(rows).T if rows else np.empty((len(kept_samples), 0)),
                      index=kept_samples, columns=names)
    df = df.groupby(level=0).mean()          # merge duplicate aliquots
    missing = sorted(wanted - set(names))
    return df, missing


# =============================================================================
#  ANALYSIS
# =============================================================================
def acidity_score(expr, cancer, genes):
    """mean of within-cancer z-scores of the given genes (per sample)."""
    genes = [g for g in genes if g in expr.columns]
    z = expr[genes].groupby(cancer).transform(
        lambda c: (c - c.mean()) / c.std(ddof=0) if c.std(ddof=0) > 0 else c * np.nan)
    return z.mean(axis=1, skipna=True), genes


def correlate(expr, cancer, score, purity, label, progress_state):
    out = []
    cts = sorted(cancer.unique())
    for ct in cts:
        idx = cancer.index[cancer == ct]
        s = score.loc[idx].to_numpy(float)
        pur = purity.reindex(idx).to_numpy(float) if purity is not None else None
        for prot in PROTEASES:
            if prot not in expr.columns:
                continue
            x = expr.loc[idx, prot].to_numpy(float)
            r, p, n = spearman(x, s)
            rec = dict(score=label, cancer=ct, protease=prot, n=n, rho=r, p=p)
            if pur is not None:
                rp, pp, npur = partial_spearman(x, s, pur)
                rec.update(rho_purity_adj=rp, p_purity_adj=pp, n_purity=npur)
            out.append(rec)
            progress_state["done"] += 1
            d, tot = progress_state["done"], progress_state["total"]
            el = time.time() - progress_state["t0"]
            eta = el * (tot - d) / d if d else float("nan")
            sys.stdout.write(f"\r  tests [{bar(d / tot)}] {100*d/tot:5.1f}%  "
                             f"{d}/{tot}  ({label}: {ct} {prot:<6s})  "
                             f"elapsed {fmt_t(el)}  ETA {fmt_t(eta)}   ")
            sys.stdout.flush()
    res = pd.DataFrame(out)
    res["fdr"] = bh_fdr(res["p"])
    if "p_purity_adj" in res:
        res["fdr_purity_adj"] = bh_fdr(res["p_purity_adj"])
    return res


def summarise(res, rho_col, fdr_col):
    g = res.assign(sig_pos=(res[fdr_col] < FDR_THRESHOLD) & (res[rho_col] > 0),
                   sig_neg=(res[fdr_col] < FDR_THRESHOLD) & (res[rho_col] < 0),
                   strong=(res[fdr_col] < FDR_THRESHOLD) & (res[rho_col] >= STRONG_RHO))
    s = g.groupby("protease").agg(n_cancers=("cancer", "size"),
                                  sig_pos=("sig_pos", "sum"),
                                  sig_neg=("sig_neg", "sum"),
                                  strong=("strong", "sum"),
                                  median_rho=(rho_col, "median"),
                                  mean_rho=(rho_col, "mean"))
    wide = res.pivot(index="cancer", columns="protease", values=rho_col)
    ranks = wide.rank(axis=1, ascending=False)
    s["mean_rank"] = ranks.mean()
    s["n_cancers_ranked_first"] = (ranks == 1).sum()
    s["pct_sig_pos"] = 100 * s["sig_pos"] / s["n_cancers"]
    s["pct_strong"] = 100 * s["strong"] / s["n_cancers"]
    return s.sort_values("median_rho", ascending=False)


def timer_agreement(timer_dir, expr, cancer, purity):
    """Compare per-marker rho from Part 1 (TIMER3) with the same pairs here."""
    files = glob.glob(os.path.join(timer_dir, "genecorr_table_*.csv"))
    rows = []
    for f in files:
        prot = os.path.basename(f)[len("genecorr_table_"):-4]
        if prot not in PROTEASES or prot not in expr.columns:
            continue
        t = pd.read_csv(f, na_values=["NA"]).dropna(subset=["rho"])
        t["cancer"] = t["cancer"].str.split(" ").str[0]
        t = t[~t["cancer"].str.contains("-")]
        for _, r in t.iterrows():
            ct, mk = r["cancer"], r["gene"]
            if ct not in set(cancer) or mk not in expr.columns:
                continue
            idx = cancer.index[cancer == ct]
            x = expr.loc[idx, prot].to_numpy(float)
            y = expr.loc[idx, mk].to_numpy(float)
            if purity is not None:
                ours, _, _ = partial_spearman(x, y, purity.reindex(idx).to_numpy(float))
            else:
                ours, _, _ = spearman(x, y)
            rows.append(dict(protease=prot, cancer=ct, marker=mk,
                             rho_timer3=r["rho"], rho_part2=ours))
    return pd.DataFrame(rows)


# =============================================================================
#  FIGURE
# =============================================================================
def make_figure(summary, main, rho_col, fdr_col, sens, out_png, title_note):
    order = summary.index.tolist()
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.1], height_ratios=[1, 1.1],
                          hspace=0.55, wspace=0.28)

    # (a) ranking
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(order))[::-1]
    cols = [HIGHLIGHT_COLOR if e == HIGHLIGHT else OTHER_COLOR for e in order]
    ax.barh(y, summary["pct_sig_pos"], color=cols, height=0.62, alpha=0.45)
    ax.barh(y, summary["pct_strong"], color=cols, height=0.62)
    for yi, e in zip(y, order):
        ax.text(summary.loc[e, "pct_sig_pos"] + 1, yi,
                f"{summary.loc[e, 'pct_sig_pos']:.0f}%  (median ρ {summary.loc[e, 'median_rho']:.2f})",
                va="center", fontsize=8, color="#374151")
    ax.set_yticks(y, order)
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel(f"% of cancer types (n = {int(summary['n_cancers'].iloc[0])})")
    ax.set_title(f"(a) Protease vs acidity score\nlight = FDR<{FDR_THRESHOLD} & ρ>0;"
                 f"  dark = FDR<{FDR_THRESHOLD} & ρ≥{STRONG_RHO}", loc="left", fontsize=10)
    ax.grid(axis="x", alpha=0.25)

    # (b) robustness table: median rho (colour + number) and rank under each
    #     acidity score / adjustment.  A table cannot overlap, however many
    #     variants there are.
    ax2 = fig.add_subplot(gs[0, 1])
    short = {}
    for v in sens.columns:
        lab = (v.replace("main (purity-adj.)", "Main\n(purity-adj.)")
                .replace("main (raw, no purity adj.)", "No purity\nadjustment")
                .replace("main (raw)", "Main\n(raw)")
                .replace("without ", "Without\n")
                .replace("HALLMARK_", "Hallmark\n").replace("_", " "))
        short[v] = lab.title() if lab.isupper() else lab
    vals = sens.loc[order].to_numpy(float)
    ranks = sens.loc[order].rank(ascending=False, method="min").to_numpy(int)
    ax2.imshow(vals, cmap=CMAP, vmin=-VMAX, vmax=VMAX, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            dark = abs(vals[i, j]) > 0.55 * VMAX
            ax2.text(j, i, f"{vals[i, j]:.2f}\n#{ranks[i, j]}", ha="center", va="center",
                     fontsize=7.5, linespacing=1.15,
                     color="white" if dark else "#111827",
                     fontweight="bold" if order[i] == HIGHLIGHT else "normal")
    ax2.set_xticks(range(vals.shape[1]), [short[v] for v in sens.columns], fontsize=7.5)
    ax2.xaxis.tick_top()
    ax2.tick_params(axis="x", length=0, pad=4)
    ax2.set_yticks(range(len(order)), order)
    for t in ax2.get_yticklabels():
        if t.get_text() == HIGHLIGHT:
            t.set_color(HIGHLIGHT_COLOR); t.set_fontweight("bold")
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.set_xticks(np.arange(-0.5, vals.shape[1]), minor=True)
    ax2.set_yticks(np.arange(-0.5, vals.shape[0]), minor=True)
    ax2.grid(which="minor", color="white", linewidth=2)
    ax2.tick_params(which="minor", length=0)
    ax2.set_title("(b) Robustness: median ρ and rank (#) under each acidity score",
                  loc="left", fontsize=10, pad=34)

    # (c) heatmap protease x cancer with significance stars
    ax3 = fig.add_subplot(gs[1, :])
    wide = main.pivot(index="protease", columns="cancer", values=rho_col).loc[order]
    sig = main.pivot(index="protease", columns="cancer", values=fdr_col).loc[order]
    wide = wide[wide.mean().sort_values(ascending=False).index]
    sig = sig[wide.columns]
    im = ax3.imshow(wide.values, cmap=CMAP, vmin=-VMAX, vmax=VMAX, aspect="auto")
    for i in range(wide.shape[0]):
        for j in range(wide.shape[1]):
            if sig.values[i, j] < FDR_THRESHOLD:
                ax3.text(j, i, "*", ha="center", va="center", fontsize=9,
                         color="white" if abs(wide.values[i, j]) > 0.6 * VMAX else "#111827")
    ax3.set_xticks(range(wide.shape[1]), wide.columns, rotation=60, fontsize=8)
    ax3.set_yticks(range(len(order)), order)
    ax3.set_title(f"(c) ρ (protease vs acidity score) per cancer type; * FDR<{FDR_THRESHOLD}"
                  f"  [{title_note}]", loc="left", fontsize=10)
    plt.colorbar(im, ax=ax3, fraction=0.02, pad=0.01, label="Spearman ρ")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_png[:-4] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _variant_labels(cols):
    out = {}
    for v in cols:
        lab = (v.replace("main (purity-adj.)", "Main (purity-adj.)")
                .replace("main (raw, no purity adj.)", "No purity adjustment")
                .replace("main (raw)", "Main (raw)")
                .replace("without ", "Without ")
                .replace("HALLMARK_", "Hallmark ").replace("_", " "))
        out[v] = lab
    return out


def _save(fig, path):
    fig.savefig(path + ".png", dpi=DPI, bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    plt.close(fig)


def make_extra_figures(summary, main, rho_col, fdr_col, sens, outdir):
    """Alternative graphical views of the robustness result (panel b)."""
    order = summary.index.tolist()
    variants = list(sens.columns)
    lab = _variant_labels(variants)
    cols = {v: VARIANT_COLORS[k % len(VARIANT_COLORS)] for k, v in enumerate(variants)}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    made = []

    # ---- 1. dot plot: one row per protease, variants dodged vertically ----
    if "dotplot" in EXTRA_FIGURES:
        fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(order) + 1.6))
        y = np.arange(len(order))[::-1]
        nv = len(variants)
        offs = np.linspace(-0.28, 0.28, nv) if nv > 1 else [0.0]
        mk = ["o", "s", "^", "D", "v", "P", "X"]
        for yi, e in zip(y, order):
            if e == HIGHLIGHT:
                ax.axhspan(yi - 0.45, yi + 0.45, color=HIGHLIGHT_COLOR, alpha=0.08, lw=0)
            ax.plot([sens.loc[e].min(), sens.loc[e].max()], [yi, yi],
                    color="#d1d5db", lw=1.5, zorder=1)
        for k, v in enumerate(variants):
            ax.scatter(sens.loc[order, v], y + offs[k], marker=mk[k % len(mk)], s=36,
                       color=cols[v], edgecolors="white", linewidths=0.6,
                       label=lab[v], zorder=3)
        ax.axvline(0, color="#9ca3af", lw=1)
        ax.set_yticks(y, order)
        for t in ax.get_yticklabels():
            if t.get_text() == HIGHLIGHT:
                t.set_color(HIGHLIGHT_COLOR); t.set_fontweight("bold")
        ax.set_xlabel("median ρ across cancer types (protease vs acidity score)")
        ax.set_title("Robustness of the protease ranking", loc="left", fontsize=10)
        ax.grid(axis="x", alpha=0.25)
        ax.legend(fontsize=8, frameon=False, ncol=min(3, nv),
                  loc="upper center", bbox_to_anchor=(0.5, -0.12))
        _save(fig, os.path.join(outdir, "extra_robustness_dotplot")); made.append("dotplot")

    # ---- 2. bump chart: rank of each protease under every variant --------
    if "bump" in EXTRA_FIGURES and len(variants) > 1:
        rk = sens.rank(ascending=False, method="min")
        fig, ax = plt.subplots(figsize=(1.45 * len(variants) + 3.2, 0.5 * len(order) + 1.6))
        x = np.arange(len(variants))
        for e in order[::-1]:
            hi = e == HIGHLIGHT
            ax.plot(x, rk.loc[e], color=HIGHLIGHT_COLOR if hi else OTHER_COLOR,
                    lw=3 if hi else 1.6, marker="o", ms=9 if hi else 7,
                    markeredgecolor="white", zorder=3 if hi else 2)
            for side, xi in (("left", x[0] - 0.12), ("right", x[-1] + 0.12)):
                col = rk.columns[0] if side == "left" else rk.columns[-1]
                ax.text(xi, rk.loc[e, col], e, ha="right" if side == "left" else "left",
                        va="center", fontsize=8.5,
                        color=HIGHLIGHT_COLOR if hi else "#374151",
                        fontweight="bold" if hi else "normal")
        ax.set_xticks(x, [lab[v].replace(" (", "\n(").replace("Hallmark ", "Hallmark\n")
                          .replace("No purity ", "No purity\n").replace("Without ", "Without\n")
                          for v in variants], fontsize=8)
        ax.set_yticks(range(1, len(order) + 1), [f"#{i}" for i in range(1, len(order) + 1)])
        ax.set_ylim(len(order) + 0.6, 0.4)
        ax.set_xlim(-0.9, len(variants) - 0.1)
        ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
        ax.tick_params(length=0)
        ax.grid(axis="x", alpha=0.25)
        ax.set_ylabel("rank (by median ρ)")
        ax.set_title("Rank of each protease under different acidity scores",
                     loc="left", fontsize=10)
        _save(fig, os.path.join(outdir, "extra_robustness_rank_chart")); made.append("bump")

    # ---- 3. grouped bar chart ---------------------------------------------
    if "grouped_bar" in EXTRA_FIGURES:
        nv = len(variants)
        fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(order) + 2), 4.6))
        x = np.arange(len(order))
        w = 0.8 / nv
        for k, v in enumerate(variants):
            ax.bar(x - 0.4 + w * (k + 0.5), sens.loc[order, v], width=w * 0.88,
                   color=cols[v], label=lab[v], edgecolor="white", linewidth=0.5)
        if HIGHLIGHT in order:
            i = order.index(HIGHLIGHT)
            ax.axvspan(i - 0.48, i + 0.48, color=HIGHLIGHT_COLOR, alpha=0.08, lw=0, zorder=0)
        ax.axhline(0, color="#6b7280", lw=1)
        ax.set_xticks(x, order)
        for t in ax.get_xticklabels():
            if t.get_text() == HIGHLIGHT:
                t.set_color(HIGHLIGHT_COLOR); t.set_fontweight("bold")
        ax.set_ylabel("median ρ across cancer types")
        ax.set_title("Median association with the acidity score under each variant",
                     loc="left", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, frameon=False, ncol=min(3, nv),
                  loc="upper center", bbox_to_anchor=(0.5, -0.1))
        _save(fig, os.path.join(outdir, "extra_robustness_grouped_bars")); made.append("grouped_bar")

    # ---- 4. distribution: rho in every cancer type -------------------------
    if "distribution" in EXTRA_FIGURES:
        fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(order) + 2), 4.8))
        data = [main.loc[main["protease"] == e, rho_col].dropna().to_numpy() for e in order]
        sigs = [main.loc[main["protease"] == e, fdr_col].to_numpy() < FDR_THRESHOLD
                for e in order]
        bp = ax.boxplot(data, positions=np.arange(len(order)), widths=0.55,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="#111827", lw=1.8))
        for k, (b, e) in enumerate(zip(bp["boxes"], order)):
            b.set_facecolor(HIGHLIGHT_COLOR if e == HIGHLIGHT else "#e5e7eb")
            b.set_alpha(0.35 if e == HIGHLIGHT else 1.0)
            b.set_edgecolor("#6b7280")
        rng = np.random.default_rng(0)
        for k, (d, sg, e) in enumerate(zip(data, sigs, order)):
            xs = k + rng.uniform(-0.17, 0.17, len(d))
            c = HIGHLIGHT_COLOR if e == HIGHLIGHT else "#374151"
            ax.scatter(xs[sg], d[sg], s=18, color=c, edgecolors="white",
                       linewidths=0.4, zorder=3)
            ax.scatter(xs[~sg], d[~sg], s=18, facecolors="white", edgecolors=c,
                       linewidths=0.8, zorder=3)
        ax.scatter([], [], s=18, color="#374151", label=f"FDR < {FDR_THRESHOLD}")
        ax.scatter([], [], s=18, facecolors="white", edgecolors="#374151",
                   label="not significant")
        ax.axhline(0, color="#9ca3af", lw=1)
        ax.axhline(STRONG_RHO, color="#9ca3af", lw=1, ls=":")
        ax.text(len(order) - 0.5, STRONG_RHO, f" ρ = {STRONG_RHO}", va="bottom",
                ha="right", fontsize=7.5, color="#6b7280")
        ax.set_xticks(np.arange(len(order)), order)
        for t in ax.get_xticklabels():
            if t.get_text() == HIGHLIGHT:
                t.set_color(HIGHLIGHT_COLOR); t.set_fontweight("bold")
        n_ct = main["cancer"].nunique()
        ax.set_ylabel("ρ (protease vs acidity score)")
        ax.set_title(f"Association in each of {n_ct} cancer types (one point = one cancer type)",
                     loc="left", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
        _save(fig, os.path.join(outdir, "extra_per_cancer_distribution")); made.append("distribution")
    return made


# =============================================================================
#  MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--timer-dir", default=None, help="folder with Part-1 TIMER3 CSVs")
    ap.add_argument("--no-purity", action="store_true")
    a = ap.parse_args()
    for d in (a.results_dir, a.figures_dir):
        os.makedirs(d, exist_ok=True)
    print("=" * 78)
    print(" PART 2: protease vs acidity score, TCGA Pan-Cancer")
    print("=" * 78)

    # ---------------------------------------------------------------- inputs
    done = phase("1/8 phenotype and purity")
    pheno = load_phenotype(os.path.join(a.data_dir, PHENO_FILE))
    purity = None
    pur_path = os.path.join(a.data_dir, PURITY_FILE)
    if not a.no_purity and os.path.exists(pur_path):
        purity = load_purity(pur_path)
    print(f"  phenotype samples: {len(pheno):,};  purity values: "
          f"{0 if purity is None else len(purity):,}")
    if purity is None:
        print("  [info] no purity file -> raw Spearman is the primary analysis")
    sig_files = sorted(set(
        glob.glob(os.path.join(a.data_dir, SIGNATURE_DIR, "*.txt")) +
        glob.glob(os.path.join(a.data_dir, SIGNATURE_DIR, "*.grp")) +
        glob.glob(os.path.join(a.data_dir, "*.grp"))))      # .grp left in data/ also works
    signatures = {}
    for f in sig_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        name = stem.split(".")[0]          # HALLMARK_GLYCOLYSIS.v2026.2.Hs -> HALLMARK_GLYCOLYSIS
        genes = [l.strip() for l in open(f)
                 if l.strip() and l.strip()[0] not in "#>"]
        # MSigDB .grp files start with the set name: drop it and any non-gene line
        genes = [GENE_ALIASES.get(g, g) for g in genes]
        genes = [g for g in genes if g not in (name, stem) and "_" not in g
                 and " " not in g and g not in PROTEASES]
        signatures[name] = genes
    if signatures:
        print(f"  extra signatures: { {k: len(v) for k, v in signatures.items()} }")
    else:
        print("  [info] no .grp/.txt signatures found -> Hallmark robustness check skipped")
    done()

    # ------------------------------------------------------------ expression
    done = phase("2/8 stream expression file (largest step)")
    wanted = set(PROTEASES) | set(ACIDITY_GENES)
    for g in signatures.values():
        wanted |= set(g)
    expr, missing = stream_expression(os.path.join(a.data_dir, EXPR_FILE),
                                      wanted, SAMPLE_TYPE_CODE)
    if missing:
        print(f"  [warn] genes not found: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
    done(f"({expr.shape[0]:,} tumours x {expr.shape[1]} genes)")

    # ------------------------------------------------------------ filtering
    done = phase("3/8 attach cancer type and filter")
    cancer = pheno.reindex(expr.index)
    n_unmatched = int(cancer.isna().sum())
    keep = cancer.notna() & ~cancer.isin(EXCLUDE_CANCERS)
    cancer = cancer[keep]
    counts = cancer.value_counts()
    small = sorted(counts[counts < MIN_N].index)
    cancer = cancer[~cancer.isin(small)]
    expr = expr.loc[cancer.index]
    print(f"  unmatched samples dropped: {n_unmatched};  excluded: {EXCLUDE_CANCERS};"
          f"  n<{MIN_N}: {small}")
    print(f"  final: {len(cancer):,} tumours in {cancer.nunique()} cancer types")
    n_table = cancer.value_counts().rename("n_tumours").to_frame()
    if purity is not None:
        n_table["n_with_purity"] = purity.reindex(cancer.index).notna().groupby(cancer).sum()
    safe_write_csv(n_table.sort_index(), os.path.join(a.results_dir, "sample_counts.csv"))
    done()

    # -------------------------------------------------------------- scores
    done = phase("4/8 acidity scores")
    score_sets = {"acidity score (7 genes)": ACIDITY_GENES,
                  f"without {'+'.join(SENSITIVITY_DROP)}":
                      [g for g in ACIDITY_GENES if g not in SENSITIVITY_DROP]}
    for k, v in signatures.items():
        score_sets[k] = v
    scores, used = {}, {}
    for k, genes in score_sets.items():
        scores[k], used[k] = acidity_score(expr, cancer, genes)
        print(f"  {k:<32s} genes used: {len(used[k])}")
    per_sample = expr[[p for p in PROTEASES if p in expr.columns]].copy()
    per_sample.insert(0, "cancer", cancer)
    if purity is not None:
        per_sample.insert(1, "purity", purity.reindex(per_sample.index))
    for k, s in scores.items():
        per_sample[f"score::{k}"] = s
    safe_write_csv(per_sample, os.path.join(a.results_dir, "per_sample_scores.csv"))
    done()

    # --------------------------------------------------------- correlations
    done = phase("5/8 correlations (all scores x cancers x proteases)")
    state = dict(done=0, t0=time.time(),
                 total=len(scores) * cancer.nunique() * len(PROTEASES))
    results = {k: correlate(expr, cancer, s, purity, k, state) for k, s in scores.items()}
    print()
    main_key = "acidity score (7 genes)"
    main_res = results[main_key]
    safe_write_csv(main_res, os.path.join(a.results_dir, "correlation_table.csv"), index=False)
    safe_write_csv(pd.concat(results.values(), ignore_index=True),
                   os.path.join(a.results_dir, "correlation_table_all_scores.csv"), index=False)
    done()

    # ------------------------------------------------------------- summary
    done = phase("6/8 ranking and sensitivity")
    use_pur = purity is not None and PRIMARY == "purity_adjusted"
    rho_col = "rho_purity_adj" if use_pur else "rho"
    fdr_col = "fdr_purity_adj" if use_pur else "fdr"
    summary = summarise(main_res, rho_col, fdr_col)
    safe_write_csv(summary.round(4), os.path.join(a.results_dir, "protease_ranking.csv"))
    sens = {}
    sens["main (" + ("purity-adj." if use_pur else "raw") + ")"] = \
        main_res.groupby("protease")[rho_col].median()
    if use_pur:
        sens["main (raw, no purity adj.)"] = main_res.groupby("protease")["rho"].median()
    for k, r in results.items():
        if k == main_key:
            continue
        sens[k] = r.groupby("protease")[rho_col].median()
    sens = pd.DataFrame(sens).loc[summary.index]
    safe_write_csv(sens.round(4), os.path.join(a.results_dir, "sensitivity_median_rho.csv"))
    # paired test: is HIGHLIGHT's rho different from each other protease,
    # comparing the two proteases cancer type by cancer type (Wilcoxon
    # signed-rank, two-sided), Holm-corrected across the comparisons.
    wide_main = main_res.pivot(index="cancer", columns="protease", values=rho_col)
    pw = []
    if HIGHLIGHT in wide_main.columns:
        for other in [c for c in wide_main.columns if c != HIGHLIGHT]:
            pair = wide_main[[HIGHLIGHT, other]].dropna()
            diff = pair[HIGHLIGHT] - pair[other]
            pval = stats.wilcoxon(pair[HIGHLIGHT], pair[other]).pvalue
            pw.append(dict(comparison=f"{HIGHLIGHT} vs {other}", n_cancers=len(pair),
                           wins=int((diff > 0).sum()), median_difference=diff.median(),
                           p_wilcoxon=pval))
        pw = pd.DataFrame(pw).sort_values("p_wilcoxon").reset_index(drop=True)
        m = len(pw)
        holm = np.maximum.accumulate([min(1.0, (m - i) * p) for i, p in enumerate(pw["p_wilcoxon"])])
        pw["p_holm"] = holm
        pw["significant_after_holm"] = pw["p_holm"] < FDR_THRESHOLD
        safe_write_csv(pw.round(6), os.path.join(a.results_dir, f"pairwise_{HIGHLIGHT}_vs_others.csv"),
                       index=False)
        print(f"\n  Paired comparison {HIGHLIGHT} vs others (Wilcoxon over cancer types, Holm):")
        print(pw[["comparison", "wins", "median_difference", "p_wilcoxon", "p_holm"]]
              .round(4).to_string(index=False))
    rank_tab = sens.rank(ascending=False).astype(int)
    safe_write_csv(rank_tab, os.path.join(a.results_dir, "sensitivity_ranks.csv"))
    print("\n  Ranking (" + ("purity-adjusted" if use_pur else "raw") + "):")
    print(summary[["pct_sig_pos", "pct_strong", "median_rho", "mean_rank",
                   "n_cancers_ranked_first"]].round(2).to_string())
    print("\n  Rank of each protease under every variant:")
    print(rank_tab.to_string())
    if a.timer_dir:
        agr = timer_agreement(a.timer_dir, expr, cancer, purity)
        if len(agr):
            r, _ = stats.spearmanr(agr["rho_timer3"], agr["rho_part2"])
            print(f"\n  Agreement with Part 1 (TIMER3): {len(agr)} pairs, "
                  f"Spearman of rho values = {r:.3f}, "
                  f"median |difference| = {np.median(np.abs(agr.rho_timer3 - agr.rho_part2)):.3f}")
            safe_write_csv(agr, os.path.join(a.results_dir, "agreement_with_timer3.csv"),
                           index=False)
    done()

    # -------------------------------------------------------------- figure
    done = phase("7/8 main figure")
    note = ("purity-adjusted partial Spearman" if use_pur else "Spearman, no purity adjustment")
    make_figure(summary, main_res, rho_col, fdr_col, sens,
                os.path.join(a.figures_dir, "part2_protease_acidity.png"), note)
    done()

    done = phase("8/8 extra figures")
    made = make_extra_figures(summary, main_res, rho_col, fdr_col, sens, a.figures_dir)
    done(f"({', '.join(made) if made else 'none selected'})")

    tot = time.time() - T_ALL
    print("\n" + "=" * 78)
    print(f" Results : ./{a.results_dir}/    Figures : ./{a.figures_dir}/")
    print(f" TOTAL WALL TIME: {fmt_t(tot)} (mm:ss)  = {tot:.1f} s")
    print("=" * 78)


if __name__ == "__main__":
    main()
