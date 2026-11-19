#!/usr/bin/env python3
"""
Part 3 -- CPTAC proteomics check of the TCGA acidity-protease result.

Question: does the TCGA (mRNA) finding hold at the PROTEIN level?
  A (main) : protease PROTEIN  vs  RNA acidity score (same 7 genes as TCGA)
  B        : protease PROTEIN  vs  PROTEIN acidity score (detected acidity proteins only)
  C        : protease mRNA vs its own protein (how well RNA reflects protein)

Data: CPTAC harmonised data through the `cptac` Python package (v1.5.x)
      proteomics  = source 'umich'  (log2 TMT ratio)
      RNA         = source 'washu'
      purity      = source 'washu'  (ESTIMATE, from RNA)  -- used for adjustment if available

Run:
  python3 part3_cptac_protein_check.py            # real CPTAC data (downloads on first run)
  python3 part3_cptac_protein_check.py --demo     # synthetic data, to test that the script runs

Outputs go to ../results_part3/ (CSV tables + figure).
"""
import argparse, os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- settings
CANCERS = ["Brca", "Ccrcc", "Coad", "Gbm", "Hnscc", "Lscc", "Luad", "Ov", "Pdac", "Ucec"]
PROTEASES = ["CTSB", "CTSL", "CTSK", "CTSD", "MMP2", "MMP9", "MMP14", "PLAU"]
ACIDITY = ["LDHA", "SLC2A1", "SLC16A3", "SLC16A1", "SLC9A1", "CA9", "CA12"]
GENE_ALIASES = {"CTSL1": "CTSL"}
PROT_SOURCE, RNA_SOURCE, PURITY_SOURCE = "umich", "washu", "washu"
RNA_FALLBACK = ["washu", "broad", "bcm"]
MIN_DETECT = 0.50     # a protein is used in a cancer only if measured in >=50% of its tumours
MIN_N = 20            # minimum tumours for a correlation
MIN_ACID_PROT = 3     # protein acidity score needs >=3 detected acidity proteins
FDR_SIG, RHO_MOD = 0.05, 0.3


# ----------------------------------------------------------------- timing helpers
T0 = time.time()
def fmt(sec):
    sec = int(sec); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

_LAST = [0.0]
def progress(i, n, start, label):
    now = time.time()
    if i not in (0, n) and now - _LAST[0] < 0.2:      # redraw at most 5 times per second
        return
    _LAST[0] = now
    el = now - start
    eta = el / i * (n - i) if i else 0
    pct = 100 * i / n
    bar = "#" * int(30 * i / n) + "-" * (30 - int(30 * i / n))
    sys.stdout.write(f"\r  {label} [{bar}] {pct:5.1f}%  {i}/{n}  elapsed {fmt(el)}  ETA {fmt(eta)}   ")
    sys.stdout.flush()
    if i == n:
        sys.stdout.write("\n")

def phase_done(name, start):
    print(f"  -> {name} finished in {fmt(time.time() - start)}  (total so far {fmt(time.time() - T0)})\n")


# ----------------------------------------------------------------- data helpers
def collapse_genes(df):
    """cptac columns are (Name, Database_ID); average isoforms to one column per gene."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.T.groupby(level=0).mean().T
    df = df.rename(columns=GENE_ALIASES)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def tumour_only(df):
    idx = df.index.astype(str)
    keep = ~(idx.str.endswith(".N") | idx.str.contains("normal", case=False))
    df = df.loc[keep]
    return df.groupby(level=0).mean()          # average any duplicated patient rows

def zscore_within(df):
    return (df - df.mean()) / df.std(ddof=1)

def purity_column(pur):
    cols = [c for c in pur.columns if "purity" in str(c).lower()]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(pur[c])]
    return num[0] if num else None

def partial_spearman(x, y, z=None):
    """Spearman of x,y; if z given, partial Spearman adjusted for z (rank-residual method)."""
    d = pd.concat([x, y] + ([z] if z is not None else []), axis=1).dropna()
    n = len(d)
    if n < MIN_N:
        return np.nan, np.nan, n
    r = d.rank()
    if z is None:
        rho, p = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
        return rho, p, n
    zc = np.column_stack([np.ones(n), r.iloc[:, 2].values])
    res = []
    for k in (0, 1):
        beta, *_ = np.linalg.lstsq(zc, r.iloc[:, k].values, rcond=None)
        res.append(r.iloc[:, k].values - zc @ beta)
    rho = np.corrcoef(res[0], res[1])[0, 1]
    t = rho * np.sqrt((n - 3) / max(1e-12, 1 - rho ** 2))
    p = 2 * stats.t.sf(abs(t), n - 3)
    return rho, p, n

def bh_fdr(p):
    p = np.asarray(p, float); out = np.full_like(p, np.nan)
    ok = ~np.isnan(p); m = ok.sum()
    if m == 0:
        return out
    ps = p[ok]; order = np.argsort(ps); ranked = ps[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    tmp = np.empty(m); tmp[order] = np.minimum(ranked, 1); out[ok] = tmp
    return out

def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); out = np.empty(m); run = 0
    for i, k in enumerate(order):
        run = max(run, min(1, (m - i) * p[k])); out[k] = run
    return out


# ----------------------------------------------------------------- loading
def patch_cptac_download():
    """cptac 1.5.x bug: get_data() crashes with
       "unsupported operand type(s) for //: 'NoneType' and 'int'"
    when it cannot find a file's size in the Zenodo record (seen for COAD and LSCC).
    Fallback: download the same URL as one plain stream (no size needed)."""
    import requests
    from cptac.tools import download_tools as dt
    if getattr(dt, "_part3_patched", False):
        return
    original = dt.get_data
    def safe_get_data(url, subfolder="", num_threads=4):
        try:
            return original(url, subfolder, num_threads)
        except TypeError:
            target = os.path.join(dt.DATA_DIR, subfolder)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            print(f"\n  [fix] plain download of {os.path.basename(subfolder)}")
            with requests.get(url, headers=dt.AUTH_HEADER, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(target, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            return subfolder
    dt.get_data = safe_get_data
    dt._part3_patched = True

NAME_ROWS = []
def record_names(cancer, raw):
    """Diagnostic: every raw proteomics column whose gene name starts with CTS or MMP or is PLAU."""
    cols = raw.columns
    names = cols.get_level_values(0) if isinstance(cols, pd.MultiIndex) else cols
    for k, nm in enumerate(names):
        nm = str(nm)
        if nm.startswith(("CTS", "MMP")) or nm == "PLAU":
            col = raw.iloc[:, k]
            NAME_ROWS.append(dict(cancer=cancer, name=nm,
                                  database_id=str(cols[k][1]) if isinstance(cols, pd.MultiIndex) else "",
                                  fraction_non_missing=round(col.notna().mean(), 3)))

def load_real():
    import cptac
    patch_cptac_download()
    print(f"  cptac version {cptac.version()}" if hasattr(cptac, "version") else "")
    print(f"  proteomics source: {PROT_SOURCE}")
    data = {}
    start = time.time()
    for i, name in enumerate(CANCERS, 1):
        progress(i - 1, len(CANCERS), start, f"loading {name:<6}")
        try:
            ca = getattr(cptac, name)()
            raw = ca.get_proteomics(source=PROT_SOURCE, tissue_type="tumor")
            record_names(name.upper(), tumour_only(raw) if not isinstance(raw.columns, pd.MultiIndex) else raw)
            prot = tumour_only(collapse_genes(raw))
            rna, rna_src, errs = None, None, []
            for src in RNA_FALLBACK:                       # washu first; others only if washu fails
                try:
                    rna = tumour_only(collapse_genes(ca.get_transcriptomics(source=src, tissue_type="tumor")))
                    rna_src = src
                    break
                except Exception as e:
                    errs.append(f"{src}: {type(e).__name__}: {e}")
            if rna is None:
                raise RuntimeError("no RNA source worked -> " + " | ".join(errs))
            if errs:
                print(f"\n  [note] {name}: RNA taken from '{rna_src}' because " + " | ".join(errs))
            try:
                pur = tumour_only(ca.get_tumor_purity(source=PURITY_SOURCE, tissue_type="tumor"))
                col = purity_column(pur)
                purity = pur[col].astype(float) if col else None
            except Exception as e:
                purity = None
            data[name.upper()] = dict(prot=prot, rna=rna, purity=purity, rna_source=rna_src)
        except Exception as e:
            import traceback
            print(f"\n  !! {name}: could not load ({type(e).__name__}: {e}). Skipped.")
            print("     --- full error (send this if a cancer type is skipped) ---")
            print("     " + traceback.format_exc().strip().replace("\n", "\n     "))
    progress(len(CANCERS), len(CANCERS), start, "loading done ")
    return data

def load_demo(seed=1):
    """Synthetic data with the same structure as cptac output (for testing only)."""
    rng = np.random.default_rng(seed); data = {}
    genes = PROTEASES + ACIDITY + ["CTSL1"]
    for name in CANCERS:
        n = rng.integers(80, 140); ids = [f"{name[:2].upper()}{k:04d}" for k in range(n)]
        acid = rng.normal(size=n); purity = pd.Series(rng.uniform(0.3, 0.95, n), index=ids)
        rna = pd.DataFrame({g: rng.normal(size=n) + (0.8 * acid if g in ACIDITY else 0.3 * acid) for g in genes}, index=ids)
        prot = rna.copy() * 0.6 + rng.normal(size=rna.shape)
        prot.loc[rng.random(n) < 0.6, "CA9"] = np.nan          # CA9 poorly detected
        prot["MMP9"] = np.nan                                  # never detected
        ids_n = [i + ".N" for i in ids[:10]]
        prot = pd.concat([prot, prot.iloc[:10].set_axis(ids_n)])  # fake normals, must be removed
        mi = lambda df: df.set_axis(pd.MultiIndex.from_tuples([(c, f"ENSP{j}") for j, c in enumerate(df.columns)],
                                                              names=["Name", "Database_ID"]), axis=1)
        data[name.upper()] = dict(prot=tumour_only(collapse_genes(mi(prot))),
                                  rna=tumour_only(collapse_genes(mi(rna))), purity=purity)
    return data


# ----------------------------------------------------------------- analysis
def acidity_score(df, min_genes):
    avail = [g for g in ACIDITY if g in df.columns and df[g].notna().mean() >= MIN_DETECT]
    if len(avail) < min_genes:
        return None, avail
    return zscore_within(df[avail]).mean(axis=1, skipna=True), avail

def run(data):
    rows, detect_rows, info_rows = [], [], []
    tasks = [(c, p) for c in data for p in PROTEASES]
    start = time.time()
    for i, (c, p) in enumerate(tasks, 1):
        d = data[c]; prot, rna, pur = d["prot"], d["rna"], d["purity"]
        if "acidA" not in d:
            d["acidA"], d["genesA"] = acidity_score(rna, len(ACIDITY) - 1)
            d["acidB"], d["genesB"] = acidity_score(prot, MIN_ACID_PROT)
            info_rows.append(dict(cancer=c, n_protein_tumours=len(prot), n_rna_tumours=len(rna),
                                  n_both=len(prot.index.intersection(rna.index)),
                                  purity_available=pur is not None, rna_source=d.get("rna_source", "demo"),
                                  acidity_genes_rna=";".join(d["genesA"]),
                                  acidity_proteins_detected=";".join(d["genesB"])))
        det = prot[p].notna().mean() if p in prot.columns else 0.0
        detect_rows.append(dict(cancer=c, gene=p, fraction_tumours_detected=round(det, 3)))
        ok = det >= MIN_DETECT
        rec = dict(cancer=c, protease=p, protein_detected_fraction=round(det, 3))
        for tag, score in (("A", d["acidA"]), ("B", d["acidB"])):
            if ok and score is not None:
                rho, pv, n = partial_spearman(prot[p], score, pur)
            else:
                rho, pv, n = np.nan, np.nan, 0
            rec.update({f"rho_{tag}": rho, f"p_{tag}": pv, f"n_{tag}": n})
        if ok and p in rna.columns:
            rho, pv, n = partial_spearman(rna[p], prot[p])
        else:
            rho, pv, n = np.nan, np.nan, 0
        rec.update(rho_rna_protein=rho, p_rna_protein=pv, n_rna_protein=n)
        rows.append(rec)
        progress(i, len(tasks), start, "correlations ")
    for c in data:   # acidity-gene detection at protein level
        for g in ACIDITY:
            prot = data[c]["prot"]
            detect_rows.append(dict(cancer=c, gene=g,
                                    fraction_tumours_detected=round(prot[g].notna().mean(), 3) if g in prot.columns else 0.0))
    res = pd.DataFrame(rows)
    for tag in ("A", "B", "rna_protein"):
        pcol = f"p_{tag}"; res[f"fdr_{tag}"] = bh_fdr(res[pcol].values)
    return res, pd.DataFrame(detect_rows), pd.DataFrame(info_rows)

def summarise(res):
    out = []
    for p in PROTEASES:
        s = res[res.protease == p]
        a = s.dropna(subset=["rho_A"])
        out.append(dict(protease=p, cancers_tested_A=len(a),
                        median_rho_A=a.rho_A.median(),
                        sig_pos_A=int(((a.fdr_A < FDR_SIG) & (a.rho_A > 0)).sum()),
                        moderate_A=int(((a.fdr_A < FDR_SIG) & (a.rho_A >= RHO_MOD)).sum()),
                        sig_neg_A=int(((a.fdr_A < FDR_SIG) & (a.rho_A < 0)).sum()),
                        median_rho_B=s.rho_B.median(),
                        cancers_tested_B=int(s.rho_B.notna().sum()),
                        median_rho_rna_protein=s.rho_rna_protein.median()))
    rk = pd.DataFrame(out).sort_values("median_rho_A", ascending=False, na_position="last")
    rk.insert(1, "rank_A", range(1, len(rk) + 1))
    # CTSB vs others, paired over cancers where both were tested
    pw = []
    wide = res.pivot(index="cancer", columns="protease", values="rho_A")
    for p in PROTEASES:
        if p == "CTSB" or p not in wide:
            continue
        pair = wide[["CTSB", p]].dropna()
        if len(pair) >= 5 and (pair.CTSB - pair[p]).abs().sum() > 0:
            stat, pv = stats.wilcoxon(pair.CTSB, pair[p])
        else:
            pv = np.nan
        pw.append(dict(other=p, n_cancers=len(pair), ctsb_higher=int((pair.CTSB > pair[p]).sum()),
                       median_diff=(pair.CTSB - pair[p]).median(), p_wilcoxon=pv))
    pw = pd.DataFrame(pw)
    okp = pw.p_wilcoxon.notna()
    pw["p_holm"] = np.nan
    if okp.any():
        pw.loc[okp, "p_holm"] = holm(pw.loc[okp, "p_wilcoxon"].values)
    return rk, pw

def figure(res, rk, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = rk.protease.tolist()
    cancers = sorted(res.cancer.unique())
    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.45)
    # (a) heatmap, analysis A
    ax = fig.add_subplot(gs[0])
    M = res.pivot(index="protease", columns="cancer", values="rho_A").reindex(index=order, columns=cancers)
    F = res.pivot(index="protease", columns="cancer", values="fdr_A").reindex(index=order, columns=cancers)
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "n.d.", ha="center", va="center", fontsize=6, color="0.5")
            elif F.values[i, j] < FDR_SIG:
                ax.text(j, i, "*", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(cancers))); ax.set_xticklabels(cancers, rotation=60, fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    for t in ax.get_yticklabels():
        if t.get_text() == "CTSB":
            t.set_color("#C2410C"); t.set_fontweight("bold")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="ρ")
    ax.set_title("(a) Protein vs RNA acidity score", fontsize=10, loc="left")
    # (b) median rho A vs B
    ax = fig.add_subplot(gs[1]); y = np.arange(len(order))
    ax.barh(y + 0.2, rk.set_index("protease").loc[order, "median_rho_A"], height=0.38, color="#1F2937", label="RNA acidity score (A)")
    ax.barh(y - 0.2, rk.set_index("protease").loc[order, "median_rho_B"], height=0.38, color="#9CA3AF", label="protein acidity score (B)")
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=9); ax.invert_yaxis(); ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlabel("median ρ across cancer types"); ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("(b) Median association", fontsize=10, loc="left")
    # (c) RNA-protein agreement
    ax = fig.add_subplot(gs[2])
    vals = [res[res.protease == p].rho_rna_protein.dropna().values for p in order]
    ax.boxplot(vals, vert=False, showfliers=False, widths=0.6)
    for k, v in enumerate(vals, 1):
        ax.scatter(v, np.full(len(v), k) + np.random.default_rng(k).uniform(-0.15, 0.15, len(v)), s=10, color="#C2410C" if order[k-1] == "CTSB" else "#374151", zorder=3)
    ax.set_yticks(range(1, len(order) + 1)); ax.set_yticklabels(order, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel("ρ (mRNA vs protein, same gene)"); ax.set_xlim(-0.2, 1)
    ax.set_title("(c) mRNA–protein agreement", fontsize=10, loc="left")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"part3_cptac_protein_check.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def save(df, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        df.to_csv(f, index=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="run on synthetic data (test only)")
    ap.add_argument("--cancers", default="", help="comma list to test only some, e.g. Lscc (test only; final run uses all)")
    ap.add_argument("--prot-source", default="umich", choices=["umich", "bcm"],
                    help="proteomics source (default umich; bcm = sensitivity check)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_part3"))
    a = ap.parse_args()
    global PROT_SOURCE, CANCERS
    PROT_SOURCE = a.prot_source
    if a.cancers:
        CANCERS = [c.strip().capitalize() for c in a.cancers.split(",")]
    suffix = "_DEMO" if a.demo else ("" if PROT_SOURCE == "umich" else f"_{PROT_SOURCE}")
    if a.cancers and not a.demo:
        suffix += "_TEST_" + "_".join(CANCERS)
    outdir = os.path.abspath(a.out + suffix); os.makedirs(outdir, exist_ok=True)
    print(f"\nPart 3 -- CPTAC proteomics check   ({'DEMO synthetic data' if a.demo else 'real CPTAC data'})")
    print(f"Output folder: {outdir}\n")

    print("Phase 1/4  Load proteomics, RNA and purity"); s = time.time()
    data = load_demo() if a.demo else load_real()
    if not data:
        sys.exit("No cancer type could be loaded. Check the cptac installation and internet connection.")
    phase_done("Phase 1", s)

    print("Phase 2/4  Acidity scores and correlations"); s = time.time()
    res, det, info = run(data)
    phase_done("Phase 2", s)

    print("Phase 3/4  Ranking and CTSB-vs-others tests"); s = time.time()
    rk, pw = summarise(res)
    phase_done("Phase 3", s)

    print("Phase 4/4  Save tables and figure"); s = time.time()
    save(res, os.path.join(outdir, "cptac_correlation_table.csv"))
    save(rk, os.path.join(outdir, "cptac_protease_ranking.csv"))
    save(pw, os.path.join(outdir, "cptac_pairwise_CTSB_vs_others.csv"))
    save(det, os.path.join(outdir, "cptac_protein_detection.csv"))
    save(info, os.path.join(outdir, "cptac_sample_info.csv"))
    if NAME_ROWS:
        save(pd.DataFrame(NAME_ROWS), os.path.join(outdir, "cptac_protein_name_check.csv"))
    figure(res, rk, outdir)
    phase_done("Phase 4", s)

    print("Summary (analysis A: protein vs RNA acidity score)")
    print(rk[["protease", "rank_A", "cancers_tested_A", "median_rho_A", "sig_pos_A", "sig_neg_A",
              "median_rho_B", "median_rho_rna_protein"]].round(3).to_string(index=False))
    print("\nCTSB vs others:")
    print(pw.round(4).to_string(index=False))
    print(f"\nPurity adjustment used in: {int(info.purity_available.sum())}/{len(info)} cancer types")
    print(f"\nTOTAL WALL TIME: {fmt(time.time() - T0)}")

if __name__ == "__main__":
    main()