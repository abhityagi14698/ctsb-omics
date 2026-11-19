#!/usr/bin/env python3
"""
Part 4 -- Acidity-stratified invasion analysis (TCGA), in the style of Sadhu et al. 2026.

Question: in acidic tumours, is CTSB (a) more highly expressed and (b) more tightly
coupled to the invasion (EMT) programme than in non-acidic tumours of the same type?

Step 1  Within each cancer type, split tumours by the acidity score:
        bottom 25% = low-acidity, top 25% = high-acidity  (also 20% and 33% as sensitivity)
Step 2  Q1  Expression: high vs low group for the 8 proteases (purity-adjusted, BH-FDR)
Step 3  Q2  Coupling: partial Spearman rho(protease, EMT score | purity) in each group;
            delta-rho = rho_high - rho_low, Fisher z test, BH-FDR; then across cancer types
            (Wilcoxon signed-rank on delta-rho; CTSB vs each other protease, Holm)

Inputs (all from your Part 2 work + ONE new MSigDB file):
  per_sample_scores.csv                                      (Part 2 results folder)
  EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz    (Part 2 data folder)
  HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION.v2026.2.Hs.grp  (NEW, from MSigDB, put in data folder)

Run from the scripts folder:
  python3 part4_acidity_stratified_invasion.py
  (options: --data ../data  --results ../results  --out ../results_part4)
  python3 part4_acidity_stratified_invasion.py --demo     # synthetic test only
"""
import argparse, glob, gzip, os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
PROTEASES = ["CTSB", "CTSL", "CTSK", "CTSD", "MMP2", "MMP9", "MMP14", "PLAU"]
ACID_COL = "score::acidity score (7 genes)"
CUTOFFS = [0.25, 0.20, 0.33]       # main = 0.25
MIN_GROUP = 15                      # minimum tumours per group
FDR = 0.05
T0 = time.time()


# ------------------------------------------------------------------ timing helpers
def fmt(s):
    s = int(s); h, r = divmod(s, 3600); m, s = divmod(r, 60); return f"{h}:{m:02d}:{s:02d}"

_last = [0.0]
def progress(i, n, start, label):
    now = time.time()
    if i not in (0, n) and now - _last[0] < 0.2:
        return
    _last[0] = now
    el = now - start; eta = el / i * (n - i) if i else 0
    bar = "#" * int(30 * i / n) + "-" * (30 - int(30 * i / n))
    sys.stdout.write(f"\r  {label} [{bar}] {100*i/n:5.1f}%  {i}/{n}  elapsed {fmt(el)}  ETA {fmt(eta)}   ")
    sys.stdout.flush()
    if i == n:
        sys.stdout.write("\n")

def done(name, start):
    print(f"  -> {name} finished in {fmt(time.time()-start)}  (total {fmt(time.time()-T0)})\n")


# ------------------------------------------------------------------ statistics
def rank_resid(v, q):
    r = pd.Series(v).rank().values
    X = np.column_stack([np.ones(len(r)), pd.Series(q).rank().values])
    b, *_ = np.linalg.lstsq(X, r, rcond=None)
    return r - X @ b

def partial_spearman(x, y, q):
    m = ~(np.isnan(x) | np.isnan(y) | np.isnan(q)); x, y, q = x[m], y[m], q[m]; n = len(x)
    if n < MIN_GROUP:
        return np.nan, np.nan, n
    ex, ey = rank_resid(x, q), rank_resid(y, q)
    rho = np.corrcoef(ex, ey)[0, 1]
    t = rho * np.sqrt((n - 3) / max(1e-12, 1 - rho ** 2))
    return rho, 2 * stats.t.sf(abs(t), n - 3), n

def group_effect(expr, group, purity):
    """OLS expr ~ 1 + group(high=1) + purity; returns coefficient (log2 units) and p."""
    m = ~(np.isnan(expr) | np.isnan(purity)); y = expr[m]; g = group[m]; q = purity[m]
    X = np.column_stack([np.ones(len(y)), g, q]); n, k = X.shape
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b; s2 = res @ res / (n - k)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    t = b[1] / se
    return b[1], 2 * stats.t.sf(abs(t), n - k)

def fisher_diff(r1, n1, r2, n2):
    """two independent (partial, 1 covariate) correlations: z test of r1 - r2"""
    if np.isnan(r1) or np.isnan(r2):
        return np.nan
    z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(1 / (n1 - 4) + 1 / (n2 - 4))
    return 2 * stats.norm.sf(abs(z))

def bh(p):
    p = np.asarray(p, float); out = np.full_like(p, np.nan); ok = ~np.isnan(p); m = ok.sum()
    if m:
        ps = p[ok]; o = np.argsort(ps); r = ps[o] * m / (np.arange(m) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]; t = np.empty(m); t[o] = np.minimum(r, 1); out[ok] = t
    return out

def holm(p):
    p = np.asarray(p, float); out = np.full_like(p, np.nan); ok = np.where(~np.isnan(p))[0]
    o = ok[np.argsort(p[ok])]; run = 0
    for i, k in enumerate(o):
        run = max(run, min(1, (len(o) - i) * p[k])); out[k] = run
    return out


# ------------------------------------------------------------------ loading
def find(patterns, roots):
    for r in roots:
        for p in patterns:
            hits = sorted(glob.glob(os.path.join(r, "**", p), recursive=True))
            if hits:
                return hits[0]
    return None

def read_grp(path):
    genes = []
    for line in open(path):
        t = line.strip()
        if t and not t.startswith("#") and not t.startswith("HALLMARK_") and not t.startswith("http"):
            genes.append(t)
    return genes

def load_emt_expression(expr_path, samples, genes):
    """read only the EMT gene rows for the needed samples from the big Xena matrix"""
    opener = gzip.open if expr_path.endswith(".gz") else open
    with opener(expr_path, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
    keep = [header[0]] + [s for s in header[1:] if s in samples]
    print(f"  matrix: {len(header)-1:,} samples in file, {len(keep)-1:,} matched to per_sample_scores.csv")
    est_rows = 20531                                  # genes in the PanCanAtlas matrix
    chunks, start, seen = [], time.time(), 0
    for ch in pd.read_csv(expr_path, sep="\t", usecols=keep, index_col=0, chunksize=1000):
        seen += len(ch)
        sub = ch[ch.index.isin(genes)]
        if len(sub):
            chunks.append(sub)
        progress(min(seen, est_rows), est_rows, start, "reading matrix")
    progress(est_rows, est_rows, start, "reading matrix")
    df = pd.concat(chunks) if chunks else pd.DataFrame()
    df = df.groupby(level=0).mean()
    return df.T                                        # samples x genes


def demo_inputs(out):
    """synthetic files with the same formats (test only)"""
    rng = np.random.default_rng(3); os.makedirs(out, exist_ok=True)
    cancers = {f"C{k:02d}": rng.integers(40, 400) for k in range(12)}
    rows, emt = [], [f"EMT{k}" for k in range(60)] + ["MMP2", "MMP14"]
    expr = {}
    for c, n in cancers.items():
        acid = rng.normal(size=n); emt_latent = rng.normal(size=n)
        for i in range(n):
            sid = f"TCGA-{c}-{i:04d}-01"
            ctsb = 15 + 0.3 * acid[i] + 0.2 * emt_latent[i] * (1 + 0.6 * acid[i]) + rng.normal(0, .5)
            rec = {"": sid, "cancer": c, "purity": rng.uniform(.3, .95), ACID_COL: acid[i]}
            for p in PROTEASES:
                rec[p] = ctsb if p == "CTSB" else 10 + 0.2 * acid[i] + 0.1 * emt_latent[i] + rng.normal(0, .6)
            rows.append(rec)
            expr[sid] = {g: 8 + emt_latent[i] + rng.normal(0, 1) for g in emt}
    pd.DataFrame(rows).to_csv(os.path.join(out, "per_sample_scores.csv"), index=False)
    m = pd.DataFrame(expr); m.index.name = "sample"
    m = pd.concat([m, pd.DataFrame(rng.normal(size=(300, m.shape[1])), columns=m.columns,
                                   index=[f"OTHER{k}" for k in range(300)])])
    m.index.name = "sample"
    m.to_csv(os.path.join(out, "demo_matrix.xena.gz"), sep="\t", compression="gzip")
    with open(os.path.join(out, "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION.demo.grp"), "w") as f:
        f.write("HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION\n# http://demo\n" + "\n".join(emt) + "\n")


# ------------------------------------------------------------------ main analysis
def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data", default=os.path.join(here, "..", "data"))
    ap.add_argument("--results", default=os.path.join(here, "..", "results"))
    ap.add_argument("--out", default=os.path.join(here, "..", "results_part4"))
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        a.data = a.results = os.path.abspath(a.out + "_DEMO_inputs"); a.out = a.out + "_DEMO"
        demo_inputs(a.data)
    os.makedirs(a.out, exist_ok=True)
    print(f"\nPart 4 -- acidity-stratified invasion analysis {'(DEMO, synthetic data)' if a.demo else ''}")

    # ---------------- Phase 1: inputs
    print("Phase 1/5  Find and read inputs"); s = time.time()
    scores_p = find(["per_sample_scores.csv"], [a.results, a.data, here])
    expr_p = find(["EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena*", "demo_matrix.xena.gz"], [a.data, here])
    grp_p = find(["HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION*"], [a.data, here])
    for name, p in [("per_sample_scores.csv", scores_p), ("expression matrix (EB++...xena.gz)", expr_p),
                    ("HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION .grp", grp_p)]:
        print(f"  {name:48s}: {p if p else 'NOT FOUND'}")
    if not (scores_p and expr_p and grp_p):
        sys.exit("\nA file is missing. Put it in the data folder (or pass --data / --results) and run again.")
    sc = pd.read_csv(scores_p, index_col=0)
    emt_all = read_grp(grp_p)
    emt = [g for g in emt_all if g not in PROTEASES]          # no protease inside its own outcome score
    print(f"  tumours: {len(sc):,} in {sc.cancer.nunique()} cancer types; EMT genes in set: {len(emt_all)} "
          f"(after removing the 8 proteases: {len(emt)})")
    done("Phase 1", s)

    # ---------------- Phase 2: EMT (invasion) score
    print("Phase 2/5  Invasion score from the Hallmark EMT genes"); s = time.time()
    E = load_emt_expression(expr_p, set(sc.index), set(emt))
    used = [g for g in emt if g in E.columns]
    print(f"  EMT genes found in the matrix: {len(used)} of {len(emt)}")
    E = E.reindex(sc.index)[used]
    Z = E.groupby(sc.cancer).transform(lambda v: (v - v.mean()) / v.std(ddof=1))
    sc["EMT_score"] = Z.mean(axis=1, skipna=True)
    with open(os.path.join(a.out, "emt_genes_used.txt"), "w") as f:
        f.write("\n".join(used))
    done("Phase 2", s)

    # ---------------- Phase 3: stratification
    print("Phase 3/5  Split tumours into low- and high-acidity groups"); s = time.time()
    grp_rows, groups = [], {}
    for cut in CUTOFFS:
        for c, d in sc.groupby("cancer"):
            lo, hi = d[ACID_COL].quantile(cut), d[ACID_COL].quantile(1 - cut)
            low, high = d[d[ACID_COL] <= lo], d[d[ACID_COL] >= hi]
            ok = len(low) >= MIN_GROUP and len(high) >= MIN_GROUP
            grp_rows.append(dict(cutoff=cut, cancer=c, n_total=len(d), n_low=len(low), n_high=len(high),
                                 acidity_low_max=lo, acidity_high_min=hi, analysed=ok))
            if ok:
                groups[(cut, c)] = (low, high)
    G = pd.DataFrame(grp_rows); G.to_csv(os.path.join(a.out, "stratified_groups.csv"), index=False)
    for cut in CUTOFFS:
        g = G[G.cutoff == cut]
        print(f"  cut-off {int(cut*100)}%: {int(g.analysed.sum())} of {len(g)} cancer types have >= {MIN_GROUP} tumours per group")
    done("Phase 3", s)

    # ---------------- Phase 4: Q1 expression + Q2 coupling
    print("Phase 4/5  Q1 expression (high vs low) and Q2 coupling with the EMT score"); s = time.time()
    q1, q2 = [], []
    tasks = [(k, p) for k in groups for p in PROTEASES]; st = time.time()
    for i, ((cut, c), p) in enumerate(tasks, 1):
        low, high = groups[(cut, c)]
        both = pd.concat([low, high]); g = np.r_[np.zeros(len(low)), np.ones(len(high))]
        coef, pv = group_effect(both[p].values.astype(float), g, both.purity.values.astype(float))
        q1.append(dict(cutoff=cut, cancer=c, protease=p, log2FC_adj=coef, p=pv,
                       mean_low=low[p].mean(), mean_high=high[p].mean()))
        rl, pl, nl = partial_spearman(low[p].values.astype(float), low.EMT_score.values, low.purity.values.astype(float))
        rh, ph, nh = partial_spearman(high[p].values.astype(float), high.EMT_score.values, high.purity.values.astype(float))
        d_all = sc[sc.cancer == c]
        ra, pa, na = partial_spearman(d_all[p].values.astype(float), d_all.EMT_score.values, d_all.purity.values.astype(float))
        q2.append(dict(cutoff=cut, cancer=c, protease=p, rho_low=rl, n_low=nl, rho_high=rh, n_high=nh,
                       delta_rho=rh - rl, p_diff=fisher_diff(rh, nh, rl, nl), rho_all=ra, p_all=pa))
        progress(i, len(tasks), st, "tests")
    Q1, Q2 = pd.DataFrame(q1), pd.DataFrame(q2)
    for cut in CUTOFFS:
        m1, m2 = Q1.cutoff == cut, Q2.cutoff == cut
        Q1.loc[m1, "fdr"] = bh(Q1.loc[m1, "p"]); Q2.loc[m2, "fdr_diff"] = bh(Q2.loc[m2, "p_diff"])
        Q2.loc[m2, "fdr_all"] = bh(Q2.loc[m2, "p_all"])
    done("Phase 4", s)

    # ---------------- Phase 5: summaries, tables, figure
    print("Phase 5/5  Summaries, tables and figure"); s = time.time()
    summ, pair = [], []
    for cut in CUTOFFS:
        for p in PROTEASES:
            a1 = Q1[(Q1.cutoff == cut) & (Q1.protease == p)]; a2 = Q2[(Q2.cutoff == cut) & (Q2.protease == p)].dropna(subset=["delta_rho"])
            w = stats.wilcoxon(a2.delta_rho).pvalue if len(a2) >= 5 and (a2.delta_rho != 0).any() else np.nan
            summ.append(dict(cutoff=cut, protease=p, n_cancers=len(a1),
                             q1_median_log2FC=a1.log2FC_adj.median(),
                             q1_sig_higher=int(((a1.fdr < FDR) & (a1.log2FC_adj > 0)).sum()),
                             q1_sig_lower=int(((a1.fdr < FDR) & (a1.log2FC_adj < 0)).sum()),
                             q2_median_rho_all=a2.rho_all.median(), q2_median_rho_low=a2.rho_low.median(),
                             q2_median_rho_high=a2.rho_high.median(), q2_median_delta_rho=a2.delta_rho.median(),
                             q2_n_delta_positive=int((a2.delta_rho > 0).sum()),
                             q2_n_delta_sig_positive=int(((a2.fdr_diff < FDR) & (a2.delta_rho > 0)).sum()),
                             q2_wilcoxon_p_delta=w))
        wide = Q2[Q2.cutoff == cut].pivot(index="cancer", columns="protease", values="delta_rho")
        rows = []
        for p in PROTEASES:
            if p == "CTSB":
                continue
            pr = wide[["CTSB", p]].dropna()
            pv = stats.wilcoxon(pr.CTSB, pr[p]).pvalue if len(pr) >= 5 else np.nan
            rows.append(dict(cutoff=cut, other=p, n_cancers=len(pr), ctsb_delta_higher=int((pr.CTSB > pr[p]).sum()),
                             median_diff=(pr.CTSB - pr[p]).median(), p_wilcoxon=pv))
        r = pd.DataFrame(rows); r["p_holm"] = holm(r.p_wilcoxon.values); pair.append(r)
    S = pd.DataFrame(summ); P = pd.concat(pair)
    Q1.to_csv(os.path.join(a.out, "q1_expression_high_vs_low.csv"), index=False)
    Q2.to_csv(os.path.join(a.out, "q2_coupling_by_group.csv"), index=False)
    S.to_csv(os.path.join(a.out, "summary_by_protease.csv"), index=False)
    P.to_csv(os.path.join(a.out, "q2_ctsb_vs_others.csv"), index=False)
    make_figure(Q1, Q2, S, a.out)
    done("Phase 5", s)

    main_s = S[S.cutoff == 0.25].set_index("protease")
    print("Summary, 25% cut-off:")
    print(main_s[["n_cancers", "q1_median_log2FC", "q1_sig_higher", "q2_median_rho_low", "q2_median_rho_high",
                  "q2_median_delta_rho", "q2_n_delta_positive", "q2_wilcoxon_p_delta"]].round(3).to_string())
    print("\nCTSB vs others (delta-rho, 25% cut-off):")
    print(P[P.cutoff == 0.25].round(4).to_string(index=False))
    print(f"\nOutputs in: {os.path.abspath(a.out)}")
    print(f"TOTAL WALL TIME: {fmt(time.time()-T0)}")


def make_figure(Q1, Q2, S, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HL = "#C2410C"
    q1 = Q1[Q1.cutoff == 0.25]; q2 = Q2[Q2.cutoff == 0.25]; s = S[S.cutoff == 0.25].set_index("protease")
    order = s.sort_values("q2_median_delta_rho", ascending=False).index.tolist()
    fig = plt.figure(figsize=(12, 5.2)); gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.45)
    # (a) Q1 heatmap
    ax = fig.add_subplot(gs[0])
    M = q1.pivot(index="protease", columns="cancer", values="log2FC_adj").reindex(order)
    F = q1.pivot(index="protease", columns="cancer", values="fdr").reindex(order)
    cols = M.mean().sort_values(ascending=False).index; M, F = M[cols], F[cols]
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if F.values[i, j] < FDR:
                ax.text(j, i + 0.1, "*", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="log2 fold change (high vs low acidity)")
    ax.set_title("(a) Expression, high vs low acidity", loc="left", fontsize=9, fontweight="bold")
    # (b) CTSB rho low vs high per cancer
    ax = fig.add_subplot(gs[1]); c = q2[q2.protease == "CTSB"].dropna(subset=["delta_rho"]).sort_values("delta_rho")
    y = np.arange(len(c))
    ax.hlines(y, c.rho_low, c.rho_high, color="0.7", lw=1)
    ax.plot(c.rho_low, y, "o", ms=4, color="0.45", label="low acidity")
    ax.plot(c.rho_high, y, "o", ms=4, color=HL, label="high acidity")
    ax.set_yticks(y); ax.set_yticklabels(c.cancer, fontsize=6.5); ax.axvline(0, color="0.5", lw=0.6)
    ax.set_xlabel("ρ (CTSB vs EMT score | purity)")
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, handletextpad=0.2)
    ax.set_title("(b) CTSB coupling to invasion", loc="left", fontsize=9, fontweight="bold")
    # (c) median delta rho per protease
    ax = fig.add_subplot(gs[2]); yy = np.arange(len(order))
    vals = s.loc[order, "q2_median_delta_rho"]
    ax.barh(yy, vals, color=[HL if p == "CTSB" else "#1F2937" for p in order], height=0.6)
    lim = max(0.05, np.nanmax(np.abs(vals.values)) * 1.15); ax.set_xlim(-lim, lim)
    for k, p in enumerate(order):                      # counts in a column outside the plot
        ax.text(1.03, k, f"{int(s.loc[p,'q2_n_delta_positive'])}/{int(s.loc[p,'n_cancers'])}",
                transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=7)
    ax.text(1.03, -0.75, "Δρ>0", transform=ax.get_yaxis_transform(), ha="left", fontsize=7, color="0.4")
    ax.set_yticks(yy); ax.set_yticklabels(order); ax.invert_yaxis(); ax.axvline(0, color="0.5", lw=0.6)
    ax.set_xlabel("median Δρ (high − low acidity)")
    ax.set_title("(c) Change in coupling", loc="left", fontsize=9, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"part4_acidity_stratified.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
