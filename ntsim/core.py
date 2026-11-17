"""Planted-truth simulation of non-targeting (NT) guide screening in a low-MOI CRISPR screen.

Everything lives in one file so it is easy to read top to bottom:

1. simulate()      - make one fake screen with known truth
2. score_test()    - negative-binomial score test of "guide cells vs control cells"
3. calib_check()   - NT calibration check (each NT guide vs the other NT cells)
4. rule_r0/r1/r2/r3 - the guide-screening rules being compared
5. diagnose()      - is the excess concentrated (off-target) or spread out (missing covariate)?
6. run_replicate() - one simulated screen through every rule, returns one row per rule
"""
import numpy as np
from scipy.stats import norm, binom

# ----------------------------------------------------------------------------
# 1. Simulator
# ----------------------------------------------------------------------------

DEFAULTS = dict(
    n_genes=300,          # genes measured
    n_nt=10,              # non-targeting guides
    n_tgt=15,             # targeting guides (each knocks down one gene)
    cells_per_guide=200,  # cells per guide (low MOI: one guide per cell)
    base_logmean=np.log(3.0), base_logsd=1.0,  # gene baseline expression (lognormal)
    lib_logsd=0.3,        # cell library-size spread
    theta=5.0,            # NB size (dispersion) for every gene
    tgt_effect=-1.5,      # log fold change of a targeting guide on its own gene
    batch_sd=0.25,        # log fold change of batch 2 vs batch 1, per gene (scenario c, d)
    imbalance=0.15,       # each guide's share of batch-2 cells ~ Uniform(0.5-imb, 0.5+imb)
    # off-target plan (scenario b, d): list of (n_genes_hit, log fold change), one per bad NT guide
    offtarget=((5, -2.0), (3, -1.5)),
)


def simulate(scenario, rng, **kw):
    """Return a dict with counts Y (cells x genes) and the true labels.

    scenario 'a': all NT guides clean, no batch effect.
    scenario 'b': a few NT guides repress a few genes (planted off-target), no batch effect.
    scenario 'c': no off-target, but a batch effect on every gene and guides spread unevenly
                  over the two batches. The analysis that omits batch is miscalibrated.
    scenario 'd': b and c together.
    """
    p = dict(DEFAULTS, **kw)
    G, K, T, m = p["n_genes"], p["n_nt"], p["n_tgt"], p["cells_per_guide"]
    n_guides = K + T
    guide = np.repeat(np.arange(n_guides), m)          # guide id per cell; 0..K-1 are NT
    N = guide.size
    base = np.exp(rng.normal(p["base_logmean"], p["base_logsd"], G))
    lib = np.exp(rng.normal(0.0, p["lib_logsd"], N))

    has_batch = scenario in ("c", "d")
    has_off = scenario in ("b", "d")

    # batch: each guide gets its own share of batch-2 cells
    share = rng.uniform(0.5 - p["imbalance"], 0.5 + p["imbalance"], n_guides)
    batch = (rng.random(N) < share[guide]).astype(int)
    batch_eff = rng.normal(0.0, p["batch_sd"], G) if has_batch else np.zeros(G)

    logfc = np.zeros((n_guides, G))
    # targeting guide j knocks down gene j (genes 0..T-1 are the targets)
    tgt_gene = np.arange(T)
    for j in range(T):
        logfc[K + j, tgt_gene[j]] = p["tgt_effect"]

    bad_guides = np.array([], dtype=int)
    off_genes = {}
    if has_off:
        # off-target genes come from the upper half of expression among non-target genes
        candidates = np.setdiff1d(np.argsort(base)[G // 2:], tgt_gene)
        bad_guides = rng.choice(K, size=len(p["offtarget"]), replace=False)
        for gidx, (n_hit, lfc) in zip(bad_guides, p["offtarget"]):
            genes = rng.choice(candidates, size=n_hit, replace=False)
            logfc[gidx, genes] = lfc
            off_genes[int(gidx)] = genes

    mu = lib[:, None] * base[None, :] * np.exp(logfc[guide] + batch[:, None] * batch_eff[None, :])
    theta = p["theta"]
    # NB via gamma-Poisson
    lam = rng.gamma(shape=theta, scale=mu / theta)
    Y = rng.poisson(lam).astype(np.int32)

    return dict(Y=Y, guide=guide, lib=lib, batch=batch, K=K, T=T, G=G,
                tgt_gene=tgt_gene, bad_guides=bad_guides, off_genes=off_genes,
                share=share, batch_eff=batch_eff, scenario=scenario)


# ----------------------------------------------------------------------------
# 2. Negative-binomial score test (null model fit in closed form)
# ----------------------------------------------------------------------------

def estimate_theta(Y, lib, groups):
    """Method-of-moments NB size per gene, with a separate mean per group (batch or none)."""
    mu = np.empty(Y.shape, dtype=float)
    for gval in np.unique(groups):
        idx = groups == gval
        rate = Y[idx].sum(0) / lib[idx].sum()
        mu[idx] = lib[idx, None] * rate[None, :]
    num = (mu ** 2).sum(0)
    den = ((Y - mu) ** 2 - Y).sum(0)
    theta = np.where(den > 0, num / np.maximum(den, 1e-12), 1000.0)
    return np.clip(theta, 0.5, 1000.0)


def score_test(Y, lib, groups, theta, a_mask, c_mask):
    """z and two-sided p for each gene: cells in a_mask (guide) vs cells in c_mask (control).

    Null model: log E[y] = log(lib) + intercept per group (batch, or one group if batch omitted).
    Fit by the Poisson MLE (closed form), NB weights with the given theta, and the efficient
    score that projects out the group intercepts.
    """
    use = a_mask | c_mask
    y, l, gr, x = Y[use], lib[use], groups[use], a_mask[use]
    U = np.zeros(Y.shape[1])
    V = np.zeros(Y.shape[1])
    for gval in np.unique(gr):
        idx = gr == gval
        rate = y[idx].sum(0) / l[idx].sum()
        mu = l[idx, None] * rate[None, :]
        denom = 1.0 + mu / theta[None, :]
        W = mu / denom
        r = (y[idx] - mu) / denom
        xa = x[idx]
        WA, Wt = W[xa].sum(0), W.sum(0)
        RA, Rt = r[xa].sum(0), r.sum(0)
        U += RA - WA / Wt * Rt
        V += WA - WA ** 2 / Wt
    z = U / np.sqrt(np.maximum(V, 1e-12))
    return z, 2 * norm.sf(np.abs(z))


# ----------------------------------------------------------------------------
# 3. NT calibration check
# ----------------------------------------------------------------------------

def nt_zmatrix(sim, kept, genes, groups, theta):
    """z-scores (len(kept) x len(genes)): each kept NT guide vs the other kept NT cells."""
    Y = sim["Y"][:, genes]
    th = theta[genes]
    g = sim["guide"]
    Z = np.empty((len(kept), len(genes)))
    for i, k in enumerate(kept):
        a = g == k
        c = np.isin(g, kept) & ~a
        Z[i], _ = score_test(Y, sim["lib"], groups, th, a, c)
    return Z


def calib_pass(P, alpha=0.1):
    """Pass if (i) no NT pair survives Bonferroni at alpha and
    (ii) the count of p < 0.01 is not above the binomial 99.9% quantile."""
    n = P.size
    bonf = (P < alpha / n).sum()
    small = (P < 0.01).sum()
    return bool(bonf == 0 and small <= binom.ppf(0.999, n, 0.01))


def worst_guide(P, order="minp"):
    """Row index of the worst guide.
    order="minp": smallest p-value first, ties broken by most p < 0.01 (main run).
    order="count": most p < 0.01 first, ties broken by smallest p (first run, kept for the record)."""
    counts = (P < 0.01).sum(1)
    if order == "count":
        return np.lexsort((P.min(1), -counts))[0]
    return np.lexsort((-counts, P.min(1)))[0]


def screen_until_pass(sim, genes, groups, theta, min_keep=2, order="minp"):
    """Drop the worst NT guide, re-test, repeat until the check passes on these genes."""
    kept = list(range(sim["K"]))
    while True:
        Z = nt_zmatrix(sim, kept, genes, groups, theta)
        P = 2 * norm.sf(np.abs(Z))
        if calib_pass(P) or len(kept) <= min_keep:
            return kept, P
        kept.pop(int(worst_guide(P, order)))


# ----------------------------------------------------------------------------
# 4. Discovery on targeting pairs
# ----------------------------------------------------------------------------

def bh(p, q=0.1):
    p = np.asarray(p).ravel()
    n = p.size
    order = np.argsort(p)
    thresh = q * np.arange(1, n + 1) / n
    ok = p[order] <= thresh
    rej = np.zeros(n, dtype=bool)
    if ok.any():
        rej[order[: np.max(np.nonzero(ok)[0]) + 1]] = True
    return rej


def targeting_pvals(sim, control_guides, genes, groups, theta):
    """p-values (T x len(genes)) for each targeting guide vs cells of control_guides."""
    Y = sim["Y"][:, genes]
    th = theta[genes]
    g = sim["guide"]
    c = np.isin(g, control_guides)
    P = np.empty((sim["T"], len(genes)))
    for j in range(sim["T"]):
        a = g == sim["K"] + j
        _, P[j] = score_test(Y, sim["lib"], groups, th, a, c)
    return P


def discovery_stats(sim, P_full):
    """P_full is T x G. BH at 0.1 over all T*G pairs. Returns false discoveries, FDP, power."""
    T, G = P_full.shape
    truth = np.zeros((T, G), dtype=bool)
    truth[np.arange(T), sim["tgt_gene"]] = True
    rej = bh(P_full).reshape(T, G)
    fd = int((rej & ~truth).sum())
    nd = int(rej.sum())
    return dict(false_disc=fd, n_disc=nd, fdp=fd / max(nd, 1),
                power=float((rej & truth).sum() / truth.sum()))


# ----------------------------------------------------------------------------
# 5. Diagnostic: concentrated (off-target-like) vs spread (covariate-like)
# ----------------------------------------------------------------------------

def spread_stat(Z, trim=30):
    """Top eigenvalue of Z Z^T / G after removing the `trim` genes with the largest max |z|.
    Removing a few genes wipes out a sparse off-target signal (and its echo in the other
    guides' tests, since every NT guide is tested against the other NT cells).
    A covariate signal is spread over many genes and survives."""
    keep = np.argsort((Z ** 2).max(0))[:-trim]
    Zt = Z[:, keep]
    M = Zt @ Zt.T / Zt.shape[1]
    return float(np.linalg.eigvalsh(M)[-1])


def conc_stat(Z, top=5):
    """Largest per-guide sum of its top-`top` z^2 (a sparse, few-gene signal)."""
    return float(np.sort(Z ** 2, axis=1)[:, -top:].sum(1).max())


def diagnose(sim, groups, theta, rng, n_null=19):
    """Compare both statistics with a null made by randomly regrouping NT cells into fake
    guides of the same sizes (this breaks any guide-specific signal). Returns the verdict:
    'covariate' if the dense statistic is above all null draws, else 'offtarget' if the sparse
    statistic is, else 'clean'. Rank p-value with 19 null draws = 0.05."""
    K = sim["K"]
    genes = np.arange(sim["G"])
    Z = nt_zmatrix(sim, list(range(K)), genes, groups, theta)
    s_obs, c_obs = spread_stat(Z), conc_stat(Z)
    nt_cells = np.nonzero(sim["guide"] < K)[0]
    s_null, c_null = [], []
    for _ in range(n_null):
        fake = dict(sim)
        g2 = sim["guide"].copy()
        g2[nt_cells] = rng.permutation(g2[nt_cells])
        fake["guide"] = g2
        Zn = nt_zmatrix(fake, list(range(K)), genes, groups, theta)
        s_null.append(spread_stat(Zn))
        c_null.append(conc_stat(Zn))
    p_spread = (1 + sum(s >= s_obs for s in s_null)) / (1 + n_null)
    p_conc = (1 + sum(c >= c_obs for c in c_null)) / (1 + n_null)
    if p_spread <= 0.05:
        verdict = "covariate"
    elif p_conc <= 0.05:
        verdict = "offtarget"
    else:
        verdict = "clean"
    return dict(verdict=verdict, p_spread=p_spread, p_conc=p_conc, Z=Z)


# ----------------------------------------------------------------------------
# 6. One replicate through every rule
# ----------------------------------------------------------------------------

def _row(rule, sim, kept, passed, P_full):
    K = sim["K"]
    bad = set(int(b) for b in sim["bad_guides"])
    dropped = set(range(K)) - set(kept)
    n_clean = K - len(bad)
    row = dict(rule=rule, n_kept=len(kept),
               clean_dropped=len(dropped - bad),
               clean_drop_rate=len(dropped - bad) / n_clean,
               planted_caught=len(dropped & bad),
               planted_catch_rate=(len(dropped & bad) / len(bad)) if bad else np.nan,
               calib_pass=passed)
    # The check "should" pass only if the analysis model is right and no bad guide is left.
    model_ok = sim["scenario"] in ("a", "b") or rule.startswith("Oracle")
    row["truth_ok"] = bool(model_ok and not (bad - dropped))
    row.update(discovery_stats(sim, P_full))
    return row


def split_rule(sim, halves, groups, theta, order="minp"):
    """R2 cross-fit: guides screened on one half of the genes are used as the control for
    calibration and discovery on the other half. A guide counts as dropped if either half drops it."""
    Pt = np.empty((sim["T"], sim["G"]))
    passes, kept_sets = [], []
    for s, h in ((0, 1), (1, 0)):
        kept_s, _ = screen_until_pass(sim, halves[s], groups, theta, order=order)
        Zh = nt_zmatrix(sim, kept_s, halves[h], groups, theta)
        passes.append(calib_pass(2 * norm.sf(np.abs(Zh))))
        Pt[:, halves[h]] = targeting_pvals(sim, kept_s, halves[h], groups, theta)
        kept_sets.append(set(kept_s))
    return sorted(kept_sets[0] & kept_sets[1]), all(passes), Pt


def split_cells_rule(sim, groups, theta, rng, order="minp"):
    """R2b: split the NT cells of every guide into two random halves. Screen guides with half 1,
    judge calibration with half 2 (all genes), then run discovery on all cells with the kept guides."""
    K, G = sim["K"], sim["G"]
    allg = np.arange(G)
    nt_idx = np.nonzero(sim["guide"] < K)[0]
    half2 = np.zeros(sim["guide"].size, dtype=bool)
    half2[rng.permutation(nt_idx)[: nt_idx.size // 2]] = True
    s1, s2 = dict(sim), dict(sim)
    g1, g2 = sim["guide"].copy(), sim["guide"].copy()
    g1[half2] = -1                                   # screening sees half 1 only
    g2[(sim["guide"] < K) & ~half2] = -1             # checking sees half 2 only
    s1["guide"], s2["guide"] = g1, g2
    kept, _ = screen_until_pass(s1, allg, groups, theta, order=order)
    Zc = nt_zmatrix(s2, kept, allg, groups, theta)
    passed = calib_pass(2 * norm.sf(np.abs(Zc)))
    Pt = targeting_pvals(sim, kept, allg, groups, theta)
    return kept, passed, Pt


def run_replicate(scenario, rep, order="minp", **kw):
    rng = np.random.default_rng([ord(scenario), rep])
    sim = simulate(scenario, rng, **kw)
    K, G = sim["K"], sim["G"]
    allg = np.arange(G)
    nobatch = np.zeros(sim["Y"].shape[0], dtype=int)   # analysis that omits batch
    nt = sim["guide"] < K
    theta = estimate_theta(sim["Y"][nt], sim["lib"][nt], nobatch[nt])
    rows = []

    # R0: keep all NT guides
    Z0 = nt_zmatrix(sim, list(range(K)), allg, nobatch, theta)
    P0 = 2 * norm.sf(np.abs(Z0))
    Pt0 = targeting_pvals(sim, list(range(K)), allg, nobatch, theta)
    rows.append(_row("R0_keep_all", sim, list(range(K)), calib_pass(P0), Pt0))

    # R1: circular - screen and judge on the same genes
    kept1, P1 = screen_until_pass(sim, allg, nobatch, theta, order=order)
    Pt1 = targeting_pvals(sim, kept1, allg, nobatch, theta)
    rows.append(_row("R1_circular", sim, kept1, calib_pass(P1), Pt1))

    # R2: split genes. Screen on half A, judge and discover on half B, then swap (cross-fit).
    perm = rng.permutation(G)
    halves = [np.sort(perm[: G // 2]), np.sort(perm[G // 2:])]
    kept2, pass2, Pt2 = split_rule(sim, halves, nobatch, theta, order)
    rows.append(_row("R2_split_genes", sim, kept2, pass2, Pt2))

    # R2b: split cells. Screen on half the NT cells, judge on the other half, discover on all.
    kept2b, pass2b, Pt2b = split_cells_rule(sim, nobatch, theta, rng, order)
    rows.append(_row("R2b_split_cells", sim, kept2b, pass2b, Pt2b))

    # R3: diagnose first. Covariate-like -> drop nothing and report the check as failed.
    # Otherwise run the cell-split rule R2b.
    dg = diagnose(sim, nobatch, theta, rng)
    if dg["verdict"] == "covariate":
        rows.append(_row("R3_diagnose_then_split_cells", sim, list(range(K)), False, Pt0))
    else:
        rows.append(_row("R3_diagnose_then_split_cells", sim, kept2b, pass2b, Pt2b))

    # Oracle: keep all guides but model the batch (only differs from R0 in c, d)
    theta_b = estimate_theta(sim["Y"][nt], sim["lib"][nt], sim["batch"][nt])
    Zb = nt_zmatrix(sim, list(range(K)), allg, sim["batch"], theta_b)
    Ptb = targeting_pvals(sim, list(range(K)), allg, sim["batch"], theta_b)
    rows.append(_row("Oracle_model_batch", sim, list(range(K)),
                     calib_pass(2 * norm.sf(np.abs(Zb))), Ptb))
    # Oracle + cell-split rule: model the batch, then run R2b
    kb, pb, Ptb2 = split_cells_rule(sim, sim["batch"], theta_b, rng, order)
    rows.append(_row("Oracle_batch_then_split_cells", sim, kb, pb, Ptb2))

    for r in rows:
        r.update(scenario=scenario, rep=rep, verdict=dg["verdict"],
                 p_spread=dg["p_spread"], p_conc=dg["p_conc"])
    return rows
