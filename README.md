# nt-guide-screen-sim

A small planted-truth simulation about screening non-targeting (NT) guides in low-MOI single-cell CRISPR screens.

The question: if you test each NT guide for miscalibration and throw out the bad ones until the calibration check looks good, is the check that comes after honest? Or does it pass by construction, even when the real problem is a missing covariate and no guide is off-target at all?

## What it simulates

Each replicate is one fake screen: 300 genes, 10 NT guides, 15 targeting guides, 200 cells per guide (one guide per cell), negative-binomial counts (size 5) with cell library sizes. Each targeting guide knocks down one gene (log fold change -1.5). Four scenarios:

- (a) all NT guides clean
- (b) two NT guides with a planted off-target effect: one represses 5 genes (log fold change -2.0), one represses 3 genes (-1.5)
- (c) no off-target guide, but a batch effect on every gene (log fold change ~ N(0, 0.25^2)) and guides spread unevenly over two batches (share in batch 2 ~ Uniform(0.35, 0.65)); the analysis omits batch
- (d) b and c together

## The test

A negative-binomial score test of "guide cells vs control cells" for each gene. The null model has a library-size offset and an intercept (per batch when batch is modeled), fit in closed form; dispersion is a method-of-moments estimate from the NT cells. The calibration check tests each NT guide against the other NT cells on every gene. It passes if no NT pair survives Bonferroni at 0.1 and the count of p < 0.01 is not above the binomial 99.9% quantile. This is a short Python stand-in for the kind of test sceptre runs, not sceptre itself (no conditional resampling).

## Rules compared

- R0 keep all NT guides
- R1 circular: drop the worst NT guide, re-test, repeat until the check passes, then judge calibration on the same genes
- R2 split genes: screen guides on a random half of the genes, judge calibration and run discovery on the other half, then swap halves (cross-fit)
- R2b split cells: split each NT guide's cells into two random halves, screen guides with one half, judge calibration with the other half, then run discovery on all cells with the kept guides
- R3 diagnose first: if the excess of small NT p-values is spread over many genes and shared across guides (covariate-like), drop nothing and report the check as failed; otherwise run R2b
- Oracle: model the batch (with and without R2b), to show what fixing the model does

"Worst guide" means the guide with the smallest NT p-value (ties broken by its count of p < 0.01). Every rule keeps at least 2 NT guides. A first run ranked guides by their count of p < 0.01 instead; its outputs are kept in `results/v1_count_order/` (made before R2b was added and when R3 used R2; `--order count` reproduces that ordering with the current rules). That order threw out many clean guides in scenario (b), because each NT guide is tested against the other NT cells, so one off-target guide makes the clean guides look bad on its genes too.

The diagnostic compares two statistics with a null made by randomly regrouping NT cells into fake guides of the same sizes (19 draws, so the smallest rank p-value is 0.05). The spread statistic is the top eigenvalue of the NT z-score matrix after removing the 30 genes with the largest |z|; a few-gene off-target signal is removed, a shared covariate signal survives. The concentration statistic is the largest per-guide sum of the top 5 squared z-scores.

## Results (200 replicates per scenario, `results/summary.csv`)

- Omitted covariate (c): the circular rule R1 makes the calibration check pass in 200 of 200 replicates (keeping all guides, R0, the check fails in 73%). It throws out 17.8% of clean NT guides along the way, and false discoveries on targeting pairs stay high (mean FDP 0.357 vs 0.307 for R0 and 0.116 when batch is modeled). This is the self-fulfilling prophecy.
- Splitting does not fix (c): the check still passes in 32.5% (split genes) and 69% (split cells) of replicates, because how unevenly a guide sits across batches is a real property of the guide, so it carries over to the held-out half.
- The diagnostic calls (c) covariate-like in 196 of 200 replicates, and R3 then reports the check as failed (passes in 2%).
- Planted off-target (b): R1, R2b and R3 all catch the planted guides (100%, 100%, 92.5%) and bring mean FDP from 0.320 (R0) to 0.101, 0.102 and 0.119. Here the circular rule works fine, since the problem really is a guide.
- Diagnostic verdicts: (a) no signal 184/200, (b) concentrated 185/200, (c) spread out 196/200, (d) spread out 192/200. In (d) R3 stops at "fix the model first" and does not remove the off-target guide (mean FDP 0.503); modeling batch and then running R2b gives 0.107.

## Outputs

- `results/replicates.csv` - one row per scenario x replicate x rule (`truth_ok` = the check should pass: model right and no planted guide left)
- `results/summary.csv` - averages over 200 replicates per scenario
- `results/diagnostic_verdicts.csv` - diagnostic verdict counts
- `figures/fig1_check_honesty_and_drops.png`, `figures/fig2_false_discoveries_and_diagnostic.png`

Each replicate's seed is (scenario letter, replicate number).

## Run

```
python3 -m pytest -q tests
python3 run_sim.py --reps 200
python3 make_figures.py
```

Needs numpy, scipy, pandas, matplotlib. CPU only; 800 replicates take under 20 minutes on one laptop core.

## Limits

Simulated data only. One data-generating family (gamma-Poisson with one dispersion for every gene), a simple effect model (fixed log fold changes on a few genes), one kind of omitted covariate (a two-level batch with uneven guide shares), and one size of screen. Real off-target effects, real confounders and real NT guide counts can look different, so these numbers say how the rules behave here and nothing more.

Made by Aarush Kandukoori
