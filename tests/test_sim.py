import os
import sys

import numpy as np
from scipy.stats import norm, kstest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ntsim.core import (simulate, estimate_theta, score_test, nt_zmatrix, calib_pass,
                        bh, run_replicate)


def _setup(scenario, seed):
    sim = simulate(scenario, np.random.default_rng(seed))
    nt = sim["guide"] < sim["K"]
    groups = np.zeros(sim["Y"].shape[0], dtype=int)
    theta = estimate_theta(sim["Y"][nt], sim["lib"][nt], groups[nt])
    return sim, groups, theta


def test_same_seed_same_data():
    a = simulate("d", np.random.default_rng(7))
    b = simulate("d", np.random.default_rng(7))
    assert np.array_equal(a["Y"], b["Y"])
    assert list(a["bad_guides"]) == list(b["bad_guides"])


def test_truth_labels():
    sa = simulate("a", np.random.default_rng(1))
    sb = simulate("b", np.random.default_rng(1))
    sc = simulate("c", np.random.default_rng(1))
    assert len(sa["bad_guides"]) == 0 and np.all(sa["batch_eff"] == 0)
    assert len(sb["bad_guides"]) == 2 and np.all(sb["batch_eff"] == 0)
    assert len(sc["bad_guides"]) == 0 and np.any(sc["batch_eff"] != 0)


def test_theta_estimate_near_truth():
    sim, groups, theta = _setup("a", 3)
    assert abs(np.median(theta) - 5.0) < 0.5


def test_score_test_null_is_uniform():
    # scenario a: NT guide vs other NT cells should give uniform p-values
    sim, groups, theta = _setup("a", 11)
    Z = nt_zmatrix(sim, list(range(sim["K"])), np.arange(sim["G"]), groups, theta)
    P = 2 * norm.sf(np.abs(Z))
    assert kstest(P.ravel(), "uniform").pvalue > 0.001
    assert abs((P < 0.05).mean() - 0.05) < 0.015


def test_score_test_finds_knockdown():
    sim, groups, theta = _setup("a", 5)
    g = sim["guide"]
    a = g == sim["K"]                       # first targeting guide, knocks down gene 0
    c = g < sim["K"]
    z, p = score_test(sim["Y"], sim["lib"], groups, theta, a, c)
    assert z[0] < -5 and p[0] < 1e-6


def test_planted_offtarget_is_visible():
    sim, groups, theta = _setup("b", 2)
    bad = int(sim["bad_guides"][0])
    genes = sim["off_genes"][bad]
    Z = nt_zmatrix(sim, list(range(sim["K"])), np.arange(sim["G"]), groups, theta)
    assert np.all(Z[bad, genes] < -2)


def test_calib_pass_logic():
    rng = np.random.default_rng(0)
    assert calib_pass(rng.random((10, 300))) in (True, False)
    P = rng.random((10, 300))
    P[0, 0] = 1e-9
    assert not calib_pass(P)


def test_bh():
    p = np.array([0.001, 0.002, 0.5, 0.9])
    assert bh(p, 0.1).tolist() == [True, True, False, False]


def test_replicate_runs():
    rows = run_replicate("b", 0)
    rules = {r["rule"] for r in rows}
    assert {"R0_keep_all", "R1_circular", "R2_split_genes", "R2b_split_cells",
            "R3_diagnose_then_split_cells", "Oracle_model_batch",
            "Oracle_batch_then_split_cells"} == rules


def test_worst_guide_orders():
    from ntsim.core import worst_guide
    P = np.full((3, 100), 0.5)
    P[0, :4] = 0.005          # guide 0: four mild hits
    P[1, 0] = 1e-12           # guide 1: one extreme hit
    assert worst_guide(P, "minp") == 1
    assert worst_guide(P, "count") == 0
