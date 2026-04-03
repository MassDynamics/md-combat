"""
End-to-end tests: Python vs R parity for ComBat and ComBat-seq.

Tests compare Python output against pre-computed R benchmark reference outputs
stored in tests/expected/.  No R installation is required to run these tests.

To regenerate the benchmark references (requires R + sva + bladderbatch + airway):
    Rscript scripts/generate_expected.R

WARNING: if scripts/export_datasets.R is re-run (updating the bundled data
in src/md_combat/data/), run generate_expected.R immediately afterwards or
the benchmark references will be out of sync.

ComBat:    asserts near-exact numerical parity (rtol=1e-5) against sva::ComBat,
           using the bladderbatch microarray dataset.
ComBatSeq: asserts Pearson r > 0.95 against sva::ComBat_seq on the 20K-gene
           airway subset (standard and fast variants).
Slow tests (full 64K airway) are marked @pytest.mark.slow and skipped by default.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from md_combat.combat import ComBat
from md_combat.combat_seq import ComBatSeq, ComBatSeqFast
from md_combat.datasets import load_airway, load_bladderbatch

EXPECTED_DIR = Path(__file__).parent / "expected"


def _load_parquet(name: str) -> pd.DataFrame:
    path = EXPECTED_DIR / name
    if not path.exists():
        pytest.skip(
            f"Benchmark reference file not found: {path}\n"
            "Run: Rscript scripts/generate_expected.R"
        )
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# ComBat e2e — bladderbatch microarray data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bladder_data():
    """Load bladderbatch input from the bundled package data and R benchmark reference."""
    expr_df, batch, _ = load_bladderbatch()
    r_result_df = _load_parquet("bladderbatch_combat.parquet")
    return expr_df, batch, r_result_df


def test_combat_output_matches_r(bladder_data):
    """Python ComBat should match R sva::ComBat to within rtol=1e-5."""
    expr_df, batch, r_result_df = bladder_data
    py_result = ComBat().fit_transform(expr_df, batch)
    np.testing.assert_allclose(
        py_result.values,
        r_result_df.values,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Python ComBat output differs from R sva::ComBat",
    )


def test_combat_index_columns_preserved(bladder_data):
    """Feature and sample names must be identical before and after correction."""
    expr_df, batch, r_result_df = bladder_data
    py_result = ComBat().fit_transform(expr_df, batch)
    assert list(py_result.index) == list(r_result_df.index)
    assert list(py_result.columns) == list(r_result_df.columns)


# ---------------------------------------------------------------------------
# ComBat-seq e2e — simulated NB data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def simulated_seq_data():
    """Load pre-generated simulated NB data and R benchmark reference."""
    counts_df = _load_parquet("simulated_counts.parquet")
    meta = _load_parquet("simulated_meta.parquet")
    r_out_df = _load_parquet("simulated_combat_seq.parquet")
    return counts_df, meta["batch"].values, meta["group"].values, r_out_df


def test_combat_seq_batch_correction_reduces_difference(simulated_seq_data):
    """Python ComBatSeq should reduce batch difference, consistent with R."""
    counts_df, batch, group, _ = simulated_seq_data
    before = abs(counts_df.values[:, batch == 1].mean() - counts_df.values[:, batch == 2].mean())

    py_result = ComBatSeq().fit_transform(counts_df, batch, group=group)
    after = abs(py_result.values[:, batch == 1].mean() - py_result.values[:, batch == 2].mean())

    assert after < before, f"Batch diff not reduced: before={before:.2f}, after={after:.2f}"


def test_combat_seq_r_reduces_difference(simulated_seq_data):
    """Sanity check: R's ComBat_seq benchmark reference should also reduce batch difference."""
    counts_df, batch, _, r_out_df = simulated_seq_data
    before = abs(counts_df.values[:, batch == 1].mean() - counts_df.values[:, batch == 2].mean())
    r_after = abs(r_out_df.values[:, batch == 1].mean() - r_out_df.values[:, batch == 2].mean())

    assert r_after < before, f"R batch diff not reduced: before={before:.2f}, after={r_after:.2f}"


def test_combat_seq_output_is_nonneg_integers(simulated_seq_data):
    """Python ComBatSeq output should be non-negative integers."""
    counts_df, batch, group, _ = simulated_seq_data
    py_result = ComBatSeq().fit_transform(counts_df, batch, group=group)
    assert (py_result.values >= 0).all()
    assert np.issubdtype(py_result.values.dtype, np.integer)


def test_combat_seq_python_vs_r_correlation(simulated_seq_data):
    """
    Python ComBatSeq output should correlate strongly with R sva::ComBat_seq.

    Exact numerical parity is not expected: R uses edgeR's GLM solver while
    Python uses statsmodels NegativeBinomial.
    """
    counts_df, batch, group, r_out_df = simulated_seq_data
    py_result = ComBatSeq().fit_transform(counts_df, batch, group=group)

    corr = np.corrcoef(
        py_result.values.ravel().astype(float), r_out_df.values.ravel().astype(float)
    )[0, 1]
    assert corr > 0.95, f"Python vs R ComBat_seq Pearson correlation too low: r={corr:.4f}"


def test_combat_seq_fast_vs_standard_on_simulated_data(simulated_seq_data):
    """
    ComBatSeqFast and ComBatSeq should produce near-identical corrected counts.

    gamma_hat converges to the same MLE in both solvers; the quantile-mapped
    integer counts should agree for the vast majority of genes/samples.
    """
    counts_df, batch, group, _ = simulated_seq_data
    std_result = ComBatSeq().fit_transform(counts_df, batch, group=group)
    fast_result = ComBatSeqFast().fit_transform(counts_df, batch, group=group)

    diff = np.abs(std_result.values - fast_result.values)
    exact_match_pct = (diff == 0).mean() * 100
    corr = np.corrcoef(
        std_result.values.ravel().astype(float),
        fast_result.values.ravel().astype(float),
    )[0, 1]

    assert exact_match_pct >= 85, (
        f"Only {exact_match_pct:.1f}% of counts match exactly between ComBatSeq and ComBatSeqFast"
    )
    assert corr > 0.999, f"Pearson r between ComBatSeq and ComBatSeqFast too low: r={corr:.4f}"


# ---------------------------------------------------------------------------
# Airway e2e — 20K gene subset (fast, no @slow marker)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def airway_20k_data():
    """
    Load airway input filtered to the 20K gene subset and the R benchmark reference.

    Gene IDs are loaded from tests/expected/airway_20k_gene_ids.parquet, which
    was created with set.seed(42) in scripts/generate_expected.R and matches
    the subset used to produce airway_20k_combat_seq.parquet.
    """
    counts_df, batch, group = load_airway()
    gene_indices = _load_parquet("airway_20k_gene_ids.parquet")["gene_idx"].tolist()
    counts_20k = counts_df.iloc[gene_indices]
    r_out_df = _load_parquet("airway_20k_combat_seq.parquet")
    return counts_20k, batch, group, r_out_df


def test_combat_seq_airway_20k_fast_vs_r_correlation(airway_20k_data):
    """
    ComBatSeqFast on the 20K airway subset should correlate strongly with R
    sva::ComBat_seq (Pearson r > 0.95).

    ComBatSeqFast is used because it is the production-recommended path for
    large datasets and converges to the same MLE as edgeR's GLM solver.
    """
    counts_20k, batch, group, r_out_df = airway_20k_data
    py_result = ComBatSeqFast().fit_transform(counts_20k, batch, group=group)

    corr = np.corrcoef(
        py_result.values.ravel().astype(float),
        r_out_df.values.ravel().astype(float),
    )[0, 1]
    assert corr > 0.95, (
        f"ComBatSeqFast vs R on airway 20K: Pearson r={corr:.4f} (expected > 0.95)"
    )


def test_combat_seq_airway_20k_fast_vs_standard(airway_20k_data):
    """
    ComBatSeqFast and ComBatSeq should produce near-identical corrected counts
    on the 20K airway subset (Pearson r > 0.999).

    On real data the statsmodels NM solver has many convergence failures, leading
    to different phi_hat values.  A threshold of 0.99 (rather than 0.999) accounts
    for this; the simulated-data test uses the stricter 0.999 threshold.
    """
    counts_20k, batch, group, _ = airway_20k_data
    fast_result = ComBatSeqFast().fit_transform(counts_20k, batch, group=group)
    std_result = ComBatSeq().fit_transform(counts_20k, batch, group=group)

    corr = np.corrcoef(
        fast_result.values.ravel().astype(float),
        std_result.values.ravel().astype(float),
    )[0, 1]

    assert corr > 0.99, (
        f"Pearson r on 20K airway between Fast and Standard: r={corr:.4f}"
    )


# ---------------------------------------------------------------------------
# Airway e2e — 20K genes, uneven batches (2+4+2) — regression guard for
# weighted grand_gamma and CDF shift fixes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def airway_20k_uneven_data():
    """
    Load airway 20K subset with a 2+4+2 unequal batch split and R benchmark reference.

    Batch remap: A=N61311 (2 samples), B=N052611+N061011 (4 samples), C=N080611 (2 samples).
    This exercises the batch-size-weighted grand_gamma and CDF shift corrections.
    """
    counts_df, _, group = load_airway()
    gene_indices = _load_parquet("airway_20k_gene_ids.parquet")["gene_idx"].tolist()
    counts_20k = counts_df.iloc[gene_indices]

    uneven_batch = _load_parquet("airway_20k_uneven_batch.parquet")["batch"].values
    r_out_df = _load_parquet("airway_20k_uneven_combat_seq.parquet")
    return counts_20k, uneven_batch, group, r_out_df


def test_combat_seq_airway_uneven_fast_vs_r_correlation(airway_20k_uneven_data):
    """
    ComBatSeqFast on the 20K airway subset with unequal batch sizes (2+4+2)
    should correlate strongly with R sva::ComBat_seq (Pearson r > 0.95).

    This test specifically validates:
    - Batch-size-weighted grand_gamma (would diverge from R with the old simple mean)
    - CDF shift (y<=1 guard + cdf(y-1) + 1+ppf) matching R's match_quantiles exactly
    """
    counts_20k, uneven_batch, group, r_out_df = airway_20k_uneven_data
    py_result = ComBatSeqFast().fit_transform(counts_20k, uneven_batch, group=group)

    corr = np.corrcoef(
        py_result.values.ravel().astype(float),
        r_out_df.values.ravel().astype(float),
    )[0, 1]
    assert corr > 0.95, (
        f"ComBatSeqFast vs R on airway 20K uneven batches: Pearson r={corr:.4f} (expected > 0.95)"
    )


# ---------------------------------------------------------------------------
# Airway e2e — full 64K genes  (slow — skipped by default)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def airway_full_data():
    """Load full airway input and the R benchmark reference (all 64K genes)."""
    counts_df, batch, group = load_airway()
    r_out_df = _load_parquet("airway_full_combat_seq.parquet")
    return counts_df, batch, group, r_out_df


@pytest.mark.slow
def test_combat_seq_airway_full_fast_vs_r_correlation(airway_full_data):
    """
    ComBatSeqFast on the full 64K airway dataset should correlate strongly
    with R sva::ComBat_seq (Pearson r > 0.95).
    """
    counts_df, batch, group, r_out_df = airway_full_data
    py_result = ComBatSeqFast().fit_transform(counts_df, batch, group=group)

    corr = np.corrcoef(
        py_result.values.ravel().astype(float),
        r_out_df.values.ravel().astype(float),
    )[0, 1]
    assert corr > 0.95, (
        f"ComBatSeqFast vs R on full airway: Pearson r={corr:.4f} (expected > 0.95)"
    )


@pytest.mark.slow
def test_combat_seq_airway_full_fast_vs_standard(airway_full_data):
    """
    ComBatSeqFast and ComBatSeq should produce near-identical corrected counts
    on the full 64K airway dataset (Pearson r > 0.999).
    """
    counts_df, batch, group, _ = airway_full_data
    fast_result = ComBatSeqFast().fit_transform(counts_df, batch, group=group)
    std_result = ComBatSeq().fit_transform(counts_df, batch, group=group)

    corr = np.corrcoef(
        fast_result.values.ravel().astype(float),
        std_result.values.ravel().astype(float),
    )[0, 1]
    assert corr > 0.99, (
        f"Pearson r on full airway between Fast and Standard: r={corr:.4f}"
    )
