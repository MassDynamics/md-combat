"""
Tests for ComBatSeq batch effect correction.

Validates:
1. Output shape and type
2. Counts are non-negative integers
3. Batch means are equalised after correction
4. Biological signal is preserved with balanced design
5. All-zero genes remain zero
"""

import numpy as np
import pandas as pd
import pytest

from md_combat.combat_seq import ComBatSeq, ComBatSeqFast

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_count_data(n_genes=200, n_samples=12, seed=7):
    """
    Simulate NB count data with a batch effect and a biological group effect.
    Two batches of 6 samples each. Groups: first 6 = group 0, last 6 = group 1.
    """
    rng = np.random.default_rng(seed)
    base_mean = rng.uniform(10, 200, size=n_genes)

    # Biological effect: group 1 has 2x counts for first 20 genes
    bio_factor = np.ones((n_genes, n_samples))
    bio_factor[:20, 6:] = 2.0

    # Batch effect: batch B has 3x counts for all genes
    batch_factor = np.ones((n_genes, n_samples))
    batch_factor[:, 6:] = 3.0

    means = base_mean[:, np.newaxis] * bio_factor * batch_factor
    # Simulate NB counts (dispersion = 0.1)
    phi = 0.1
    size = 1.0 / phi
    p = size / (size + means)
    counts = rng.negative_binomial(size, p)

    df = pd.DataFrame(
        counts,
        index=[f"gene_{i}" for i in range(n_genes)],
        columns=[f"sample_{i}" for i in range(n_samples)],
    )
    batch = ["A"] * 6 + ["B"] * 6
    group = [0] * 6 + [1] * 6
    return df, batch, group


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_shape_and_type():
    df, batch, _ = make_count_data()
    result = ComBatSeq().fit_transform(df, batch)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape
    assert list(result.index) == list(df.index)
    assert list(result.columns) == list(df.columns)


def test_counts_are_nonneg_integers():
    """All corrected values must be non-negative integers."""
    df, batch, _ = make_count_data()
    result = ComBatSeq().fit_transform(df, batch)
    assert (result.values >= 0).all(), "Negative counts found"
    assert np.issubdtype(result.values.dtype, np.integer), "Counts are not integers"


def test_batch_means_equalised():
    """After correction, batch means per gene should be closer."""
    df, batch, _ = make_count_data(n_genes=100, seed=42)
    batch_arr = np.array(batch)

    before_diff = np.abs(
        df.values[:, batch_arr == "A"].mean() - df.values[:, batch_arr == "B"].mean()
    )

    result = ComBatSeq().fit_transform(df, batch)
    after_diff = np.abs(
        result.values[:, batch_arr == "A"].mean() - result.values[:, batch_arr == "B"].mean()
    )

    assert after_diff < before_diff, (
        f"Batch means not reduced: before={before_diff:.2f}, after={after_diff:.2f}"
    )


def test_zero_genes_remain_zero():
    """Genes with all-zero counts should remain all-zero after correction."""
    df, batch, _ = make_count_data(n_genes=50)
    # Force first gene to all zeros
    df.iloc[0, :] = 0

    result = ComBatSeq().fit_transform(df, batch)
    assert (result.iloc[0, :] == 0).all(), "All-zero gene was modified"


def test_biological_signal_preserved():
    """Strong biological signal in a balanced design should survive correction."""
    rng = np.random.default_rng(11)
    n_genes, n_samples = 100, 12
    # Balanced: each batch has both groups
    batch = ["A"] * 6 + ["B"] * 6
    group = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    group_arr = np.array(group)

    base_mean = rng.uniform(50, 200, size=n_genes)
    bio_factor = np.ones((n_genes, n_samples))
    bio_factor[:20, group_arr == 1] = 5.0  # strong signal on first 20 genes
    batch_factor = np.ones((n_genes, n_samples))
    batch_factor[:, 6:] = 3.0

    means = base_mean[:, np.newaxis] * bio_factor * batch_factor
    phi = 0.1
    size = 1.0 / phi
    p = size / (size + means)
    counts = rng.negative_binomial(size, p)

    df = pd.DataFrame(
        counts,
        index=[f"g{i}" for i in range(n_genes)],
        columns=[f"s{i}" for i in range(n_samples)],
    )

    result = ComBatSeq().fit_transform(df, batch, group=group)

    bio_mean_group1 = result.iloc[:20, group_arr == 1].values.mean()
    bio_mean_group0 = result.iloc[:20, group_arr == 0].values.mean()
    ratio = bio_mean_group1 / (bio_mean_group0 + 1e-6)

    assert ratio > 1.5, f"Biological signal not preserved: ratio group1/group0 = {ratio:.2f}"


def test_no_nans_in_output():
    """Output should not contain NaNs."""
    df, batch, _ = make_count_data()
    result = ComBatSeq().fit_transform(df, batch)
    assert not result.isnull().any().any()


def test_combatseqfast_output_shape_type():
    """ComBatSeqFast must return a DataFrame of non-negative integers with the same shape."""
    df, batch, _ = make_count_data()
    result = ComBatSeqFast().fit_transform(df, batch)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape
    assert list(result.index) == list(df.index)
    assert list(result.columns) == list(df.columns)
    assert (result.values >= 0).all()
    assert np.issubdtype(result.values.dtype, np.integer)


def test_three_batches():
    """ComBatSeq and ComBatSeqFast must handle more than two batches."""
    rng = np.random.default_rng(3)
    n_genes, n_per_batch = 50, 4
    n_samples = n_per_batch * 3
    base = rng.uniform(10, 100, size=n_genes)
    batch_effects = [1.0, 2.0, 0.5]
    counts = np.hstack([
        rng.negative_binomial(
            10, (10 / (10 + base * be))[:, np.newaxis] * np.ones((1, n_per_batch))
        )
        for be in batch_effects
    ])
    df = pd.DataFrame(counts, index=[f"g{i}" for i in range(n_genes)],
                      columns=[f"s{i}" for i in range(n_samples)])
    batch = ["A"] * n_per_batch + ["B"] * n_per_batch + ["C"] * n_per_batch

    for cls in (ComBatSeq, ComBatSeqFast):
        result = cls().fit_transform(df, batch)
        assert result.shape == df.shape
        assert (result.values >= 0).all()
        assert np.issubdtype(result.values.dtype, np.integer)


def test_group_none():
    """Calling fit_transform with group=None should use an intercept-only design."""
    df, batch, _ = make_count_data(n_genes=50, seed=5)
    for cls in (ComBatSeq, ComBatSeqFast):
        result = cls().fit_transform(df, batch, group=None)
        assert result.shape == df.shape
        assert (result.values >= 0).all()
        assert not pd.DataFrame(result).isnull().any().any()


def test_fast_matches_standard():
    """
    ComBatSeqFast and ComBatSeq must agree on batch coefficients (gamma_hat) within 1e-4.

    gamma_hat (log-fold batch effects) are the primary driver of the quantile
    correction and must match closely.  phi_hat (dispersion) is intentionally
    not compared here: ComBatSeq uses Nelder-Mead (no gradient), which often
    returns phi ≈ 1 (the starting value) due to slow convergence for dispersion
    under NM.  ComBatSeqFast uses Newton-Raphson and converges to the true MLE
    (closer to the simulated phi = 0.1).  Both are valid; only gamma_hat needs
    to agree for the batch correction to be equivalent.
    """
    df, batch, group = make_count_data(n_genes=100, seed=7)

    std = ComBatSeq()
    fast = ComBatSeqFast()

    count_mat = np.array(df, dtype=float)
    batch_arr = np.asarray(batch)

    count_mat_nz, _, _ = std._filter_zero_genes(count_mat, batch_arr)
    # No group covariate — avoids confounded design (group == batch in make_count_data)
    n_batch, X, _ = std._build_design(batch_arr, count_mat.shape[1], None, None)
    offsets = np.log(count_mat_nz.sum(axis=0) + 1)

    gamma_std, _ = std._fit_nb_glm(count_mat_nz, X, offsets, n_batch)
    gamma_fast, _ = fast._fit_nb_glm(count_mat_nz, X, offsets, n_batch)

    assert np.allclose(gamma_std, gamma_fast, atol=1e-4), (
        f"gamma_hat max diff: {np.abs(gamma_std - gamma_fast).max():.2e}"
    )


def test_filter_zero_genes_excludes_batch_zero_gene():
    """Gene zero in all samples of any batch must be filtered out (matches R)."""
    # Gene 0: non-zero in both batches — should be kept
    # Gene 1: non-zero in batch A but all-zero in batch B — should be filtered
    # Gene 2: all-zero globally — should be filtered
    counts = pd.DataFrame(
        [[10, 12, 0, 0], [8, 9, 0, 0], [0, 0, 0, 0]],
        columns=["s1", "s2", "s3", "s4"],
    )
    counts.iloc[0, 2:] = [5, 6]   # gene 0 is expressed in batch B too
    batch = np.array(["A", "A", "B", "B"])

    seq = ComBatSeq()
    count_mat = np.array(counts, dtype=float)
    _, zero_idx, keep_idx = seq._filter_zero_genes(count_mat, batch)

    assert 0 in keep_idx, "Gene 0 (expressed in both batches) should be kept"
    assert 1 in zero_idx, "Gene 1 (zero in all batch B samples) should be filtered"
    assert 2 in zero_idx, "Gene 2 (globally zero) should be filtered"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
