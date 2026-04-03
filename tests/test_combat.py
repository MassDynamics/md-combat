"""
Tests for ComBat batch effect correction.

Validates:
1. Output shape and type
2. Batch means are equalised after correction
3. Reference batch is unchanged
4. Mean-only mode doesn't change within-batch variances
5. Biological signal (covariate) is preserved
6. Zero-variance rows pass through unchanged
7. Helper functions produce correct values
"""

import numpy as np
import pandas as pd
import pytest

from md_combat._helpers import _aprior as aprior
from md_combat._helpers import _bprior as bprior
from md_combat._helpers import _it_sol as it_sol
from md_combat._helpers import _postmean as postmean
from md_combat._helpers import _postvar as postvar
from md_combat.combat import combat

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_data(n_features=50, n_samples=12, seed=42):
    """
    Simulate data with a known batch effect and a biological group effect.
    Two batches of 6 samples each. Biological groups: first 6 = group 0, last 6 = group 1.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(5, 1, size=(n_features, n_samples))
    bio = np.zeros((n_features, n_samples))
    bio[:10, 6:] = 2.0
    batch_effect = np.zeros((n_features, n_samples))
    batch_effect[:, 6:] = 3.0

    data = base + bio + batch_effect
    df = pd.DataFrame(
        data,
        index=[f"feat_{i}" for i in range(n_features)],
        columns=[f"sample_{i}" for i in range(n_samples)],
    )
    batch = ["A"] * 6 + ["B"] * 6
    group = [0] * 6 + [1] * 6
    return df, batch, group


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


def test_aprior_bprior_positive():
    gamma = np.array([0.1, 0.3, -0.1, 0.2, 0.0, 0.4])
    assert aprior(gamma) > 0
    assert bprior(gamma) > 0


def test_aprior_raises_on_zero_variance():
    """_aprior and _bprior must raise when all delta values are identical."""
    with pytest.raises(ValueError, match="zero variance"):
        aprior(np.ones(10))


def test_bprior_raises_on_zero_variance():
    with pytest.raises(ValueError, match="zero variance"):
        bprior(np.ones(10))


def test_postmean_shape():
    g_hat = np.array([0.5, -0.2, 0.1])
    result = postmean(g_hat, g_bar=0.0, n=np.array([6, 6, 6]), d_star=np.ones(3), t2=0.5)
    assert result.shape == (3,)


def test_postvar_positive():
    sum2 = np.array([1.0, 2.0, 0.5])
    result = postvar(sum2, n=np.array([6, 6, 6]), a=2.0, b=1.0)
    assert np.all(result > 0)


def test_it_sol_converges():
    rng = np.random.default_rng(0)
    sdat = rng.normal(0, 1, size=(20, 6))
    g_hat = rng.normal(0, 0.3, size=20)
    d_hat = np.ones(20)
    result = it_sol(sdat, g_hat, d_hat, g_bar=0.0, t2=0.5, a=2.0, b=1.0)
    assert result.shape == (2, 20)
    assert np.all(result[1, :] > 0)


# ---------------------------------------------------------------------------
# Main ComBat tests
# ---------------------------------------------------------------------------


def test_output_shape_and_type():
    df, batch, _ = make_data()
    result = combat(df, batch)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape
    assert list(result.index) == list(df.index)
    assert list(result.columns) == list(df.columns)


def test_batch_means_equalised():
    """After correction, batch means should be much closer."""
    df, batch, _ = make_data()
    batch_arr = np.array(batch)

    before_diff = np.abs(
        df.values[:, batch_arr == "A"].mean() - df.values[:, batch_arr == "B"].mean()
    )

    result = combat(df, batch)
    after_diff = np.abs(
        result.values[:, batch_arr == "A"].mean() - result.values[:, batch_arr == "B"].mean()
    )

    assert after_diff < before_diff * 0.1, (
        f"Batch means not equalised: before={before_diff:.3f}, after={after_diff:.3f}"
    )


def test_reference_batch_unchanged():
    """Reference batch samples should be identical before and after."""
    df, batch, _ = make_data()
    result = combat(df, batch, ref_batch="A")
    batch_arr = np.array(batch)
    ref_cols = df.columns[batch_arr == "A"]
    np.testing.assert_array_almost_equal(df[ref_cols].values, result[ref_cols].values, decimal=10)


def test_mean_only_preserves_within_batch_variance():
    """In mean_only mode, within-batch variances should be unchanged."""
    df, batch, _ = make_data()
    batch_arr = np.array(batch)
    result = combat(df, batch, mean_only=True)

    for lvl in ["A", "B"]:
        mask = batch_arr == lvl
        var_before = np.var(df.values[:, mask], axis=1)
        var_after = np.var(result.values[:, mask], axis=1)
        np.testing.assert_allclose(
            var_before,
            var_after,
            rtol=1e-6,
            err_msg=f"Within-batch variance changed in batch {lvl}",
        )


def test_biological_signal_preserved():
    """A strong biological signal should survive batch correction with balanced design."""
    rng = np.random.default_rng(99)
    n_features, n_samples = 50, 12
    batch = ["A"] * 6 + ["B"] * 6
    group = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    group_arr = np.array(group)

    base = rng.normal(5, 1, size=(n_features, n_samples))
    bio = np.zeros((n_features, n_samples))
    bio[:10, group_arr == 1] = 3.0
    batch_effect = np.zeros((n_features, n_samples))
    batch_effect[:, 6:] = 4.0

    data = base + bio + batch_effect
    df = pd.DataFrame(
        data,
        index=[f"feat_{i}" for i in range(n_features)],
        columns=[f"s{i}" for i in range(n_samples)],
    )

    mod = group_arr.reshape(-1, 1).astype(float)
    result = combat(df, batch, mod=mod)

    bio_feats = result.index[:10]
    diff_after = (
        result.loc[bio_feats, result.columns[group_arr == 1]].values.mean()
        - result.loc[bio_feats, result.columns[group_arr == 0]].values.mean()
    )
    assert diff_after > 1.0, f"Biological signal lost: mean diff = {diff_after:.3f}"


def test_zero_variance_rows_unchanged():
    """Features with zero variance in any batch should pass through unchanged."""
    df, batch, _ = make_data()
    batch_arr = np.array(batch)
    df.iloc[0, batch_arr == "A"] = 5.0

    result = combat(df, batch)
    np.testing.assert_array_equal(df.iloc[0].values, result.iloc[0].values)


def test_no_nans_in_output():
    """Output should not contain NaNs for clean input data."""
    df, batch, _ = make_data()
    result = combat(df, batch)
    assert not result.isnull().any().any()


def test_non_parametric_runs():
    """Non-parametric mode should run and return correct shape."""
    df, batch, _ = make_data(n_features=10, n_samples=8)
    batch_small = ["A"] * 4 + ["B"] * 4
    result = combat(df, batch_small, par_prior=False)
    assert result.shape == df.shape
    assert not result.isnull().any().any()


def test_three_batches():
    """Should handle more than two batches."""
    rng = np.random.default_rng(7)
    data = rng.normal(5, 1, size=(20, 9))
    data[:, 3:6] += 2
    data[:, 6:] -= 1
    df = pd.DataFrame(data)
    batch = ["A"] * 3 + ["B"] * 3 + ["C"] * 3
    result = combat(df, batch)
    assert result.shape == df.shape
    assert not result.isnull().any().any()


def test_single_sample_batch_raises():
    """Single-sample batch must raise — user must pass mean_only=True explicitly."""
    df, _, _ = make_data(n_features=10, n_samples=7)
    batch = ["A"] * 6 + ["B"]  # batch B has 1 sample
    with pytest.raises(ValueError, match="mean_only=True"):
        combat(df, batch)


def test_single_sample_batch_allowed_with_mean_only():
    """Single-sample batch is valid when mean_only=True is explicitly set."""
    df, _, _ = make_data(n_features=10, n_samples=7)
    batch = ["A"] * 6 + ["B"]
    result = combat(df, batch, mean_only=True)
    assert result.shape == df.shape
    assert not result.isnull().any().any()


def test_confounded_covariate_raises():
    """A covariate that perfectly mirrors batch must raise ValueError."""
    df, batch, _ = make_data()
    batch_arr = np.array(batch)
    # mod column = 0 for batch A, 1 for batch B — perfectly confounded with batch
    mod = (batch_arr == "B").astype(float).reshape(-1, 1)
    with pytest.raises(ValueError, match="confounded with batch"):
        combat(df, batch, mod=mod)


def test_ref_batch_not_found_raises():
    """Passing an unknown ref_batch must raise ValueError."""
    df, batch, _ = make_data()
    with pytest.raises(ValueError, match="ref_batch"):
        combat(df, batch, ref_batch="Z")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
