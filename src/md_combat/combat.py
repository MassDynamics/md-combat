"""
ComBat batch effect correction.

Python translation of sva::ComBat (Johnson et al. 2007).
  Johnson WE, Li C, Rabinovic A (2007). Adjusting batch effects in microarray
  expression data using empirical Bayes methods. Biostatistics, 8(1), 118-127.
  Source: https://bioconductor.org/packages/sva (GPL-3)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from md_combat._helpers import _aprior, _bprior, _it_sol, _postmean


def _validate_batch_sizes(
    batch_sizes: np.ndarray,
    batch_levels: np.ndarray,
    mean_only: bool,
) -> None:
    """Raise if any batch has only one sample and mean_only correction is not requested."""
    single = batch_levels[batch_sizes == 1]
    if len(single) > 0 and not mean_only:
        raise ValueError(
            f"Batch(es) {list(single)} contain only one sample. "
            "ComBat cannot estimate within-batch variance for single-sample batches. "
            "If you want to correct means only (no variance scaling), "
            "pass mean_only=True explicitly."
        )


class ComBat:
    """
    Adjust for batch effects using an empirical Bayes framework.

    Follows the sklearn estimator convention: fit_transform() applies correction
    in one step. Fitted attributes are stored on the instance after calling
    fit_transform().

    Parameters
    ----------
    par_prior : bool
        If True (default), use parametric EB adjustments.
        If False, use non-parametric (Monte-Carlo) adjustments.
    mean_only : bool
        If True, only correct the mean (no variance/scale adjustment).
    ref_batch : scalar or None
        If given, use this batch as the reference (it will not be changed).
    """

    def __init__(self, par_prior: bool = True, mean_only: bool = False, ref_batch=None) -> None:
        self.par_prior = par_prior
        self.mean_only = mean_only
        self.ref_batch = ref_batch

        # Fitted attributes (set by fit_transform)
        self.gamma_star_: np.ndarray | None = None
        self.delta_star_: np.ndarray | None = None
        self.gamma_bar_: np.ndarray | None = None
        self.t2_: np.ndarray | None = None
        self.a_prior_: np.ndarray | None = None
        self.b_prior_: np.ndarray | None = None

    def fit_transform(
        self,
        dat: pd.DataFrame,
        batch: list | np.ndarray | pd.Series,
        mod: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Apply ComBat batch correction.

        Parameters
        ----------
        dat : pd.DataFrame, shape (n_features, n_samples)
            Expression / intensity matrix. Rows = features, columns = samples.
            Should be log-transformed and normalised before calling.
        batch : array-like, length n_samples
            Batch label for each sample.
        mod : np.ndarray or None
            Model matrix of biological covariates to preserve (excluding batch).
            Rows = samples, columns = covariates.

        Returns
        -------
        pd.DataFrame
            Batch-corrected matrix, same shape and index/columns as `dat`.
        """
        data = np.array(dat, dtype=float)
        batch = np.asarray(batch)
        batch_levels = np.unique(batch)
        n_batch = len(batch_levels)

        batch_idx = {lvl: i for i, lvl in enumerate(batch_levels)}
        batch_int = np.array([batch_idx[b] for b in batch])  # noqa: F841

        print(f"Found {n_batch} batches")

        # --- 1. Find zero-variance rows ---
        zero_rows = set()
        for lvl in batch_levels:
            mask = batch == lvl
            if mask.sum() > 1:
                variances = np.var(data[:, mask], axis=1, ddof=1)
                zero_rows |= set(np.where(variances == 0)[0])

        keep_rows = np.array([i for i in range(data.shape[0]) if i not in zero_rows])

        if len(zero_rows) > 0:
            print(
                f"Found {len(zero_rows)} features with zero variance in a batch; "
                "these will not be adjusted."
            )
            data_orig = data.copy()
            data = data[keep_rows, :]

        n_features, n_samples = data.shape

        # --- 2. Validate batch sizes ---
        batch_sizes = np.array([np.sum(batch == lvl) for lvl in batch_levels])
        mean_only = self.mean_only
        _validate_batch_sizes(batch_sizes, batch_levels, mean_only)
        if mean_only:
            print("Using the 'mean only' version of ComBat")

        # --- 3. Build batch indicator matrix ---
        batchmod = np.zeros((n_samples, n_batch))
        for i, lvl in enumerate(batch_levels):
            batchmod[batch == lvl, i] = 1.0

        ref = None
        if self.ref_batch is not None:
            if self.ref_batch not in batch_levels:
                raise ValueError(f"ref_batch '{self.ref_batch}' not found in batch labels")
            ref = batch_idx[self.ref_batch]
            batchmod[:, ref] = 1.0
            print(f"Using batch '{self.ref_batch}' as reference (it will not change)")

        # --- 4. Build full design matrix ---
        if mod is None:
            design = batchmod.copy()
        else:
            mod = np.asarray(mod, dtype=float)
            design = np.hstack([batchmod, mod])

        intercept_cols = np.where(np.all(design == 1, axis=0))[0]
        if ref is not None:
            intercept_cols = intercept_cols[intercept_cols != ref]
        keep_cols = [c for c in range(design.shape[1]) if c not in intercept_cols]
        design = design[:, keep_cols]

        n_covariates = design.shape[1] - n_batch
        print(f"Adjusting for {n_covariates} covariate(s) or covariate level(s)")

        rank = np.linalg.matrix_rank(design)
        if rank < design.shape[1]:
            if design.shape[1] == n_batch + 1:
                raise ValueError(
                    "The covariate is confounded with batch! "
                    "Remove the covariate and rerun ComBat."
                )
            if design.shape[1] > n_batch + 1:
                covar_only = design[:, n_batch:]
                if np.linalg.matrix_rank(covar_only) < covar_only.shape[1]:
                    raise ValueError(
                        "The covariates are confounded with each other! "
                        "Please remove one or more covariates."
                    )
                else:
                    raise ValueError(
                        "At least one covariate is confounded with batch! "
                        "Please remove confounded covariates and rerun ComBat."
                    )

        # --- 5. Check for missing values ---
        has_na = np.any(np.isnan(data))
        if has_na:
            n_na = int(np.sum(np.isnan(data)))
            print(f"Found {n_na} missing values")

        # --- 6. Standardise data ---
        print("Standardizing data across features")
        XtX = design.T @ design
        XtY = design.T @ data.T
        B_hat = np.linalg.solve(XtX, XtY)

        if ref is not None:
            grand_mean = B_hat[ref, :]
        else:
            grand_mean = (batch_sizes / n_samples) @ B_hat[:n_batch, :]

        if ref is not None:
            ref_mask = batch == batch_levels[ref]
            ref_data = data[:, ref_mask]
            ref_design = design[ref_mask, :]
            resid_ref = ref_data - (ref_design @ B_hat).T
            var_pooled = np.sum(resid_ref**2, axis=1) / ref_mask.sum()
        else:
            resid = data - (design @ B_hat).T
            var_pooled = np.sum(resid**2, axis=1) / n_samples

        stand_mean = np.outer(grand_mean, np.ones(n_samples))
        if design.shape[1] > n_batch:
            tmp = design.copy()
            tmp[:, :n_batch] = 0
            stand_mean += (tmp @ B_hat).T

        sd = np.sqrt(var_pooled)[:, np.newaxis]
        s_data = (data - stand_mean) / sd

        # --- 7. Estimate batch effect parameters ---
        print("Fitting L/S model and finding priors")
        batch_design = design[:, :n_batch]
        gamma_hat = np.linalg.solve(
            batch_design.T @ batch_design,
            batch_design.T @ s_data.T,
        )

        delta_hat = np.ones((n_batch, n_features))
        if not mean_only:
            for i, lvl in enumerate(batch_levels):
                mask = batch == lvl
                delta_hat[i, :] = np.var(s_data[:, mask], axis=1, ddof=1)

        # --- 8. Find empirical Bayes priors ---
        gamma_bar = np.mean(gamma_hat, axis=1)
        t2 = np.var(gamma_hat, axis=1, ddof=1)
        # a_prior / b_prior are only needed for variance scaling (not mean_only).
        if not mean_only:
            a_prior = np.array([_aprior(delta_hat[i, :]) for i in range(n_batch)])
            b_prior = np.array([_bprior(delta_hat[i, :]) for i in range(n_batch)])

        # --- 9. Find EB adjustments ---
        gamma_star = np.zeros_like(gamma_hat)
        delta_star = np.ones_like(delta_hat)

        if self.par_prior:
            print("Finding parametric adjustments")
            for i, lvl in enumerate(batch_levels):
                mask = batch == lvl
                batch_s_data = s_data[:, mask]

                if mean_only:
                    n_i = np.sum(~np.isnan(batch_s_data), axis=1)
                    gamma_star[i, :] = _postmean(
                        gamma_hat[i, :], gamma_bar[i], n_i, delta_hat[i, :], t2[i]
                    )
                    delta_star[i, :] = 1.0
                else:
                    temp = _it_sol(
                        batch_s_data,
                        gamma_hat[i, :],
                        delta_hat[i, :],
                        gamma_bar[i],
                        t2[i],
                        a_prior[i],
                        b_prior[i],
                    )
                    gamma_star[i, :] = temp[0, :]
                    delta_star[i, :] = temp[1, :]
        else:
            print("Finding non-parametric adjustments")
            for i, lvl in enumerate(batch_levels):
                mask = batch == lvl
                batch_s_data = s_data[:, mask]

                g_star_i = np.zeros(n_features)
                d_star_i = np.zeros(n_features)

                for feat in range(n_features):
                    g = np.delete(gamma_hat[i, :], feat)
                    d = np.delete(delta_hat[i, :], feat)
                    x = batch_s_data[feat, :]
                    x = x[~np.isnan(x)]

                    weights = np.array(
                        [np.prod(norm.pdf(x, g[k], np.sqrt(d[k]))) for k in range(len(g))]
                    )
                    weights_sum = weights.sum()
                    if weights_sum == 0:
                        weights = np.ones(len(g)) / len(g)
                    else:
                        weights /= weights_sum

                    g_star_i[feat] = np.dot(weights, g)
                    d_star_i[feat] = np.dot(weights, d)

                gamma_star[i, :] = g_star_i
                delta_star[i, :] = d_star_i if not mean_only else np.ones(n_features)

        if ref is not None:
            gamma_star[ref, :] = 0.0
            delta_star[ref, :] = 1.0

        # Store fitted attributes
        self.gamma_star_ = gamma_star
        self.delta_star_ = delta_star
        self.gamma_bar_ = gamma_bar
        self.t2_ = t2
        self.a_prior_ = a_prior if not mean_only else None
        self.b_prior_ = b_prior if not mean_only else None

        # --- 10. Apply batch correction ---
        print("Adjusting the data")
        bayes_data = s_data.copy()

        for i, lvl in enumerate(batch_levels):
            mask = batch == lvl
            bayes_data[:, mask] = (
                bayes_data[:, mask] - gamma_star[i, :][:, np.newaxis]
            ) / np.sqrt(delta_star[i, :])[:, np.newaxis]

        bayes_data = bayes_data * sd + stand_mean

        if ref is not None:
            ref_mask = batch == batch_levels[ref]
            bayes_data[:, ref_mask] = data[:, ref_mask]

        # --- 11. Restore zero-variance rows ---
        if len(zero_rows) > 0:
            data_orig[keep_rows, :] = bayes_data
            bayes_data = data_orig

        return pd.DataFrame(bayes_data, index=dat.index, columns=dat.columns)


def combat(
    dat: pd.DataFrame,
    batch: list | np.ndarray | pd.Series,
    mod: np.ndarray | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch=None,
) -> pd.DataFrame:
    """
    Functional shim around ComBat — preserves the original function API.

    See ComBat.fit_transform() for full documentation.
    """
    return ComBat(
        par_prior=par_prior, mean_only=mean_only, ref_batch=ref_batch
    ).fit_transform(dat, batch, mod)
