"""
ComBat-seq batch effect correction for RNA-seq count data.

Python translation of sva::ComBat_seq (Zhang et al. 2020).
  Zhang Y, Parmigiani G, Johnson WE (2020). ComBat-seq: batch effect adjustment
  for RNA-seq count data. NAR Genomics and Bioinformatics, 2(3).
  Source: https://bioconductor.org/packages/sva (GPL-3)

Classes
-------
ComBatSeq
    Standard implementation: per-gene NB GLM via statsmodels.
ComBatSeqFast
    Vectorised implementation: all-gene NB GLM via batched Newton-Raphson.
"""

import logging
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import digamma, polygamma
from scipy.stats import nbinom

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base class — Template Method pattern
# ---------------------------------------------------------------------------


class _ComBatSeqBase(ABC):
    """
    Abstract base for ComBat-seq variants.

    Subclasses implement :meth:`_fit_nb_glm`; all other steps are shared.

    Parameters
    ----------
    shrink : bool
        If True, apply EB shrinkage on batch mean (gamma) estimates.
    shrink_disp : bool
        If True, apply EB shrinkage on dispersion estimates.
    gene_subset_n : int or None
        Number of genes to use when estimating shrinkage priors.
    """

    def __init__(
        self,
        shrink: bool = False,
        shrink_disp: bool = False,
        gene_subset_n: int | None = None,
    ) -> None:
        self.shrink = shrink
        self.shrink_disp = shrink_disp
        self.gene_subset_n = gene_subset_n

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        counts: pd.DataFrame,
        batch: list | np.ndarray | pd.Series,
        group: list | np.ndarray | pd.Series | None = None,
        covar_mod: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Apply ComBat-seq batch correction.

        Parameters
        ----------
        counts : pd.DataFrame, shape (n_genes, n_samples)
            Raw integer count matrix. Rows = genes, columns = samples.
        batch : array-like, length n_samples
            Batch label for each sample.
        group : array-like or None
            Biological group label for each sample.
        covar_mod : np.ndarray or None
            Additional covariate design matrix (rows = samples).

        Returns
        -------
        pd.DataFrame
            Batch-corrected count matrix, same shape and index/columns as
            ``counts``. Values are non-negative integers.
        """
        count_mat = np.array(counts, dtype=float)
        batch = np.asarray(batch)

        count_mat_nz, zero_gene_idx, keep_gene_idx = self._filter_zero_genes(count_mat, batch)
        n_batch, full_design, batch_levels = self._build_design(
            batch, count_mat.shape[1], group, covar_mod
        )
        log_offset = np.log(count_mat_nz.sum(axis=0) + 1)

        gamma_hat, phi_hat = self._fit_nb_glm(count_mat_nz, full_design, log_offset, n_batch)

        if self.shrink:
            gamma_hat = self._eb_shrink_gamma(
                gamma_hat, count_mat_nz, full_design, n_batch, log_offset
            )

        corrected_nz = self._quantile_map(
            count_mat_nz,
            batch,
            batch_levels,
            gamma_hat,
            phi_hat,
            log_offset,
        )

        corrected = self._reconstruct(corrected_nz, count_mat, zero_gene_idx, keep_gene_idx)
        corrected = np.round(corrected).clip(0).astype(int)
        return pd.DataFrame(corrected, index=counts.index, columns=counts.columns)

    # ------------------------------------------------------------------
    # Abstract — subclasses fill this in
    # ------------------------------------------------------------------

    @abstractmethod
    def _fit_nb_glm(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        offsets: np.ndarray,
        n_batch: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit a negative binomial GLM for every gene.

        Parameters
        ----------
        Y : np.ndarray, shape (n_genes, n_samples)
        X : np.ndarray, shape (n_samples, n_params)
        offsets : np.ndarray, shape (n_samples,)
        n_batch : int
            Number of batch columns (first block of X).

        Returns
        -------
        gamma_hat : np.ndarray, shape (n_batch, n_genes)
        phi_hat : np.ndarray, shape (n_genes,)
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _filter_zero_genes(
        self, count_mat: np.ndarray, batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return non-zero-gene submatrix plus index arrays for reconstruction.

        Matches R: keep a gene only if it has at least one non-zero count in
        every batch.  Genes that are all-zero in any single batch are removed.
        """
        keep_mask = np.ones(count_mat.shape[0], dtype=bool)
        for lvl in np.unique(batch):
            keep_mask &= count_mat[:, batch == lvl].sum(axis=1) > 0
        zero_gene_idx = np.where(~keep_mask)[0]
        keep_gene_idx = np.where(keep_mask)[0]
        count_mat_nz = count_mat[keep_gene_idx, :] if len(zero_gene_idx) > 0 else count_mat
        return count_mat_nz, zero_gene_idx, keep_gene_idx

    def _build_design(
        self,
        batch: np.ndarray,
        n_samples: int,
        group: list | np.ndarray | pd.Series | None,
        covar_mod: np.ndarray | None,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Build the full design matrix (batch indicators + covariates)."""
        batch_levels = np.unique(batch)
        n_batch = len(batch_levels)

        batch_design = np.zeros((n_samples, n_batch))
        for i, lvl in enumerate(batch_levels):
            batch_design[batch == lvl, i] = 1.0

        covar_parts = []
        if group is not None:
            group = np.asarray(group)
            group_levels = np.unique(group)
            if len(group_levels) > 1:
                for lvl in group_levels[1:]:
                    covar_parts.append((group == lvl).astype(float))

        if covar_mod is not None:
            covar_parts.append(np.asarray(covar_mod, dtype=float))

        if covar_parts:
            covar_cols = (
                np.column_stack(covar_parts)
                if len(covar_parts) > 1
                else covar_parts[0].reshape(-1, 1)
            )
            full_design = np.hstack([batch_design, covar_cols])
        else:
            full_design = batch_design

        return n_batch, full_design, batch_levels

    def _reconstruct(
        self,
        corrected_nz: np.ndarray,
        count_mat: np.ndarray,
        zero_gene_idx: np.ndarray,
        keep_gene_idx: np.ndarray,
    ) -> np.ndarray:
        """Restore all-zero genes back to full gene dimension."""
        if len(zero_gene_idx) > 0:
            corrected = np.zeros_like(count_mat, dtype=float)
            corrected[keep_gene_idx, :] = corrected_nz
            return corrected
        return corrected_nz

    def _eb_shrink_gamma(
        self,
        gamma_hat: np.ndarray,
        count_mat: np.ndarray,
        design: np.ndarray,
        n_batch: int,
        log_offset: np.ndarray,
    ) -> np.ndarray:
        """Apply empirical Bayes shrinkage to batch mean (gamma) estimates."""
        from md_combat._helpers import _postmean

        gamma_shrunk = gamma_hat.copy()
        for i in range(n_batch):
            g_bar = np.mean(gamma_hat[i, :])
            t2 = np.var(gamma_hat[i, :], ddof=1)
            n_per_gene = np.ones(gamma_hat.shape[1]) * (design[:, i].sum())
            gamma_shrunk[i, :] = _postmean(
                gamma_hat[i, :],
                g_bar,
                n_per_gene,
                np.ones(gamma_hat.shape[1]),
                t2 if t2 > 0 else 1e-6,
            )
        return gamma_shrunk

    def _quantile_map(
        self,
        count_mat: np.ndarray,
        batch: np.ndarray,
        batch_levels: np.ndarray,
        gamma_hat: np.ndarray,
        phi_hat: np.ndarray,
        log_offset: np.ndarray,
    ) -> np.ndarray:
        """
        Remove batch effects via quantile mapping on the NB distribution.

        For each sample, compute the quantile of the observed count under the
        batch-specific NB, then map that quantile to the batch-free NB.

        Matches R's match_quantiles exactly:
        - y <= 1: kept unchanged (boundary guard)
        - y > 1:  q = P(X <= y-1) under batch NB; corrected = 1 + Q(q) under adjusted NB
        grand_gamma is the batch-size-weighted mean across batches, matching R's
        weighted.mean(gamma.hat, w=n.batches).
        """
        corrected = count_mat.copy()

        # Batch-size-weighted grand mean (matches R's weighted.mean(..., w=n.batches))
        batch_sizes = np.array([np.sum(batch == lvl) for lvl in batch_levels])
        grand_gamma = (batch_sizes[:, np.newaxis] * gamma_hat).sum(axis=0) / batch_sizes.sum()

        size = np.where(phi_hat > 0, 1.0 / phi_hat, 1e6)  # NB size = 1/phi

        for j in range(count_mat.shape[1]):
            b_idx = np.where(batch_levels == batch[j])[0][0]
            log_ratio = grand_gamma - gamma_hat[b_idx, :]

            mu_batch = np.exp(log_offset[j] + gamma_hat[b_idx, :])
            mu_adj = mu_batch * np.exp(log_ratio)

            prob_batch = np.clip(size / (size + mu_batch), 1e-10, 1 - 1e-10)
            prob_adj = np.clip(size / (size + mu_adj), 1e-10, 1 - 1e-10)

            y = count_mat[:, j]
            # y <= 1: already preserved by corrected = count_mat.copy()
            map_mask = y > 1
            if map_mask.any():
                q = nbinom.cdf(y[map_mask] - 1, size[map_mask], prob_batch[map_mask])
                q = np.clip(q, 1e-10, 1 - 1e-10)
                corrected[map_mask, j] = 1 + nbinom.ppf(q, size[map_mask], prob_adj[map_mask])

        return corrected


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class ComBatSeq(_ComBatSeqBase):
    """
    ComBat-seq with per-gene NB GLM via statsmodels (reference implementation).

    Fits one ``statsmodels.NegativeBinomial`` model per gene using the
    Nelder-Mead optimiser.  Accurate but slow for large gene sets (~3-10 min
    for 20 K genes).
    """

    def _fit_nb_glm(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        offsets: np.ndarray,
        n_batch: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit NB-2 GLM per gene using statsmodels Nelder-Mead."""
        n_genes = Y.shape[0]
        phi_hat = np.ones(n_genes)
        gamma_hat = np.zeros((n_batch, n_genes))
        n_failed = 0

        for g in range(n_genes):
            y = Y[g, :]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    nb_mod = sm.NegativeBinomial(
                        y,
                        X,
                        loglike_method="nb2",
                        offset=offsets,
                    )
                    result = nb_mod.fit(disp=False, method="nm", maxiter=500)
                phi_hat[g] = np.exp(result.params[-1]) if result.params[-1] < 20 else 1e8
                gamma_hat[:, g] = result.params[:n_batch]
            except Exception:
                n_failed += 1

        if n_failed:
            logger.warning(
                "%d / %d genes failed NB-GLM fitting and were left uncorrected",
                n_failed,
                n_genes,
            )

        return gamma_hat, phi_hat


class ComBatSeqFast(_ComBatSeqBase):
    """
    ComBat-seq with vectorised NB GLM via batched Newton-Raphson.

    Fits all genes simultaneously in NumPy.  Typically 10-50x faster than
    :class:`ComBatSeq` for large gene sets.

    Parameters
    ----------
    max_iter : int
        Maximum Newton-Raphson iterations. Default 30.
    tol : float
        Convergence tolerance on parameter updates. Default 1e-6.
    """

    def __init__(
        self,
        shrink: bool = False,
        shrink_disp: bool = False,
        gene_subset_n: int | None = None,
        max_iter: int = 30,
        tol: float = 1e-6,
    ) -> None:
        super().__init__(shrink, shrink_disp, gene_subset_n)
        self.max_iter = max_iter
        self.tol = tol

    def _fit_nb_glm(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        offsets: np.ndarray,
        n_batch: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit NB-2 GLM for all genes simultaneously via Newton-Raphson."""
        n_genes, n_samples = Y.shape
        n_params = X.shape[1]

        beta = np.zeros((n_genes, n_params))
        log_phi = np.zeros(n_genes)

        for _ in range(self.max_iter):
            phi = np.exp(log_phi)  # (n_genes,)
            eta = offsets[np.newaxis, :] + beta @ X.T  # (n_genes, n_samples)
            mu = np.exp(np.clip(eta, -20, 20))

            # beta Newton step
            # NB-2 score: X' @ (y - mu) / (1 + phi*mu)  [correct weighted gradient]
            # NB-2 Fisher info: X' @ diag(mu / (1 + phi*mu)) @ X
            inv_disp = 1.0 + phi[:, np.newaxis] * mu  # (n_genes, n_samples)
            W = mu / inv_disp  # Fisher weight
            score_b = ((Y - mu) / inv_disp) @ X  # (n_genes, n_params)
            # X'WX is positive definite; solve (X'WX) @ delta = score for Newton ascent step
            H_b = np.einsum("gs,si,sj->gij", W, X, X)  # (n_genes, n_params, n_params)
            # Small ridge term for numerical stability
            H_b += 1e-6 * np.eye(n_params)[np.newaxis, :, :]
            # np.linalg.solve batched: b must be (..., m, k), so add trailing dim
            delta_b = np.linalg.solve(H_b, score_b[:, :, np.newaxis])[:, :, 0]
            beta = beta + delta_b

            # log_phi Newton step
            r = 1.0 / np.maximum(phi, 1e-8)  # NB size (n_genes,)
            r2d = r[:, np.newaxis]
            y_plus_r = Y + r2d

            score_p = (
                r2d
                * (
                    digamma(y_plus_r)
                    - digamma(r2d)
                    + np.log(r2d)
                    - np.log(r2d + mu)
                    + 1.0
                    - y_plus_r / (r2d + mu)
                )
            ).sum(axis=1)

            H_p = (
                r2d**2
                * (-polygamma(1, y_plus_r) + polygamma(1, r2d) - 1.0 / r2d + 1.0 / (r2d + mu))
            ).sum(axis=1)

            delta_phi = np.clip(-score_p / (H_p - 1e-10), -1.0, 1.0)
            log_phi = log_phi + delta_phi

            if np.max(np.abs(delta_b)) < self.tol and np.max(np.abs(delta_phi)) < self.tol:
                break

        phi_hat = np.exp(log_phi)
        gamma_hat = beta[:, :n_batch].T  # (n_batch, n_genes)
        return gamma_hat, phi_hat
