"""
End-to-end tests: Python vs R parity for ComBat and ComBat-seq.

R is invoked as a subprocess (Rscript). Tests are auto-skipped if:
  - Rscript is not found on PATH
  - sva or bladderbatch are not installed in the detected R

For reproducible CI/CD, run inside a container that has R + sva pre-installed.
See CLAUDE.md for the recommended Dockerfile stub.

ComBat:    asserts near-exact numerical parity (rtol=1e-5) against sva::ComBat,
           using the bladderbatch microarray dataset.
ComBatSeq: asserts batch correction is applied against simulated NB data run
           through sva::ComBat_seq. Exact integer equality is not expected —
           R and Python use different NB GLM solvers.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from md_combat.combat import ComBat
from md_combat.combat_seq import ComBatSeq

# ---------------------------------------------------------------------------
# Runtime R detection
# ---------------------------------------------------------------------------

def _find_rscript() -> str | None:
    """Return path to Rscript binary, or None if not found."""
    # Try standard name first, then versioned quick-links created by rig
    for candidate in ["Rscript", "Rscript4.5", "Rscript4.4", "Rscript4.3"]:
        path = shutil.which(candidate)
        if path:
            return path
    return None


_RSCRIPT_OPTS = ["--no-save", "--no-restore", "--quiet"]


def _r_has_packages(rscript: str, *packages: str) -> bool:
    """Return True if all packages are loadable in the given Rscript."""
    check = "; ".join(f'stopifnot(requireNamespace("{p}", quietly=TRUE))' for p in packages)
    result = subprocess.run(
        [rscript, *_RSCRIPT_OPTS, "-e", check],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _rscript_version(rscript: str) -> str:
    result = subprocess.run([rscript, "--version"], capture_output=True, text=True)
    return (result.stdout + result.stderr).splitlines()[0]


# ---------------------------------------------------------------------------
# Shared fixture: detect R at session start, skip everything if unavailable
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rscript() -> str:
    path = _find_rscript()
    if path is None:
        pytest.skip("Rscript not found on PATH — install R to run e2e tests")
    if not _r_has_packages(path, "sva", "bladderbatch", "Biobase"):
        pytest.skip(
            f"R found ({path}) but sva/bladderbatch not installed. "
            "Run: Rscript dependencies.R"
        )
    return path


def _run_r(rscript: str, script: str, timeout: int = 120) -> None:
    """Run an R script string as a subprocess; raise on failure."""
    result = subprocess.run(
        [rscript, *_RSCRIPT_OPTS, "-e", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"R script failed:\n{result.stderr}")


def _read_csv_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path, index_col=0).values


# ---------------------------------------------------------------------------
# ComBat e2e — bladderbatch microarray data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bladder_data(rscript, tmp_path_factory):
    """
    Export bladderbatch data from R and run R's ComBat.
    Returns (expr_df, batch_array, r_result_df).
    """
    tmp = tmp_path_factory.mktemp("bladder")
    expr_csv   = tmp / "expr.csv"
    batch_csv  = tmp / "batch.csv"
    result_csv = tmp / "result.csv"

    _run_r(rscript, f"""
library(sva); library(bladderbatch); library(Biobase)
data(bladderdata, package="bladderbatch")
expr  <- exprs(bladderEset)
batch <- pData(bladderEset)$batch
write.csv(as.data.frame(expr),  "{expr_csv}")
write.csv(data.frame(batch=batch), "{batch_csv}", row.names=FALSE)
corrected <- ComBat(dat=expr, batch=batch)
write.csv(as.data.frame(corrected), "{result_csv}")
""")

    expr_df      = pd.read_csv(expr_csv, index_col=0)
    batch_arr    = pd.read_csv(batch_csv)["batch"].values
    r_result_df  = pd.read_csv(result_csv, index_col=0)

    return expr_df, batch_arr, r_result_df


def test_combat_output_matches_r(bladder_data):
    """Python ComBat should match R sva::ComBat to within rtol=1e-5."""
    expr_df, batch_arr, r_result_df = bladder_data
    py_result = ComBat().fit_transform(expr_df, batch_arr)
    np.testing.assert_allclose(
        py_result.values,
        r_result_df.values,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Python ComBat output differs from R sva::ComBat",
    )


def test_combat_index_columns_preserved(bladder_data):
    """Feature and sample names must be identical before and after correction."""
    expr_df, batch_arr, r_result_df = bladder_data
    py_result = ComBat().fit_transform(expr_df, batch_arr)
    assert list(py_result.index)   == list(r_result_df.index)
    assert list(py_result.columns) == list(r_result_df.columns)


# ---------------------------------------------------------------------------
# ComBat-seq e2e — simulated NB data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def simulated_seq_data(rscript, tmp_path_factory):
    """
    Simulate NB count data in R (set.seed(42)), run R's ComBat_seq,
    and export both input and output for Python comparison.
    """
    tmp = tmp_path_factory.mktemp("seq")
    counts_csv = tmp / "counts.csv"
    meta_csv   = tmp / "meta.csv"
    r_out_csv  = tmp / "r_out.csv"

    # Balanced design: each batch contains both groups (avoids confounding)
    _run_r(rscript, f"""
library(sva)
set.seed(42)
n_genes <- 200; n_samples <- 12
batch <- c(rep(1,6), rep(2,6))
group <- c(1,1,1,2,2,2, 1,1,1,2,2,2)   # balanced: 3 of each group per batch
base_mean <- runif(n_genes, 10, 200)
bio    <- matrix(1, n_genes, n_samples); bio[1:20, group==2]  <- 2
bfact  <- matrix(1, n_genes, n_samples); bfact[, batch==2] <- 3
means  <- base_mean * bio * bfact
counts <- matrix(rnbinom(n_genes*n_samples, mu=means, size=10),
                 nrow=n_genes, ncol=n_samples)
rownames(counts) <- paste0("gene_",  seq_len(n_genes))
colnames(counts) <- paste0("sample_",seq_len(n_samples))
write.csv(as.data.frame(counts), "{counts_csv}")
write.csv(data.frame(batch=batch, group=group), "{meta_csv}", row.names=FALSE)
r_out <- ComBat_seq(counts, batch=batch, group=group)
write.csv(as.data.frame(r_out), "{r_out_csv}")
""")

    counts_df  = pd.read_csv(counts_csv, index_col=0)
    meta       = pd.read_csv(meta_csv)
    r_out_df   = pd.read_csv(r_out_csv,  index_col=0)

    return counts_df, meta["batch"].values, meta["group"].values, r_out_df


def test_combat_seq_r_ran(simulated_seq_data):
    """Fixture loading itself confirms R's ComBat_seq completed without error."""
    _, _, _, r_out_df = simulated_seq_data
    assert r_out_df.shape[0] > 0


def test_combat_seq_batch_correction_reduces_difference(simulated_seq_data):
    """Python ComBatSeq should reduce batch difference, consistent with R."""
    counts_df, batch, group, _ = simulated_seq_data
    before = abs(counts_df.values[:, batch==1].mean() - counts_df.values[:, batch==2].mean())

    py_result = ComBatSeq().fit_transform(counts_df, batch, group=group)
    after = abs(py_result.values[:, batch==1].mean() - py_result.values[:, batch==2].mean())

    assert after < before, f"Batch diff not reduced: before={before:.2f}, after={after:.2f}"


def test_combat_seq_r_reduces_difference(simulated_seq_data):
    """Sanity check: R's ComBat_seq should also reduce batch difference."""
    counts_df, batch, _, r_out_df = simulated_seq_data
    before = abs(counts_df.values[:, batch==1].mean() - counts_df.values[:, batch==2].mean())
    r_after = abs(r_out_df.values[:, batch==1].mean() - r_out_df.values[:, batch==2].mean())

    assert r_after < before, f"R batch diff not reduced: before={before:.2f}, after={r_after:.2f}"


def test_combat_seq_output_is_nonneg_integers(simulated_seq_data):
    """Python ComBatSeq output should be non-negative integers."""
    counts_df, batch, group, _ = simulated_seq_data
    py_result = ComBatSeq().fit_transform(counts_df, batch, group=group)
    assert (py_result.values >= 0).all()
    assert np.issubdtype(py_result.values.dtype, np.integer)
