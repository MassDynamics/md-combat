"""
Bundled reference datasets for md_combat.

Datasets are stored as parquet files in the ``data/`` sub-directory and
loaded lazily on first call.  No network access or R installation is
required at runtime.
"""

from pathlib import Path

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).parent / "data"


def load_bladderbatch() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load the bladderbatch microarray dataset.

    Originally published in Dyrskjot et al. (2004).  Distributed via the
    Bioconductor ``bladderbatch`` package (GPL-2).

    Returns
    -------
    expr_df : pd.DataFrame, shape (n_genes, n_samples)
        Log2-normalised expression matrix. Rows = genes, columns = samples.
    batch : np.ndarray, shape (n_samples,)
        Integer batch labels (1–5).
    cancer : np.ndarray, shape (n_samples,)
        Cancer status (0 = normal, 1 = cancer).
    """
    # parquet stored as genes x samples (rows = genes)
    expr_df = pd.read_parquet(_DATA_DIR / "bladderbatch_expr.parquet")
    meta = pd.read_parquet(_DATA_DIR / "bladderbatch_meta.parquet")
    return expr_df, meta["batch"].values, meta["cancer"].values


def load_airway() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load the airway RNA-seq count dataset.

    Himes et al. (2014), GEO accession GSE52778.  Distributed via the
    Bioconductor ``airway`` package (LGPL-2.1+).

    The dataset contains 8 samples from 4 human airway smooth muscle cell
    lines (``cell``), each treated with dexamethasone (``dex``) or untreated.
    Cell line is used as the batch variable; dex treatment is the group variable.

    Returns
    -------
    counts_df : pd.DataFrame, shape (n_genes, n_samples)
        Raw integer count matrix. Rows = genes, columns = samples.
    batch : np.ndarray, shape (n_samples,)
        Cell-line labels (used as batch).
    group : np.ndarray, shape (n_samples,)
        Dexamethasone treatment labels: ``"trt"`` or ``"untrt"``.
    """
    counts_df = pd.read_parquet(_DATA_DIR / "airway_counts.parquet")
    meta = pd.read_parquet(_DATA_DIR / "airway_meta.parquet")
    return counts_df, meta["cell"].values, meta["dex"].values
