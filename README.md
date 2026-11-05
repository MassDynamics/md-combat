# md_combat

Python implementation of batch effect correction for genomics data.

Provides `ComBat` (microarray / continuous data) and `ComBatSeq` (RNA-seq
count data), translated from the R package
[`sva`](https://bioconductor.org/packages/sva) (Bioconductor, GPL-3).

---

## Background

Batch effects are systematic, non-biological differences between samples
processed in different experimental batches (different runs, labs, or dates).
Left uncorrected they can dominate downstream analysis.

- **ComBat** — empirical Bayes (EB) framework for continuous expression data
  (e.g. microarray, log-normalised RNA-seq). Estimates and removes additive and
  multiplicative batch effects. Reference: Johnson et al., *Biostatistics*
  2007.
- **ComBatSeq** — negative binomial GLM adaptation for raw RNA-seq count data.
  Preserves the integer nature of counts and handles library-size differences.
  Reference: Zhang et al., *NAR Genomics and Bioinformatics* 2020.

---

## Installation

```bash
pip install md-combat          # from PyPI (when published)

# or from source
git clone https://github.com/your-org/md_combat
cd md_combat
uv sync
```

Requirements: Python ≥ 3.11, numpy, pandas, scipy, statsmodels.

---

## How to use

### ComBat (microarray / log-expression)

```python
import pandas as pd
from md_combat import ComBat, load_bladderbatch

# dat:   DataFrame of shape (n_features, n_samples)
#        rows = genes/probes, columns = samples
#        should be log-transformed and normalised before calling
# batch: list or array of length n_samples, one label per sample

dat, batch, cancer = load_bladderbatch()
batch = [f"batch{int(b)}" for b in batch]  # 1.0, 2.0, ... -> "batch1", "batch2", ...
corrected = ComBat().fit_transform(dat, batch)
```

**Preserve a biological covariate** (avoids removing true biology when
building the correction model):

```python
import numpy as np

# mod: (n_samples, n_covariates) design matrix — do NOT include an intercept
group = np.random.default_rng(0).integers(0, 2, size=dat.shape[1])  # random 0/1 per sample, e.g. treatment vs control
mod   = np.array(group).reshape(-1, 1).astype(float)
corrected = ComBat().fit_transform(dat, batch, mod=mod)
```

**Mean-only correction** (use when a batch has very few samples or you only
want to shift means without rescaling variance):

```python
corrected = ComBat(mean_only=True).fit_transform(dat, batch)
```

**Reference batch** (keep one batch unchanged and adjust all others toward it):

```python
corrected = ComBat(ref_batch="batch1").fit_transform(dat, batch)
```

**Non-parametric EB** (slower but makes no distributional assumption on the
batch effects):

```python
corrected = ComBat(par_prior=False).fit_transform(dat, batch)
```

**Functional API** (drop-in replacement for scripts that used the original
`combat()` function):

```python
from md_combat import combat

corrected = combat(dat, batch)
corrected = combat(dat, batch, mod=mod, mean_only=True, ref_batch="batch1")
```

**Inspect fitted parameters** after correction:

```python
model = ComBat()
corrected = model.fit_transform(dat, batch)

model.gamma_star_   # (n_batch, n_features) — additive batch effects
model.delta_star_   # (n_batch, n_features) — multiplicative batch effects
model.gamma_bar_    # (n_batch,)             — prior mean for gamma
model.t2_           # (n_batch,)             — prior variance for gamma
model.a_prior_      # (n_batch,)             — inverse-gamma shape prior
model.b_prior_      # (n_batch,)             — inverse-gamma scale prior
```

---

### ComBatSeq (RNA-seq raw counts)

```python
from md_combat import ComBatSeq, load_airway

# counts: DataFrame of shape (n_genes, n_samples), non-negative integers
# batch:  list or array of batch labels
# group:  biological condition per sample (airway: dexamethasone "trt" / "untrt")

counts, batch, group = load_airway()
corrected = ComBatSeq().fit_transform(counts, batch)
```

**With a biological group** (recommended for balanced designs):

```python
# each cell line (batch) has one treated and one untreated sample
corrected = ComBatSeq().fit_transform(counts, batch, group=group)
```

**With additional covariates**:

```python
import numpy as np

covar = np.column_stack([
    (group == "trt").astype(float),                          # treatment as 0/1
    np.random.default_rng(0).normal(size=counts.shape[1]),   # e.g. a continuous covariate
])
corrected = ComBatSeq().fit_transform(counts, batch, covar_mod=covar)
```

The output is a `DataFrame` of the same shape as `counts`, with non-negative
integer values.

---

## Input requirements

| | ComBat | ComBatSeq |
|---|---|---|
| Data type | `float` (log-scale recommended) | Non-negative integers |
| Shape | `(n_features, n_samples)` | `(n_genes, n_samples)` |
| Min samples per batch | 2 (or 1 with `mean_only=True`) | 2 |
| Missing values | `NaN` tolerated | Not supported |

---

## R setup (for e2e tests)

The unit tests have no R dependency. The e2e parity tests require R and the
`sva` / `bladderbatch` Bioconductor packages. Follow these steps once per
machine.

### 1 — Install R

**macOS (recommended):**

```bash
# Install rig — the R version manager (requires sudo password once)
brew install --cask rig

# Install the latest R release
rig add release

# Verify
Rscript --version
```

`brew` not installed? See https://brew.sh

**Other platforms:** download the installer from https://cran.r-project.org.
After installation `Rscript` should be on your `PATH`.

### 2 — Install required R packages

```bash
# From the repo root
Rscript dependencies.R
```

This installs `sva`, `bladderbatch`, and `Biobase` via Bioconductor. It takes
a few minutes the first time (some packages compile from source).

### 3 — Verify

```bash
Rscript -e "library(sva); library(bladderbatch); cat('R env OK\n')"
```

Benchmark reference Parquet files are committed in the repository; missing files cause failures,
not skips.

---

## Application note and demo notebook

- **Application note (Markdown):** [docs/APP_NOTE_PYTHON_R_PARITY.md](docs/APP_NOTE_PYTHON_R_PARITY.md) — Python vs R parity on full bladderbatch and full airway. Optional YAML for static-site builds: [docs/APP_NOTE_SSG_METADATA.yaml](docs/APP_NOTE_SSG_METADATA.yaml).
- **Figures:** produced by [examples/combat_comparison.ipynb](examples/combat_comparison.ipynb). Run with `uv sync --extra dev` and Jupyter. **Warning:** the notebook runs **ComBatSeq** (standard) on the **full ~64k-gene airway** matrix by default; that step can take **tens of minutes to hours**. Outputs are written to `docs/figures/parity/`.

---

## End-to-end parity with R

The test suite includes end-to-end tests that compare Python output to
**pre-computed R benchmark reference** Parquet files under `tests/expected/`.

### How it works

Tests in `tests/test_combat_e2e.py` load inputs from the bundled datasets and
benchmark reference outputs generated by `scripts/generate_expected.R`. **No R runtime is
required** when running tests — only when regenerating those references.

### ComBat parity

Uses the real `bladderbatch` microarray dataset (57 samples, 5 batches) from
Bioconductor:

```python
np.testing.assert_allclose(py_result, r_result, rtol=1e-5, atol=1e-8)
```

Because `ComBat` is a line-for-line translation of the R source, numerical
agreement is near-exact (differences are floating-point rounding only), with
the two deliberate exceptions below.

### Known differences from `sva::ComBat`

1. **Mean-only shrinkage (`mean_only=True`, parametric).** Both shrink each
   batch's mean offset towards the average across batches with
   `(t2 * n * gamma_hat + d * gamma_bar) / (t2 * n + d)`. sva's mean-only
   branch passes `n = 1, d = 1` for every batch. md_combat passes `d = 1` and
   `n` = the number of samples in the batch, which is the usual empirical Bayes
   posterior mean (a mean over `n` standardised values has variance `1/n`).
   md_combat therefore shrinks less and removes more of each batch mean, and
   the gap grows with batch size. On a 479 x 75 metabolomics benchmark it was
   up to 1.24 log2 (batches of 1–4 samples) and 0.99 log2 (batches of 25).
   With `n` forced to 1, the two agree to 1e-13. Full ComBat
   (`mean_only=False`) is unaffected and matches sva, with and without
   `ref_batch` and with categorical, numeric and mixed `mod`.
2. **Single-sample batches.** sva prints a note and switches to mean-only
   ComBat on its own. md_combat raises `ValueError` and asks for
   `mean_only=True` explicitly, so the change in method is never silent.

Neither difference is covered by the R reference outputs in `tests/expected/`,
which only hold full ComBat without `mod` or `ref_batch`.

### ComBatSeq behaviour that matches `sva::ComBat_seq`

- **Genes all-zero in a batch** are left out of the fit and returned with their
  original counts, not batch-adjusted. A warning reports how many of them have
  counts in another batch; filter low-expressed genes before ComBat-Seq to
  avoid them. (Before 2026-11-05 these genes came back as all zeros; see
  `IMPLEMENTATION.md`, ComBat-seq Step 6.)
- **Single-sample batches** raise `ValueError`, as in R.
- **Confounded designs** (a covariate or `group` nested in the batch, or
  covariates confounded with each other) raise `ValueError` with R's messages,
  instead of being fitted silently through the small ridge term in the NB fit.
- All-ones columns in `covar_mod` are dropped, as in R.

### ComBatSeq parity

Uses simulated NB count data with the same random seed in both R and Python
(`set.seed(42)` / `np.random.default_rng(42)`). The assertion checks that
Python reduces batch differences in the same direction as R:

```python
assert after_diff < before_diff
```

Exact integer equality is **not** asserted, because R uses `edgeR`'s
Cox-Reid dispersion estimator while Python uses `statsmodels.NegativeBinomial`.
The algorithms are equivalent in intent but produce different numerical
estimates. See `IMPLEMENTATION.md` for the full explanation.

### Running the e2e tests

```bash
uv run pytest tests/test_combat_e2e.py -v
```

Regenerating benchmark reference outputs requires R; see `scripts/generate_expected.R` and
`CLAUDE.md`.

---

## Running all tests

```bash
uv sync --extra dev

# Unit tests only (no R required)
uv run pytest tests/test_combat.py tests/test_combat_seq.py -v

# Full suite including e2e
uv run pytest -v

# With coverage
uv run pytest --cov=md_combat --cov-report=term-missing
```

---

## Project structure

```
src/md_combat/
├── __init__.py       # public API: ComBat, combat, ComBatSeq
├── _helpers.py       # shared EB helpers: _aprior, _bprior, _postmean, _postvar, _it_sol
├── combat.py         # ComBat class + combat() functional shim
└── combat_seq.py     # ComBatSeq class

tests/
├── test_combat.py      # unit tests for ComBat and helpers
├── test_combat_seq.py  # unit tests for ComBatSeq
└── test_combat_e2e.py  # R parity tests via subprocess

IMPLEMENTATION.md  # line-by-line R ↔ Python translation reference
```

---

## References

- Johnson WE, Li C, Rabinovic A (2007). Adjusting batch effects in microarray
  expression data using empirical Bayes methods. *Biostatistics*, 8(1), 118–127.
- Zhang Y, Parmigiani G, Johnson WE (2020). ComBat-seq: batch effect adjustment
  for RNA-seq count data. *NAR Genomics and Bioinformatics*, 2(3).
- Leek JT et al. (2012). The `sva` package for removing batch effects and other
  unwanted variation in high-throughput experiments. *Bioinformatics*, 28(6).
