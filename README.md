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

Requirements: Python ≥ 3.13, numpy, pandas, scipy, statsmodels.

---

## How to use

### ComBat (microarray / log-expression)

```python
import pandas as pd
from md_combat import ComBat

# dat:   DataFrame of shape (n_features, n_samples)
#        rows = genes/probes, columns = samples
#        should be log-transformed and normalised before calling
# batch: list or array of length n_samples, one label per sample

corrected = ComBat().fit_transform(dat, batch)
```

**Preserve a biological covariate** (avoids removing true biology when
building the correction model):

```python
import numpy as np

# mod: (n_samples, n_covariates) design matrix — do NOT include an intercept
group = [0, 0, 1, 1, 0, 0, 1, 1]           # e.g. treatment vs control
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
from md_combat import ComBatSeq

# counts: DataFrame of shape (n_genes, n_samples), non-negative integers
# batch:  list or array of batch labels

corrected = ComBatSeq().fit_transform(counts, batch)
```

**With a biological group** (recommended for balanced designs):

```python
group     = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]   # balanced across batches
corrected = ComBatSeq().fit_transform(counts, batch, group=group)
```

**With additional covariates**:

```python
import numpy as np

covar = np.array([[1, 0], [0, 1], ...])   # (n_samples, n_covariates)
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

## End-to-end parity with R

The test suite includes end-to-end tests that run the original R
implementations and compare their output to Python's.

### How it works

Tests in `tests/test_combat_e2e.py` use **subprocess + Rscript**:

1. Python writes the input data to a temp CSV file.
2. `Rscript` is called as a subprocess with an inline R script that loads the
   data, runs `sva::ComBat` or `sva::ComBat_seq`, and writes the result to
   another CSV.
3. Python reads the result back and asserts parity.

This avoids any native-library binding issues (no `rpy2` required at runtime)
and works with any R version on `PATH`.

### ComBat parity

Uses the real `bladderbatch` microarray dataset (57 samples, 5 batches) from
Bioconductor:

```python
np.testing.assert_allclose(py_result, r_result, rtol=1e-5, atol=1e-8)
```

Because `ComBat` is a line-for-line translation of the R source, numerical
agreement is near-exact (differences are floating-point rounding only).

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
# Requires R + sva + bladderbatch installed
Rscript dependencies.R          # install R packages (first time only)

uv run pytest tests/test_combat_e2e.py -v
```

Tests skip automatically if `Rscript` is not found on `PATH`.

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
