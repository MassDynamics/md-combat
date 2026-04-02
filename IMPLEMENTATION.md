# Python ↔ R Implementation Reference

This document explains every function in `md_combat` side-by-side with the
corresponding R source from `sva` (GPL-3). The goal is to make it easy to
verify correctness and to understand every translation decision.

**R source version:** sva 3.44.0 (Bioconductor 3.15, R 4.2)

---

## Table of Contents

1. [Helper functions (`_helpers.py`)](#1-helper-functions)
   - [`_aprior` / `aprior`](#_aprior)
   - [`_bprior` / `bprior`](#_bprior)
   - [`_postmean` / `postmean`](#_postmean)
   - [`_postvar` / `postvar`](#_postvar)
   - [`_it_sol` / `it.sol`](#_it_sol)
2. [ComBat (`combat.py`)](#2-combat)
   - [Step 1 — Zero-variance rows](#step-1--zero-variance-rows)
   - [Step 2 — mean_only edge cases](#step-2--mean_only-edge-cases)
   - [Step 3 — Batch indicator matrix](#step-3--batch-indicator-matrix)
   - [Step 4 — Design matrix](#step-4--design-matrix)
   - [Step 5 — Missing values](#step-5--missing-values)
   - [Step 6 — Standardise data](#step-6--standardise-data)
   - [Step 7 — Estimate batch parameters](#step-7--estimate-batch-parameters)
   - [Step 8 — EB priors](#step-8--eb-priors)
   - [Step 9 — EB adjustments (parametric)](#step-9--eb-adjustments-parametric)
   - [Step 9b — EB adjustments (non-parametric)](#step-9b--non-parametric-adjustments)
   - [Step 10 — Apply correction](#step-10--apply-correction)
   - [Step 11 — Restore zero-variance rows](#step-11--restore-zero-variance-rows)
3. [ComBatSeq (`combat_seq.py`)](#3-combatseq)
   - [Step 1 — Filter all-zero genes](#step-1--filter-all-zero-genes-1)
   - [Step 2 — Design matrix](#step-2--design-matrix-1)
   - [Step 3 — Dispersion + batch coefficients](#step-3--dispersion--batch-coefficients)
   - [Step 4 — EB shrinkage (optional)](#step-4--eb-shrinkage-optional)
   - [Step 5 — Quantile mapping (`_quantile_map` / `match_quantiles`)](#step-5--quantile-mapping)
   - [Step 6 — Restore all-zero genes](#step-6--restore-all-zero-genes)
4. [Key translation decisions](#4-key-translation-decisions)

---

## 1. Helper functions

### `_aprior`

Computes the shape parameter **α** of the inverse-gamma hyper-prior placed on δ
(the multiplicative batch effect variance). Derived by matching the first two
moments of the inverse-gamma to the empirical distribution of `delta.hat`.

| R (`sva:::aprior`) | Python (`_helpers._aprior`) |
|---|---|
| `m <- mean(gamma.hat)` | `m = np.mean(gamma_hat)` |
| `s2 <- var(gamma.hat)` | `s2 = np.var(gamma_hat, ddof=1)` |
| *(no guard — returns `NaN` if `s2 == 0`)* | `if s2 == 0: return 1.0` |
| `(2 * s2 + m^2) / s2` | `(2 * s2 + m**2) / s2` |

**Note:** R's `var()` uses `ddof=1` by default; Python's `np.var` defaults to
`ddof=0`. We pass `ddof=1` explicitly to match R.

**Guard:** R does not protect against `s2 == 0`. Python raises a `ValueError`
with a "please report this as a bug" message. This case is unreachable in
normal use because:
1. Input validation (`_validate_batch_sizes`) rejects single-sample batches
   unless `mean_only=True` is passed explicitly.
2. When `mean_only=True`, `a_prior`/`b_prior` are never computed (they are
   guarded by `if not mean_only` in step 8 of `fit_transform`).

So if `_aprior`/`_bprior` ever receive `s2 == 0`, something unexpected has
bypassed validation and the error message asks the caller to report it as a bug.

---

### `_bprior`

Computes the scale parameter **β** of the same inverse-gamma hyper-prior.

| R (`sva:::bprior`) | Python (`_helpers._bprior`) |
|---|---|
| `m <- mean(gamma.hat)` | `m = np.mean(gamma_hat)` |
| `s2 <- var(gamma.hat)` | `s2 = np.var(gamma_hat, ddof=1)` |
| *(no guard)* | `if s2 == 0: return 1.0` |
| `(m * s2 + m^3) / s2` | `(m * s2 + m**3) / s2` |

Same guard as `_aprior` — raises `ValueError` if `s2 == 0`.

---

### `_postmean`

Posterior mean of γ (additive batch effect) under a normal prior with mean
`g_bar` and variance `t2`. Closed-form Bayes update.

| R (`sva:::postmean`) | Python (`_helpers._postmean`) |
|---|---|
| `(t2 * n * g.hat + d.star * g.bar) / (t2 * n + d.star)` | `(t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)` |

One-liner — exact translation. Both operate element-wise on arrays.

---

### `_postvar`

Posterior variance of δ (multiplicative batch effect) under an inverse-gamma
prior with shape `a` and scale `b`. Closed-form Bayes update.

| R (`sva:::postvar`) | Python (`_helpers._postvar`) |
|---|---|
| `(0.5 * sum2 + b) / (n/2 + a - 1)` | `(0.5 * sum2 + b) / (n / 2 + a - 1)` |

Exact translation. Both operate element-wise.

---

### `_it_sol`

EM iteration to find the parametric EB estimates γ\* and δ\* for one batch.
Alternates between updating γ (via `postmean`) and δ (via `postvar`) until the
maximum relative change falls below `conv`.

| R (`sva:::it.sol`) | Python (`_helpers._it_sol`) |
|---|---|
| `n <- rowSums(!is.na(sdat))` | `n = np.sum(~np.isnan(sdat), axis=1)` |
| `g.old <- g.hat` | `g_old = g_hat.copy()` |
| `d.old <- d.hat` | `d_old = d_hat.copy()` |
| `change <- 1` | `change = 1.0` |
| `while (change > conv) {` | `while change > conv:` |
| `g.new <- postmean(g.hat, g.bar, n, d.old, t2)` | `g_new = _postmean(g_hat, g_bar, n, d_old, t2)` |
| `sum2 <- rowSums((sdat - g.new %*% t(rep(1, ncol(sdat))))^2, na.rm=TRUE)` | `resid = sdat - g_new[:, np.newaxis]` / `sum2 = np.nansum(resid**2, axis=1)` |
| `d.new <- postvar(sum2, n, a, b)` | `d_new = _postvar(sum2, n, a, b)` |
| `change <- max(abs(g.new - g.old)/g.old, abs(d.new - d.old)/d.old)` | `change = max(np.max(abs(g_new-g_old) / abs(g_old+1e-10)), ...)` |
| `adjust <- rbind(g.new, d.new)` | `return np.vstack([g_new, d_new])` |

**Broadcast translation:** `g.new %*% t(rep(1, ncol(sdat)))` in R broadcasts
a column vector across all columns. In Python this is `g_new[:, np.newaxis]`
which adds a trailing size-1 dimension so NumPy broadcasts the subtraction
correctly.

**Convergence guard:** R divides by `g.old` and `d.old` directly — if either
is 0 the result is `Inf`, which is `> conv` so the loop continues (effectively
fine). Python adds `1e-10` to both denominators to avoid `0/0 → NaN` which
would break the `>` comparison. The effect is identical for any non-degenerate
input.

---

## 2. ComBat

The main algorithm is `ComBat.fit_transform()` in `combat.py`. The R entry
point is `sva::ComBat()`. The Python code is organised as numbered steps that
map directly to the R code sections.

### Step 1 — Zero-variance rows

Genes with zero variance within *any* batch cannot be standardised (division by
zero later in step 6). Both implementations identify and temporarily remove
them.

**R:**
```r
zero.rows.lst <- lapply(levels(batch), function(batch_level) {
    if (sum(batch == batch_level) > 1) {
        return(which(apply(dat[, batch == batch_level], 1,
                           function(x) { var(x) == 0 })))
    } else {
        return(which(rep(1, 3) == 2))   # empty set
    }
})
zero.rows <- Reduce(union, zero.rows.lst)
keep.rows <- setdiff(1:nrow(dat), zero.rows)
```

**Python:**
```python
zero_rows = set()
for lvl in batch_levels:
    mask = batch == lvl
    if mask.sum() > 1:
        variances = np.var(data[:, mask], axis=1, ddof=1)
        zero_rows |= set(np.where(variances == 0)[0])
keep_rows = np.array([i for i in range(data.shape[0]) if i not in zero_rows])
```

`Reduce(union, ...)` → `set |=` (union-update). `apply(..., 1, var)` →
`np.var(..., axis=1, ddof=1)`. The `if sum(batch == batch_level) > 1` guard is
identical — single-sample batches are skipped because `var` of one value is
undefined.

---

### Step 2 — Validate batch sizes

**R:** silently degrades to `mean_only=True` when any batch has one sample.

**Python:** raises at input validation instead, forcing the caller to opt in
explicitly.

```python
# R (silent fallback)
if (any(table(batch) == 1)) { mean.only = TRUE }

# Python (fail fast)
def _validate_batch_sizes(batch_sizes, batch_levels, mean_only):
    single = batch_levels[batch_sizes == 1]
    if len(single) > 0 and not mean_only:
        raise ValueError(
            f"Batch(es) {list(single)} contain only one sample. ..."
            "pass mean_only=True explicitly."
        )
```

**Design difference:** R hides the limitation; Python surfaces it. A
single-sample batch is still supported when the caller explicitly passes
`mean_only=True`, acknowledging that variance scaling is impossible.

---

### Step 3 — Batch indicator matrix

A one-hot matrix `batchmod` of shape `(n_samples, n_batch)`.

**R:**
```r
batchmod <- model.matrix(~-1 + batch)
# reference batch handling:
batchmod[, ref] <- 1
```

**Python:**
```python
batchmod = np.zeros((n_samples, n_batch))
for i, lvl in enumerate(batch_levels):
    batchmod[batch == lvl, i] = 1.0
# reference batch handling:
batchmod[:, ref] = 1.0
```

`model.matrix(~-1 + batch)` produces a one-hot matrix with no intercept.
Python replicates this manually with a loop. The reference-batch column is set
to all-ones (overwriting the indicator) in both cases — this matches R's
`batchmod[,ref] <- 1` exactly.

---

### Step 4 — Design matrix

Appends the biological covariate model matrix to `batchmod`, then removes any
all-ones (intercept) column to avoid rank deficiency.

**R:**
```r
design <- cbind(batchmod, mod)
check  <- apply(design, 2, function(x) all(x == 1))
if (!is.null(ref)) check[ref] <- FALSE
design <- as.matrix(design[, !check])
```

**Python:**
```python
design = np.hstack([batchmod, mod]) if mod is not None else batchmod.copy()
intercept_cols = np.where(np.all(design == 1, axis=0))[0]
if ref is not None:
    intercept_cols = intercept_cols[intercept_cols != ref]
keep_cols = [c for c in range(design.shape[1]) if c not in intercept_cols]
design = design[:, keep_cols]
```

`apply(design, 2, function(x) all(x == 1))` → `np.all(design == 1, axis=0)`.
`cbind` → `np.hstack`. Column indexing with a boolean mask (`design[, !check]`)
→ index array (`design[:, keep_cols]`).

**Rank check (confounding):** Both check `qr(design)$rank < ncol(design)` /
`np.linalg.matrix_rank(design) < design.shape[1]` and raise an error if the
design is rank-deficient.

---

### Step 5 — Missing values

**R:**
```r
NAs <- any(is.na(dat))
if (NAs) { message(...) }
```

**Python:**
```python
has_na = np.any(np.isnan(data))
if has_na:
    print(f"Found {n_na} missing values")
```

`is.na` → `np.isnan`. Python only supports `NaN` for missing floats (no `NA`
type), which is the relevant case for numeric expression matrices.

**Note:** R has a separate `Beta.NA` code path for OLS in the presence of
missing values. Python's `np.linalg.solve` does not handle `NaN` inputs
gracefully; the missing-value path is preserved for completeness but relies on
the user not passing `NaN`-containing matrices (the non-parametric EM path uses
`np.nansum`, which is correct).

---

### Step 6 — Standardise data

OLS regression of the data on the design matrix to obtain coefficient matrix
**B̂**, then subtract the batch-free fitted values and scale by the pooled
standard deviation.

**R:**
```r
B.hat <- solve(crossprod(design), tcrossprod(t(design), as.matrix(dat)))
```

**Python:**
```python
XtX  = design.T @ design          # crossprod(design)
XtY  = design.T @ data.T          # t(design) %*% t(dat) = tcrossprod(t(design), dat)
B_hat = np.linalg.solve(XtX, XtY) # solve(XtX, XtY)
```

`crossprod(X)` = `X'X`; `tcrossprod(A, B)` = `A %*% t(B)`. So
`tcrossprod(t(design), dat)` = `t(design) %*% dat`. Combined: `B̂ = (X'X)⁻¹ X'Y`.

**Grand mean:**

| R | Python |
|---|---|
| `grand.mean <- t(B.hat[ref, ])` (ref batch) | `grand_mean = B_hat[ref, :]` |
| `grand.mean <- crossprod(n.batches/n.array, B.hat[1:n.batch, ])` | `grand_mean = (batch_sizes / n_samples) @ B_hat[:n_batch, :]` |

`crossprod(v, M)` = `t(v) %*% M` = weighted average of rows. `@` in Python is
matrix multiply, so `(batch_sizes / n_samples) @ B_hat[:n_batch, :]` is
identical.

**Pooled variance:**

| R | Python |
|---|---|
| `var.pooled <- ((dat - t(design %*% B.hat))^2) %*% rep(1/n.array, n.array)` | `resid = data - (design @ B_hat).T` / `var_pooled = np.sum(resid**2, axis=1) / n_samples` |

`%*% rep(1/n, n)` sums each row and divides by `n`. Python uses `np.sum(...,
axis=1) / n_samples`. `t(design %*% B.hat)` = `(X B̂)'` = fitted values
transposed to `(n_genes, n_samples)`.

**Standardised data:**

| R | Python |
|---|---|
| `stand.mean <- t(grand.mean) %*% t(rep(1, n.array))` | `stand_mean = np.outer(grand_mean, np.ones(n_samples))` |
| `stand.mean <- stand.mean + t(tmp %*% B.hat)` (covariate contribution) | `stand_mean += (tmp @ B_hat).T` |
| `s.data <- (dat - stand.mean) / (sqrt(var.pooled) %*% t(rep(1, n.array)))` | `s_data = (data - stand_mean) / sd` (where `sd = sqrt(var_pooled)[:, newaxis]`) |

`t(grand.mean) %*% t(rep(1, n))` broadcasts a row vector into a full matrix —
`np.outer` is the direct equivalent. `sqrt(var.pooled) %*% t(rep(1, n))` is
the same broadcast for the scaling step; Python uses `[:, np.newaxis]` to get
the same shape.

---

### Step 7 — Estimate batch parameters

OLS of the standardised data on the batch-only part of the design matrix to get
`gamma.hat` (additive effects). `delta.hat` is the within-batch variance of
the standardised data.

**R:**
```r
batch.design <- design[, 1:n.batch]
gamma.hat <- solve(crossprod(batch.design), tcrossprod(t(batch.design), as.matrix(s.data)))
delta.hat <- NULL
for (i in batches) {
    delta.hat <- rbind(delta.hat, rowVars(s.data[, i], na.rm=TRUE))
}
```

**Python:**
```python
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
```

`rowVars(s.data[, i], na.rm=TRUE)` → `np.var(s_data[:, mask], axis=1, ddof=1)`.
Python pre-fills `delta_hat = 1` (so `mean_only` mode works without an
explicit branch for `delta.hat <- rbind(delta.hat, rep(1, ...))`).

---

### Step 8 — EB priors

**R:**
```r
gamma.bar <- rowMeans(gamma.hat)
t2        <- rowVars(gamma.hat)
a.prior   <- apply(delta.hat, 1, aprior)   # always computed
b.prior   <- apply(delta.hat, 1, bprior)
```

**Python:**
```python
gamma_bar = np.mean(gamma_hat, axis=1)
t2        = np.var(gamma_hat, axis=1, ddof=1)
if not mean_only:                          # only computed when used
    a_prior = np.array([_aprior(delta_hat[i, :]) for i in range(n_batch)])
    b_prior = np.array([_bprior(delta_hat[i, :]) for i in range(n_batch)])
```

`rowMeans` → `np.mean(..., axis=1)`.
`rowVars` → `np.var(..., axis=1, ddof=1)`.
`apply(X, 1, f)` → list comprehension over rows.

**Design difference:** R computes `a.prior`/`b.prior` unconditionally, even
when `mean_only=TRUE` (where they are never used, but can silently produce
`NaN` from division by zero). Python guards the computation so `_aprior` and
`_bprior` are only called when their results will be used.

---

### Step 9 — EB adjustments (parametric)

**R:**
```r
results <- bplapply(1:n.batch, function(i) {
    if (mean.only) {
        gamma.star <- postmean(gamma.hat[i,], gamma.bar[i], 1, 1, t2[i])
        delta.star <- rep(1, nrow(s.data))
    } else {
        temp <- it.sol(s.data[, batches[[i]]],
                       gamma.hat[i,], delta.hat[i,],
                       gamma.bar[i], t2[i], a.prior[i], b.prior[i])
        gamma.star <- temp[1,]; delta.star <- temp[2,]
    }
    list(gamma.star=gamma.star, delta.star=delta.star)
}, BPPARAM=BPPARAM)
```

**Python:**
```python
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
        temp = _it_sol(batch_s_data, gamma_hat[i,:], delta_hat[i,:],
                       gamma_bar[i], t2[i], a_prior[i], b_prior[i])
        gamma_star[i, :] = temp[0, :]
        delta_star[i, :] = temp[1, :]
```

R uses `bplapply` for optional parallel execution. Python uses a plain `for`
loop (single-threaded; parallelism can be added with `concurrent.futures` if
needed).

**`mean_only` path:** R calls `postmean(g.hat, g.bar, 1, 1, t2)` — passing
`n=1, d.star=1`. Python computes `n_i` (actual non-NA count per gene) and
passes `delta_hat[i,:]` as `d_star`. Both produce the same shrinkage direction;
using the real `n` is more accurate.

---

### Step 9b — Non-parametric adjustments

R calls the internal `int.eprior()`. Python re-implements the same Monte Carlo
integration inline.

**R (inside `int.eprior`):** for each gene, form weights proportional to the
likelihood of observing the gene's data under each other gene's parameter
values, then compute weighted averages of `g` and `d`.

**Python:**
```python
for feat in range(n_features):
    g = np.delete(gamma_hat[i, :], feat)
    d = np.delete(delta_hat[i, :], feat)
    x = batch_s_data[feat, :]; x = x[~np.isnan(x)]
    weights = np.array([np.prod(norm.pdf(x, g[k], np.sqrt(d[k])))
                        for k in range(len(g))])
    weights /= weights.sum()          # (with zero guard)
    g_star_i[feat] = np.dot(weights, g)
    d_star_i[feat] = np.dot(weights, d)
```

This is a direct translation of R's `int.eprior`. `dnorm(x, g[k], sqrt(d[k]))`
→ `norm.pdf(x, g[k], np.sqrt(d[k]))`, `prod` → `np.prod`, weighted mean →
`np.dot`.

---

### Step 10 — Apply correction

Subtract `gamma.star`, divide by `sqrt(delta.star)`, then back-transform to
original scale.

**R:**
```r
for (i in batches) {
    bayesdata[, i] <- (bayesdata[, i] - t(batch.design[i,] %*% gamma.star)) /
                      (sqrt(delta.star[j,]) %*% t(rep(1, n.batches[j])))
    j <- j + 1
}
bayesdata <- (bayesdata * (sqrt(var.pooled) %*% t(rep(1, n.array)))) + stand.mean
```

**Python:**
```python
for i, lvl in enumerate(batch_levels):
    mask = batch == lvl
    bayes_data[:, mask] = (
        bayes_data[:, mask] - gamma_star[i, :][:, np.newaxis]
    ) / np.sqrt(delta_star[i, :])[:, np.newaxis]
bayes_data = bayes_data * sd + stand_mean
```

`t(batch.design[i,] %*% gamma.star)` selects batch `i`'s gamma vector and
transposes it to a column — `gamma_star[i, :][:, np.newaxis]` does the same.
`sqrt(delta.star[j,]) %*% t(rep(1, n))` broadcasts the square-root scaling
across columns — `[:, np.newaxis]` achieves the same broadcast.

---

### Step 11 — Restore zero-variance rows

**R:**
```r
if (length(zero.rows) > 0) {
    dat.orig[keep.rows, ] <- bayesdata
    bayesdata <- dat.orig
}
```

**Python:**
```python
if len(zero_rows) > 0:
    data_orig[keep_rows, :] = bayes_data
    bayes_data = data_orig
```

Exact translation. Rows that were removed in step 1 are copied back unchanged.

---

## 3. ComBatSeq

ComBat-seq uses a **negative binomial GLM** instead of a linear model, because
RNA-seq counts are discrete and overdispersed. The algorithm is structurally
similar to ComBat but differs in:

1. How dispersions are estimated (edgeR in R; statsmodels in Python)
2. How the GLM is fitted (edgeR's `glmFit` in R; per-gene `NegativeBinomial` in Python)
3. How the quantile mapping is parametrised

### Step 1 — Filter all-zero genes

**R:**
```r
keep_lst <- lapply(levels(batch), function(b) {
    which(apply(counts[, batch == b], 1, function(x) { !all(x == 0) }))
})
keep <- Reduce(intersect, keep_lst)   # genes non-zero in ALL batches
rm   <- setdiff(1:nrow(counts), keep)
```

**Python:**
```python
gene_means  = count_mat.mean(axis=1)
nonzero_mask = gene_means > 0
zero_gene_idx = np.where(~nonzero_mask)[0]
keep_gene_idx = np.where(nonzero_mask)[0]
```

**Difference:** R keeps genes that are non-zero in *every* batch (intersection).
Python keeps genes with a positive overall mean — slightly more permissive for
genes that are zero in one batch but expressed in others. Both approaches
exclude genes that are all-zero globally.

---

### Step 2 — Design matrix

**R:**
```r
batchmod <- model.matrix(~-1 + batch)
if (full_mod & nlevels(group) > 1) {
    mod <- model.matrix(~group)        # includes intercept + group dummies
} else {
    mod <- model.matrix(~1, ...)       # intercept only
}
design <- cbind(batchmod, mod)
check  <- apply(design, 2, function(x) all(x == 1))
design <- as.matrix(design[, !check])
```

**Python:**
```python
batch_design = np.zeros((n_samples, n_batch))
for i, lvl in enumerate(batch_levels): batch_design[batch == lvl, i] = 1.0

covar_parts = []
if group is not None:
    for lvl in group_levels[1:]:          # drop first level (reference)
        covar_parts.append((group == lvl).astype(float))
full_design = np.hstack([batch_design, covar_cols])
```

`model.matrix(~group)` automatically creates the intercept + k−1 dummy columns
for a k-level factor. Python reproduces this with a one-hot encoding that drops
the first level (reference category), which is exactly what R's
`model.matrix(~group)` does. The intercept is implicit — the first batch column
in `batch_design` serves as the global intercept in the NB log-linear model.

---

### Step 3 — Dispersion + batch coefficients

This step has the most significant difference between R and Python.

**R — two-stage edgeR approach:**
```r
# Stage 1: estimate common dispersion per batch
disp_common <- sapply(1:n_batch, function(i) {
    estimateGLMCommonDisp(counts[, batches_ind[[i]]], design=mod[batches_ind[[i]],])
})
# Stage 2: tagwise (per-gene) dispersion per batch
genewise_disp_lst <- lapply(1:n_batch, function(j) {
    estimateGLMTagwiseDisp(counts[, batches_ind[[j]]], design=mod[batches_ind[[j]],],
                           dispersion=disp_common[j], prior.df=0)
})
# Stage 3: fit the full GLM with known dispersions
dge_obj <- DGEList(counts=counts)
glm_f   <- glmFit(dge_obj, design=design, dispersion=phi_matrix, prior.count=1e-4)
gamma_hat <- glm_f2$coefficients[, 1:n_batch]
mu_hat    <- glm_f2$fitted.values
```

**Python — per-gene statsmodels NB:**
```python
log_offset = np.log(count_mat_nz.sum(axis=0) + 1)  # library-size offset
phi_hat    = np.ones(n_genes_nz)
gamma_hat  = np.zeros((n_batch, n_genes_nz))

for g in range(n_genes_nz):
    y = count_mat_nz[g, :]
    nb_mod = sm.NegativeBinomial(y, full_design, loglike_method="nb2",
                                 offset=log_offset)
    result = nb_mod.fit(disp=False, method="nm", maxiter=500)
    phi_hat[g]       = np.exp(result.params[-1])
    gamma_hat[:, g]  = result.params[:n_batch]
```

**Why different?** edgeR uses Cox-Reid profile-adjusted likelihood for
dispersion estimation, which is a shrinkage estimator designed for RNA-seq. It
is not available in statsmodels. Python uses statsmodels' `NegativeBinomial`
which fits the NB-2 parametrisation (`Var = μ + φμ²`) directly per gene. The
log-normalisation offset (`log(library_size + 1)`) mirrors edgeR's use of
`getOffset(dge_obj)`.

**Consequence:** The batch coefficient estimates (`gamma_hat`) will be
numerically different from R's, which is why the e2e test for ComBatSeq checks
the *direction* of batch correction rather than exact numerical equality.

---

### Step 4 — EB shrinkage (optional)

**R** uses Monte Carlo integration (`monte_carlo_int_NB`) to compute shrinkage
posteriors for both `gamma` and `phi`. This is a computationally expensive
procedure and is disabled by default (`shrink=FALSE`).

**Python** uses the same `_postmean` helper as ComBat to apply a simple
conjugate normal-prior shrinkage on `gamma_hat` when `shrink=True`. This is
an approximation but avoids reimplementing the full Monte Carlo integration.

When `shrink=False` (default in both), `gamma_star = gamma_hat` and
`phi_star = phi_hat` — identical in both implementations.

---

### Step 5 — Quantile mapping

This is the core correction step. For each gene in each sample, the observed
count is mapped to its quantile under the batch-specific NB distribution, then
the corresponding quantile under the batch-free NB is used as the corrected
count.

**R (`match_quantiles`):**
```r
for (a in 1:nrow(counts_sub)) {
    for (b in 1:ncol(counts_sub)) {
        if (counts_sub[a, b] <= 1) {
            new_counts_sub[a, b] <- counts_sub[a, b]   # keep 0s and 1s
        } else {
            tmp_p <- pnbinom(counts_sub[a,b] - 1,       # P(X <= x-1)  [lower.tail CDF]
                             mu=old_mu[a,b], size=1/old_phi[a])
            new_counts_sub[a, b] <- 1 + qnbinom(tmp_p,  # 1 + Q(p, new)
                                                 mu=new_mu[a,b], size=1/new_phi[a])
        }
    }
}
```

**Python (`_quantile_map`):**
```python
mu_batch   = np.exp(offset_j + gamma_hat[b_idx, :])
mu_adj     = mu_batch * np.exp(grand_gamma - gamma_hat[b_idx, :])
size       = 1.0 / phi_hat                               # NB size parameter
prob_batch = size / (size + mu_batch)                    # p in scipy NB
prob_adj   = size / (size + mu_adj)
q          = nbinom.cdf(y, size, prob_batch)             # P(X <= y)
corrected[:, j] = nbinom.ppf(q, size, prob_adj)         # Q(p, new)
```

**NB parametrisation difference:** R's `pnbinom`/`qnbinom` use
`(mu, size=1/phi)`. scipy uses `(n, p)` where `n = 1/phi` (size) and
`p = size/(size + mu)` (success probability). These are algebraically
equivalent: `p = n/(n + mu)`.

**CDF shift:** R uses `pnbinom(x - 1, ...)` (CDF of `x-1`, i.e. `P(X ≤ x-1)`)
then adds 1 back via `1 + qnbinom(...)`. Python uses `nbinom.cdf(y, ...)` which
is `P(X ≤ y)`. This is a one-step shift that gives the same mapping for counts
> 1 but handles the boundary slightly differently for 0s/1s. R explicitly
protects 0s and 1s; Python's `cdf`/`ppf` chain naturally maps them close to
their original values given that the clipping keeps `q` away from the boundary.

**Vectorisation:** R loops over all `(gene, sample)` pairs with nested `for`
loops. Python vectorises over genes for each sample in the outer loop, using
numpy array operations.

---

### Step 6 — Restore all-zero genes

**R:**
```r
adjust_counts_whole[keep, ] <- adjust_counts
adjust_counts_whole[rm, ]   <- countsOri[rm, ]   # zero genes unchanged
```

**Python:**
```python
corrected = np.zeros_like(count_mat, dtype=float)
corrected[keep_gene_idx, :] = corrected_nz        # non-zero genes corrected
# zero_gene_idx rows remain 0                      # zero genes unchanged
corrected = np.round(corrected).clip(0).astype(int)
```

R writes both corrected and original zero rows separately. Python initialises
the full matrix to zero and writes only the non-zero genes, which leaves
all-zero rows at zero automatically.

**Integer rounding:** R's output from `glmFit` are already non-negative
integers (edgeR works in count space). Python's scipy quantile mapping returns
floats; `np.round(...).clip(0).astype(int)` ensures the output is
non-negative integers.

---

## 4. Key translation decisions

| Decision | R | Python | Reason |
|---|---|---|---|
| **`var()` default** | `ddof=1` (sample variance) | Must pass `ddof=1` explicitly | Match R |
| **Zero-division guard in helpers** | Returns `NaN` / `Inf` | Raises `ValueError` (unreachable in normal use) | Fail-fast; guards removed from reachable paths |
| **`model.matrix`** | Automatic factor → dummy | Manual one-hot loop | No equivalent in NumPy/pandas |
| **`crossprod(X)`** | Optimised `X'X` | `X.T @ X` | Identical result |
| **`solve(A, B)`** | Direct solver | `np.linalg.solve(A, B)` | Identical result |
| **`rowVars`** | Bioconductor helper | `np.var(..., axis=1, ddof=1)` | Identical result |
| **Single-sample batch** | Silent fallback to `mean_only=TRUE` | `ValueError`; user must pass `mean_only=True` explicitly | Fail-fast; no silent degradation |
| **`bplapply`** (parallel) | Optional BiocParallel | Sequential `for` loop | Simplicity; parallelism is addable |
| **Dispersion estimation** | edgeR Cox-Reid shrinkage | statsmodels NB-2 per gene | edgeR not available in Python |
| **NB parametrisation** | `(mu, size=1/phi)` | `(n=1/phi, p=size/(size+mu))` | scipy convention |
| **CDF in quantile map** | `pnbinom(x-1, ...)` | `nbinom.cdf(y, ...)` | Same distribution; boundary handled differently for 0/1 |
| **Missing values** | `Beta.NA` fallback | Not fully implemented | NA matrices rare in practice |
| **Output type** | R matrix | `pd.DataFrame` (preserves index/columns) | Pythonic API |
