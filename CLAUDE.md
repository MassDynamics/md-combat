# md_combat

Python translation of R's `sva::ComBat` and `sva::ComBat_seq` for batch effect correction.

## Commands

```bash
# Set up Python env
uv sync --extra dev

# Unit tests (no R needed)
uv run pytest tests/test_combat.py tests/test_combat_seq.py -v

# End-to-end tests (requires R + sva on PATH — see below)
uv run pytest tests/test_combat_e2e.py -v

# All tests
uv run pytest -v

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Coverage
uv run pytest --cov=md_combat --cov-report=term-missing
```

## R Environment (local dev)

The e2e tests call `Rscript` as a subprocess. R version is auto-detected at
runtime — tests skip cleanly if R or the required packages are not available.

### 1 — Install R

**macOS (recommended: use `rig` to manage R versions)**

```bash
# Install rig — requires your sudo password once
brew install --cask rig

# Install the latest R release
rig add release

# Verify
Rscript --version
```

If `brew` is not installed: https://brew.sh

**Alternative (macOS / Windows / Linux):** download the installer directly from
https://cran.r-project.org and follow the prompts. After installation,
`Rscript` should be on your `PATH`.

### 2 — Install required R packages

```bash
# From the repo root — installs sva, bladderbatch, Biobase via Bioconductor
Rscript dependencies.R
```

This takes a few minutes the first time (compiles some packages from source).

### 3 — Verify

```bash
Rscript -e "library(sva); library(bladderbatch); cat('R env OK\n')"
```

### 4 — Record your R binary path (optional)

If you have multiple R versions installed (via `rig`) and want to pin which one
the e2e tests use, create `.claude-r-env` in the repo root:

```bash
# .claude-r-env is gitignored — safe to store local paths here
export RSCRIPT_EXEC="Rscript4.5"   # or full path e.g. /usr/local/bin/Rscript
```

The e2e tests use plain `Rscript` by default and fall back through
`Rscript4.5`, `Rscript4.4`, `Rscript4.3` automatically.

## CI/CD

The unit tests have no R dependency and run anywhere Python is available.

For the e2e tests in CI, the recommended approach is a Docker image with R +
sva pre-installed. Minimal Dockerfile stub:

```dockerfile
FROM rocker/r-ver:4.5

RUN Rscript -e "install.packages('BiocManager')" && \
    Rscript -e "BiocManager::install(c('sva','bladderbatch','Biobase'), ask=FALSE)"

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
COPY . /app
WORKDIR /app
RUN uv sync --extra dev
CMD ["uv", "run", "pytest", "-v"]
```

The e2e tests auto-discover `Rscript` on PATH — no other config needed inside
the container.
