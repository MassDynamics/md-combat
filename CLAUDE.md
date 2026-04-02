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

```bash
# Install R packages (only needed once per R installation)
Rscript dependencies.R

# Verify
Rscript -e "library(sva); library(bladderbatch); cat('OK\n')"
```

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
