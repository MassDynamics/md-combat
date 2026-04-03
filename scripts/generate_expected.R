#!/usr/bin/env Rscript
#
# Generate R benchmark reference outputs (Parquet) for e2e tests.
#
# Run once from the repo root:
#   Rscript scripts/generate_expected.R
#
# WARNING: If you re-run scripts/export_datasets.R (which regenerates the
# bundled dataset parquet files in src/md_combat/data/), you MUST re-run
# this script immediately afterwards.  Otherwise the benchmark references will be
# out of sync with the bundled data and all e2e tests will fail.
#
# Requirements:
#   install.packages("arrow")
#   Rscript dependencies.R
#   BiocManager::install(c("airway", "SummarizedExperiment"))
#
# Outputs written to tests/expected/:
#   bladderbatch_combat.parquet       (R sva::ComBat on bladderbatch)
#   simulated_counts.parquet          (200-gene NB input, set.seed(42))
#   simulated_meta.parquet            (batch / group for simulated data)
#   simulated_combat_seq.parquet      (R sva::ComBat_seq on simulated data)
#   airway_20k_gene_ids.parquet           (the 20 000 gene IDs sampled, set.seed(42))
#   airway_20k_combat_seq.parquet         (R sva::ComBat_seq on the 20K subset, equal batches)
#   airway_20k_uneven_batch.parquet       (batch labels for the 2+4+2 unequal split)
#   airway_20k_uneven_combat_seq.parquet  (R sva::ComBat_seq on the 20K subset, unequal batches)
#   airway_full_combat_seq.parquet        (R sva::ComBat_seq on all 64K genes — slow)

library(arrow)
library(sva)
library(bladderbatch)
library(Biobase)
library(airway)
library(SummarizedExperiment)

out_dir <- file.path("tests", "expected")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# 1. bladderbatch ComBat
# ---------------------------------------------------------------------------
message("1/4  bladderbatch sva::ComBat ...")
data(bladderdata, package = "bladderbatch")
expr  <- exprs(bladderEset)
batch <- pData(bladderEset)$batch
corrected <- ComBat(dat = expr, batch = batch)
write_parquet(
  as.data.frame(corrected),
  file.path(out_dir, "bladderbatch_combat.parquet")
)
message("     ", nrow(corrected), " genes x ", ncol(corrected), " samples  -- done")

# ---------------------------------------------------------------------------
# 2. Simulated NB data — ComBat_seq
# ---------------------------------------------------------------------------
message("2/4  Simulated sva::ComBat_seq ...")
set.seed(42)
n_genes   <- 200
n_samples <- 12
batch_sim <- c(rep(1, 6), rep(2, 6))
group_sim <- c(1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2)   # balanced: no confounding
base_mean <- runif(n_genes, 10, 200)
bio   <- matrix(1, n_genes, n_samples); bio[1:20, group_sim == 2]  <- 2
bfact <- matrix(1, n_genes, n_samples); bfact[, batch_sim == 2] <- 3
means     <- base_mean * bio * bfact
counts_sim <- matrix(
  rnbinom(n_genes * n_samples, mu = means, size = 10),
  nrow = n_genes, ncol = n_samples
)
rownames(counts_sim) <- paste0("gene_",   seq_len(n_genes))
colnames(counts_sim) <- paste0("sample_", seq_len(n_samples))
r_out_sim <- ComBat_seq(counts_sim, batch = batch_sim, group = group_sim)

write_parquet(as.data.frame(counts_sim),                     file.path(out_dir, "simulated_counts.parquet"))
write_parquet(data.frame(batch = batch_sim, group = group_sim), file.path(out_dir, "simulated_meta.parquet"))
write_parquet(as.data.frame(r_out_sim),                      file.path(out_dir, "simulated_combat_seq.parquet"))
message("     ", nrow(r_out_sim), " genes x ", ncol(r_out_sim), " samples  -- done")

# ---------------------------------------------------------------------------
# 3. Airway 20K gene subset — ComBat_seq
# ---------------------------------------------------------------------------
message("3/4  Airway 20K subset sva::ComBat_seq ...")
data(airway)
counts_aw <- assay(airway, "counts")
meta_aw   <- colData(airway)
batch_aw  <- as.character(meta_aw$cell)
group_aw  <- as.character(meta_aw$dex)

set.seed(42)
idx_20k    <- sort(sample(nrow(counts_aw), 20000))
counts_20k <- counts_aw[idx_20k, ]

# Save 0-based integer positions (Python iloc) alongside gene names for readability
write_parquet(
  data.frame(gene_idx = as.integer(idx_20k - 1L), gene_id = rownames(counts_20k)),
  file.path(out_dir, "airway_20k_gene_ids.parquet")
)

r_out_20k  <- ComBat_seq(counts_20k, batch = batch_aw, group = group_aw)
write_parquet(as.data.frame(r_out_20k),        file.path(out_dir, "airway_20k_combat_seq.parquet"))
message("     ", nrow(r_out_20k), " genes x ", ncol(r_out_20k), " samples  -- done")

# ---------------------------------------------------------------------------
# 3b. Airway 20K subset — uneven batches (2+4+2) — exercises weighted grand_gamma
#     Remap: A = N61311 (2), B = N052611 + N061011 (4), C = N080611 (2)
# ---------------------------------------------------------------------------
message("3b/4  Airway 20K uneven-batch sva::ComBat_seq ...")
batch_uneven <- ifelse(batch_aw %in% c("N052611", "N061011"), "B",
               ifelse(batch_aw == "N61311", "A", "C"))
r_out_uneven <- ComBat_seq(counts_20k, batch = batch_uneven, group = group_aw)
write_parquet(data.frame(batch = batch_uneven),  file.path(out_dir, "airway_20k_uneven_batch.parquet"))
write_parquet(as.data.frame(r_out_uneven),       file.path(out_dir, "airway_20k_uneven_combat_seq.parquet"))
message("     batches: ", paste(table(batch_uneven), collapse="+"),
        "  ", nrow(r_out_uneven), " genes x ", ncol(r_out_uneven), " samples  -- done")

# ---------------------------------------------------------------------------
# 4. Airway full — ComBat_seq  (slow — only used by @pytest.mark.slow tests)
# ---------------------------------------------------------------------------
message("4/4  Airway full sva::ComBat_seq (this takes several minutes) ...")
r_out_full <- ComBat_seq(counts_aw, batch = batch_aw, group = group_aw)
write_parquet(as.data.frame(r_out_full), file.path(out_dir, "airway_full_combat_seq.parquet"))
message("     ", nrow(r_out_full), " genes x ", ncol(r_out_full), " samples  -- done")

message("\nAll benchmark reference files written to ", normalizePath(out_dir), "/")
