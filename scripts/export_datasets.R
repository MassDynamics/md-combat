#!/usr/bin/env Rscript
#
# Export bundled datasets to parquet files for md_combat.
#
# Run once from the repo root:
#   Rscript scripts/export_datasets.R
#
# Requirements:
#   install.packages("arrow")
#   BiocManager::install(c("bladderbatch", "Biobase", "airway", "SummarizedExperiment"))
#
# Outputs (committed to the repo):
#   src/md_combat/data/bladderbatch_expr.parquet
#   src/md_combat/data/bladderbatch_meta.parquet
#   src/md_combat/data/airway_counts.parquet
#   src/md_combat/data/airway_meta.parquet

library(arrow)
library(bladderbatch)
library(Biobase)
library(airway)
library(SummarizedExperiment)

out_dir <- file.path("src", "md_combat", "data")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# bladderbatch — microarray (ComBat)
# ---------------------------------------------------------------------------
message("Exporting bladderbatch ...")
data(bladderdata, package = "bladderbatch")

# expr: genes x samples — store as genes x samples (rows = genes, few columns)
expr <- exprs(bladderEset)                    # genes x samples matrix
write_parquet(
  as.data.frame(expr),                        # rows = genes, columns = samples
  file.path(out_dir, "bladderbatch_expr.parquet")
)

meta_bb <- pData(bladderEset)[, c("batch", "cancer", "outcome"), drop = FALSE]
write_parquet(
  as.data.frame(meta_bb),
  file.path(out_dir, "bladderbatch_meta.parquet")
)
message("  bladderbatch: ", nrow(expr), " genes x ", ncol(expr), " samples")

# ---------------------------------------------------------------------------
# airway — RNA-seq counts (ComBatSeq)
# ---------------------------------------------------------------------------
message("Exporting airway ...")
data(airway)

counts <- assay(airway, "counts")             # genes x samples integer matrix
write_parquet(
  as.data.frame(counts),                      # rows = genes, columns = samples
  file.path(out_dir, "airway_counts.parquet")
)

meta_aw <- as.data.frame(colData(airway)[, c("cell", "dex"), drop = FALSE])
write_parquet(
  meta_aw,
  file.path(out_dir, "airway_meta.parquet")
)
message("  airway: ", nrow(counts), " genes x ", ncol(counts), " samples")

message("Done. Files written to ", out_dir)
