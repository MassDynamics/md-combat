if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(c("sva", "bladderbatch", "Biobase"), ask = FALSE)
if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv")
renv::snapshot()
