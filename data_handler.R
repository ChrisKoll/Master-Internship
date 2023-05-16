library(anndata)
library(edgeR)

adata_path <- '/media/sf_Share/sample_data.h5ad'
adata <- anndata::read_h5ad(adata_path)

count_data <- as.matrix(adata$X)
# Add pseudocounts to matrix (Every cell)
# --> Here: 0.5
# Generally low expression (Many zeros)
count_data <- count_data + 0.5

# Create a DGEList object from the count matrix
dge <- DGEList(counts = count_data)

# Perform library size normalization
dge <- calcNormFactors(dge)

# Access the normalized counts
normalized_counts <- cpm(dge, log = FALSE)

min_value <- min(normalized_counts)
max_value <- max(normalized_counts)

# Perform min-max normalization
normalized_data <- (normalized_counts - min_value) / (max_value - min_value)

adata$X <- normalized_data

write(adata, file = "normalized.h5ad")