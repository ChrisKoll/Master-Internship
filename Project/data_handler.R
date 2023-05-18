# Check for installed packages
packages <- c("anndata", "edgeR")

all_installed <- TRUE
for (pkg in packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
        
        all_installed <- FALSE
        break
        
    }
}


# === Functions ================================================================

cpm_normalization <- function(count_data) {
    # Normalizes the raw count data with cpm
    
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
    
    return(normalized_counts)
}

min0_max1_nomalization <- function(count_data) {
    # Normalizes count data for min0 - max1
    
    min_value <- min(count_data)
    max_value <- max(count_data)
    
    # Perform min-max normalization
    normalized_counts <- (count_data - min_value) / (max_value - min_value)
    
    return(normalized_counts)
}

calc_stats <- function(count_data) {
    # Calculates specific statistical information for a given count matrix
    
    # Calculate statistical interesting values
    min_value <- min(count_data)
    max_value <- max(count_data)
    average_value <- mean(count_data)
    
    frequency <- table(count_data)
    most_common <- names(frequency)[frequency == max(frequency)]
    least_common <- names(frequency)[frequency == min(frequency)]
    
    summary <- c(min_value, max_value, average_value, most_common, least_common)
    
    return(summary)
}

plot_stats <- function(counts) {
    # Gathers various statistical information
    
    summary <- matrix(nrow = 0, ncol = 0)
    for (matrix in counts) {
        result <- calc_stats(matrix)
        summary <- c(summary, result)
    }
    
    return(summary)
}


# === Script ===================================================================

if (all_installed == TRUE) {
    
    # Load libraries
    suppressPackageStartupMessages(library(anndata))
    suppressPackageStartupMessages(library(edgeR))
    
    # Get cmd line arguments
    args <- commandArgs(trailingOnly = TRUE)
    
    # Only one arguments can be provided
    # --> File path
    if (length(args) == 0) {
        
        print("Error: No arguments provided...")
        
    } else if (length(args) == 1) {
        
        file_path <- args
        # Normalizes the path for every operating system
        file_path <- normalizePath(path = file_path)
        
        # Create anndata object
        adata <- anndata::read_h5ad(file_path)
        # Convert anndata to matrix
        # --> X provides the raw count matrix
        count_data <- as.matrix(adata$X)
        
        cpm_counts <- cpm_normalization(count_data)
        min0_max1_counts <- min0_max1_nomalization(cpm_counts)
        # Substitute count matrix with normalized matrix
        adata$X <- min0_max1_counts
        
        counts <- list(count_data, cpm_counts, min0_max1_counts)
        plot_stats(counts)
        
        # Create export file
        file_name <- paste0("normalized_",
                            strsplit(basename(args), '\\.')[[1]][1],
                            ".h5ad")
        export_path <- file.path(dirname(args), file_name)
        # Write anndata file
        write(adata, file = export_path)
        
    } else {
        
        print("Error: Passed too many arguments...")
        
    }
    
} else {
    
    print("Error: Not all required packages are installed...")
    
}
