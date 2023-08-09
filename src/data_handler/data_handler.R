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
    
    summary <- c(min_value, max_value, average_value)
    
    return(summary)
}

plot_stats <- function(counts, directory) {
    # Gathers various statistical information and saves them in a pdf file
    
    pdf(file.path(directory, "stats.pdf"), width = 8, height = 11)
    
    for (i in 1:length(counts)) {
        result <- calc_stats(counts[[i]])
        
        # Plot table
        data <- data.frame(
            Minimum = result[1],
            Maximum = result[2],
            Average = result[3]
        )
        cat("\n")
        print(data, row.names = FALSE, quote = FALSE)
        
        # Plot bar plot
        frequency <- as.data.frame(table(counts[i]))
        barplot(frequency$Freq, 
                names.arg = frequency$Var1,
                col = "blue",
                xlab = "Expression",
                ylab = "Quantity",
                main = paste("Counts", as.character(1)))
        
        # === Not needed! ===
        # Plot table
        # most_common <- names(frequency)[frequency == max(frequency)]
        # least_common <- names(frequency)[frequency == min(frequency)]
        
        # data <- data.frame(
        #     Most_common = most_common,
        #     Least_common = least_common
        # )
        # print(data)
        
        cat("\n")
        print(paste("Stats computed for", as.character(i), "out of",
                    as.character(length(counts))),
              row.names = FALSE, quote = FALSE)
    }
    
    dev.off()
}


# === Script ===================================================================

# Get cmd line arguments
args <- commandArgs(trailingOnly = TRUE)
    
# Check for flags in args
if ("--path" %in% args) {
        
    # Load required packages
    if (require(anndata)) {
            
        if (require(edgeR)) {
                
            # Get index of flag in args
            flag_index <- which(args == "--path")
                
            # Check if the flag parameter is provided
            if (flag_index + 1 <= length(args)) {
                # Get the value of the flag parameter
                path_arg <- args[flag_index + 1]
                    
            } 
                
        } else {
            
            print("Error: The package 'edgeR' could not be loaded or attached")
            
        }
        
    } else {
        
        print("Error: The package 'anndata' could not be loaded or attached")
        
    }
        
}
        
if ("--stat" %in% args) {
    
    stat_arg <- TRUE
            
} else {
  
  stat_arg <- FALSE
  
}

if (file.exists(path_arg)) {
    
    # Normalizes the path for every operating system
    file_path <- normalizePath(path = path_arg)
    
    # Create anndata object
    adata <- anndata::read_h5ad(file_path)
    # Convert anndata to matrix
    # --> X provides the raw count matrix
    count_data <- as.matrix(adata$X)
    
    cpm_counts <- cpm_normalization(count_data)
    min0_max1_counts <- min0_max1_nomalization(cpm_counts)
    # Substitute count matrix with normalized matrix
    adata$X <- min0_max1_counts
    
    if (stat_arg == TRUE) {
        
        counts <- list(count_data, cpm_counts, min0_max1_counts)
        plot_stats(counts, dirname(file_path))
        
    }
    
    # Create export file
    file_name <- paste0("normalized_",
                        strsplit(basename(file_path), '\\.')[[1]][1],
                        ".h5ad")
    export_path <- file.path(dirname(file_path), file_name)
    # Write anndata file
    write_h5ad(adata, file = export_path)
    
} else {
    
    print("Error: The provided path is invalid")

}
