# =============================================================================
# scLENS (single-cell LENS) Module - Advanced Dimensionality Reduction
# =============================================================================
# 
# scLENS is a sophisticated method for dimensionality reduction in single-cell 
# RNA sequencing (scRNA-seq) data that uses Random Matrix Theory (RMT) to 
# distinguish biological signals from technical noise.
#
# Key Features:
# - Random Matrix Theory-based signal detection
# - Signal robustness testing through perturbation analysis
# - GPU acceleration for large datasets
# - Automatic parameter optimization
# - Integration with downstream analysis tools
#
# Main Concepts:
# - Eigenvalue decomposition: Breaking down data into principal components
# - Marchenko-Pastur distribution: Theoretical noise distribution
# - Signal robustness: Testing if signals persist under data perturbation
# - Denoising: Removing technical noise while preserving biological signal
# =============================================================================

module scLENS

# =============================================================================
# PACKAGE IMPORTS AND DEPENDENCIES
# =============================================================================
# Each package serves a specific purpose in the analysis pipeline

using DataFrames      # For handling tabular data (cells × genes matrices)
using CUDA           # GPU acceleration for matrix operations
using SparseArrays   # Efficient storage for sparse matrices (many zeros)
using CSV            # Reading/writing comma-separated value files
using StatsBase      # Statistical functions (mean, std, quantiles, etc.)
using ProgressMeter  # Progress bars for long-running computations
using Random:shuffle, shuffle!  # Randomization functions for perturbation tests
using LinearAlgebra  # Matrix operations (eigenvalues, matrix multiplication)
using Distributions:Normal  # Statistical distributions for modeling
using NaNStatistics: histcounts, nanmaximum  # Histogram and NaN-safe statistics
using UMAP:UMAP_     # Uniform Manifold Approximation and Projection
using Distances:CosineDist  # Distance metrics for similarity calculations
using JLD2:save as jldsave, load as jldload  # Julia data format for efficient storage
using CairoMakie     # High-quality plotting and visualization
using ColorSchemes   # Color palettes for plots
using Muon           # Single-cell data format support (AnnData)
using MatrixMarket   # Reading Matrix Market format files
using GZip           # Handling compressed files

# =============================================================================
# DATA INPUT/OUTPUT FUNCTIONS
# =============================================================================

"""
The `read_file` function reads count matrix files in either CSV or JLD2 format.

## Arguments
- `test_file::String`: The path to the file containing data (CSV or JLD2 format). For CSV files, rows should represent cells and columns should represent genes. For JLD2 files, the file must contain a variable named `"data"`, which is a DataFrame.
- `gid_file=nothing`: (optional) Path to a CSV file containing new gene names. If `gid_file=nothing`, gene names from the original data will not be modified.

## File Formats
- **CSV format**: Rows represent cells, and columns represent genes. The first row must contain gene names or IDs, and the first column must contain cell IDs.
- **JLD2 format**: The file should contain a variable named `"data"` as a DataFrame. The first column in this DataFrame should be named `:cell` and must represent cell IDs.

## Using a Gene Dictionary
To change gene names, you can provide a second argument to the `scLENS.read_file` function as follows:

```julia
ndf = scLENS.read_file("data/Z8eq.csv.gz", gid_file="path/to/gene_id.csv")
```

The `gene_id_file` must be in **CSV format** and should contain two columns:
- `"gene"`: Original gene names
- `"gene_ID"`: Corresponding new gene names

This file should follow the structure of the `gene_dictionary/gene_id.csv` file.

## Example
```julia
# Load the compressed CSV file into a dataframe
ndf = scLENS.read_file("data/Z8eq.csv.gz")

# Alternatively, load with a gene dictionary
ndf = scLENS.read_file("data/Z8eq.csv.gz", gid_file="gene_dictionary/gene_id.csv")
```
"""
function read_file(test_file::String; gid_file=nothing)
    """
    Main data loading function that handles different file formats.
    
    This function is the entry point for getting your single-cell data into Julia.
    It can handle compressed CSV files and Julia's native JLD2 format.
    """
    # Extract the base filename for logging
    fname_base = splitext(split(test_file, "/")[end])[1]
    println(fname_base)
    
    # Check file extension and load accordingly
    ndf = if occursin("csv", test_file)
        # Load CSV file (handles compressed .gz files automatically)
        ndf = CSV.read(test_file, DataFrame, buffer_in_memory=true)
        
        # Verify proper format - first column should be cell IDs
        if !("cell" in names(ndf))
            println("Warning: The first column should contain cell IDs, and the column name should be 'cell.' However, the 'cell' column was not found. It is recommended to check the file.")
        end
        
        # Rename first column to standardized 'cell' name
        rename!(ndf, names(ndf)[1] => :cell)
        
        # Apply gene name mapping if provided
        tmp_gname = change_gname(names(ndf)[2:end], gid_file)
        rename!(ndf, ["cell"; tmp_gname], makeunique=true)
        return ndf
        
    elseif occursin("jld2", test_file)
        # Load Julia's native JLD2 format (faster and more efficient)
        ndf = jldload(test_file, "data")
        
        # Same validation as CSV
        if !("cell" in names(ndf))
            println("Warning: The first column should contain cell IDs, and the column name should be 'cell.' However, the 'cell' column was not found. It is recommended to check the file.")
        end
        
        # Apply gene name mapping
        tmp_gname = change_gname(names(ndf)[2:end], gid_file)
        rename!(ndf, ["cell"; tmp_gname], makeunique=true)
        return ndf
    end
end

function change_gname(g_names, inp_file=nothing)
    """
    Helper function to map gene names using a dictionary file.
    
    This is useful for converting between different gene naming conventions
    (e.g., Ensembl IDs to gene symbols, or vice versa).
    
    Parameters:
    - g_names: Current gene names in the dataset
    - inp_file: Path to CSV file with gene name mappings
    
    Returns: Vector of updated gene names
    """
    if isnothing(inp_file)
        # No mapping file provided, return original names
        return g_names
    else
        # Load the gene mapping dictionary
        gene_iid = CSV.read(inp_file, DataFrame, buffer_in_memory=true)
        
        # Create mapping dictionary: gene_ID -> gene
        g_dict = Dict(gene_iid.gene_ID .=> gene_iid.gene)
        
        # Apply mapping, keeping original names if not found in dictionary
        return [s in keys(g_dict) ? g_dict[s] : s for s in g_names]
    end
end

# =============================================================================
# DATA STRUCTURE CONVERSION FUNCTIONS
# =============================================================================

function df2sparr(inp_df::DataFrame; for_df=false)
    """
    Convert DataFrame to sparse matrix for efficient storage and computation.
    
    Single-cell data is typically very sparse (lots of zeros), so sparse matrices
    save memory and computation time. This function handles the conversion while
    preserving the data structure.
    
    Parameters:
    - inp_df: DataFrame with cells as rows, genes as columns
    - for_df: Whether to use Int64 (for DataFrames) or UInt32 (for computation)
    
    Returns: Sparse matrix in CSC (Compressed Sparse Column) format
    """
    if issparse(inp_df[:, 2])
        # Data is already sparse, just reorganize the sparse structure
        if for_df
            # Use Int64 for DataFrame compatibility
            a_ = Vector{Int64}([])      # Column indices
            b_ = Vector{Int64}([])      # Row indices  
            c_ = Vector{Float32}([])    # Values
            
            # Extract sparse elements column by column
            for (i, s) in enumerate(eachcol(inp_df[!, 2:end]))
                append!(a_, ones(Int64, lastindex(s.nzind)) .* i)  # Column i
                append!(b_, Int64.(s.nzind))                       # Row indices
                append!(c_, Float32.(s.nzval))                     # Values
            end
            sparse(b_, a_, c_, size(inp_df, 1), size(inp_df, 2) - 1)
        else
            # Use UInt32 for memory efficiency in computations
            a_ = Vector{UInt32}([])
            b_ = Vector{UInt32}([])
            c_ = Vector{Float32}([])
            
            for (i, s) in enumerate(eachcol(inp_df[!, 2:end]))
                append!(a_, ones(UInt32, lastindex(s.nzind)) .* i)
                append!(b_, UInt32.(s.nzind))
                append!(c_, Float32.(s.nzval))
            end
            sparse(b_, a_, c_, size(inp_df, 1), size(inp_df, 2) - 1)
        end
    else
        # Data is dense, convert to sparse
        if for_df
            SparseMatrixCSC{Float32, Int64}(mat_(inp_df))
        else
            SparseMatrixCSC{Float32, UInt32}(mat_(inp_df))
        end
    end
end

# =============================================================================
# UTILITY FUNCTIONS FOR DATA ANALYSIS
# =============================================================================

# Helper functions to extract numeric matrices and calculate sparsity
mat_(x::DataFrame) = Matrix{Float32}(x[!, 2:end])        # Extract gene expression matrix
mat_(x::SubDataFrame) = Matrix{Float32}(x[!, 2:end])     # For DataFrame subsets

# Functions to calculate sparsity (proportion of zero values)
sparsity_(A::Matrix) = 1 - length(nonzeros(SparseMatrixCSC(A))) / length(A)
sparsity_(A::SparseMatrixCSC) = 1 - length(nonzeros(A)) / length(A)
sparsity_(A::Vector) = 1 - sum(.!iszero.(A)) / length(A)
sparsity_(A::SparseVector) = 1 - sum(.!iszero.(A)) / length(A)
sparsity_(A::DataFrame) = issparse(A[!, 2]) ? 1 - length(nonzeros(df2sparr(A))) / length(df2sparr(A)) : sparsity_(mat_(A))

# =============================================================================
# DATA PREPROCESSING AND QUALITY CONTROL
# =============================================================================

"""
`preprocess(tmp_df; min_tp_c=0, min_tp_g=0, max_tp_c=Inf, max_tp_g=Inf,
    min_genes_per_cell=200, max_genes_per_cell=0, min_cells_per_gene=15, mito_percent=5.,
    ribo_percent=0.)`

The `preprocess` function filters and cleans a given count matrix DataFrame to retain high-quality cells and genes based on various criteria.

## Arguments
- `tmp_df`: A DataFrame where each row represents a cell, and each column (except the first) represents a gene. The first column contains cell IDs, and subsequent columns contain gene expression values.
- `min_tp_c`: Minimum total counts per cell. Cells with fewer counts are filtered out.
- `max_tp_c`: Maximum total counts per cell. Cells with counts exceeding this value are filtered out.
- `min_tp_g`: Minimum total counts per gene. Genes with fewer counts are filtered out.
- `max_tp_g`: Maximum total counts per gene. Genes with counts exceeding this value are filtered out.
- `min_genes_per_cell`: Minimum number of genes per cell. Only cells with at least this number of expressed genes are retained.
- `max_genes_per_cell`: Maximum number of genes per cell. Cells with more than this number of expressed genes are filtered out.
- `min_cells_per_gene`: Minimum number of cells per gene. Only genes expressed in at least this number of cells are retained.
- `mito_percent`: Upper threshold for mitochondrial gene expression as a percentage of total cell expression. Cells exceeding this threshold are filtered out.
- `ribo_percent`: Upper threshold for ribosomal gene expression as a percentage of total cell expression. Cells exceeding this threshold are filtered out.

## Example
```julia
# Basic preprocessing with default filtering parameters
filtered_df = preprocess(tmp_df)

# Advanced preprocessing with custom mitochondrial and ribosomal thresholds
filtered_df = preprocess(tmp_df, mito_percent=10, ribo_percent=5)
```

This function is designed to improve data quality by filtering cells and genes that do not meet specified quality criteria, enabling more reliable downstream analyses.
"""
function preprocess(tmp_df; min_tp_c=0, min_tp_g=0, max_tp_c=Inf, max_tp_g=Inf,
                   min_genes_per_cell=200, max_genes_per_cell=0, min_cells_per_gene=15, mito_percent=5.,
                   ribo_percent=0.)
    """
    Quality control function that filters out low-quality cells and genes.
    
    This is a crucial step in scRNA-seq analysis that removes:
    - Dead or dying cells (very high or low gene counts)
    - Doublets (cells with too many detected genes)
    - Poorly expressed genes (detected in few cells)
    - Cells with high mitochondrial gene expression (stressed/dying cells)
    
    The function implements standard QC metrics used across the field.
    """

    # Store original identifiers
    cell_name = tmp_df.cell
    gene_name = names(tmp_df)[2:end]
    size_ndf = (size(tmp_df))
    sp_ndf = sparsity_(tmp_df)
    
    println("Inp_spec")
    println("data size: $size_ndf, sparsity: $sp_ndf")

    # Convert to appropriate matrix format based on sparsity
    X = if sp_ndf < 0.3
        # If less than 30% sparse, use dense matrix
        mat_(tmp_df)
    else
        # If more sparse, use sparse matrix for efficiency
        SparseMatrixCSC{Float32, Int64}(df2sparr(tmp_df))
    end

    # Update DataFrame to match matrix format
    tmp_df = if issparse(X)
        DataFrame(X, gene_name)
    else
        tmp_df[!, 2:end]
    end

    # =============================================================================
    # GENE-LEVEL FILTERING
    # =============================================================================
    # Remove genes that are poorly expressed across the dataset
    
    # Helper function to count non-zero values along dimensions
    c_nonz(X, dim) = sum(X .!= 0, dims=dim)[:]
    
    # Calculate gene-level statistics
    n_cell_counts = c_nonz(X, 1)          # Number of cells expressing each gene
    n_cell_counts_sum = sum(X, dims=1)[:] # Total expression per gene
    
    # Apply gene filtering criteria
    bidx_1 = n_cell_counts_sum .> min_tp_g      # Minimum total expression
    bidx_2 = n_cell_counts_sum .< max_tp_g      # Maximum total expression  
    bidx_3 = n_cell_counts .>= min_cells_per_gene  # Minimum expressing cells
    
    # Combine gene filters
    fg_idx = bidx_1[:] .& bidx_2[:] .& bidx_3[:]

    # =============================================================================
    # CELL-LEVEL FILTERING
    # =============================================================================
    # Remove low-quality cells based on multiple criteria
    
    # Calculate cell-level statistics
    n_gene_counts = c_nonz(X, 2)          # Number of genes per cell
    n_gene_counts_sum = sum(X, dims=2)[:] # Total expression per cell
    
    # Basic expression filters
    bidx_1 = n_gene_counts_sum .> min_tp_c      # Minimum total counts
    bidx_2 = n_gene_counts_sum .< max_tp_c      # Maximum total counts
    bidx_3 = n_gene_counts .>= min_genes_per_cell  # Minimum detected genes
    
    # Identify mitochondrial and ribosomal genes using regex patterns
    bidx_mito = occursin.(r"^(?i)mt-.", gene_name)    # Mitochondrial genes (mt-)
    bidx_ribo = occursin.(r"^(?i)RP[SL].", gene_name) # Ribosomal genes (RPS/RPL)
    
    # Mitochondrial gene filter
    if mito_percent == 0
        bidx_4 = ones(Bool, size(bidx_1))  # Skip filtering if threshold is 0
    else
        # Calculate mitochondrial gene percentage per cell
        bidx_4 = sum(X[:, bidx_mito], dims=2)[:] ./ sum(X, dims=2)[:] .< mito_percent / 100
    end

    # Ribosomal gene filter
    if ribo_percent == 0
        bidx_5 = ones(Bool, size(bidx_1))  # Skip filtering if threshold is 0
    else
        # Calculate ribosomal gene percentage per cell
        bidx_5 = sum(X[:, bidx_ribo], dims=2)[:] ./ sum(X, dims=2)[:] .< ribo_percent / 100
    end

    # Maximum genes per cell filter (to remove potential doublets)
    if max_genes_per_cell == 0
        bidx_6 = ones(Bool, size(bidx_1))  # Skip filtering if threshold is 0
    else
        bidx_6 = n_gene_counts .< max_genes_per_cell
    end

    # Combine all cell filters
    fc_idx = bidx_1[:] .& bidx_2[:] .& bidx_3[:] .& bidx_4[:] .& bidx_5[:] .& bidx_6[:]

    # =============================================================================
    # APPLY FILTERS AND CREATE OUTPUT
    # =============================================================================
    
    if any(fc_idx) && any(fg_idx)
        # Apply both cell and gene filters
        oo_X = X[fc_idx, fg_idx]
        
        # Remove any genes that have zero expression after cell filtering
        nn_idx = (sum(oo_X, dims=1) .!= 0)[:]
        oo_X = oo_X[:, nn_idx]
        norm_gene = gene_name[fg_idx][nn_idx]

        # Sort genes by mean expression (optional, for consistency)
        s_idx = sortperm(mean(oo_X, dims=1)[:])
        o_df = DataFrame(oo_X[:, s_idx], norm_gene[s_idx])
        insertcols!(o_df, 1, :cell => cell_name[fc_idx])

        # Report final dimensions
        size_odf = (size(o_df))
        sp_odf = sparsity_(o_df)
        println("After filtering>> data size: $size_odf, sparsity: $sp_odf")
        return o_df
    else
        println("There is no high quality cells and genes")
        return nothing
    end
end

# =============================================================================
# RANDOM MATRIX GENERATION FOR NULL HYPOTHESIS TESTING
# =============================================================================

function _random_matrix(X; dims=1)
    """
    Generate a randomized version of the input matrix for null hypothesis testing.
    
    This function creates control data by shuffling the original matrix while 
    preserving certain structural properties. This is crucial for Random Matrix
    Theory analysis to establish what patterns arise from noise alone.
    
    Parameters:
    - X: Input matrix (sparse or dense)
    - dims: Dimension along which to randomize (1=rows, 2=columns)
    
    Returns: Randomized matrix with same sparsity pattern
    """
    if issparse(X)
        if dims == 1
            # Randomize along rows (shuffle which cells express each gene)
            nz_row, nz_col, nz_val = findnz(X)
            
            # Count how many values each column (gene) has
            ldict = countmap(nz_col)
            n_ldict = keys(ldict)
            
            # For each gene, randomly assign its expression values to different cells
            row_i = vcat([sample(1:lastindex(X, 1), ldict[s], replace=false) for s in n_ldict]...)
            sparse(row_i, nz_col, nz_val)
            
        elseif dims == 2
            # Randomize along columns (shuffle which genes are expressed in each cell)
            nz_row, nz_col, nz_val = findnz(X)
            
            # Count how many values each row (cell) has  
            ldict = countmap(nz_row)
            n_ldict = keys(ldict)
            
            # For each cell, randomly assign its expression values to different genes
            col_i = vcat([sample(1:lastindex(X, 2), ldict[s], replace=false) for s = 1:n_ldict]...)
            sparse(nz_row, col_i, nz_val)
        end
    else
        # For dense matrices, use Julia's built-in shuffle function
        mapslices(shuffle!, X, dims=dims)
    end
end

function random_nz(pre_df; rmix=true, mix_p=nothing)
    """
    Create a randomized version of the dataset for establishing noise baselines.
    
    This function generates a control dataset that maintains the same sparsity
    structure as the original but removes biological correlations. This is
    essential for Random Matrix Theory analysis.
    
    Parameters:
    - pre_df: Original DataFrame or matrix
    - rmix: Whether to apply additional randomization
    - mix_p: Proportion of values to keep (for partial randomization)
    
    Returns: Randomized dataset in same format as input
    """
    # Convert input to sparse matrix format
    tmp_X, return_mat = if typeof(pre_df) <: DataFrame
        df2sparr(pre_df), false
    else
        sparse(pre_df), true
    end

    # Extract sparse matrix components
    nz_row, nz_col, nz_val = findnz(tmp_X)
    nz_idx = sparse(nz_row, nz_col, true)  # Boolean mask of non-zero positions
    
    # Optional: keep only a subset of values (for robustness testing)
    if !isnothing(mix_p)
        tmp_i = findall(nz_idx)[sample(1:sum(nz_idx), Int(ceil(sum(nz_idx) * (1 - mix_p))), replace=false)]
        nz_idx[tmp_i] .= false
    end

    # Shuffle the values while keeping positions fixed
    tmp_X.nzval .= shuffle(nz_val)
    
    # Apply additional matrix-level randomization if requested
    tmp_X = if rmix
        _random_matrix(tmp_X)
    else
        SparseMatrixCSC{Float32, Int64}(tmp_X)
    end

    # Return in original format
    if !return_mat
        # Convert back to DataFrame
        tmp_df = DataFrame(tmp_X, names(pre_df)[2:end])
        insertcols!(tmp_df, 1, :cell => pre_df.cell)
        return tmp_df
    else
        return tmp_X
    end
end

# =============================================================================
# DATA NORMALIZATION AND SCALING FUNCTIONS
# =============================================================================

function scaled_gdata(X; dim=1, position_="mean")
    """
    Standardize data using z-score normalization with options for robust statistics.
    
    This function implements several normalization strategies:
    - Standard z-score: (x - mean) / std
    - Robust z-score: (x - median) / std  
    - Centering only: (x - mean)
    
    For sparse matrices, special handling preserves sparsity while accounting
    for the implicit zeros.
    
    Parameters:
    - X: Input matrix to normalize
    - dim: Dimension to normalize along (1=genes, 2=cells)
    - position_: Center statistic ("mean", "median", or "cent")
    
    Returns: Normalized matrix
    """
    # Calculate central tendency statistic
    tmp_mean = if position_ == "mean"
        mean(X, dims=dim)
    elseif position_ == "median"
        if issparse(X)
            # For sparse matrices, only calculate median if more than half the values are non-zero
            mapslices(x -> nnz(x) > length(x) / 2 ? median(x) : Float32(0.0), X, dims=dim)
        else
            mapslices(median, X, dims=dim)
        end
    elseif position_ == "cent"
        mean(X, dims=dim)
    end

    if position_ == "cent"
        # Only center, don't scale
        return @. (X - tmp_mean)
    else
        # Calculate standard deviation
        tmp_std = std(X, dims=dim)
        
        if isnothing(tmp_mean)
            # Only scale by standard deviation
            if dim == 1
                return X ./ tmp_std
            elseif dim == 2
                return tmp_std ./ X
            end
        elseif issparse(X) && (position_ == "median")
            # Special handling for sparse matrices with median centering
            new_nz = zeros(length(X.nzval))
            
            # Process each column separately to maintain sparsity
            for i = 1:lastindex(X, 2)
                sub_ii = X.colptr[i]:X.colptr[i+1]-1
                new_nz[sub_ii] = (X.nzval[sub_ii] .- tmp_mean[i]) ./ (tmp_std[i])
                X[:, i].nzind
            end
            
            # Update sparse matrix values
            new_X = copy(X)
            new_X.nzval .= new_nz

            # Handle implicit zeros (positions that were originally zero)
            nz_a = sparse(-tmp_mean ./ tmp_std) .* iszero.(new_X)
            return new_X + nz_a
        else
            # Standard z-score normalization
            return @. (X - tmp_mean) / (tmp_std)
        end
    end
end

# =============================================================================
# LINEAR ALGEBRA AND EIGENVALUE COMPUTATION
# =============================================================================

function _wishart_matrix(X; device="gpu")
    """
    Compute the sample covariance matrix (Wishart matrix) efficiently.
    
    For Random Matrix Theory analysis, we need the covariance matrix X*X' or X'*X.
    This function computes it efficiently using either GPU or CPU.
    
    The Wishart matrix has eigenvalues that follow the Marchenko-Pastur distribution
    under the null hypothesis (pure noise).
    
    Parameters:
    - X: Input data matrix
    - device: "gpu" for CUDA acceleration, "cpu" for standard computation
    
    Returns: Covariance matrix normalized by number of samples
    """
    if device == "gpu"
        # Use GPU acceleration for large matrices
        gpu_x = CuMatrix{Float32}(X)
        out = CuArray{Float32, 2}(undef, size(X, 1), size(X, 1))
        mul!(out, gpu_x, gpu_x')  # Matrix multiplication: X * X'
        return Matrix(out) ./ size(X, 2)  # Normalize by sample size
    elseif device == "cpu"
        # CPU computation
        X = if issparse(X)
            Matrix{eltype(X)}(X)  # Convert sparse to dense for multiplication
        else
            X
        end
        out = Array{eltype(X), 2}(undef, size(X, 1), size(X, 1))
        mul!(out, X, X')
        return out ./ size(X, 2)
    end
end

function corr_mat(X, Y; device="gpu")
    """
    Compute correlation matrix between two datasets efficiently.
    
    This function calculates X'*Y, which is needed for comparing
    eigenvectors between original and randomized data.
    
    Parameters:
    - X, Y: Input matrices
    - device: "gpu" or "cpu"
    
    Returns: Correlation matrix X'*Y
    """
    if device == "gpu"
        gpu_x = cu(X)
        gpu_y = cu(Y)
        out = CuArray{Float32, 2}(undef, size(gpu_x, 2), size(gpu_y, 2))
        mul!(out, gpu_x', gpu_y)
        return Matrix(out)
    elseif device == "cpu"
        return X' * Y
    end
end

function _get_eigen(Y; device="gpu")
    """
    Compute eigenvalues and eigenvectors efficiently.
    
    Eigenvalue decomposition is the core of PCA and Random Matrix Theory analysis.
    This function handles both GPU and CPU computation with fallback for
    numerical stability.
    
    Parameters:
    - Y: Symmetric matrix (covariance matrix)
    - device: "gpu" or "cpu"
    
    Returns: (eigenvalues, eigenvectors)
    """
    if device == "gpu"
        # Use CUDA's optimized symmetric eigenvalue solver
        tmp_L, tmp_V = CUDA.CUSOLVER.syevd!('V', 'U', cu(Y))
        tmp_L, tmp_V = Array{Float32, 1}(tmp_L), Array{Float32, 2}(tmp_V)
        
        # Fallback to CPU if GPU computation fails (NaN values)
        if !isnothing(findfirst(isnan.(tmp_L)))
            tmp_L, tmp_V = eigen(convert.(Float64, Y))
        end
        return tmp_L, tmp_V
    elseif device == "cpu"
        tmp_L, tmp_V = eigen(Y)
        return tmp_L, tmp_V
    end
end

# =============================================================================
# RANDOM MATRIX THEORY (RMT) ANALYSIS FUNCTIONS
# =============================================================================

function _mp_parameters(L)
    """
    Calculate parameters of the Marchenko-Pastur distribution.
    
    The Marchenko-Pastur (MP) distribution describes the eigenvalue spectrum
    of random matrices. By comparing our data's eigenvalues to this theoretical
    distribution, we can identify which eigenvalues represent real signal
    versus noise.
    
    Parameters:
    - L: Vector of eigenvalues
    
    Returns: Dictionary with MP distribution parameters
    - gamma: Aspect ratio parameter
    - b_plus/b_minus: Upper and lower bounds of MP distribution
    - s: Scale parameter
    - peak: Mode of the distribution
    """
    # Calculate first and second moments of eigenvalue distribution
    moment_1 = mean(L)          # First moment (mean)
    moment_2 = mean(L .^ 2)     # Second moment
    
    # Derive MP distribution parameters
    gamma = moment_2 / moment_1^2 - 1  # Aspect ratio parameter
    s = moment_1                        # Scale parameter
    sigma = moment_2                    # Variance parameter
    
    # Calculate support bounds of MP distribution
    b_plus = s * (1 + sqrt(gamma))^2   # Upper bound
    b_minus = s * (1 - sqrt(gamma))^2  # Lower bound
    
    # Peak of the distribution
    x_peak = s * (1.0 - gamma)^2.0 / (1.0 + gamma)
    
    # Return as dictionary for easy access
    dic = Dict("moment_1" => moment_1,
              "moment_2" => moment_2,
              "gamma" => gamma,
              "b_plus" => b_plus,
              "b_minus" => b_minus,
              "s" => s,
              "peak" => x_peak,
              "sigma" => sigma)
end

function _marchenko_pastur(x, y)
    """
    Evaluate the Marchenko-Pastur probability density function.
    
    This function computes the theoretical noise distribution for a given
    eigenvalue x and set of MP parameters y.
    
    The MP distribution has the form:
    pdf(x) = sqrt((b+ - x)(x - b-)) / (2πγsx)  for b- < x < b+
    pdf(x) = 0  otherwise
    
    Parameters:
    - x: Eigenvalue to evaluate
    - y: Dictionary of MP parameters
    
    Returns: Probability density at x
    """
    if y["b_minus"] < x < y["b_plus"]
        # Within the support of the distribution
        pdf = sqrt((y["b_plus"] - x) * (x - y["b_minus"])) /
              (2 * y["s"] * pi * y["gamma"] * x)
    else
        # Outside the support
        pdf = 0
    end
end

function _mp_pdf(x, L)
    """
    Vectorized evaluation of Marchenko-Pastur PDF.
    
    This function evaluates the MP distribution at multiple points,
    useful for plotting and comparison with empirical eigenvalue distribution.
    
    Parameters:
    - x: Vector of points to evaluate
    - L: Eigenvalues used to fit MP parameters
    
    Returns: Vector of PDF values
    """
    _marchenko_pastur.(x, Ref(_mp_parameters(L)))
end

function _mp_calculation(L, Lr, eta=1, eps=1e-6, max_iter=10000)
    """
    Iteratively refine the Marchenko-Pastur fit to identify noise eigenvalues.
    
    This function implements an iterative algorithm to separate signal from noise
    eigenvalues by fitting the MP distribution to the bulk of the spectrum.
    
    The algorithm:
    1. Fit MP distribution to randomized data (Lr)
    2. Use these bounds to select noise eigenvalues from real data (L)
    3. Iteratively refine the bounds until convergence
    
    Parameters:
    - L: Real data eigenvalues
    - Lr: Randomized data eigenvalues  
    - eta: Learning rate for gradient descent
    - eps: Convergence tolerance
    - max_iter: Maximum iterations
    
    Returns: (noise_eigenvalues, upper_bound, lower_bound)
    """
    converged = false
    iter = 0
    loss_history = []
    
    # Initialize with MP parameters from randomized data
    mpp_Lr = _mp_parameters(Lr)
    b_plus = mpp_Lr["b_plus"]
    b_minus = mpp_Lr["b_minus"]
    
    # Select eigenvalues within MP bounds from real data
    L_updated = L[b_minus .< L .< b_plus]
    new_mpp_L = _mp_parameters(L_updated)
    new_b_plus = new_mpp_L["b_plus"]
    new_b_minus = new_mpp_L["b_minus"]

    # Iterative refinement
    while ~converged
        # Calculate loss (difference between bounds)
        loss = (1 - new_b_plus / b_plus)^2
        push!(loss_history, loss)
        iter += 1
        
        if loss <= eps
            converged = true
        elseif iter == max_iter
            println("Max interactions exceeded!")
            converged = true
        else
            # Gradient descent update
            gradient = new_b_plus - b_plus
            new_b_plus = b_plus + eta * gradient
            
            # Update eigenvalue selection
            L_updated = L[new_b_minus .< L .< new_b_plus]
            b_plus = new_b_plus
            b_minus = new_b_minus
            
            # Recalculate MP parameters
            up_mpp_L = _mp_parameters(L_updated)
            new_b_plus = up_mpp_L["b_plus"]
            new_b_minus = up_mpp_L["b_minus"]
        end
    end
    
    b_plus = new_b_plus
    b_minus = new_b_minus
    return L[new_b_minus .< L .< new_b_plus], b_plus, b_minus
end

function _tw(L, L_mp)
    """
    Apply Tracy-Widom correction for finite-size effects.
    
    The Tracy-Widom distribution describes fluctuations of the largest eigenvalue
    in random matrices. This correction helps determine the threshold above which
    eigenvalues represent true signal rather than noise.
    
    Parameters:
    - L: All eigenvalues
    - L_mp: Noise eigenvalues (within MP bounds)
    
    Returns: (signal_threshold, gamma, p, sigma)
    """
    gamma = _mp_parameters(L_mp)["gamma"]
    p = length(L) / gamma                              # Effective degrees of freedom
    sigma = 1 / p^(2/3) * gamma^(5/6) * (1 + sqrt(gamma))^(4/3)  # TW correction
    lambda_c = mean(L_mp) * (1 + sqrt(gamma))^2 + sigma           # Signal threshold
    return lambda_c, gamma, p, sigma
end

function mp_check(test_L, p_val=0.05)
    """
    Perform Kolmogorov-Smirnov test to validate MP distribution fit.
    
    This function tests whether the eigenvalue distribution follows the
    expected Marchenko-Pastur distribution, validating our noise model.
    
    Parameters:
    - test_L: Eigenvalues to test
    - p_val: Significance level for KS test
    
    Returns: Dictionary with test statistic and pass/fail result
    """
    # Create histogram of empirical distribution
    bin_x = LinRange(minimum(test_L) - 1, maximum(test_L) + 1, 100)
    count_ = histcounts(test_L, bin_x)
    pdf_arr = count_ ./ sum(count_)      # Empirical PDF
    cdf_arr = cumsum(pdf_arr)           # Empirical CDF

    new_binx = (bin_x[2:end] .+ bin_x[1:end-1]) ./ 2

    # Calculate theoretical MP distribution
    mp_pdf = x -> _mp_pdf(x, test_L)
    c_cdf2 = cumsum(mp_pdf.(new_binx))   # Theoretical CDF
    nc_cdf2 = c_cdf2 ./ maximum(c_cdf2)  # Normalized theoretical CDF
    
    # Kolmogorov-Smirnov test statistic
    D_ = maximum(abs.(cdf_arr .- nc_cdf2))
    
    # Critical value for KS test
    c_α = sqrt(-1/2 * log(p_val))
    m = length(cdf_arr)
    n = length(nc_cdf2)

    return Dict(:ks_static => D_, :pass => D_ <= c_α * sqrt((m + n) / m / n))
end

# =============================================================================
# EIGENVALUE DECOMPOSITION FOR DIFFERENT MATRIX SIZES
# =============================================================================

function get_eigvec(X; device="gpu")
    """
    Compute eigenvalues and eigenvectors with automatic dimension handling.
    
    This function automatically chooses the most efficient approach based on
    matrix dimensions (N×M):
    - If N > M: Compute M×M matrix X'X (gene-gene covariance)
    - If N ≤ M: Compute N×N matrix XX' (cell-cell covariance)
    
    This optimization is crucial for large single-cell datasets where one
    dimension is much larger than the other.
    
    Parameters:
    - X: Data matrix (cells × genes)
    - device: "gpu" or "cpu"
    
    Returns: (eigenvalues, eigenvectors)
    """
    N, M = size(X)  # N = cells, M = genes
    
    if N > M
        # More cells than genes: compute gene-gene covariance
        Y = _wishart_matrix(X', device=device)  # M×M matrix
        ev_, V = _get_eigen(Y, device=device)
        L = ev_
        
        # Keep only positive eigenvalues (numerical stability)
        positive_idx = L .> 0
        L = L[positive_idx]
        V = V[:, positive_idx]
        
        # Sort by eigenvalue magnitude (largest first)
        nLi = sortperm(L, rev=true)
        nL = L[nLi]
        nVs = V[:, nLi]

        # Transform back to cell space (N×M matrix)
        mul_X = nVs .* sqrt.(1 ./ nL)'
        new_nVs = try
            # Try GPU acceleration
            mapslices(s -> s / norm(s), Matrix(cu(X) * cu(mul_X)), dims=1)
        catch
            # Fallback to CPU
            mapslices(s -> s / norm(s), Matrix{Float32}(X * mul_X), dims=1)
        end

        return nL, new_nVs
    else
        # More genes than cells: compute cell-cell covariance
        Y = _wishart_matrix(X, device=device)  # N×N matrix
        ev_, V = _get_eigen(Y, device=device)
        L = ev_
        
        # Keep only positive eigenvalues
        positive_idx = L .> 0
        L = L[positive_idx]
        V = V[:, positive_idx]

        # Sort by eigenvalue magnitude
        nLi = sortperm(L, rev=true)
        nL = L[nLi]
        nVs = V[:, nLi]
        return nL, nVs
    end
end

function get_sigev(X, Xr; device="gpu")
    """
    Identify signal eigenvalues using Random Matrix Theory.
    
    This is the core function that separates biological signal from technical noise
    by comparing the eigenvalue spectrum of real data (X) with randomized data (Xr).
    
    The algorithm:
    1. Compute eigenvalues for both real and randomized data
    2. Fit Marchenko-Pastur distribution to identify noise eigenvalues
    3. Apply Tracy-Widom correction to determine signal threshold
    4. Return signal and noise components separately
    
    Parameters:
    - X: Real data matrix
    - Xr: Randomized control matrix
    - device: "gpu" or "cpu"
    
    Returns: (signal_eigenvalues, signal_eigenvectors, all_eigenvalues, 
              noise_eigenvalues, signal_threshold, noise_eigenvalues_sorted, 
              noise_eigenvectors)
    """
    n, m = size(X)
    
    if n > m
        # More cells than genes: work in gene space
        Y = _wishart_matrix(X', device=device)
        ev_, V = _get_eigen(Y, device=device)
        Yr = _wishart_matrix(Xr', device=device)
        evr_, _ = _get_eigen(Yr, device=device)

        L = ev_    # Real eigenvalues
        Lr = evr_  # Randomized eigenvalues

        # Fit MP distribution and find noise eigenvalues
        L_mp, _, b_min = _mp_calculation(L, Lr[1:end-1])
        
        # Apply Tracy-Widom correction for signal threshold
        lambda_c, _ = _tw(L, L_mp)
        println("(Using $device) number of signal ev: $(sum(L .> lambda_c))")

        # Separate signal and noise components
        sel_L = L[L .> lambda_c]           # Signal eigenvalues
        sel_Vs = V[:, L .> lambda_c]       # Signal eigenvectors (gene space)

        noiseL = L[b_min .<= L .<= lambda_c]    # Noise eigenvalues
        noiseV = V[:, b_min .<= L .<= lambda_c] # Noise eigenvectors (gene space)

        # Sort by magnitude
        nLi = sortperm(sel_L, rev=true)
        nLi2 = sortperm(noiseL, rev=true)

        nL = sel_L[nLi]
        nVs = sel_Vs[:, nLi]

        snL = noiseL[nLi2]
        noiseVs = noiseV[:, nLi2]
        
        # Transform eigenvectors back to cell space
        mul_X = nVs .* sqrt.(1 ./ nL)'
        mul_X2 = noiseVs .* sqrt.(snL)'
        
        # Signal eigenvectors in cell space
        new_nVs = mapslices(s -> s / norm(s), Matrix(cu(X) * cu(mul_X)), dims=1)
        CUDA.reclaim()  # Free GPU memory
        
        # Noise eigenvectors in cell space
        new_noiseV = try
            mapslices(s -> s / norm(s), Matrix(cu(X) * cu(mul_X2)), dims=1)    
        catch
            mapslices(s -> s / norm(s), Matrix(X * mul_X2), dims=1)
        end
        
        return nL, new_nVs, L, L_mp, lambda_c, snL, new_noiseV
        
    else
        # More genes than cells: work in cell space
        Y = _wishart_matrix(X, device=device)
        ev_, V = _get_eigen(Y, device=device)
        Yr = _wishart_matrix(Xr, device=device)
        evr_, _ = _get_eigen(Yr, device=device)
        L = ev_
        Lr = evr_

        # Same RMT analysis as above
        L_mp, _, b_min = _mp_calculation(L, Lr[1:end-1])
        lambda_c, _ = _tw(L, L_mp)
        println("(Using $device) number of signal ev: $(sum(L .> lambda_c))")

        sel_L = L[L .> lambda_c]
        sel_Vs = V[:, L .> lambda_c]

        noiseL = L[b_min .<= L .<= lambda_c]
        noiseV = V[:, b_min .<= L .<= lambda_c]

        nLi = sortperm(sel_L, rev=true)
        nLi2 = sortperm(noiseL, rev=true)

        nL = sel_L[nLi]
        nVs = sel_Vs[:, nLi]
        
        return nL, nVs, L, L_mp, lambda_c, noiseL[nLi2], noiseV[:, nLi2]
    end
end

# =============================================================================
# ALTERNATIVE NORMALIZATION METHODS
# =============================================================================

function zscore_with_l2(X)
    """
    Z-score normalization with L2 norm regularization.
    
    This function implements a robust normalization that:
    1. Standardizes each gene by its standard deviation
    2. Centers the data
    3. Normalizes by L2 distance from the center
    
    This approach is more robust to outliers than standard z-score.
    
    Parameters:
    - X: Input matrix
    
    Returns: L2-normalized z-scored matrix
    """
    # Step 1: Scale each gene by its standard deviation
    std_ = std(X, dims=1)[:]
    X_norm = X * spdiagm(1. ./ std_)
    
    # Step 2: Calculate mean after scaling
    mu = mean(X_norm, dims=1)

    # Step 3: Calculate L2 norms
    l2X = sqrt.(sum(X_norm .^ 2, dims=2)[:])    # L2 norm of each cell
    l2mu = norm(mu)                              # L2 norm of mean
    
    # L2 distance from mean for each cell
    l2norm_ = sqrt.(l2X .^ 2 .- 2 .* (X_norm * mu')[:] .+ l2mu^2)
    
    # Step 4: Normalize by L2 distance
    (Matrix(X_norm) .- mu) ./ (l2norm_ / mean(l2norm_))
end

# Helper functions for different normalization strategies
proj_l = x -> issparse(x) ? spdiagm(1 ./ sum(x, dims=2)[:]) * x : x ./ sum(x, dims=2)  # Library size normalization
norm_l = x -> issparse(x) ? spdiagm(mean(sqrt.(sum(x .^ 2, dims=2)[:])) ./ sqrt.(sum(x .^ 2, dims=2)[:])) * x : x ./ sqrt.(sum(x .^ 2, dims=2)) * mean(sqrt.(sum(x .^ 2, dims=2)))  # L2 normalization

# =============================================================================
# MAIN scLENS ALGORITHM
# =============================================================================

"""
`sclens(inp_df; device_="gpu", th=70, l_inp=nothing, p_step=0.001, return_scaled=true, n_perturb=20, centering="mean")`

The `sclens` is a function for dimensionality reduction and noise filtering in scRNA-seq data, designed to detect biologically meaningful signals without extensive parameter tuning. 

## Arguments
- `device_`: Specifies the device to be used, either `"cpu"` or `"gpu"`. If `"gpu"` is not available, the function will automatically fall back to `"cpu"`. Note that using `"gpu"` requires an Nvidia graphics card and a compatible driver.
- `th`: The threshold angle (in degrees) used in the signal robustness test. After perturbation, any changes in signal angle greater than this threshold are filtered out. Acceptable values range from 0 to 90. A value of 90 means no filtering, while 0 filters out all signals. Modifying this value is generally not recommended.
- `p_step`: The decrement level for sparsity in the signal robustness test. Increasing `p_step` allows faster computation but reduces the accuracy of the signal robustness test.
- `n_perturb`: Specifies the number of perturbations to perform during the signal robustness test. Increasing this value enhances the accuracy of the test but increases computation time.
- `centering`: Determines whether to center the data on the mean or median during the z-score scaling in log normalization. Only `"mean"` and `"median"` are allowed.

## Output
The function returns a dictionary containing the following keys:

- `:pca`: A DataFrame containing the PC score matrix after applying Random Matrix Theory filtering, in a cell-by-PC format.
- `:pca_n1`: A DataFrame of the PC score matrix after completing the signal robustness test, also in a cell-by-PC format.
- `:L`: The eigenvalues of the data.
- `:L_mp`: The noise eigenvalues.
- `:λ`: The eigenvalue threshold obtained using Random Matrix Theory (RMT).
- `:robustness_scores`: The robustness scores of each signal obtained after the signal robustness test, represented as a dictionary containing:
    - `:m_scores`: Mean scores of each signal' robustness.
    - `:sd_scores`: Standard deviation scores of each signal' robustness.
- `:signal_ev`: The signal eigenvalues, distinguishing significant signals from noise.
- `:signal_evec`: The signal eigenvector matrix.
- `:cell_id`: Barcodes
- `:gene_id`: Gene ID.
- `:sig_id` : 
- `:gene_basis`: 
- `:pass` : 
- `:rec_vals` : 


## Example
```julia
# Basic sclens run with default parameters
result = sclens(inp_df)

# Advanced run with a custom threshold and CPU as the device
result = sclens(inp_df, device_="cpu", th=45, p_step=0.005)
```
"""
function sclens(inp_df; device_="gpu", th=60, p_step=0.001, n_perturb=20, centering="mean")
    """
    Main scLENS algorithm: Random Matrix Theory-based dimensionality reduction.
    
    This function implements the complete scLENS pipeline:
    1. Data preprocessing and normalization
    2. Random Matrix Theory analysis to identify signal
    3. Signal robustness testing through perturbation
    4. Return denoised, low-dimensional representation
    
    The algorithm is designed to automatically separate biological signal from 
    technical noise without requiring extensive parameter tuning.
    """
    
    # =============================================================================
    # STEP 1: DEFINE NORMALIZATION PIPELINE
    # =============================================================================
    
    # Pre-scaling: log-transform after library size normalization
    pre_scale = x -> log1p.(proj_l(x))
    
    # Final normalization strategy based on centering method
    logn_scale = if centering == "mean"
        # Mean-centered L2 normalization
        x -> scaled_gdata(zscore_with_l2(x), position_="cent")
    elseif centering == "median"
        # Median-centered robust normalization
        x -> issparse(x) ? norm_l(scaled_gdata(Matrix{Float32}(x), position_="median")) : norm_l(scaled_gdata(x, position_="median"))
    else
        # Fallback to mean centering with warning
        println("Warning: The specified centering method is not supported in the current algorithm. scLENS will automatically use mean centering.")
        x -> issparse(x) ? scaled_gdata(norm_l(scaled_gdata(Matrix{Float32}(x), position_="mean")), position_="cent") : scaled_gdata(norm_l(scaled_gdata(x, position_="mean")), position_="cent")
    end

    # =============================================================================
    # STEP 2: DATA PREPARATION AND SPARSITY ANALYSIS
    # =============================================================================
    
    println("Extracting matrices")
    X_ = df2sparr(inp_df)  # Convert to sparse matrix

    # Extract sparsity structure for perturbation analysis
    nz_row, nz_col, nz_val = findnz(X_)
    nzero_idx = sparse(nz_row, nz_col, ones(Float32, length(nz_row)))
    N, M = size(X_)

    # Find potential zero positions for adding noise during perturbation
    z_idx1, z_idx2 = begin
        # Generate random positions
        sample_idx = [(i, j) for (i, j) in zip(rand(UInt32(1):UInt32(N), length(nz_val)), rand(UInt32(1):UInt32(M), length(nz_val)))]
        # Find which positions are currently zero
        z_idset = [(i, j) for (i, j) in zip(nz_row, nz_col)]
        nzz_ = setdiff(sample_idx, z_idset)
        [s[1] for s in nzz_], [s[2] for s in nzz_]    
    end
    GC.gc()  # Garbage collection to free memory

    # =============================================================================
    # STEP 3: DATA NORMALIZATION WITH OPTIONAL RECONSTRUCTION VALUES
    # =============================================================================
    
    # Dictionary to store values needed for data reconstruction
    rec_vals = Dict{String, Union{VecOrMat{Float64}}}()
    
    # Apply normalization pipeline (different for mean vs median centering)
    scaled_X = if centering == "mean"
        # Detailed normalization with stored intermediate values for reconstruction
        rec_vals["TGC"] = Vector{Float64}(sum(X_, dims=2)[:])  # Total gene counts per cell
        n_mat = spdiagm(1 ./ rec_vals["TGC"]) * X_             # Library size normalization

        mat2 = log1p.(n_mat)                                   # Log transformation
        rec_vals["mat2_mean"] = mean(mat2, dims=1)             # Gene means
        rec_vals["mat2_std"] = std(mat2, dims=1)               # Gene standard deviations

        mat3_no = mat2 * spdiagm(1. ./ rec_vals["mat2_std"][:])  # Scale by gene std
        mup = mean(mat3_no, dims=1)                              # Updated means

        # L2 normalization calculations
        l2X = sqrt.(sum(mat3_no .^ 2, dims=2)[:])
        l2mu = norm(mup)
        l2norm_ = sqrt.(l2X .^ 2 .- 2 .* (mat3_no * mup')[:] .+ l2mu^2)
        rec_vals["norm_tgc"] = l2norm_
    
        mat4 = (Matrix(mat3_no) .- mup) ./ (rec_vals["norm_tgc"] / mean(rec_vals["norm_tgc"]))

        rec_vals["cent_"] = mean(mat4, dims=1)  # Final centering values
        mat4 .- rec_vals["cent_"]               # Center the data
    else
        # Simpler pipeline for median centering
        logn_scale(pre_scale(X_))
    end

    # =============================================================================
    # STEP 4: RANDOM MATRIX THEORY ANALYSIS
    # =============================================================================
    
    # Generate randomized control data
    X_r = df2sparr(random_nz(inp_df, rmix=true))
    println("Extracting Signals...")
    GC.gc()
    
    # Apply RMT to identify signal vs noise
    nL, nV, L, L_mp, lambda_c, _, noiseV = get_sigev(scaled_X, logn_scale(pre_scale(X_r)), device=device_)
 
    # Validate that eigenvalues follow Marchenko-Pastur distribution
    mpC_ = mp_check(L_mp)
    println("Calculating noise baseline...")

    # =============================================================================
    # STEP 5: SIGNAL ROBUSTNESS TESTING SETUP
    # =============================================================================
    
    # Establish baseline for random correlations
    nm = min(N, M)
    model_norm = Normal(0, sqrt(1 / nm))  # Theoretical distribution for random correlations
    p_tharr = [maximum(abs.(rand(model_norm, nm))) for _ = 1:5000]  # Sample maximum correlations
    p_th = mean(p_tharr)  # Threshold for significant correlations
    println("spth_: $p_th")
    
    # =============================================================================
    # STEP 6: DETERMINE OPTIMAL PERTURBATION LEVEL
    # =============================================================================
    
    p_ = 0.999  # Starting sparsity level
    println("Calculating sparsity level for the perturbation...")
    
    # Calculate baseline eigenvectors from sparsity pattern alone
    Vr2 = if N > M 
        get_eigvec(logn_scale(pre_scale(nzero_idx))', device=device_)[end]
    else
        get_eigvec(logn_scale(pre_scale(nzero_idx)), device=device_)[end]
    end
    
    n_2 = round(Int, lastindex(Vr2, 2) / 2)  # Use half the eigenvectors for comparison
    tank_ = zeros(5, 0)  # Storage for correlation tracking
    tank_n = 5           # Number of values to track
    
    # Iteratively find sparsity level where noise eigenvectors become uncorrelated
    while true
        # Add random values at selected sparsity level
        sple_idx = sample(UInt32(1):UInt32(lastindex(z_idx1)), Int(round((1 - p_) * M * N)), replace=false)
        GC.gc()
        
        # Calculate eigenvectors with added noise
        nV_2 = if N > M
            get_eigvec(logn_scale(pre_scale(
                sparse(vcat(nz_row, z_idx1[sple_idx]), vcat(nz_col, z_idx2[sple_idx]), ones(Float32, length(nz_col) + length(sple_idx)), N, M)))', device=device_)[end]
        else
            get_eigvec(logn_scale(pre_scale(
                sparse(vcat(nz_row, z_idx1[sple_idx]), vcat(nz_col, z_idx2[sple_idx]), ones(Float32, length(nz_col) + length(sple_idx)), N, M))), device=device_)[end]
        end
        
        # Calculate correlations between baseline and perturbed eigenvectors
        d_arr = try
            nanmaximum(abs.(corr_mat(Vr2, nV_2[:, end-n_2:end], device=device_)), dims=1)[:]
        catch
            nanmaximum(abs.(corr_mat(Vr2, nV_2[:, end-n_2:end], device="cpu")), dims=1)[:]
        end
        
        # Track the correlations
        tmp_A = sort(d_arr)
        tank_ = hcat(tank_, tmp_A[1:5])
        ppj_ = if size(tank_, 2) < tank_n
            tank_[2, :]
        else
            tank_[2, end-(tank_n-1):end]
        end
        println(ppj_[end])

        # Check if correlations are below noise threshold
        if (sum(ppj_ .< p_th) > (tank_n - 1)) | (p_ < 0.9)
            p_ += (tank_n - 1) * p_step
            break
        end
        p_ -= p_step
    end
    println("Selected perturb sparisty: $p_")
    
    # Clean up memory
    Vr2 = nothing
    nzero_idx = nothing

    # =============================================================================
    # STEP 7: SIGNAL ROBUSTNESS TESTING
    # =============================================================================
    
    # Generate multiple perturbed datasets
    nV_set = Matrix[]    # Store eigenvectors from each perturbation
    nL_set = Vector[]    # Store eigenvalues from each perturbation
    min_s = size(nV, 2)  # Number of signal components
    min_pc = Int(ceil(min_s * 1.5))  # Number of components to track
    
    @showprogress "perturbing..." for _ in 1:n_perturb
        # Create perturbed dataset
        sple_idx = sample(UInt32(1):UInt32(lastindex(z_idx1)), Int(round((1 - p_) * M * N)), replace=false)
        GC.gc()
        tmp_X = sparse(vcat(nz_row, z_idx1[sple_idx]), vcat(nz_col, z_idx2[sple_idx]), vcat(nz_val, ones(Float32, lastindex(sple_idx))), N, M)
        
        # Calculate eigenvectors for perturbed data
        tmp_nL, tmp_nV = get_eigvec(logn_scale(pre_scale(tmp_X)), device=device_)
        push!(nV_set, tmp_nV[:, 1:min(min_pc, size(tmp_nV, 2))])
        push!(nL_set, tmp_nL[1:min(min_pc, size(tmp_nV, 2))])
    end

    # =============================================================================
    # STEP 8: ROBUSTNESS ANALYSIS AND SIGNAL SELECTION
    # =============================================================================
    
    if iszero(min_s)
        # No signal found
        println("warning: There is no signal")
        results = Dict(:L => L, :L_mp => L_mp,
        :λ => lambda_c, :cell_id => string.(inp_df.cell))
        return results
    else
        # Analyze signal robustness
        th_ = cos(deg2rad(th))  # Convert angle threshold to cosine
        println("Finding robust signals...")
        
        # Find best matching eigenvectors across perturbations
        a_b = hcat([[s[2] for s in argmax(abs.(nV' * j), dims=2)] for j in nV_set]...)

        # Extract matched eigenvectors and calculate correlations
        sub_nVset = [nV_set[s][:, a_b[:, s]] for s = 1:lastindex(nV_set)]
        a_ = hcat([maximum(abs.(nV' * j), dims=2) for j in sub_nVset]...)  # Correlations with original
        
        # Calculate pairwise correlations between perturbations
        b_vec = []
        for i = 1:n_perturb, j = i+1:n_perturb
            push!(b_vec, maximum(abs.(sub_nVset[i]' * sub_nVset[j]), dims=2)[:])
        end
        b_ = hcat(b_vec...)

        # Robust statistics to handle outliers
        q1_val = mapslices(x -> quantile(x, 0.25), b_, dims=2)[:]
        q3_val = mapslices(x -> quantile(x, 0.75), b_, dims=2)[:]
        iqr_val = mapslices(iqr, b_, dims=2)[:]
        
        # Filter outliers using IQR method
        filt_b_ = [b_[s, :][q1_val[s] - 1.5 * iqr_val[s] .<= b_[s, :] .<= q3_val[s] + 1.5 * iqr_val[s]] for s = 1:length(iqr_val)]

        # Calculate robustness scores
        m_score = median.(filt_b_)   # Median correlation across perturbations
        sd_score = std.(filt_b_)     # Standard deviations
        rob_score = m_score          # Use median as robustness score

        # Select robust signals based on threshold
        sig_id = findall(rob_score .> th_)
        println("Number of filtered signal: $(size(sig_id, 1))")
    
        # =============================================================================
        # STEP 9: CONSTRUCT OUTPUT
        # =============================================================================
        
        println("Reconstructing reduced data...")
        
        # Create PC score matrices
        Xout0 = nV .* (sqrt.(nL))'       # All signals
        Xout1 = nV[:, sig_id] .* sqrt.(nL[sig_id])'  # Robust signals only
        
        # Calculate gene basis (for reconstruction)
        tmp_gmat = if device_ == "gpu"
            tmp_X = CuArray{Float32, 2}(undef, size(nV, 2), size(scaled_X, 2))
            mul!(tmp_X, cu(nV'), cu(scaled_X))
            sqrt.(nL) .^ -1 .* Matrix(tmp_X) ./ sqrt.(size(scaled_X, 2))
        else device_ == "cpu"
            sqrt.(nL) .^ -1 .* nV' * scaled_X ./ sqrt.(size(scaled_X, 2))
        end
        
        # Create output DataFrames
        df_X0 = DataFrame(Xout0, :auto)
        insertcols!(df_X0, 1, :cell => inp_df.cell)
        df_X1 = DataFrame(Xout1, :auto)
        insertcols!(df_X1, 1, :cell => inp_df.cell)
 
        # Compile results dictionary
        results = Dict(:pca => df_X0, :pca_n1 => df_X1, :sig_id => sig_id, :L => L, :L_mp => L_mp,
        :λ => lambda_c, :robustness_scores => Dict(:b_ => b_, :rob_score => rob_score, :m_scores => m_score, :sd_scores => sd_score), :signal_evec => nV, :signal_ev => nL,
        :cell_id => inp_df.cell, :gene_id => names(inp_df)[2:end], :gene_basis => tmp_gmat,
        :pass => mpC_[:pass], :rec_vals => rec_vals)
        return results
    end
end

# =============================================================================
# DOWNSTREAM ANALYSIS FUNCTIONS
# =============================================================================

"""
`apply_umap!(input_dict; k=15, nc=2, md=0.1, metric=CosineDist())`

The `apply_umap!` function applies UMAP (Uniform Manifold Approximation and Projection) to the results from `scLENS`, stored in `input_dict`, and adds the UMAP-transformed coordinates and graph object to `input_dict`.

## Arguments
- `input_dict`: The dictionary output from `scLENS`, containing the processed data to which UMAP will be applied.
- `k`: Number of nearest neighbors considered for UMAP. This parameter influences how local relationships are preserved in the embedding.
- `nc`: The number of output dimensions for UMAP, defining the dimensionality of the transformed space (e.g., 2 or 3 for visualization).
- `md`: Minimum distance between points in the UMAP embedding. Smaller values will allow points to be closer together in the low-dimensional space, preserving more local detail.
- `metric`: The distance metric used to measure cell-to-cell distances in the PCA space. While `CosineDist` is used by default, it is generally recommended not to change this metric for consistent results.

## Output
After executing this function:
- `:umap` key in `input_dict` contains the UMAP-transformed coordinates.
- `:umap_obj` key in `input_dict` contains the UMAP graph object with the underlying connectivity information.

## Example
```julia
# Apply UMAP to the scLENS results with default parameters
apply_umap!(input_dict)

# Customize UMAP parameters, if needed
apply_umap!(input_dict, k=10, nc=3, md=0.2)
```

This function integrates UMAP embeddings into the `scLENS` results, facilitating visualization and further analysis in the reduced-dimensional space.
"""
function apply_umap!(l_dict; k=15, nc=2, md=0.1, metric=CosineDist())
    """
    Apply UMAP dimensionality reduction to scLENS results.
    
    UMAP creates a low-dimensional embedding that preserves both local and 
    global structure, making it ideal for visualization and downstream analysis.
    
    This function modifies the input dictionary in-place, adding UMAP results.
    """
    # Use robust signals if available, otherwise fall back to all signals
    pca_y = mat_(l_dict[:pca_n1])
    
    # Create UMAP model
    model = if size(pca_y, 2) > nc
        # Use robust signals for UMAP
        UMAP_(pca_y', nc, metric=metric, n_neighbors=k, min_dist=md)
    else
        # Fall back to first 3 components if not enough robust signals
        UMAP_(mat_(l_dict[:pca])[:, 1:3]', metric=metric, n_neighbors=k, min_dist=md)
    end

    # Store results in the dictionary
    l_dict[:umap] = Matrix(model.embedding')  # UMAP coordinates
    l_dict[:umap_obj] = model                 # UMAP object (contains graph)
end

function get_denoised_df(out_ours; device_="gpu")
    """
    Reconstruct denoised gene expression data using robust signals.
    
    This function projects the data back to gene space using only the robust
    signals, effectively removing noise while preserving biological structure.
    
    Parameters:
    - out_ours: scLENS results dictionary
    - device_: "gpu" or "cpu" for computation
    
    Returns: DataFrame with denoised gene expression data
    """
    # Extract gene basis for robust signals only
    g_mat = out_ours[:gene_basis][out_ours[:sig_id], :]
    Xout0 = Matrix{Float32}(out_ours[:pca_n1][!, 2:end])
    
    # Reconstruct expression data: PC_scores × Gene_basis
    d_mean = if device_ == "gpu"
        Xout = CuArray{Float32, 2}(undef, size(Xout0, 1), size(g_mat, 2))
        mul!(Xout, cu(Xout0), cu(g_mat))
        Matrix{Float32}(Xout) .* sqrt(size(out_ours[:gene_basis], 2))
    elseif device_ == "cpu"
        Xout = Array{Float32, 2}(undef, size(Xout0, 1), size(g_mat, 2))
        mul!(Xout, Xout0, g_mat)
        Xout .* sqrt(size(out_ours[:gene_basis], 2))
    else
        println("Warning: wrong device")
        return nothing
    end
    
    # Create output DataFrame
    odf = DataFrame(d_mean, out_ours[:gene_id])
    insertcols!(odf, 1, :cell => out_ours[:cell_id])
    odf
end

function save_anndata(fn, input; device_="gpu")
    """
    Save scLENS results in AnnData format for interoperability with Python tools.
    
    AnnData is the standard format for single-cell data, compatible with
    scanpy, scvi-tools, and other Python packages.
    
    Parameters:
    - fn: Output filename
    - input: scLENS results dictionary
    - device_: Device for denoising computation
    
    Creates h5ad file with:
    - X: Denoised expression matrix
    - obs: Cell metadata
    - var: Gene metadata  
    - obsm: Dimensionality reduction results (PCA, UMAP)
    - obsp: Connectivity graphs
    - uns: Additional metadata
    """
    # Prepare cell metadata
    out_ldf = if haskey(input, :l_df)
        input[:l_df]  # Use existing metadata if available
    else
        DataFrame(:cell => input[:cell_id])  # Minimal metadata
    end
    
    # Generate denoised expression data
    denoised_df = scLENS.get_denoised_df(input, device_=device_)

    # Create AnnData object with conditional fields
    tmp_adata = if haskey(input, :umap)
        if haskey(input, :ic)
            # Full dataset with UMAP and clustering results
            AnnData(X=scLENS.mat_(denoised_df),
            obs = out_ldf,
            var = DataFrame(:gene => names(denoised_df)[2:end]),
            obsm=Dict("X_pca" => Matrix(input[:pca_n1][!, 2:end]), "X_umap" => input[:umap]),
            obsp=Dict("connectivities" => input[:graph].weights),
            uns=Dict("ic_stat" => input[:ic], "n_cluster" => input[:n_cluster])
            )
        else
            # Dataset with UMAP but no clustering
            AnnData(X=scLENS.mat_(denoised_df),
            obs = out_ldf,
            var = DataFrame(:gene => names(denoised_df)[2:end]),
            obsm=Dict("X_pca" => Matrix(input[:pca_n1][!, 2:end]), "X_umap" => input[:umap])
            )
        end
    else
        if haskey(input, :ic)
            # Dataset with clustering but no UMAP
            AnnData(X=scLENS.mat_(denoised_df),
            obs = out_ldf,
            var = DataFrame(:gene => names(denoised_df)[2:end]),
            obsm=Dict("X_pca" => Matrix(input[:pca_n1][!, 2:end])),
            obsp=Dict("connectivities" => input[:graph].weights),
            uns=Dict("ic_stat" => input[:ic], "n_cluster" => input[:n_cluster])
            )
        else
            # Minimal dataset with only PCA
            AnnData(X=scLENS.mat_(denoised_df),
            obs = out_ldf,
            var = DataFrame(:gene => names(denoised_df)[2:end]),
            obsm=Dict("X_pca" => Matrix(input[:pca_n1][!, 2:end]))
            )
        end
    end
    
    # Write to h5ad format
    writeh5ad(fn, tmp_adata)
end

# =============================================================================
# DATA FORMAT CONVERSION UTILITIES
# =============================================================================

"""
`tenx2jld2(p_dir, out_name="out_jld2/out.jld2", mode="gz")`

The `tenx2jld2` function converts 10x Genomics data from compressed `gz` format to `JLD2` format, facilitating efficient storage and access within Julia.

## Arguments
- `p_dir`: Path to the directory containing the 10x data files. The directory should include the following files:
  - `matrix.mtx.gz`
  - `features.tsv.gz`
  - `barcodes.tsv.gz`
- `out_name`: The name of the output file, including the path where the converted data will be saved. The default location is `out_jld2/out.jld2`.
- `mode`: Specifies the file format of the 10x data. Set to `"gz"` by default for compatibility with `.gz` compressed files.

## Usage Example
```julia
# Convert 10x data from gz format to JLD2 format
scLENS.tenx2jld2("/path/to/10x/data", "output_data.jld2")
```

## Loading the JLD2 Data
Once the data has been converted, you can load it back into Julia as a DataFrame:

```julia
using JLD2

# Load the DataFrame
df = JLD2.load("output_data.jld2", "data")
```

The converted JLD2 file will contain the data under the variable name `"data"`, allowing for easy access and analysis within Julia.
"""
function tenx2jld2(p_dir, out_name="out_jld2/out.jld2", mode="gz")
    """
    Convert 10x Genomics data to efficient Julia format.
    
    10x Genomics produces data in Matrix Market format with separate files
    for the count matrix, gene names, and cell barcodes. This function
    combines them into a single, efficient JLD2 file.
    
    The JLD2 format is much faster to load than CSV and preserves
    Julia data types exactly.
    """
    if mode == "gz"
        println("loading matrix file..")
        
        # Load the sparse count matrix
        M = try
            # Try compressed format first
            tmp_f = joinpath(p_dir, "matrix.mtx.gz")
            f_obj_ = GZip.open(tmp_f)
            tmp_obj = readlines(f_obj_)
            
            # Parse Matrix Market format
            a_tmp = [parse.(Int, split(s, " ")) for s in tmp_obj[3:end]]
            I = [s[1] for s in a_tmp[2:end]]  # Row indices
            J = [s[2] for s in a_tmp[2:end]]  # Column indices  
            K = [s[3] for s in a_tmp[2:end]]  # Values
            GZip.close(f_obj_)
            
            # Create sparse matrix (genes × cells in 10x format)
            sparse(I, J, K, a_tmp[1][1], a_tmp[1][2])
        catch
            # Fallback to uncompressed format
            mmread(joinpath(p_dir, "matrix.mtx"))
        end
        
        println("loading cell_id file..")
        # Load cell barcodes
        cells_ = try
            values(CSV.read(joinpath((p_dir, "barcodes.tsv.gz")), DataFrame, header=false, buffer_in_memory=true)[!, 1])
        catch
            values(CSV.read(joinpath((p_dir, "barcodes.tsv")), DataFrame, header=false)[!, 1])
        end
        
        println("loading gene_id file..")
        # Load gene names (use second column which contains gene symbols)
        gene_ = try
            values(CSV.read(joinpath(p_dir, "features.tsv.gz"), DataFrame, header=false, buffer_in_memory=true)[!, 2])
        catch
            values(CSV.read(joinpath(p_dir, "features.tsv"), DataFrame, header=false)[!, 2])
        end
        
        println("constructing DataFrame...")
        # Create DataFrame in scLENS format (cells × genes)
        # Note: 10x format is genes × cells, so we transpose with M'
        ndf = DataFrame(M', gene_, makeunique=true)
        insertcols!(ndf, 1, :cell => cells_)
        
        # Create output directory if needed
        if !isdir(dirname(out_name)) & !isempty(dirname(out_name))
            mkdir(dirname(out_name))
        end
        
        println("Saving...")
        # Save in compressed JLD2 format
        jldsave(out_name, Dict("data" => ndf); compress=true)
        println("JLD2 file has been successfully saved as: $out_name")
    else
        println("Currently, only 10x gz files are supported.")
    end
end

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

function plot_embedding(inp, l_inp=nothing)
    """
    Create a scatter plot of UMAP embedding with optional color coding.
    
    This function generates publication-ready plots of the low-dimensional
    embedding, with optional coloring by cell types or other metadata.
    
    Parameters:
    - inp: scLENS results dictionary (must contain :umap key)
    - l_inp: Optional labels for color coding
    
    Returns: Makie Figure object
    """
    xlabel_ = "UMAP 1"; ylabel_ = "UMAP 2"; title_ = ""
    
    CairoMakie.activate!()  # Use Cairo backend for high-quality output
    
    inp1 = inp[:umap]  # Extract UMAP coordinates
    
    # Use provided labels or default to single group
    label = if isnothing(l_inp)
        ones(Int, length(inp[:cell_id]))
    else
        l_inp
    end

    # Create figure and axis
    fig = Figure()
    ax = Axis(fig[1, 1], title=title_, xlabel=xlabel_, ylabel=ylabel_, xgridvisible=false, ygridvisible=false)

    if isnothing(label)
        # Simple scatter plot without grouping
        scatter!(ax, inp1[:, 1], inp1[:, 2])
    else
        # Colored scatter plot with legend
        tmp_df1 = DataFrame(x = inp1[:, 1], y = inp1[:, 2], type = label)
        unique_labels = unique(tmp_df1.type)
        
        # Use colorblind-friendly palette
        clist = get(ColorSchemes.tab20, collect(LinRange(0, 1, max(2, length(unique_labels)))))
        sc_list = []
        
        # Plot each group separately for legend
        for (i, ul) in enumerate(unique_labels)
            indices = findall(tmp_df1.type .== ul)
            push!(sc_list, scatter!(ax, tmp_df1.x[indices], tmp_df1.y[indices], color = clist[i], markersize = 5))
        end
        
        # Add legend
        Legend(fig[1, 2], sc_list, string.(unique_labels))
    end

    return fig
end
 
function plot_stability(l_dict)
    """
    Visualize signal robustness scores.
    
    This plot shows the stability of each principal component, helping users
    understand which signals are reliable vs noisy.
    
    Parameters:
    - l_dict: scLENS results dictionary
    
    Returns: Makie Figure object
    """
    CairoMakie.activate!()

    m_scores = l_dict[:robustness_scores][:m_scores]  # Mean robustness scores
    sd_scores = l_dict[:robustness_scores][:sd_scores]  # Standard deviations
    nPC = 1:length(m_scores)  # PC numbers
    color_map = CairoMakie.colormap("RdBu")  # Red-blue color scheme

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="nPC", ylabel="Stability", title= "$(length(l_dict[:sig_id])) robust signals were detected")
    
    # Scatter plot with color coding by stability
    scatter!(ax, nPC, m_scores, color = 1 .- m_scores, colormap=color_map, markersize = 10)
    
    # Add error bars for uncertainty
    errorbars!(nPC, m_scores, sd_scores,
    sd_scores, color = :grey, whiskerwidth=10)

    return fig
end
 
function plot_mpdist(out_ours; dx=2000)
    """
    Visualize the fit of the Marchenko-Pastur distribution to eigenvalues.
    
    This diagnostic plot shows:
    - Histogram of all eigenvalues (blue)
    - Histogram of noise eigenvalues (gray)  
    - Theoretical MP distribution (black line)
    
    A good fit indicates successful noise modeling.
    
    Parameters:
    - out_ours: scLENS results dictionary
    - dx: Number of points for theoretical curve
    
    Returns: Makie Figure object
    """
    L = out_ours[:L]       # All eigenvalues
    L_mp = out_ours[:L_mp]  # Noise eigenvalues
    
    # Create x-axis for theoretical curve
    x = LinRange(0, round(maximum(L) + 0.5), dx)
    lmp_max = maximum(L_mp)
    y = _mp_pdf(x, L_mp)   # Theoretical MP PDF

    CairoMakie.activate!()

    fig = Figure()
    ax = Axis(fig[1, 1],
        xlabel = "Eigenvalue",
        ylabel = "Probability density",
        title = "$(size(out_ours[:pca], 2) - 1) signals were detected"
    )

    # Histogram for all eigenvalues
    hista = hist!(ax, L, bins = 200, normalization = :pdf, color = :blue)
    
    # Histogram for noise eigenvalues
    histb = hist!(ax, L_mp, bins = 200, normalization = :pdf, color = :gray)
    
    # Theoretical MP distribution
    lin = lines!(ax, x[x .< lmp_max + 0.5], y[x .< lmp_max + 0.5], color = :black, linewidth=2)

    # Add legend
    Legend(fig[1, 2],
    [hista, histb, lin],
    ["eigenvalues", "eigenvalues between [a,b]", "fitted MP dist. pdf"])
    return fig
end

# =============================================================================
# END OF MODULE
# =============================================================================

end

# =============================================================================
# SUMMARY OF scLENS MODULE
# =============================================================================
# 
# This module implements scLENS, a sophisticated method for single-cell RNA-seq
# analysis that uses Random Matrix Theory to separate biological signal from
# technical noise without requiring extensive parameter tuning.
#
# ## Key Innovations:
# 
# 1. **Random Matrix Theory**: Uses eigenvalue statistics to automatically
#    distinguish signal from noise based on theoretical distributions
#
# 2. **Signal Robustness Testing**: Perturbs data to test which signals
#    persist under noise, ensuring biological relevance
#
# 3. **GPU Acceleration**: Leverages CUDA for fast computation on large datasets
#
# 4. **Automatic Parameter Selection**: Minimizes user input while maintaining
#    statistical rigor
#
# ## Main Workflow:
# 
# 1. **Data Loading**: `read_file()` - Load and format single-cell data
# 2. **Quality Control**: `preprocess()` - Filter low-quality cells/genes  
# 3. **Dimensionality Reduction**: `sclens()` - Main algorithm
# 4. **Visualization**: `apply_umap!()` - Create 2D embeddings
# 5. **Export**: `save_anndata()` - Save results for other tools
#
# ## Key Functions:
# 
# - `sclens()`: Main algorithm implementing RMT-based dimensionality reduction
# - `get_sigev()`: Random Matrix Theory analysis for signal detection
# - `apply_umap!()`: UMAP embedding for visualization
# - `get_denoised_df()`: Reconstruct noise-free expression data
# - `plot_*()`: Visualization functions for quality control and results
#
# ## Output Interpretation:
# 
# - `:pca_n1`: Robust principal components (use for downstream analysis)
# - `:robustness_scores`: How stable each component is across perturbations
# - `:λ`: Threshold separating signal from noise eigenvalues
# - `:pass`: Whether eigenvalues follow expected noise distribution
#
# The method is designed to be parameter-free while providing interpretable
# quality metrics to assess the reliability of the dimensionality reduction.
# =============================================================================
