# =============================================================================
# scICE (single-cell Iterative Clustering and Ensemble) Analysis Package
# =============================================================================
# This file contains functions for single-cell RNA sequencing data clustering
# using graph-based methods and ensemble clustering approaches.
# 
# Key concepts:
# - Graph-based clustering: Represents cells as nodes in a graph, edges show similarity
# - Ensemble clustering: Runs clustering multiple times and finds consensus
# - Leiden clustering: A community detection algorithm for finding cell clusters
# - Inconsistency metric: Measures how stable/reliable the clustering results are
# =============================================================================

# Load required Julia packages
using PyCall        # Interface to Python libraries (igraph, scipy)
using DataFrames    # For handling tabular data (like R's data.frames)
using ProgressMeter # For showing progress bars during long computations
using StatsBase     # Statistical functions (mean, median, etc.)
using Distributed   # For parallel computing across multiple CPU cores
using SparseArrays  # For efficient storage of sparse matrices (mostly zeros)
using LinearAlgebra: dot, I as L_I  # Linear algebra operations
using CairoMakie    # For creating plots and figures

# =============================================================================
# PYTHON INTEGRATION AND DEPENDENCY CHECKING
# =============================================================================
# This section checks if Python is properly installed and accessible from Julia
# PyCall allows Julia to use Python libraries like igraph and scipy

try
    # Try to import Python's sys module to test if Python is working
    pyimport("sys")
    println("Python is properly installed and accessible.")
    return true
catch e
    # If Python import fails, provide installation instructions for different OS
    println("""
    Python is not properly installed or not accessible.
    Error: $e

    To install Python, follow the instructions below for your operating system:

    Windows:
      1. Download the Python installer from https://www.python.org/downloads/
      2. Run the installer and make sure to select 'Add Python to PATH'.
      3. After installation, restart your terminal or IDE.

    Linux (Debian/Ubuntu):
      1. Open a terminal.
      2. Run the following commands:
         sudo apt update
         sudo apt install python3

    MacOS:
      1. Open a terminal.
      2. Run the following command:
         brew install python
         (Note: You need Homebrew installed. Visit https://brew.sh/ if not installed)
    """)
    return false
end

# =============================================================================
# PYTHON PACKAGE INSTALLATION FUNCTION
# =============================================================================
# This function automatically installs required Python packages if they're missing
function pip_installation(pkg_name::String)
    try
        # Try to import the Python package
        pyimport(pkg_name)
        println("Python package '$pkg_name' is already installed.")
    catch
        # If package is missing, attempt to install it using pip
        println("Python package '$pkg_name' is not installed. Attempting to install.")
        try
            # Run pip install command using Julia's run() function
            run(`$(PyCall.python) -m pip install $pkg_name`)
            println("Python package '$pkg_name' has been successfully installed.")
            pyimport(pkg_name)  # Try importing again after installation
        catch e
            println("Failed to install Python package '$pkg_name'.")
            println("Error: ", e)
            return nothing
        end
    end
end

# Install required Python packages for graph analysis
pip_installation("igraph")  # For graph-based clustering algorithms
pip_installation("scipy")   # For scientific computing and sparse matrices

# =============================================================================
# PARALLEL COMPUTING SETUP
# =============================================================================
# Set up multiple CPU cores for parallel processing to speed up computations

# Get number of cores from environment variable, default to 10
n_cores = parse(Int, get(ENV, "NUM_CORES", "10")) 

# Add worker processes (subtract existing processes to avoid duplicates)
addprocs(n_cores - nprocs() + 1)

# Load packages on all worker processes (needed for parallel computing)
@everywhere using SimpleWeightedGraphs: SimpleWeightedGraph  # Graph data structures
@everywhere using PyCall           # Python interface on all workers
@everywhere using StatsBase:mean   # Statistical functions on all workers
@everywhere const ig = pyimport("igraph")  # Import igraph on all workers

# =============================================================================
# GRAPH CONVERSION FUNCTION
# =============================================================================
# Converts Julia adjacency matrices to Python igraph format for clustering
function graph2ig(adj_m, weighted=true)
    """
    Convert a Julia sparse adjacency matrix to Python igraph Graph object.
    
    Parameters:
    - adj_m: Sparse adjacency matrix representing cell-cell similarities
    - weighted: Whether to preserve edge weights (similarities) or just connections
    
    Returns: Python igraph Graph object that can be used for clustering
    """
    # Import scipy.sparse for handling sparse matrices in Python
    sp_sparse = pyimport("scipy.sparse")
    
    # Extract non-zero elements from sparse matrix
    # a, b = row and column indices, c = values at those positions
    a, b, c = findnz(adj_m)
    
    # Create coordinate (COO) format sparse matrix for Python
    # Subtract 1 from indices because Python uses 0-based indexing, Julia uses 1-based
    coo_mat = sp_sparse.coo_matrix((c, (a .-1, b .-1)), shape=[size(adj_m,1), size(adj_m,1)])
    
    if weighted
        # Create weighted undirected graph preserving similarity values
        return ig.Graph.Weighted_Adjacency(coo_mat, mode="undirected")
    else
        # Create unweighted graph (just connections, no similarity values)
        return ig.Graph.Adjacency(coo_mat, mode="undirected")
    end
end

# =============================================================================
# CLUSTERING FUNCTION (runs on each worker process)
# =============================================================================
@everywhere function clust_graph(igg; gamma=0.8, objective_function="CPM", n_iter=5, beta=0.1, init_mem=nothing)
    """
    Perform Leiden clustering on a graph using Python igraph.
    
    Parameters:
    - igg: igraph Graph object
    - gamma: Resolution parameter (higher = more clusters)
    - objective_function: "CPM" (Constant Potts Model) or "modularity" 
    - n_iter: Number of iterations for optimization
    - beta: Randomness parameter for algorithm
    - init_mem: Initial cluster assignments (for iterative refinement)
    
    Returns: Vector of cluster assignments for each node (cell)
    """
    if igg.is_weighted()
        # Use edge weights in clustering if graph is weighted
        igl = igg.community_leiden(
            resolution_parameter=gamma,
            weights="weight",
            objective_function=objective_function,
            n_iterations=n_iter,
            beta=beta,
            initial_membership=init_mem
        )
        Vector{Int16}(igl.membership)  # Convert to Julia Int16 vector
    else
        # Clustering without considering edge weights
        igl = igg.community_leiden(
            resolution_parameter=gamma,
            objective_function=objective_function,
            n_iterations=n_iter,
            beta=beta,
            initial_membership=init_mem
        )
        Vector{Int16}(igl.membership)
    end
end

# =============================================================================
# ENSEMBLE CLUSTERING ANALYSIS
# =============================================================================
# This function analyzes multiple clustering results to find patterns
function extract_arr(inp_arr)
    """
    Analyze an ensemble of clustering results to find unique cluster patterns.
    
    In ensemble clustering, we run the same algorithm many times with slight
    randomness and look for consistent patterns across runs.
    
    Parameters:
    - inp_arr: Matrix where each row is one clustering result
    
    Returns: Dictionary with:
    - :arr: Unique clustering patterns found
    - :parr: Probability/frequency of each pattern
    """
    # Convert each row (clustering result) to a vector
    a_ = [Vector(s) for s in eachrow(inp_arr)]
    
    # Count how often each unique clustering pattern appears
    c_dict_ = countmap(a_)  # Creates Dict: pattern => count
    n_c = collect(values(c_dict_))      # Get counts
    arr_t = collect(keys(c_dict_))      # Get unique patterns

    # Calculate probabilities (frequencies)
    prob_arr = n_c ./ sum(n_c)
    
    # Sort by probability (most common patterns first)
    sort_i = sortperm(prob_arr, rev=true)
    arr_ = arr_t[sort_i]
    n_c = n_c[sort_i]
    prob_arr2 = prob_arr[sort_i]
    
    # Normalize probabilities to sum to 1
    prob_arr2 = prob_arr2 ./ sum(prob_arr2)

    return Dict(:arr => arr_, :parr => prob_arr2)
end

# =============================================================================
# SIMILARITY CALCULATION BETWEEN CLUSTERING RESULTS
# =============================================================================
@everywhere function simmat_v2(inp_a::Vector{Int16}, inp_b::Vector{Int16}; d::Float64=0.9, flag::Bool=true)::Union{Float64, Vector{Float64}}
    """
    Calculate similarity between two clustering results using a random walk-based metric.
    
    This function implements a sophisticated similarity measure that considers:
    - How well clusters in one result correspond to clusters in another
    - The size and structure of each cluster
    - A damping parameter (d) similar to PageRank algorithm
    
    Parameters:
    - inp_a, inp_b: Two clustering results to compare
    - d: Damping factor (0.9 means 90% random walk, 10% teleportation)
    - flag: If true, return average similarity; if false, return per-cell similarities
    
    Returns: Similarity score between 0 and 1 (1 = identical clustering)
    """
    n = length(inp_a)  # Number of cells
    
    # Get unique cluster IDs from both clustering results
    g_idx_a = unique(inp_a)
    g_idx_b = unique(inp_b)

    # Create lists to store which cells belong to each cluster
    gg_idx_a = [Int[] for _ in g_idx_a]  # One list per cluster in result A
    gg_idx_b = [Int[] for _ in g_idx_b]  # One list per cluster in result B
    
    # Populate the cluster membership lists
    for i in 1:n
        id1_ = inp_a[i] + 1  # Add 1 because clusters are 0-indexed
        id2_ = inp_b[i] + 1
        push!(gg_idx_a[id1_], i)  # Cell i belongs to cluster id1_ in result A
        push!(gg_idx_b[id2_], i)  # Cell i belongs to cluster id2_ in result B
    end
    
    # Calculate cluster size penalties (smaller clusters get higher weights)
    c_size1 = d ./ length.(gg_idx_a)  # Weight inversely proportional to cluster size
    c_size2 = d ./ length.(gg_idx_b)
   
    # Cache for avoiding recalculation of same cluster pair similarities
    unique_ecs_vals = fill(NaN, length(g_idx_a), length(g_idx_b))
    ecs = zeros(n)       # Earth mover's distance for each cell
    ppr1 = zeros(n)      # Personalized PageRank vector for result A
    ppr2 = zeros(n)      # Personalized PageRank vector for result B

    # Calculate similarity for each cell
    for i in 1:n
        i1 = inp_a[i] + 1  # Cluster ID in result A
        i2 = inp_b[i] + 1  # Cluster ID in result B
        
        # Check if we've already calculated this cluster pair
        if isnan(unique_ecs_vals[i1, i2])
            # Get all cells in the relevant clusters
            nei1 = gg_idx_a[i1]  # Cells in same cluster as cell i in result A
            nei2 = gg_idx_b[i2]  # Cells in same cluster as cell i in result B
            all_ = BitSet(vcat(nei1, nei2))  # All cells involved in comparison
            
            # Create personalized PageRank vectors
            # PPR represents "probability mass" distributed among cells
            for idx in nei1 
                @inbounds ppr1[idx] = c_size1[i1]  # Distribute mass to cluster members
            end
            ppr1[i] = 1.0 - d + c_size1[i1]  # Extra mass for the reference cell
            
            for idx in nei2
                @inbounds ppr2[idx] = c_size2[i2]
            end
            ppr2[i] = 1.0 - d + c_size2[i2]
            
            # Calculate Earth Mover's Distance between the two distributions
            escore = 0
            for j in all_
                escore += abs(ppr2[j] - ppr1[j])  # Sum of absolute differences
            end
            ecs[i] = escore

            # Reset vectors for next calculation (important for memory efficiency)
            for idx in nei1
                @inbounds ppr1[idx] = 0.0
            end
            for idx in nei2
                @inbounds ppr2[idx] = 0.0
            end
            
            # Cache the result for this cluster pair
            unique_ecs_vals[i1, i2] = ecs[i]
        else
            # Use cached result
            ecs[i] = unique_ecs_vals[i1, i2]
        end
    end

    if flag
        # Return average similarity across all cells
        return mean(1 .- 1 / (2 * d) .* ecs)
    else
        # Return per-cell similarities
        return 1 .- 1 / (2 * d) .* ecs
    end
end

# =============================================================================
# MEMBER-ELEMENT-INCONSISTENCY (MEI) CALCULATION
# =============================================================================
function get_mei_from_array(inp, n_workers=nworkers())
    """
    Calculate Member-Element-Inconsistency (MEI) scores for ensemble clustering.
    
    MEI measures how inconsistently each cell is clustered across multiple runs.
    - Low MEI: Cell is consistently placed in the same cluster
    - High MEI: Cell jumps between different clusters across runs
    
    This helps identify which cells have ambiguous cluster membership.
    """
    if length(inp[:arr]) == 1
        # Only one unique clustering pattern found - all cells are consistent
        return ones(length(inp[:arr][1]))
    else
        # Use parallel workers for faster computation
        wp = CachingPool(workers()[1:n_workers])
        
        # Calculate pairwise similarities between all clustering patterns
        # Weight by pattern frequencies (more common patterns matter more)
        tmp_S = let nu_mem=inp[:arr], p_mem=inp[:parr]
            hcat(vcat(pmap(i -> [simmat_v2(nu_mem[i], nu_mem[j], flag=false) .* (p_mem[i] + p_mem[j]) 
                                for j=i+1:lastindex(nu_mem)], wp, 1:lastindex(nu_mem))...)...)
        end
        clear!(wp)  # Clean up worker pool
        
        # Average inconsistency across all pattern pairs
        return sum(tmp_S, dims=2)[:] ./ (length(inp[:parr]) - 1)    
    end
end

# =============================================================================
# INCONSISTENCY COEFFICIENT (IC) CALCULATION  
# =============================================================================
@everywhere function get_ic2(inp_c; n_workers=nworkers())
    """
    Calculate the Inconsistency Coefficient (IC) for ensemble clustering quality.
    
    IC measures the overall stability of a clustering solution:
    - IC ≈ 1: Very stable, consistent clustering
    - IC > 1: Unstable, inconsistent clustering
    - IC < 1: Over-stable (may indicate over-clustering)
    
    Returns:
    - nu_mem: Matrix of clustering patterns  
    - S_ab: Similarity matrix between patterns
    - prob_arr: Normalized probabilities of each pattern
    - IC value: Overall inconsistency score
    """
    # Convert clustering patterns to matrix format (rows = patterns, cols = cells)
    nu_mem = hcat(inp_c[:arr]...)'
    
    # Use parallel processing to calculate pairwise similarities
    wp = CachingPool(workers()[1:n_workers])
    tmp_S = let nu_mem=nu_mem
        pmap(i -> [simmat_v2(nu_mem[i,:], nu_mem[j,:]) for j=i+1:size(nu_mem,1)], wp, 1:size(nu_mem,1))
    end
    clear!(wp)
    
    # Normalize probabilities to sum to 1
    prob_arr = if sum(inp_c[:parr]) != 1
        inp_c[:parr] ./ sum(inp_c[:parr])
    else
        inp_c[:parr]
    end

    # Build full similarity matrix (symmetric)
    S_ab = zeros(size(nu_mem,1), size(nu_mem,1))
    for i = 1:size(nu_mem,1)
        S_ab[i, i+1:end] = tmp_S[i]  # Fill upper triangle
    end
    S_ab = S_ab + S_ab' + L_I  # Make symmetric and add identity (diagonal = 1)

    # Calculate weighted inconsistency coefficient
    # This is a quadratic form: prob_arr' * S_ab * prob_arr
    return nu_mem, S_ab, prob_arr, dot(S_ab * prob_arr, prob_arr)
end

# =============================================================================
# BEST REPRESENTATIVE CLUSTERING SELECTION
# =============================================================================
function get_best_l(inp)
    """
    Find the "best" representative clustering from an ensemble.
    
    The best clustering is the one most similar to all other clusterings
    (highest average similarity score).
    """
    nu_mem = hcat(inp[:arr]...)'
    
    # Calculate pairwise similarities (same as in get_ic2)
    wp = CachingPool(workers()[1:nworkers()])
    tmp_S = let nu_mem=nu_mem
            pmap(i -> [simmat_v2(nu_mem[i,:], nu_mem[j,:]) for j=i+1:size(nu_mem,1)], wp, 1:size(nu_mem,1))
    end
    clear!(wp)

    # Build similarity matrix
    S_ab = zeros(size(nu_mem,1), size(nu_mem,1))
    for i = 1:size(nu_mem,1)
        S_ab[i, i+1:end] = tmp_S[i]
    end
    S_ab = S_ab + S_ab' + L_I

    # Return the clustering with highest total similarity to all others
    nu_mem[argmax(sum(S_ab, dims=2)[:]), :]
end

# =============================================================================
# MAIN CLUSTERING FUNCTION WITH AUTOMATIC PARAMETER OPTIMIZATION
# =============================================================================
function clustering!(a_dict, r=[1,20]; n_workers=nworkers(), l_ground=nothing, n_steps=11, n_trials=15, 
                     n_boot=100, β=0.1, Nc=10, dN=2, max_iter=150, remove_threshold=1.15, 
                     g_type="umap", obj_fun="CPM", val_tol=1e-8)
    """
    Main clustering function that automatically finds optimal parameters for multiple cluster numbers.
    
    This is the core function that:
    1. Tests different numbers of clusters (from r[1] to r[2])
    2. For each cluster number, finds the optimal gamma (resolution) parameter
    3. Runs ensemble clustering to assess stability
    4. Returns the best clustering solutions
    
    Parameters:
    - a_dict: Dictionary containing graph objects (UMAP, SNN, KNN, etc.)
    - r: Range of cluster numbers to test [min, max]
    - n_workers: Number of parallel workers to use
    - l_ground: Ground truth labels (if available) for validation
    - n_steps: Number of gamma values to test in initial search
    - n_trials: Number of clustering runs for ensemble
    - n_boot: Number of bootstrap samples for IC estimation
    - β: Randomness parameter for Leiden algorithm
    - Nc: Initial number of iterations
    - dN: Iteration increment for refinement
    - max_iter: Maximum iterations before stopping
    - remove_threshold: IC threshold for removing unstable results
    - g_type: Type of graph to use ("umap", "snn", "knn", etc.)
    - obj_fun: Objective function ("CPM" or "modularity")
    - val_tol: Convergence tolerance for parameter search
    """
    
    # Define the range of cluster numbers to test
    t_range = collect(r[1]:r[2])
    
    # Extract the appropriate graph object based on g_type
    rmt_g, rigg, _ = if typeof(a_dict) <: Dict
        if g_type == "umap"
            # Use UMAP graph for clustering
            SimpleWeightedGraph(a_dict[:umap_obj].graph), graph2ig(a_dict[:umap_obj].graph), Array(a_dict[:umap_obj].embedding'), a_dict[:umap_obj], a_dict[:umap_obj].graph
        elseif g_type == "snn"
            # Use Shared Nearest Neighbor graph
            ig_ = graph2ig(adjacency_matrix(a_dict[:snn_obj][1]))
            if haskey(a_dict, :umap)
                a_dict[:snn_obj][1], ig_, a_dict[:umap]
            else
                a_dict[:snn_obj][1], ig_, nothing
            end
        elseif g_type == "knn"
            # Use K-Nearest Neighbor graph
            ig_ = graph2ig(adjacency_matrix(a_dict[:knn_obj][1]))
            if haskey(a_dict, :umap)
                a_dict[:knn_obj][1], ig_, a_dict[:umap]
            else
                a_dict[:knn_obj][1], ig_, nothing
            end
        elseif g_type == "harmony"
            # Use Harmony-corrected graph (for batch effect correction)
            if haskey(a_dict, :umap)
                a_dict[:harmony_graph][1], a_dict[:harmony_graph][2], a_dict[:harmony_umap]
            else
                a_dict[:harmony_graph][1], a_dict[:harmony_graph][2], nothing
            end
        elseif g_type == "testg"
            # Use test graph
            ig_ = graph2ig(adjacency_matrix(a_dict[:testg_obj][1]))
            if haskey(a_dict, :umap)
                a_dict[:testg_obj][1], ig_, a_dict[:umap]
            else
                a_dict[:testg_obj][1], ig_, nothing
            end
        end
    elseif typeof(a_dict[2]) <: PyObject && ((typeof(a_dict[1]) <: SimpleWeightedGraph) | (typeof(a_dict[1]) <: SimpleGraph))
        # Direct graph input
        a_dict[1], a_dict[2], nothing
    end

    # Set up parallel workers with the graph object
    wp = CachingPool(workers()[1:n_workers])
    @everywhere rg_all = $rigg

    # Define parameter search ranges based on objective function
    start_g, end_g = if obj_fun == "modularity"
        0, 10  # Modularity values typically range 0-10
    elseif obj_fun == "CPM"
        min(log(val_tol), -13), 0  # CPM uses log scale, very small to 1
    end

    # Initialize variables for binary search algorithm
    g_dict = Dict{Int,Vector{Float64}}()  # Store gamma ranges for each cluster number
    left, right = start_g, end_g
    left_b, right_b = start_g, end_g
    a, b = start_g, end_g
    gam_xarr = Matrix{Float64}(undef, 3, 0)  # Store [n_clusters, gamma, IC] during search
    Nc_pre = 3; N_cls = 10; β_d = 0.01;  # Initial search parameters
    
    # =============================================================================
    # BINARY SEARCH FOR OPTIMAL GAMMA VALUES
    # =============================================================================
    # For each desired cluster number, find the gamma values that produce it
    
    @showprogress "range_searching..." for i = 1:Int(ceil(length(t_range)/2))
        y = t_range[i]  # Current target cluster number
        if haskey(g_dict, y)
            break  # Skip if already found
        end

        # Binary search to find gamma range that produces y clusters
        left, right = a, b
        flag_ = if obj_fun == "modularity"
            !isapprox(left, right, atol=val_tol)
        else obj_fun == "CPM"
            !isapprox(exp(left), exp(right), atol=val_tol)
        end
        
        while flag_
            mid = (left + right) / 2  # Midpoint for binary search
            g_ = if obj_fun == "modularity"
                mid
            elseif obj_fun == "CPM"
                exp(mid)  # Convert from log scale
            end

            # Run clustering with current gamma value
            X = let gam=g_
                Matrix(hcat(pmap(i -> clust_graph(rg_all, gamma=gam, objective_function=obj_fun, 
                                                n_iter=Nc_pre, beta=β_d), wp, 1:N_cls)...)')
            end
            
            # Count actual number of clusters found
            ff_i = maximum(X, dims=2)[:] .+ 1  # Add 1 because clusters are 0-indexed
            nc_ = median(ff_i)  # Use median across multiple runs
            
            # Adjust search range based on result
            if nc_ < y
                left = mid   # Need higher gamma (more clusters)
            else
                right = mid  # Need lower gamma (fewer clusters)
            end
            
            # Store this data point for analysis
            gam_xarr = hcat(gam_xarr, [nc_, mid, get_ic2(extract_arr(X), n_workers=n_workers)[end]^-1])
            
            # Check convergence
            flag_ = if obj_fun == "modularity"
                !isapprox(left, right, atol=val_tol)
            else obj_fun == "CPM"
                !isapprox(exp(left), exp(right), atol=val_tol)
            end
        end
        
        # Search for upper bound of gamma range
        left_b = copy(right)
        flag_ = if obj_fun == "modularity"
            !isapprox(left_b, right_b, atol=val_tol)
        else obj_fun == "CPM"
            !isapprox(exp(left_b), exp(right_b), atol=val_tol)
        end
        
        while flag_
            mid = (left_b + right_b) / 2
            g_ = if obj_fun == "modularity"
                mid
            elseif obj_fun == "CPM"
                exp(mid)
            end

            X = let gam=g_
                Matrix(hcat(pmap(i -> clust_graph(rg_all, gamma=gam, objective_function=obj_fun, 
                                                n_iter=Nc_pre, beta=β_d), wp, 1:N_cls)...)')
            end
            ff_i = maximum(X, dims=2)[:] .+ 1
            nc_ = median(ff_i)
            
            if nc_ > y
                right_b = mid
            else
                left_b = mid 
            end
            gam_xarr = hcat(gam_xarr, [nc_, mid, get_ic2(extract_arr(X), n_workers=n_workers)[end]^-1])

            flag_ = if obj_fun == "modularity"
                !isapprox(left, right, atol=val_tol)
            else obj_fun == "CPM"
                !isapprox(exp(left_b), exp(right_b), atol=val_tol)
            end
        end
        
        # Store the gamma range for this cluster number
        g_dict[y] = [left, right_b]

        # Now search for the opposite end of the range (high cluster number)
        left = sum(g_dict[y])/2
        right = b
        y = t_range[end-(i-1)]  # Target high cluster number
        if haskey(g_dict, y)
            break
        end

        # Repeat binary search for high cluster number
        flag_ = if obj_fun == "modularity"
            !isapprox(left, right, atol=val_tol)
        else obj_fun == "CPM"
            !isapprox(exp(left), exp(right), atol=val_tol)
        end
        
        while flag_
            mid = (left + right) / 2
            g_ = if obj_fun == "modularity"
                mid
            elseif obj_fun == "CPM"
                exp(mid)
            end

            X = let gam=g_
                Matrix(hcat(pmap(i -> clust_graph(rg_all, gamma=gam, objective_function=obj_fun, 
                                                n_iter=Nc_pre, beta=β_d), wp, 1:N_cls)...)')
            end
            ff_i = maximum(X, dims=2)[:] .+ 1
            nc_ = median(ff_i)
            if nc_ < y
                left = mid 
            else
                right = mid
            end
            gam_xarr = hcat(gam_xarr, [nc_, mid, get_ic2(extract_arr(X), n_workers=n_workers)[end]^-1])
            flag_ = if obj_fun == "modularity"
                !isapprox(left, right, atol=val_tol)
            else obj_fun == "CPM"
                !isapprox(exp(left), exp(right), atol=val_tol)
            end
        end

        # Upper bound search for high cluster number
        left_b = right
        right_b = b
        flag_ = if obj_fun == "modularity"
            !isapprox(left_b, right_b, atol=val_tol)
        else obj_fun == "CPM"
            !isapprox(exp(left_b), exp(right_b), atol=val_tol)
        end
        
        while flag_
            mid = (left_b + right_b) / 2
            g_ = if obj_fun == "modularity"
                mid
            elseif obj_fun == "CPM"
                exp(mid)
            end

            X = let gam=g_
                Matrix(hcat(pmap(i -> clust_graph(rg_all, gamma=gam, objective_function=obj_fun, 
                                                n_iter=Nc_pre, beta=β_d), wp, 1:N_cls)...)')
            end
            ff_i = maximum(X, dims=2)[:] .+ 1
            nc_ = median(ff_i)
            if nc_ > y
                right_b = mid
            else
                left_b = mid 
            end
            gam_xarr = hcat(gam_xarr, [nc_, mid, get_ic2(extract_arr(X), n_workers=n_workers)[end]^-1])
            flag_ = if obj_fun == "modularity"
                !isapprox(left, right, atol=val_tol)
            else obj_fun == "CPM"
                !isapprox(exp(left_b), exp(right_b), atol=val_tol)
            end
        end
        g_dict[y] = [left, right_b]
        
        # Update search bounds for next iteration
        left, = sum(g_dict[t_range[i]])/2
        right = sum(g_dict[t_range[end-(i-1)]])/2
        left_b = sum(g_dict[t_range[i]])/2
        right_b = sum(g_dict[t_range[end-(i-1)]])/2
        a, b = left, right_b
    end
    
    # Sort gamma exploration results by gamma value
    gam_xarr = gam_xarr[:, sortperm(gam_xarr[2,:])]
    
    # Convert gamma ranges to appropriate scale
    gam_range = if obj_fun == "CPM" 
        Dict(s.first => (s.second[2] - s.second[1]) > 0 ? exp.(s.second) : exp.([g_dict[max(1, s.first-1)][2],
        g_dict[min(maximum(t_range), s.first+1)][1]]) for s in g_dict)
    elseif obj_fun == "modularity" 
        gam_range = Dict(s.first => (s.second[2] - s.second[1]) > 0 ? s.second : [g_dict[max(1, s.first-1)][2],
        g_dict[min(maximum(t_range), s.first+1)][1]] for s in g_dict)
    else
        println("Warning!")
        return nothing
    end
    
    # =============================================================================
    # FILTERING OUT UNSTABLE CLUSTER NUMBERS
    # =============================================================================
    # Remove cluster numbers that show high inconsistency during exploration
    
    tmp_dict = Dict(i => Matrix{Float64}(undef, 0, 3) for i in t_range)
    for i = 1:lastindex(gam_xarr, 2)
        g_ = if obj_fun == "CPM"
            exp(gam_xarr[2, i])
        elseif obj_fun =="modularity"
            gam_xarr[2, i]
        end
        # Find which cluster numbers this gamma value can produce
        n_ = [x.first for x in gam_range if x.second[1] <= g_ <= x.second[2]]
        if isempty(n_)
            continue
        end
        for n in n_
            if n in t_range
                tmp_dict[n] = vcat(tmp_dict[n], gam_xarr[:, i]')
            end
        end
    end
    
    # Remove cluster numbers with consistently high inconsistency
    filter!(x -> size(x.second, 1) > 1 ? (minimum(x.second[:, 3]) >= remove_threshold) : true, tmp_dict)

    ex_n = sort(collect(keys(tmp_dict)))
    println("$ex_n are removed from the candidates")

    # =============================================================================
    # DETAILED ANALYSIS FOR REMAINING CLUSTER NUMBERS
    # =============================================================================
    # For each viable cluster number, do detailed ensemble clustering analysis
    
    # Initialize result storage
    list_l = Vector{Dict{Symbol, Vector}}([])          # Clustering patterns
    list_l_best = Vector{Vector{Int16}}([])            # Best representative clusterings
    list_incons = Vector{Float64}([])                  # Inconsistency coefficients
    list_allic = Vector{Vector{Float64}}([])           # IC bootstrap distributions
    list_gam = Vector{Float64}([])                     # Optimal gamma values
    list_mn = Vector{Float64}([])                      # Mean cluster numbers
    list_k = Vector{Int}([])                           # Number of iterations used
    
    # Get final list of cluster numbers to analyze
    all_n = sort(intersect(collect(keys(gam_range)), setdiff(t_range, vcat(1, ex_n))))
    pbar = Progress(length(all_n), desc="zoomin_steps: ", showspeed=true)    
    tmp_get_ic = x -> x[end]^-1  # Helper function to extract IC
    
    for i in all_n
        # Get gamma range for this cluster number
        st_g, en_g = gam_range[i]
        
        # Calculate step size for detailed search
        d_g = if obj_fun == "modularity"
            (en_g - st_g) ./ n_steps
        elseif obj_fun == "CPM"
            (log(en_g) - log(st_g)) ./ n_steps
        end
        
        # Create array of gamma values to test
        g_x = try
            if obj_fun == "modularity"
                LinRange(round(st_g, digits=2), round(en_g, digits=2), n_steps)
                if st_g != en_g
                    st_g:d_g:en_g
                else
                    st_g-d_g : d_g /n_steps*2 : st_g + d_g
                end

            elseif obj_fun == "CPM"
                if st_g != en_g
                    exp.(log(st_g):d_g:log(en_g))
                else
                    exp.(log(st_g)-d_g : d_g /n_steps*2 : log(st_g) + d_g)
                end
            end
        catch
            []
        end
        mn_arr = zeros(size(g_x, 1))

        # Run ensemble clustering for each gamma value
        X_list = Vector{Matrix}(undef, size(g_x, 1))
        for (ii, g) in enumerate(g_x)
            X_list[ii] = let gam=g
                Matrix(hcat(pmap(i -> clust_graph(rg_all, gamma=gam, objective_function=obj_fun, 
                                                n_iter=Nc, beta=β), wp, 1:n_trials)...)')
            end
            mn_arr[ii] = median(maximum(X_list[ii], dims=2)) + 1
        end
        clear!(wp)
        
        # Filter to keep only results that produce the target cluster number
        b_idx = mn_arr .== i
        g_x = g_x[b_idx]
        X_list = X_list[b_idx]
        mn_arr = mn_arr[b_idx]
        
        # Extract clustering patterns and calculate inconsistencies
        e_arr = extract_arr.(X_list)
        b_ic = tmp_get_ic.(get_ic2.(e_arr, n_workers=n_workers))
        one_i = findfirst(b_ic .== 1)  # Perfect consistency found?
        k = Nc

        if isempty(g_x)
            # No gamma values produced the target cluster number
            ProgressMeter.next!(pbar, showvalues=[(:searching, i)])
            continue
        elseif !isnothing(one_i)
            # Found perfect consistency - use this result
            g_x = g_x[one_i]
            X_list = X_list[one_i]
            mn_arr = mn_arr[one_i]

            # Bootstrap estimation of IC distribution
            ic_mat = zeros(n_boot)
            for j = 1:n_boot
                smp_i = sample(1:size(X_list, 1), size(X_list, 1))
                X_arr = extract_arr(X_list[smp_i, :])
                ic_mat[j] = get_ic2(X_arr, n_workers=n_workers)[end]^-1
            end
            ic_median = median(ic_mat)
        
            X_earr = extract_arr(X_list)
            best_l = get_best_l(X_earr)
        
            # Store results
            push!(list_gam, g_x)
            push!(list_mn, mn_arr)
            push!(list_incons, ic_median)
            push!(list_allic, ic_mat)
            push!(list_l_best, best_l)
            push!(list_l, X_earr)
            push!(list_k, k)
            ProgressMeter.next!(pbar, showvalues=[(:searching, i), (:γ, g_x), (:nClusterings, size(X_list, 1)),
            (:IC, median(ic_mat)), (:N_Iterations, k)])
            continue
        end

        # =============================================================================
        # ITERATIVE REFINEMENT FOR BETTER CONSISTENCY
        # =============================================================================
        # If no perfect consistency found, iteratively refine clustering
        
        nk = 1
        tank_bic_all = repeat(ones(length(b_ic)) .* 2, 1, 10)  # Track IC history
        tank_bic = repeat(b_ic, 10)
        ustable_id = ones(Bool, length(g_x))  # Track which gamma values are still changing
        
        while true
            k += dN  # Increase iteration count
            
            # Refine clustering for unstable gamma values
            for i = 1:lastindex(g_x)
                X_list[i] = if ustable_id[i]
                    let gam=g_x[i], init_mem = X_list[i], dN = dN
                        Matrix(hcat(pmap(i -> clust_graph(rg_all, gamma=gam, objective_function=obj_fun, 
                                                        n_iter=dN, beta=β, init_mem=init_mem[i, :]), wp, 1:n_trials)...)')
                    end
                else
                    X_list[i]  # Keep unchanged if already stable
                end
            end
            clear!(wp)
            
            # Recalculate inconsistencies
            mn_arr = median.(maximum.(X_list, dims=2))[:] .+ 1
            e_arr = extract_arr.(X_list)
            b_ic_t = tmp_get_ic.(get_ic2.(e_arr, n_workers=n_workers))
            
            # Update IC history
            tank_bic_all[:, 1:end-1] = tank_bic_all[:, 2:end]
            tank_bic_all[:, end] = b_ic_t
            last_bic = b_ic_t
            diff_bic = diff(tank_bic_all, dims=2)
            stable_idx = iszero.(diff_bic)  # Which gamma values have stabilized
            ustable_id = (sum(stable_idx, dims=2)[:, 1] .!= size(stable_idx, 2)) .& (tank_bic_all[:, 1] .- tank_bic_all[:, end] .>= 0)
            one_i = tank_bic_all[:, end] .== 1  # Perfect consistency achieved
            
            # Selection criteria for continuing or stopping
            b_idx = (last_bic .<= quantile(last_bic, 0.5)) .| ustable_id
            b_idx[argmin(last_bic)] = true  # Always keep the best result
            
            if length(b_idx) == 1
                # Only one gamma value left
                g_x = g_x[1]
                X_list = X_list[1]
                mn_arr = mn_arr[1]
                b_ic_t = b_ic_t[1]
                tank_bic = tank_bic_all[1, :]
                break
            elseif any(one_i)
                # Perfect consistency achieved
                sel_i = findfirst(one_i)
                g_x = g_x[sel_i]
                X_list = X_list[sel_i]
                mn_arr = mn_arr[sel_i]
                b_ic_t = b_ic_t[sel_i]
                tank_bic = tank_bic_all[sel_i, :]
                break
            elseif all(stable_idx)
                # All gamma values have stabilized
                sel_i = argmin(b_ic_t)
                g_x = g_x[sel_i]
                X_list = X_list[sel_i]
                mn_arr = mn_arr[sel_i]
                b_ic_t = b_ic_t[sel_i]
                tank_bic = tank_bic_all[sel_i, :]
                break
            elseif k >= max_iter
                # Maximum iterations reached
                sel_i = argmin(b_ic_t)
                g_x = g_x[sel_i]
                X_list = X_list[sel_i]
                mn_arr = mn_arr[sel_i]
                b_ic_t = b_ic_t[sel_i]
                tank_bic = tank_bic_all[sel_i, :]
                break
            elseif k > 100
                # After many iterations, accept good enough results
                if median(b_ic_t) > 1.1
                    sel_i = argmin(b_ic_t)
                    g_x = g_x[sel_i]
                    X_list = X_list[sel_i]
                    mn_arr = mn_arr[sel_i]
                    b_ic_t = b_ic_t[sel_i]
                    tank_bic = tank_bic_all[sel_i, :]
                    break
                end
            else
                # Continue with selected gamma values
                g_x = g_x[b_idx]
                X_list = X_list[b_idx]
                mn_arr = mn_arr[b_idx]
                b_ic_t = b_ic_t[b_idx]
                tank_bic_all = tank_bic_all[b_idx, :]
            end
            nk += 1
        end

        # Ensure we have single values (not vectors)
        if typeof(g_x) <: Vector
            g_x = g_x[1]
            X_list = X_list[1]
            mn_arr = mn_arr[1]
        end
        
        # Bootstrap estimation of final IC
        ic_mat = zeros(n_boot)
        for j = 1:n_boot
            smp_i = sample(1:size(X_list, 1), size(X_list, 1))
            X_arr = extract_arr(X_list[smp_i, :])
            ic_mat[j] = get_ic2(X_arr, n_workers=n_workers)[end]^-1
        end
        ic_median = median(ic_mat)
       
        X_earr = extract_arr(X_list)
        best_l = get_best_l(X_earr)

        # Store final results
        push!(list_gam, g_x)
        push!(list_mn, mn_arr)
        push!(list_incons, ic_median)
        push!(list_allic, ic_mat)
        push!(list_l_best, best_l)
        push!(list_l, X_earr)
        push!(list_k, k)
        ProgressMeter.next!(pbar, showvalues=[(:searching, i), (:γ, g_x), (:nClusterings, size(X_list, 1)),
        (:IC, median(ic_mat)), (:N_Iterations, k), (:N_loops, nk)])
    end

    # =============================================================================
    # FINALIZE RESULTS
    # =============================================================================
    
    if isempty(findall(mean.(list_incons) .< 2))
        # No stable clustering found
        @everywhere begin
            rigg = nothing
            GC.gc()  # Garbage collection
        end
        return nothing
    else
        # Store all results in the input dictionary
        a_dict[:gamma] = list_gam                    # Optimal gamma values
        a_dict[:labels] = list_l                     # Clustering patterns
        a_dict[:ic] = list_incons                    # Inconsistency coefficients
        a_dict[:ic_vec] = list_allic                 # IC bootstrap distributions
        a_dict[:n_cluster] = list_mn                 # Number of clusters
        a_dict[:l_ground] = l_ground                 # Ground truth (if provided)
        a_dict[:graph] = rmt_g                       # Graph object
        a_dict[:best_l] = list_l_best                # Best representative clusterings
        a_dict[:n_iter] = list_k                     # Iterations used
        a_dict[:mei] = get_mei_from_array.(list_l, Ref(n_workers))  # Member inconsistency

        # Clean up parallel workers
        @everywhere begin
            rigg = nothing
            GC.gc()
        end
    end
end

# =============================================================================
# RESULT EXTRACTION AND VISUALIZATION FUNCTIONS
# =============================================================================

function get_rlabel!(input, th=1.005)
    """
    Extract reliable clustering labels based on inconsistency threshold.
    
    Only returns clusterings with IC below the threshold (more stable results).
    
    Parameters:
    - input: Results dictionary from clustering!()
    - th: IC threshold for accepting results (default 1.005, very strict)
    
    Returns: DataFrame with cell IDs and cluster labels for reliable clusterings
    """
    # Select only results below inconsistency threshold
    b_idx = input[:ic] .< th
    labels_ = Dict(Int.(input[:n_cluster][b_idx]) .=> input[:best_l][b_idx])
    c_names = input[:pca].cell  # Get cell names from PCA results
    
    # Create DataFrame with cluster labels
    out_df = DataFrame(Dict(["l_"*string(s.first) => s.second for s in labels_]))
    insertcols!(out_df, 1, :cell_id => c_names)
    input[:l_df] = out_df
    out_df
end

function plot_ic(out_, th=1.005; fig_size=(550,500))
    """
    Create a boxplot showing IC distributions for different cluster numbers.
    
    This plot helps visualize which cluster numbers give stable results.
    - Lower IC values = more stable clustering
    - Red line shows the threshold for acceptable stability
    
    Parameters:
    - out_: Results dictionary from clustering!()
    - th: IC threshold to show as reference line
    - fig_size: Plot dimensions
    
    Returns: Makie Figure object
    """
    # Prepare data for plotting
    x = repeat(out_[:n_cluster], inner=length(out_[:ic_vec][1]))  # Cluster numbers
    y = vcat(out_[:ic_vec]...)  # IC values from bootstrap
    
    # Create figure
    fig = Figure(size=fig_size)
    ax = Axis(fig[1, 1], xlabel="Number of clusters", ylabel="IC",
        limits=((0.5, maximum(out_[:n_cluster]) + 0.5), nothing))
    
    # Add threshold line
    hlines!(ax, [th]; xmin = 0.02, xmax = 0.98, color=:red)
    
    # Create boxplot
    boxplot!(ax, x, y)
    fig
end

using CSV
function save_plot_ic(out_, output_path::Union{String,Nothing}=nothing; th=1.005)
    """
    Save IC plot data to CSV file or return as DataFrame.
    
    This function extracts the data used in plot_ic() and either saves it
    to a file or returns it for further analysis.
    
    Parameters:
    - out_: Results dictionary from clustering!()
    - output_path: If provided, save data to this CSV file path
    - th: IC threshold value to include in data
    
    Returns: DataFrame with plot data (if output_path is nothing)
    """
    # Prepare the same data as plot_ic()
    x = repeat(out_[:n_cluster], inner=length(out_[:ic_vec][1]))
    y = vcat(out_[:ic_vec]...)
    
    # Create DataFrame with the plot data
    df = DataFrame(
        n_clusters = x,        # Number of clusters for each data point
        ic_values = y,         # IC values from bootstrap sampling
        threshold = fill(th, length(x))  # Threshold for reference
    )
    
    # If output path is provided, save to CSV
    if !isnothing(output_path)
        CSV.write(output_path, df)
        return nothing
    end
    
    # Otherwise return the DataFrame
    return df
end

# =============================================================================
# END OF FILE
# =============================================================================
# 
# Summary of what this file does:
# 
# 1. SETUP: Installs Python dependencies and sets up parallel processing
# 
# 2. GRAPH CONVERSION: Converts Julia matrices to Python igraph format
# 
# 3. CLUSTERING: Implements Leiden algorithm for community detection
# 
# 4. ENSEMBLE ANALYSIS: Runs clustering multiple times and analyzes consistency
# 
# 5. SIMILARITY METRICS: Calculates how similar different clustering results are
# 
# 6. PARAMETER OPTIMIZATION: Automatically finds best parameters for each cluster number
# 
# 7. STABILITY ASSESSMENT: Uses IC and MEI metrics to evaluate clustering quality
# 
# 8. RESULT EXTRACTION: Provides functions to extract and visualize final results
# 
# The main workflow is:
# 1. clustering!() - Run the full analysis
# 2. get_rlabel!() - Extract stable cluster labels  
# 3. plot_ic() - Visualize stability across different cluster numbers
# 
# Key concepts for single-cell analysis:
# - Each cell is a node in a graph
# - Edge weights represent cell-cell similarity
# - Clustering finds groups of similar cells (cell types)
# - Ensemble methods improve robustness
# - IC measures help choose reliable cluster numbers
# =============================================================================