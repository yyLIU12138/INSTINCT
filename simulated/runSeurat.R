library(Seurat)
library(Matrix)
library(ggplot2)
library(reticulate)
use_condaenv("epi_integration", required = TRUE)
anndata <- import("anndata")
options(warn=-1)

num_clusters <- 5
num_iters <- 8

scenario <- 1

slice_name_list <- c(
    'Tech_0_0_Bio_0_0.5',
    'Tech_0_0.1_Bio_0_0',
    'Tech_0_0.1_Bio_0_0.5'
)

slice_index_list <- seq_along(slice_name_list)

name_concat <- slice_name_list[1]
for (mode in slice_name_list[-1]) {
    name_concat <- paste(name_concat, mode, sep = "_")
}

save_dir <- paste0('C:/University/FINAL/results/simulated/scenario_', scenario, '/T_', name_concat, '/')

if (!dir.exists(file.path(save_dir, 'comparison/Seurat/'))) {
    dir.create(file.path(save_dir, 'comparison/Seurat/'), recursive = TRUE)
}

print('----------Seurat----------')

for (j in 1:num_iters) {
    
    seed <- 1233 + j
    
    cat(sprintf('Iteration %d\n', j))
    
    seurat_list <- lapply(slice_name_list, function(mode) {
        
        file_path <- sprintf(paste0(save_dir, 'filtered_spot_level_slice_', mode, '.h5ad'))
        adata <- anndata$read_h5ad(file_path)
        counts_matrix <- as(adata$X, "CsparseMatrix")
        seurat_obj <- CreateSeuratObject(counts = t(counts_matrix))
        return(seurat_obj)
        
    })
    
    names(seurat_list) <- slice_name_list
    
    
    # normalize and identify variable features for each dataset independently
    seurat_list <- lapply(X = seurat_list, FUN = function(x) {
        x <- NormalizeData(x)
        x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
    })
    
    # select features that are repeatedly variable across datasets for integration run PCA on each
    # dataset using these features
    features <- SelectIntegrationFeatures(object.list = seurat_list)
    
    seurat_list <- lapply(X = seurat_list, FUN = function(x) {
        x <- ScaleData(x, features = features, verbose = FALSE)
        x <- RunPCA(x, npcs = 100, features = features, seed.use = seed, verbose = FALSE)
    })
    
    cas.anchors <- FindIntegrationAnchors(object.list = seurat_list, dims = 1:100,
                                          anchor.features = features, reduction = "rpca")
    print(cas.anchors)
    
    rm(seurat_list)
    gc()
    
    cas.combined <- IntegrateData(anchorset = cas.anchors, dims = 1:100)
    
    DefaultAssay(cas.combined) <- "integrated"
    
    cas.combined <- ScaleData(cas.combined, verbose = FALSE)
    cas.combined <- RunPCA(cas.combined, seed.use = seed, npcs = 30, verbose = FALSE)
    
    # Save PCA embeddings to CSV file
    pca_embeddings <- cas.combined[["pca"]]@cell.embeddings
    file_path <- file.path(save_dir, sprintf('comparison/Seurat/Seurat_embed_%d.csv', j))
    write.csv(pca_embeddings, file = file_path, row.names = FALSE, col.names = FALSE)
    
    rm(cas.anchors)
    rm(cas.combined)
    gc()
}

print('----------Done----------')



