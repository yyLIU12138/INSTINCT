library(ArchR)
library(SummarizedExperiment)
library(Matrix)
set.seed(1)
addArchRGenome("mm10")

# idx <- 1

labels <- c('GSM5238385', 'GSM5238386', 'GSM5238387')
inputFiles = c("D:/Data/spCASdata/HumanMouse_Deng2022/GSM5238385_ME11_50um.fragments.tsv.gz",
               "D:/Data/spCASdata/HumanMouse_Deng2022/GSM5238386_ME13_50um.fragments.tsv.gz",
               "D:/Data/spCASdata/HumanMouse_Deng2022/GSM5238387_ME13_50um_2.fragments.tsv.gz")
sampleNames = c('GSM5238385_ME11_50um', 'GSM5238386_ME13_50um', 'GSM5238387_ME13_50um_2')

for (idx in 1:3){
    
    save_dir <- paste0('C:/University/FINAL/results/HumanMouse_Deng2022/[0, 1, 2]/', labels[idx], '/')
    if (!dir.exists(save_dir)) {
        dir.create(save_dir, recursive = TRUE)
    }
    
    
    ArrowFiles <- createArrowFiles(
        inputFiles = inputFiles[idx],
        sampleNames = sampleNames[idx],
        addTileMat = TRUE,
        addGeneScoreMat = TRUE,
        minTSS = 0,
        minFrags = 0,
        maxFrags = Inf,
        force = TRUE,
    )
    
    proj <- ArchRProject(
        ArrowFiles = ArrowFiles,
        outputDirectory = "ArchR_outputfiles",
        copyArrows = TRUE,
    )
    
    gene_score_mat <- getMatrixFromProject(ArchRProj = proj, useMatrix = "GeneScoreMatrix")
    
    gene_score_matrix <- assay(gene_score_mat, "GeneScoreMatrix")
    writeMM(gene_score_matrix, file = paste0(save_dir, "gene_score_matrix.mtx"))
    
    row_names <- rowData(gene_score_mat)$name
    writeLines(row_names, con = paste0(save_dir, "gene_names.txt"))
    
    col_names <- colnames(gene_score_mat)
    writeLines(col_names, con = paste0(save_dir, "spot_names.txt"))
        
}



