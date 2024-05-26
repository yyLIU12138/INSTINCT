options(warn=-1)
library(scABC)
library(chromVAR)
library(motifmatchr)
library(SummarizedExperiment)
library(Matrix)
# library(ggplot2)
library(BiocParallel)
library(BSgenome.Hsapiens.UCSC.hg19)
library(JASPAR2016)
library(data.table)
library(parallel)
set.seed(1234)

# archive_file <- "C:/University/FINAL/data/SNP_files/1000G_Phase3_plinkfiles.tgz"
# extract_dir <- "C:/University/FINAL/data/SNP_files/"
# untar(archive_file, exdir = extract_dir)


# find and return indices for overlaps
find.index <- function(df1,df2,type='reg'){
    #colnames(df1) <- colnames(df2) <- c('V1','V2','V3')
    library(GenomicRanges)
    df1.gr = GRanges (IRanges(start = df1$V2, end = df1$V3), seqnames=df1$V1)
    if(type=='reg'){
        df2.gr = GRanges(IRanges(start=df2$V2, end = df2$V3), seqnames = df2$V1)
    }
    if(type=='pos'){
        df2.gr = GRanges(IRanges(start=df2$V4, end = df2$V4), seqnames = df2$V1)
    }
    df1.git  = GNCList(df1.gr)
    df2.git  = GNCList(df2.gr)
    overlap_git = findOverlaps(df2.git, df1.git)
    overlap_git
    temp <- as.data.frame(overlap_git)
    colnames(temp) <- c('df2','df1')
    return(temp)
}


mode_list <- c('E11_0', 'E13_5', 'E15_5', 'E18_5')
mode_index <- 4
mode = mode_list[mode_index]


# Load data
data_path <- "C:/University/FINAL/data/SNP_files/"
for(chr in 1:22){
    if(chr==1){
        # https://doi.org/10.5281/zenodo.7768714 -> 1000G_Phase3_plinkfiles.tgz
        bim <- fread(paste0(data_path, '1000G_EUR_Phase3_plink/1000G.EUR.QC.1.bim'))
    }else{
        bim <- rbind(bim,fread(paste0(data_path, '1000G_EUR_Phase3_plink/1000G.EUR.QC.',chr,'.bim')))
    }
} ## 1000G bim file
bim$V1 <- paste0('chr',bim$V1)


bg_mode <- 'all'

inputpath <- paste0("C:/University/FINAL/results/MouseBrain_Jiang2023/vertical/", mode, '/INSTINCT/GRCh37hg19/S2/')
outputpath <- paste0("C:/University/FINAL/results/MouseBrain_Jiang2023/vertical/", mode, '/INSTINCT/SNPs_2/S2_', bg_mode, '/')
if (!dir.exists(outputpath)) {
    dir.create(outputpath, recursive = TRUE)
}


bg_name <- paste0('bg_', bg_mode)
bed_file <- fread(paste0(inputpath, bg_name, '.bed'), header = FALSE, sep = "\t", quote = "")
chromatin <- sub(":.*", "", bed_file$V1)
positions <- gsub(".*:", "", bed_file$V1)
start <- as.numeric(gsub("-.*", "", positions))
end <- as.numeric(gsub(".*-", "", positions))
bg1 <- data.frame(V1 = chromatin, V2 = start, V3 = end)
ind.bg <- find.index(bg1,bim,type='pos')
snpset <- list()
snpset[[1]] <- bim$V2[ind.bg$df2]


## 2. specific annotations
peak <- list()
name <- c('Midbrain', 'Diencephalon_and_hindbrain', 'Subpallium_1', 'Subpallium_2', 'Cartilage_1', 'Cartilage_2', 
          'Cartilage_3', 'Cartilage_4', 'Mesenchyme', 'Muscle', 'Thalamus', 'DPallm', 'DPallv')
for(k in 1:length(name)){
    print(name[k])
    bed_file <- fread(paste0(inputpath,name[k],'.bed'), header = FALSE, sep = "\t", quote = "")
    chromatin <- sub(":.*", "", bed_file$V1)
    positions <- gsub(".*:", "", bed_file$V1)
    start <- as.numeric(gsub("-.*", "", positions))
    end <- as.numeric(gsub(".*-", "", positions))
    peak[[k]] <- data.frame(V1 = chromatin, V2 = start, V3 = end)
    ind.temp <- find.index(peak[[k]],bim,type='pos')
    snpset[[k+1]] <- bim$V2[ind.temp$df2]
}


for(chr in 1:22){
    print(chr)
    bim2 <- fread(paste0(data_path, '1000G_EUR_Phase3_plink/1000G.EUR.QC.',chr,'.bim')) ## 1000G bim file for each chromosome
    for(j in 1:(length(name)+1)){
        index <- which(bim2$V2%in%snpset[[j]])
        anno <- rep(0,nrow(bim2))
        anno[index] <- 1
        if(j==1){
            anno1 <- cbind(rep(1,nrow(bim2)),anno)
        }else{
            anno1 <- cbind(anno1,anno)
        }
    }
    colnames(anno1) <- c('base',bg_name,name)
    # write.table(anno1,paste0(outputpath,'bg.mm',chr,'.annot'),quote=F,row.names=F,col.names=T,sep='\t')
}



















