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


inputpath <- paste("C:/University/FINAL/results/MouseBrain_Jiang2023/vertical", mode, 'INSTINCT/GRCh37hg19/S2/', sep = "/")
outputpath <- paste("C:/University/FINAL/results/MouseBrain_Jiang2023/vertical", mode, 'INSTINCT/SNPs/S2/', sep = "/")
if (!dir.exists(outputpath)) {
    dir.create(outputpath, recursive = TRUE)
}


## 1. bg
name <- c('bg_union', 'bg_all')
for(k in 1:2){
    if (file.exists(paste0(inputpath,name[k],'.bed'))){
        print(name[k])
        bed_file <- fread(paste0(inputpath,name[k],'.bed'), header = FALSE, sep = "\t", quote = "")
        chromatin <- sub(":.*", "", bed_file$V1)
        positions <- gsub(".*:", "", bed_file$V1)
        start <- as.numeric(gsub("-.*", "", positions))
        end <- as.numeric(gsub(".*-", "", positions))
        peak <- data.frame(V1 = chromatin, V2 = start, V3 = end)
    }else{
        next
    }
    ind.temp <- find.index(peak,bim,type='pos')
    bim1 <- bim[ind.temp$df2,c(1,4,2)]
    colnames(bim1) <- c("CHR","POS","SNP")
    print(dim(bim1))
    # if (nrow(bim1) > 0){
    #     write.table(bim1,paste0(outputpath,gsub(" ","_",name[k]),'.anno'),sep="\t",quote=F,col.names=T,row.names=F)
    # }else{
    #     print('Skipped !')
    # }
}

## 2. specific annotations
name <- c('Midbrain', 'Diencephalon_and_hindbrain', 'Basal_plate_of_hindbrain', 'Subpallium_1', 'Subpallium_2',
          'Cartilage_1', 'Cartilage_2', 'Cartilage_3', 'Cartilage_4', 'Mesenchyme', 'Muscle', 'Thalamus', 'DPallm', 'DPallv')
for(k in 1:14){
    if (file.exists(paste0(inputpath,name[k],'.bed'))){
        print(name[k])
        bed_file <- fread(paste0(inputpath,name[k],'.bed'), header = FALSE, sep = "\t", quote = "")
        chromatin <- sub(":.*", "", bed_file$V1)
        positions <- gsub(".*:", "", bed_file$V1)
        start <- as.numeric(gsub("-.*", "", positions))
        end <- as.numeric(gsub(".*-", "", positions))
        peak <- data.frame(V1 = chromatin, V2 = start, V3 = end)
    }else{
        next
    }
    ind.temp <- find.index(peak,bim,type='pos')
    bim1 <- bim[ind.temp$df2,c(1,4,2)]
    colnames(bim1) <- c("CHR","POS","SNP")
    print(dim(bim1))
    # if (nrow(bim1) > 0){
    #     write.table(bim1,paste0(outputpath,gsub(" ","_",name[k]),'.anno'),sep="\t",quote=F,col.names=T,row.names=F)
    # }else{
    #     print('Skipped !')
    # }
}











