library(ArchR)
set.seed(1)
addArchRGenome("mm10")

save_dir <- 'C:/University/FINAL/results/HumanMouse_Deng2022/data_quality/'
if (!dir.exists(save_dir)) {
    dir.create(save_dir, recursive = TRUE)
}

inputFiles = c("D:/Data/spCASdata/HumanMouse_Deng2022/GSM5238385_ME11_50um.fragments.tsv.gz",
               "D:/Data/spCASdata/HumanMouse_Deng2022/GSM5238386_ME13_50um.fragments.tsv.gz",
               "D:/Data/spCASdata/HumanMouse_Deng2022/GSM5238387_ME13_50um_2.fragments.tsv.gz")
sampleNames = c('spatial-ATAC-seq ME11', 
                'spatial-ATAC-seq ME13(1)', 
                'spatial-ATAC-seq ME13(2)')
ArrowFiles <- createArrowFiles(
    inputFiles = inputFiles,
    sampleNames = sampleNames,
    addTileMat = TRUE,
    minTSS = 0,
    minFrags = 0,
    maxFrags = Inf,
    minFragSize = 0,
    maxFragSize = Inf,
    force = TRUE,
)

proj <- ArchRProject(
    ArrowFiles = ArrowFiles,
    outputDirectory = "ArchR_outputfiles",
    copyArrows = TRUE #This is recommended so that you maintain an unaltered copy for later usage.
)

proj@cellColData[, "log10(nFrags)"] = log10(proj@cellColData$nFrags)
proj@cellColData[, "TSSRatio"] = proj@cellColData$ReadsInTSS / (proj@cellColData$nFrags * 2)
write.csv(proj@cellColData, "ArchR_outputfiles/cellColData.csv", row.names = TRUE)

p1 <- plotGroups(
    ArchRProj = proj, 
    groupBy = "Sample", 
    colorBy = "cellColData", 
    pal = unlist(list("spatial-ATAC-seq ME11"='#3B4CBB',
                      "spatial-ATAC-seq ME13(1)"='#B30A26',
                      "spatial-ATAC-seq ME13(2)"='#3BB30A')),
    maxCells = 10000,
    name = "TSSRatio",
    plotAs = "ridges"
)
# plotPDF(p1, name = "DataQuality-TSSratioRidge.pdf", ArchRProj = proj, addDOC = FALSE, width = 5, height = 5)
ggsave(paste0(save_dir, "DataQuality-TSSratioRidge.pdf"), plot = p1, width = 5, height = 4)


p2 <- plotGroups(
    ArchRProj = proj,
    groupBy = "Sample",
    colorBy = "cellColData",
    pal = unlist(list("spatial-ATAC-seq ME11"='#3B4CBB',
                      "spatial-ATAC-seq ME13(1)"='#B30A26',
                      "spatial-ATAC-seq ME13(2)"='#3BB30A')),
    maxCells = 10000,
    name = "TSSRatio",
    plotAs = "violin",
    alpha = 0.4,
    addBoxPlot = TRUE
   )
# plotPDF(p2, name = "DataQuality-TSSratioViolin.pdf", ArchRProj = proj, addDOC = FALSE, width = 5, height = 5)
ggsave(paste0(save_dir, "DataQuality-TSSratioViolin.pdf"), plot = p2, width = 4, height = 5)


p3 <- plotGroups(
    ArchRProj = proj, 
    groupBy = "Sample", 
    colorBy = "cellColData", 
    pal = unlist(list("spatial-ATAC-seq ME11"='#3B4CBB',
                      "spatial-ATAC-seq ME13(1)"='#B30A26',
                      "spatial-ATAC-seq ME13(2)"='#3BB30A')),
    maxCells = 10000,
    name = "TSSEnrichment",
    plotAs = "ridges"
)
# plotPDF(p3, name = "DataQuality-TSSenrichmentRidge.pdf", ArchRProj = proj, addDOC = FALSE, width = 5, height = 5)
ggsave(paste0(save_dir, "DataQuality-TSSenrichmentRidge.pdf"), plot = p3, width = 5, height = 4)


p4 <- plotGroups(
    ArchRProj = proj,
    groupBy = "Sample",
    pal = unlist(list("spatial-ATAC-seq ME11"='#3B4CBB',
                      "spatial-ATAC-seq ME13(1)"='#B30A26',
                      "spatial-ATAC-seq ME13(2)"='#3BB30A')),
    maxCells = 10000,
    name = "TSSEnrichment",
    plotAs = "violin",
    alpha = 0.4,
    addBoxPlot = TRUE
   )
# plotPDF(p4, name = "DataQuality-TSSenrichmentViolin.pdf", ArchRProj = proj, addDOC = FALSE, width = 5, height = 5)
ggsave(paste0(save_dir, "DataQuality-TSSenrichmentViolin.pdf"), plot = p4, width = 4, height = 5)


p5 <- plotTSSEnrichment(ArchRProj = proj, 
                        pal = unlist(list("spatial-ATAC-seq ME11"='#3B4CBB',
                                          "spatial-ATAC-seq ME13(1)"='#B30A26',
                                          "spatial-ATAC-seq ME13(2)"='#3BB30A')),)
# plotPDF(p5, name = "DataQuality-TSSenrichmentLine.pdf", ArchRProj = proj, addDOC = FALSE, width = 5, height = 5)
ggsave(paste0(save_dir, "DataQuality-TSSenrichmentLine.pdf"), plot = p5, width = 5, height = 5)


p6 <- plotGroups(
    ArchRProj = proj, 
    groupBy = "Sample", 
    colorBy = "cellColData",
    pal = unlist(list("spatial-ATAC-seq ME11"='#3B4CBB',
                      "spatial-ATAC-seq ME13(1)"='#B30A26',
                      "spatial-ATAC-seq ME13(2)"='#3BB30A')),
    maxCells = 10000,
    name = "log10(nFrags)",
    plotAs = "ridges"
)
# plotPDF(p6, name = "DataQuality-log10nFragsRidge.pdf", ArchRProj = proj, addDOC = FALSE, width = 5, height = 5)
ggsave(paste0(save_dir, "DataQuality-log10nFragsRidge.pdf"), plot = p6, width = 5, height = 4)


p7 <- plotGroups(
    ArchRProj = proj,
    groupBy = "Sample",
    colorBy = "cellColData",
    pal = unlist(list("spatial-ATAC-seq ME11"='#3B4CBB',
                      "spatial-ATAC-seq ME13(1)"='#B30A26',
                      "spatial-ATAC-seq ME13(2)"='#3BB30A')),
    maxCells = 10000,
    name = "log10(nFrags)",
    plotAs = "violin",
    alpha = 0.4,
    addBoxPlot = TRUE
   )
# plotPDF(p7, name = "DataQuality-log10nFragsViolin.pdf", ArchRProj = proj, addDOC = FALSE, width = 5, height = 5)
ggsave(paste0(save_dir, "DataQuality-log10nFragsViolin.pdf"), plot = p7, width = 4, height = 5)




