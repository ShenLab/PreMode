# prepare reference
reference <- read.delim('/share/terra/Users/gz2294/Data/mutRate/hg19_mutrate_3mer.txt', na.strings = c("NA", "."))
blacklist <- read.delim("/share/terra/Users/gz2294/Data/mutRate/GENCODEV19_blacklist.txt", header = F)
reference = reference[!reference$GeneID %in% blacklist$V1,]
gnomAD <- read.table("/share/terra/Users/gz2294/Data/gnomad.v2.1.1.lof_metrics.by_gene.txt", sep = "\t", na.strings = "NA", header = T)
reference$pLI <- gnomAD$pLI[match(reference$GeneID, gnomAD$gene_id)]

cohort.files <- c("/share/terra/Users/gz2294/Data/DDD/DDD_DNVs_Anno_wSpliceAI.b37.txt",
                  "/share/terra/Users/gz2294/Data/SPARK_20210508/AllDNVs.no_duplicates_no_twins.hg38.ensembl75_canonical_only.anno.txt",
                  "/share/terra/Users/gz2294/Data/PCGC.anno.hg19/PCGC_DNVs_Anno_wSpliceAI.b37.txt")
cohort.names <- c("NDD", "ASD", "CHD")
cohort.sizes <- c(31058, 30003, 3966)
for (i in 1:3) {
  DNVs <- read.delim(cohort.files[i], na.strings = ".") #DNVs
  DNVs <- DNVs[!is.na(DNVs$GeneID),]
  # annotate variants
  source('~/Pipeline/annotate.var.class.R')
  DNVs <- annotate.all.in.one(DNVs, reference)
  if ("CADD13_phred" %in% colnames(DNVs)) {
    cadd_column <- list("CADD13_phred"=20)
  } else {
    cadd_column <- list("CADD13"=20)
  }
  DNVs <- annotate.Dmis.advanced(DNVs, cadd_column, Dmis_name = "Dmis", cover = T)
  # run extTADA
  geneset <- unique(reference$GeneID)
  # extTADA scripts
  file_list <- list.files(path = "/share/terra/Users/gz2294/ld1/extTADA/model/", pattern = "*.R", full.names = TRUE)
  for (j in 1:length(file_list)) {
    source(file_list[j], local=T)  #read all R codes in the "model" folder
  }
  result <- gene_set_exttada(geneset, DNVs, cohort.sizes[i], reference, Dmis_def = "CADD20")
  saveRDS(result, file = paste0("cohort/", cohort.names[i], ".result.CADD20.RDS"))
}

