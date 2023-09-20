cohorts <- c("CHD", "ASD", "NDD")
good.pfams <- read.csv('../scripts/pfams.txt', header = F)$V1
good.pfams <- good.pfams[startsWith(good.pfams, "PF")]
for (i in 1:length(good.pfams)) {
  good.pfams[i] <- strsplit(good.pfams[i], '\\.')[[1]][1]
}
good.pfams <- unique(good.pfams)
good.pfams <- good.pfams[good.pfams != "PF_IPR000719"]
good.pfams.dnvs <- list()
cohort.files <- c("NDD"="/share/terra/Users/gz2294/Data/DDD/DDD_DNVs_Anno_wSpliceAI.b37.txt",
                  "ASD"="/share/terra/Users/gz2294/Data/SPARK_20210508/AllDNVs.no_duplicates_no_twins.hg38.ensembl75_canonical_only.anno.txt",
                  "CHD"="/share/terra/Users/gz2294/Data/PCGC.anno.hg19/PCGC_DNVs_Anno_wSpliceAI.b37.txt")
dir.create('cohort.cases/')
for (c in cohorts) {
  print(paste0("Begin ", c))
  tmp <- readRDS(paste0('cohort/', c, ".result.CADD20.RDS"))
  tmp <- tmp$dataFDR.full.posterior
  cohort.cases <- read.delim(cohort.files[c], na.strings = '.')
  if (!'CADD13_phred' %in% colnames(cohort.cases)) {
    cohort.cases$CADD13_phred = cohort.cases$CADD13
  }
  print(paste0('origin dnv variants: ', dim(cohort.cases)[1]))
  source('~/Pipeline/dnv.table.to.uniprot.R')
  # be careful, we split different transcripts
  cohort.cases <- dnv.table.seperate.multiple.effects(cohort.cases, split.chr = ',', carno.only = F)
  cohort.cases <- dnv.table.seperate.multiple.effects(cohort.cases, split.chr = ';', carno.only = F)
  
  ensembl <- read.csv('~/Data/Protein/ensembl.canonical.csv')
  cohort.cases$uniprotID <- ensembl$UniProtKB.Swiss.Prot.ID[match(cohort.cases$GeneID, ensembl$Gene.stable.ID)]
  
  ensembl.uniprot <- read.delim('~/Data/Protein/uniprot.ID/swissprot.ID.mapping.tsv')
  
  to.fix <- which(cohort.cases$uniprotID == "" | is.na(cohort.cases$uniprotID))
  for (i in 1:length(to.fix)) {
    cohort.cases$uniprotID[to.fix[i]] <- ensembl.uniprot$Entry[
      grep(paste0("\\b", cohort.cases$HGNC[to.fix[i]], "\\b"), ensembl.uniprot$Gene.Names)[1]
    ][1]
  }
  
  cohort.cases <- cohort.cases[!is.na(cohort.cases$uniprotID),]
  print(paste0('left dnv variants: ', dim(cohort.cases)[1]))
  cohort.cases <- uniprot.table.add.annotation.parallel(cohort.cases, annotation.to.add = 'pfam')
  # save cohort cases
  cohort.cases <- cohort.cases[cohort.cases$GeneEff == "missense",]
  cohort.cases$extTADA.PP <- tmp$PP[match(cohort.cases$GeneID, tmp$Gene)]
  cohort.cases$BF.mis <- tmp$BF[match(cohort.cases$GeneID, tmp$Gene), 2]
  cohort.cases$BF.LGD <- tmp$BF[match(cohort.cases$GeneID, tmp$Gene), 1]
  cohort.cases$cohort <- c
  write.csv(cohort.cases, paste0('cohort/', c, '.missense.csv'))
  good.pfams.dnvs[[paste0(c, '.missense')]] <- cohort.cases
  # check if any in our good pfams
  for (pfam in (good.pfams)) {
    good.pfams.cases <- cohort.cases[grep(pfam, cohort.cases$pfam),]
    # good.pfams.cases$uniprotID <- tmp$uniprotID[match(good.pfams.cases$GeneID, tmp$Gene)]
    if (dim(good.pfams.cases)[1] > 0) {
      if (!is.null(dim(good.pfams.dnvs[[pfam]])[1])) {
        for (col in colnames(good.pfams.dnvs[[pfam]])) {
          if (col %in% colnames(good.pfams.cases)) {
            if (typeof(good.pfams.cases[1,col]) != typeof(good.pfams.dnvs[[pfam]][1,col])) {
              good.pfams.cases[,col] <- as.character(good.pfams.cases[,col])
              good.pfams.dnvs[[pfam]][,col] <- as.character(good.pfams.dnvs[[pfam]][,col])
            }
          }
        }
      }
      good.pfams.dnvs[[pfam]] <- dplyr::bind_rows(good.pfams.dnvs[[pfam]], good.pfams.cases)
    } 
  }
  # add IonChannel
  IonChannel <- rbind(read.csv('~/Data/DMS/Ion_Channel/all.af2update.csv'))
  IonChannel.genes <- unique(IonChannel$HGNC[!is.na(IonChannel$HGNC)])
  good.pfams.cases <- cohort.cases[cohort.cases$HGNC %in% IonChannel.genes,]
  # print(sum(tmp$dn_mis))
  good.pfams.cases <- good.pfams.cases[good.pfams.cases$GeneEff == "missense",]
  good.pfams.cases$extTADA.PP <- tmp$PP[match(good.pfams.cases$GeneID, tmp$Gene)]
  good.pfams.cases$BF.mis <- tmp$BF[match(good.pfams.cases$GeneID, tmp$Gene), 2]
  good.pfams.cases$BF.LGD <- tmp$BF[match(good.pfams.cases$GeneID, tmp$Gene), 1]
  if (dim(good.pfams.cases)[1] > 0) {
    good.pfams.cases$cohort <- c
    if (!is.null(dim(good.pfams.dnvs[["IonChannel"]])[1])) {
      for (col in colnames(good.pfams.dnvs[["IonChannel"]])) {
        if (col %in% colnames(good.pfams.cases)) {
          if (typeof(good.pfams.cases[1,col]) != typeof(good.pfams.dnvs[["IonChannel"]][1,col])) {
            good.pfams.cases[,col] <- as.character(good.pfams.cases[,col])
            good.pfams.dnvs[["IonChannel"]][,col] <- as.character(good.pfams.dnvs[["IonChannel"]][,col])
          }
        }
      }
    }
    good.pfams.dnvs[["IonChannel"]] <- dplyr::bind_rows(good.pfams.dnvs[["IonChannel"]], good.pfams.cases)
  } 
}
saveRDS(good.pfams.dnvs, "good.pfams.dnvs.RDS")

good.pfams.dnvs <- readRDS('good.pfams.dnvs.RDS')
for (pfam in names(good.pfams.dnvs)) {
  print(paste0("Begin ", pfam))
  pfam.data <- good.pfams.dnvs[[pfam]]
  pfam.data$ENST <- as.character(as.data.frame(strsplit(pfam.data$TransIDs, '\\.'))[1,])
  pfam.data <- pfam.data[!is.na(pfam.data$AAChg) & !is.na(pfam.data$uniprotID),]
  # we don't know the score
  pfam.data$score <- NA
  # pfam.data <- dnv.table.seperate.multiple.effects(pfam.data, split.chr = ';', carno.only = F)
  pfam.data <- dnv.table.to.uniprot.by.af2.uniprotID.parallel(
    pfam.data, 'VarID', 'score', 'uniprotID', 'AAChg')
  print(paste0("dropped ", 
               dim(pfam.data$result)[1] - dim(pfam.data$result.noNA)[1], 
               " out of ", 
               dim(pfam.data$result)[1],
               " variants"))
  print(paste0("Finished ", pfam))
  pfam.data <- pfam.data$result.noNA
  pfam.data <- pfam.data[!is.na(pfam.data$sequence) & pfam.data$sequence != "NA",]
  pfam.data <- pfam.data[!is.na(pfam.data$alt) & pfam.data$alt != "NA",]
  write.csv(pfam.data, file = paste0('cohort/', pfam, '.csv'))
}

# for (pfam in c('IonChannel', good.pfams)) {
#   pfam.data <- read.csv(paste0('cohort/', pfam, '.csv'), row.names = 1)
#   pfam.data$CADD13_phred[pfam.data$cohort=='ASD'] <- pfam.data$CADD13[pfam.data$cohort=='ASD']
#   write.csv(pfam.data, file = paste0('cohort/', pfam, '.csv'))
# }
