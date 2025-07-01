source('/share/vault/Users/gz2294/Pipeline/uniprot.table.add.annotation.R')
ALL <- read.csv('ALL.csv', row.names = 1)

ALL <- uniprot.table.add.annotation.parallel(ALL, "InterPro")
# remove glazer
ALL <- ALL[ALL$data_source != "glazer",]

good.uniprotIDs <- data.frame(
  uniprotID=c("P15056", "P21802", "P07949", 
              "P04637", "Q09428", "O00555",
              "Q14654", "Q99250", "Q14524"))
good.uniprotIDs.df <- data.frame()
frac <- 0.8
for (seed in 0:4) {
  split.dir <- paste0('ICC.seed.', seed, '/')
  dir.create(split.dir)
  for (i in 1:dim(good.uniprotIDs)[1]) {
    gene.itan <- ALL[ALL$uniprotID==good.uniprotIDs$uniprotID[i],]
    # prepare some types
    pfams <- unique(unlist(strsplit(gene.itan$InterPro,";")))
    # pick ratio% of variants as training
    for (pfam in pfams) {
      set.seed(seed)
      GO.itan <- ALL[grep(pfam, ALL$InterPro),]
      GO.itan.training <- GO.itan[!GO.itan$uniprotID %in% gene.itan$uniprotID,]
      if (dim(GO.itan.training)[1] > 0) {
        GO.itan.training$split <- 'train'
      }
      # select only the data in domain as train/test
      gene.itan.domain <- gene.itan[grep(pfam, gene.itan$InterPro),]
      # random select testing and validation
      # select equal amount of gof and lof for testing
      gof.training <- sample(which(gene.itan.domain$score==1), size = floor(sum(gene.itan.domain$score==1)*frac))
      lof.training <- sample(which(gene.itan.domain$score==-1), size = floor(sum(gene.itan.domain$score==-1)*frac))
      # select equal amount of gof and lof for validation
      if (length(gof.training) > 0 & length(lof.training) > 0) {
        gene.itan.domain.training <- gene.itan.domain[c(gof.training, lof.training),]
        gene.itan.domain.training$split <- 'train'
        gof.val <- sample(which(gene.itan.domain.training$score==1), size = floor(sum(gene.itan.domain$score==1)*(1-frac)))
        lof.val <- sample(which(gene.itan.domain.training$score==-1), size = floor(sum(gene.itan.domain$score==-1)*(1-frac)))
        gene.itan.domain.training$split[c(gof.val, lof.val)] <- 'val'
        
        GO.itan.testing <- gene.itan.domain[-c(gof.training, lof.training),]
        if (dim(GO.itan.testing)[1] > 0) {
          # first save the gene itself
          dir.create(paste0(split.dir, good.uniprotIDs$uniprotID[i], '.', pfam, ".", "self"))
          write.csv(gene.itan.domain.training[sample(dim(gene.itan.domain.training)[1]),], paste0(split.dir, good.uniprotIDs$uniprotID[i], ".", pfam, ".self", "/training.csv"))
          write.csv(GO.itan.testing, paste0(split.dir, good.uniprotIDs$uniprotID[i], ".", pfam, ".self", "/testing.csv"))
          good.uniprotIDs.df <- rbind(good.uniprotIDs.df,
                                      data.frame(dataID=paste0(good.uniprotIDs$uniprotID[i], ".", pfam, ".self"),
                                                 uniprotID=paste0(good.uniprotIDs$uniprotID[i]),
                                                 pfam=pfam,
                                                 gof.training=sum(gene.itan.domain.training$score==1),
                                                 lof.training=sum(gene.itan.domain.training$score==-1),
                                                 gof.testing=sum(GO.itan.testing$score==1),
                                                 lof.testing=sum(GO.itan.testing$score==-1),
                                                 seed=seed))
          # next concatenate and shuffle
          GO.itan.training <- dplyr::bind_rows(gene.itan.domain.training, GO.itan.training)
          GO.itan.training <- GO.itan.training[sample(dim(GO.itan.training)[1]),]
          GO.itan.testing <- GO.itan.testing[sample(dim(GO.itan.testing)[1]),]
          # save the training files
          dir.create(paste0(split.dir, good.uniprotIDs$uniprotID[i], '.', pfam, ".", pfam))
          write.csv(GO.itan.training, paste0(split.dir, good.uniprotIDs$uniprotID[i], ".", pfam, ".", pfam, "/training.csv"))
          write.csv(GO.itan.testing, paste0(split.dir, good.uniprotIDs$uniprotID[i], ".", pfam, ".", pfam, "/testing.csv"))
          
          good.uniprotIDs.df <- rbind(good.uniprotIDs.df,
                                      data.frame(dataID=paste0(good.uniprotIDs$uniprotID[i], ".", pfam, ".", pfam),
                                                 uniprotID=paste0(good.uniprotIDs$uniprotID[i]),
                                                 pfam=pfam,
                                                 gof.training=sum(GO.itan.training$score==1),
                                                 lof.training=sum(GO.itan.training$score==-1),
                                                 gof.testing=sum(GO.itan.testing$score==1),
                                                 lof.testing=sum(GO.itan.testing$score==-1),
                                                 seed=seed))
        } 
      }
    }
  }
}
entry.list <- read.delim('/share/vault/Users/gz2294/Data/Protein/InterPro/entry.list')
good.uniprotIDs.df$name <- entry.list$ENTRY_NAME[match(good.uniprotIDs.df$pfam, entry.list$ENTRY_AC)]
write.csv(good.uniprotIDs.df, file = "good.uniprotIDs.InterPros.csv")
