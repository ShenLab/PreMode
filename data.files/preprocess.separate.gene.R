source('../analysis/prepare.biochem.R')
ALL <- read.csv('../analysis/figs/ALL.csv', row.names = 1)
ALL$score.label <- NULL
gof.lof.df <- data.frame(uniprotIDs=as.character(unique(unlist(strsplit(ALL$uniprotID, split = ";")))), gof=0, lof=0)
for (i in 1:dim(gof.lof.df)[1]) {
  gene <- ALL[grep(gof.lof.df$uniprotIDs[i], ALL$uniprotID),]
  gof.lof.df$gof[i] <- sum(gene$score==1)
  gof.lof.df$lof[i] <- sum(gene$score==-1)
}

good.uniprotIDs <- gof.lof.df[gof.lof.df$gof >= 15 & gof.lof.df$lof >= 15, ]

# split by random
ratio <- 0.8
good.uniprotIDs$gof.training <- NA
good.uniprotIDs$gof.testing <- NA
good.uniprotIDs$lof.training <- NA
good.uniprotIDs$lof.testing <- NA
for (seed in 0:4) {
  split.dir <- paste0('ICC.seed.', seed, '/')
  dir.create(split.dir)
  for (i in 1:dim(good.uniprotIDs)[1]) {
    gene <- ALL[grep(good.uniprotIDs$uniprotIDs[i], ALL$uniprotID),]
    gene.gof <- gene[gene$score == 1,]
    gene.lof <- gene[gene$score == -1,]
    set.seed(seed)
    # pick ratio% of variants as training
    if (floor(dim(gene.gof)[1] * ratio) > 0 & 
        floor(dim(gene.lof)[1] * ratio) > 0) {
      gene.gof.training <- sample(dim(gene.gof)[1], floor(dim(gene.gof)[1] * ratio))
      gene.lof.training <- sample(dim(gene.lof)[1], floor(dim(gene.lof)[1] * ratio))
      gene.training <- rbind(gene.gof[gene.gof.training,], gene.lof[gene.lof.training,])
      gene.testing <- rbind(gene.gof[-gene.gof.training,], gene.lof[-gene.lof.training,])
      
      gene.training <- gene.training[sample(dim(gene.training)[1]),]
      gene.testing <- gene.testing[sample(dim(gene.testing)[1]),]
      
      dir.create(paste0(split.dir, good.uniprotIDs$uniprotIDs[i]))
      
      uid <- good.uniprotIDs$uniprotIDs[i]
      if (uid == 'Q14524') {
        uid <- 'Q14524.clean'
      }
      
      if (!file.exists(paste0(split.dir, uid, "/training.csv")) | uid=='Q99250') {
        print(uid)
        write.csv(gene.training, paste0(split.dir, uid, "/training.csv"))
      }
      if (!file.exists(paste0(split.dir, uid, "/testing.csv")) | uid=='Q99250') {
        print(uid)
        write.csv(gene.testing, paste0(split.dir, uid, "/testing.csv"))
      }
      
      good.uniprotIDs$gof.training[i] <- sum(tmp.training$score==1)
      good.uniprotIDs$lof.training[i] <- sum(tmp.training$score==-1)
      
      good.uniprotIDs$gof.testing[i] <- sum(tmp.testing$score==1)
      good.uniprotIDs$lof.testing[i] <- sum(tmp.testing$score==-1)
      
    }
  }
}
write.csv(good.uniprotIDs, file = "sup.data.1.csv")