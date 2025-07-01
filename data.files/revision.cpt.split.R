icc.all <- read.csv(('../analysis/figs/ALL.csv'), row.names = 1)
icc.all$score.label <- NULL
# do random split in all genes, keep it the same as seed 0 1 2 3 4 results 
nine.genes <- read.csv('../scripts/gene.itan.txt', header = F)$V1
nine.genes <- gsub('.clean.itan.split|.itan.split', '', nine.genes)
icc.no.nine.genes <- icc.all[!icc.all$uniprotID %in% nine.genes,]
# get a table of all genes that have sufficient data
icc.noitan <- icc.all[!grepl('Itan', icc.all$data_source),]
icc.itan <- icc.all[grepl('Itan', icc.all$data_source),]
# check the number of genes
genes <- unique(icc.noitan$uniprotID)
good.genes <- c()
for (uid in genes) {
  icc.noitan.gene <- icc.noitan[icc.noitan$uniprotID == uid,]
  # check number of gof and lof
  n.gof <- sum(icc.noitan.gene$score == 1)
  n.lof <- sum(icc.noitan.gene$score == -1)
  # add to list if n.gof >= 5 and n.lof >= 5 and not in icc.itan
  if (n.gof >= 4 & n.lof >= 4 & !uid %in% icc.itan$uniprotID) {
    good.genes <- c(good.genes, uid)
  }
}
# take the good genes, make the split
good.genes.train <- icc.all[!icc.all$uniprotID %in% good.genes,]
good.genes.test <- icc.all[icc.all$uniprotID %in% good.genes,]
for (g in good.genes) {
  itan.all.g <- icc.all[icc.all$uniprotID == g,]
  write.csv(itan.all.g, paste0('ICC.seed.0/ALL.itan.nogeneoverlap/', g, '.csv'))
}
write.csv(good.genes.train, 'ICC.seed.0/ALL.itan.nogeneoverlap/training.csv')
write.csv(good.genes.test, 'ICC.seed.0/ALL.itan.nogeneoverlap/testing.csv')

good.genes <- c()
for (uid in genes) {
  icc.gene <- icc.all[icc.all$uniprotID == uid,]
  # check number of gof and lof
  n.gof <- sum(icc.gene$score == 1)
  n.lof <- sum(icc.gene$score == -1)
  # add to list if n.gof >= 5 and n.lof >= 5 and not in icc.itan
  if (n.gof >= 10 & n.lof >= 10) {
    good.genes <- c(good.genes, uid)
  }
}
# take the good genes, make the split
good.genes.train <- icc.all[!icc.all$uniprotID %in% good.genes,]
good.genes.test <- icc.all[icc.all$uniprotID %in% good.genes,]
for (g in good.genes) {
  itan.all.g <- icc.all[icc.all$uniprotID == g,]
  write.csv(itan.all.g, paste0('ICC.seed.0/ALL.nogeneoverlap/', g, '.csv'))
}
write.csv(good.genes.train, 'ICC.seed.0/ALL.nogeneoverlap/training.csv')
write.csv(good.genes.test, 'ICC.seed.0/ALL.nogeneoverlap/testing.csv')


good.genes <- c()
for (uid in genes) {
  icc.gene <- icc.all[icc.all$uniprotID == uid,]
  # check number of gof and lof
  n.gof <- sum(icc.gene$score == 1)
  n.lof <- sum(icc.gene$score == -1)
  # add to list if n.gof >= 5 and n.lof >= 5 and not in icc.itan
  if (n.gof + n.lof >= 10) {
    good.genes <- c(good.genes, uid)
  }
}
# take the good genes, make the split
good.genes.train <- icc.all[icc.all$uniprotID %in% good.genes & !icc.all$uniprotID %in% nine.genes,]
good.genes.test <- icc.all[icc.all$uniprotID %in% nine.genes,]
write.csv(good.genes.train, 'ICC.seed.0/ALL.nogeneoverlap.large.gene/training.csv')
write.csv(good.genes.test, 'ICC.seed.0/ALL.nogeneoverlap.large.gene/testing.csv')



