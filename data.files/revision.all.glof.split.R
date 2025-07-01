icc.all <- read.csv(gzfile('../analysis/figs/ALL.csv.gz'), row.names = 1)
# do random split in all genes, keep it the same as seed 0 1 2 3 4 results 
nine.genes <- read.csv('../scripts/gene.txt', header = F)$V1
icc.no.nine.genes <- icc.all[!icc.all$uniprotID %in% gsub('.clean', '', nine.genes),]
my.bind.rows <- function (df1, df2) {
  for (c in colnames(df1)[colnames(df1) %in% colnames(df2)]) {
    if(typeof(df1[,c])!=typeof(df2[,c])) {
      df1[,c] <- as.character(df1[,c])
      df2[,c] <- as.character(df2[,c])
    }
  }
  result <- dplyr::bind_rows(df1, df2)
  result
}
for (seed in 0:4) {
  # for other genes in icc, split by 80-20 for each gene randomly
  training <- c()
  testing <- c()
  for (uid in unique(icc.no.nine.genes$uniprotID)) {
    uid.gof <- which(icc.no.nine.genes$uniprotID==uid & icc.no.nine.genes$score==1)
    uid.lof <- which(icc.no.nine.genes$uniprotID==uid & icc.no.nine.genes$score==-1)
    set.seed(seed)
    if (length(uid.gof) > 0) {
      uid.gof.trn <- sample(uid.gof, floor(length(uid.gof)*0.8))
      uid.gof.tst <- uid.gof[!uid.gof %in% uid.gof.trn]
    } else {
      uid.gof.trn <- c()
      uid.gof.tst <- c()
    }
    if (length(uid.lof) > 0) {
      uid.lof.trn <- sample(uid.lof, length(uid.lof)*0.8)
      uid.lof.tst <- uid.lof[!uid.lof %in% uid.lof.trn]
    } else {
      uid.lof.trn <- c()
      uid.lof.tst <- c()
    }
    training <- c(training, uid.gof.trn, uid.lof.trn)
    testing <- c(testing, uid.gof.tst, uid.lof.tst)
  }
  # aggregate results
  icc.train <- icc.no.nine.genes[training,]
  icc.test <- icc.no.nine.genes[testing,]
  for (uid in nine.genes) {
    icc.train <- my.bind.rows(icc.train, read.csv(paste0('ICC.seed.', seed, '/', uid, '/training.csv'), row.names = 1))
    icc.test <- my.bind.rows(icc.test, read.csv(paste0('ICC.seed.', seed, '/', uid, '/testing.csv'), row.names = 1))
  }
  set.seed(seed)
  # shuffle
  icc.train <- icc.train[sample(dim(icc.train)[1]),]
  icc.test <- icc.test[sample(dim(icc.test)[1]),]
  # write
  dir.create(paste0('ICC.seed.', seed, '/ALL/'), showWarnings = F)
  # drop score.labels column
  icc.train$score.label <- NULL
  icc.test$score.label <- NULL
  write.csv(icc.train, paste0('ICC.seed.', seed, '/ALL/training.csv'))
  write.csv(icc.test, paste0('ICC.seed.', seed, '/ALL/testing.csv'))
}
