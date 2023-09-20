library(caret)
pfams <- c('PF07714', 'PF00130', 'PF02196', 
           'PF07679', 'PF00047', 'PF00028', 
           'PF17756', 'PF00520', 'PF01007', 
           'IonChannel', 'IonChannel',
           'IPR000719', 'IPR001806')
log.dir <- '5genes.all.mut/CHPs.v4.esm.torchmdnet.small.TriAttn.StarPool.1dim/'
for (pfam in pfams) {
  training <- read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8/',
                       pfam, '.chps.even.uniprotID/training.csv'))
  testing <- read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8/', 
                       pfam, '.chps.even.uniprotID/testing.csv'))
  training <- training[,colnames(training) %in% colnames(testing)]
  for (col in colnames(training)) {
    if (typeof(training[,col]) != typeof(testing[,col])) {
      training[,col] <- as.character(training[,col])
      testing[,col] <- as.character(testing[,col])
    }
  }
  all <- dplyr::bind_rows(training, testing)
  write.csv(all, file = paste0('5genes.all.mut/', pfam, '.meta.csv'))
}

for (pfam in pfams) {
  logits.mat <- NULL
  for (fold in 0:4) {
    logits.file <- read.csv(paste0(log.dir,
                          pfam, '.fold.', fold, '.csv'))
    logits.mat <- cbind(logits.mat, logits.file$logits)
  }
  logits.mat <- logits.mat[logits.file$score %in% c(-1, 1), ]
  colnames(logits.mat) <- paste0('model.', 0:4)
  train.score <- logits.file$score[logits.file$score %in% c(-1, 1)] * 0.5 + 0.5
  set.seed(0)
  meta_model_fit <- train(logits.mat, 
                          as.factor(train.score),
                          method='ada')
  saveRDS(meta_model_fit, file = paste0(log.dir, pfam, '.meta.RDS'))
}

