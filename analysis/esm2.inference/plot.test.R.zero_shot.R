library(ggplot2)
# setwd('/share/terra/Users/gz2294/RESCVE.final/')
args <- commandArgs(trailingOnly = T)
log.dir <- args[1]
source('/share/terra/Users/gz2294/Pipeline/AUROC.R')
alphabet <- c('<cls>', '<pad>', '<eos>', '<unk>',
              'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
              'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
              'X', 'B', 'U', 'Z', 'O', '.', '-',
              '<null_1>', '<mask>')

test.result <- read.csv(paste0('~/Data/DMS/MAVEDB/', log.dir, 'testing.csv'))
logits <- read.csv(paste0(log.dir, 'testing.logits.csv'))
logits <- logits[,2:34]
colnames(logits) <- alphabet
score <- c()
rank.ref <- c()
rank.alt <- c()
for (k in 1:dim(logits)[1]) {
  score <- c(score, logits[k, test.result$alt[k]] - logits[k, test.result$ref[k]])
  rank.ref <- c(rank.ref, rank(-logits[k,])[test.result$ref[k]])
  rank.alt <- c(rank.alt, rank(-logits[k,])[test.result$alt[k]])
}
scores <- colnames(test.result)[startsWith(colnames(test.result), 'score.')]
for (score.i in scores) {
  result <- plot.R2(test.result[,score.i], score)
  print(paste0("R for ", score.i, ": ", round(result$R2, 3)))
}


