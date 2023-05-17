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

test.result <- read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8/', log.dir, 'testing.csv'))
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
result <- plot.AUC(test.result$score, score)
aucs <- result$auc

print(paste0("AUC GoF/LoF: ", aucs))

result <- plot.AUC(test.result$score[test.result$score!="-1"], score[test.result$score!="-1"])
aucs <- result$auc
print(paste0("AUC GoF/Beni: ", aucs))

result <- plot.AUC(-test.result$score[test.result$score!="1"], score[test.result$score!="1"])
aucs <- result$auc
print(paste0("AUC LoF/Beni: ", aucs))

