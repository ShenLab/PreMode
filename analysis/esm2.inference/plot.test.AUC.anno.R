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

test.result <- rbind(read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8/', log.dir, 'testing.csv'), row.names = 1),
                     read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8/', log.dir, 'training.csv'), row.names = 1))

source('~/Pipeline/uniprot.table.add.annotation.R')
test.result <- uniprot.table.add.annotation.parallel(test.result, 'EVE')
test.result <- uniprot.table.add.annotation.parallel(test.result, 'dbnsfp')
test.result <- uniprot.table.add.annotation.parallel(test.result, 'gMVP')

aucs <- c()
result <- plot.AUC(test.result$score, test.result$REVEL)
aucs <- c(aucs, result$auc)
result <- plot.AUC(test.result$score, test.result$PrimateAI)
aucs <- c(aucs, result$auc)
result <- plot.AUC(test.result$score, test.result$EVE)
aucs <- c(aucs, result$auc)

print(paste0(c("REVEL: ", "PrimateAI: ", "EVE: "), round(aucs, 3)))
