# visualize with dssp secondary structure 
library(ggplot2)
library(patchwork)
library(bio3d)
genes <- c("P60484")
gene <- genes[1]
# plot original scores
original <- rbind(read.csv('~/Data/DMS/MAVEDB/PTEN.bin/train.seed.0.csv'),
                  read.csv('~/Data/DMS/MAVEDB/PTEN.bin/test.seed.0.csv'))
# 
# af2.seqs <- read.csv('~/Data/Protein/alphafold2_v4/swissprot_and_human.full.seq.csv', row.names = 1)
# aa.dict <- c('L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
#              'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C')
log.dir <- '5genes.all.mut/CHPs.v4.esm.torchmdnet.small.TriAttn.StarPool.1dim/'
folds <- c(0:4)
gene.result <- read.csv(paste0(log.dir, gene, '.pretrain.csv'), row.names = 1)
pretrain.result <- gene.result
pretrain.training.file <- read.csv(paste0('~/Data/DMS/ClinVar.HGMD.PrimateAI.syn/all.clinvar.4.splits/training.csv'))[,c("HGNC", "uniprotID", "pos.orig", "ref", "alt", "score", "data_source")]
pretrain.training.file$score[pretrain.training.file$score!=0] <- 1
pretrain.training.file <- pretrain.training.file[pretrain.training.file$uniprotID == gene,]
# assemble.logits <- 0
to.plot <- data.frame()
for (fold in folds) {
  # gene.result <- read.csv(paste0(log.dir, gene, '.fold.', fold, '.csv'), row.names = 1)
  training.file <- read.csv(paste0('~/Data/DMS/MAVEDB/PTEN.bin/train.seed.', fold, '.csv'))[,c("HGNC", "pos.orig", "ref", "alt", "score.1", "score.2")]
  testing.file <- read.csv(paste0('PreMode.inference/PTEN.stab/test.fold.', fold, '.csv'))[,c("HGNC", "pos.orig", "ref", "alt", "score.1", "logits")]
  testing.file.2 <- read.csv(paste0('PreMode.inference/PTEN.bin/test.fold.', fold, '.annotated.csv'))[,c("HGNC", "pos.orig", "ref", "alt", "score.1", "score.2", "logits.0", "logits.1")]
  testing.file$aaChg <- paste0('p.', testing.file$ref, testing.file$pos.orig, testing.file$alt)
  
  # test the zero shot performance of logits.0, logits.1 to predict logits.2
  source('~/Pipeline/AUROC.R')
  testing.file$score.2 <- testing.file.2$score.2
  testing.file$logits.2 <- testing.file.2$logits.1
  testing.file$logits.1 <- testing.file$logits
  testing.file$logits.0 <- pretrain.result$logits[match(testing.file$aaChg, pretrain.result$aaChg)]
  # testing.file$zero.shot <- (testing.file$logits.0 * dim(pretrain.training.file)[1] +
  #                              testing.file$logits.1 * dim(original)[1]) / (dim(pretrain.training.file)[1] + dim(original)[1])
  testing.file$zero.shot <- (testing.file$logits.0 + testing.file$logits.1) / 2
  a1 <- plot.AUC(testing.file$score.2[!is.na(testing.file$score.2)],
                 testing.file$zero.shot[!is.na(testing.file$score.2)])
  # this means train on stability assay can improve prediction in enzyme activity assay
  a2 <- plot.AUC(testing.file$score.2[!is.na(testing.file$score.2)],
                 testing.file$logits.1[!is.na(testing.file$score.2)])
  a3 <- plot.AUC(testing.file$score.2[!is.na(testing.file$score.2)],
                 testing.file$logits.0[!is.na(testing.file$score.2)])
  a4 <- plot.AUC(testing.file$score.2[!is.na(testing.file$score.2)],
                 testing.file$logits.2[!is.na(testing.file$score.2)])
  # print(paste0('together auc: ', a1$auc))
  # print(paste0('stability auc: ', a2$auc))
  # print(paste0('pretrain auc: ', a3$auc))
  # print(paste0('transfer auc: ', a4$auc))
  to.plot <- rbind(to.plot,
                   data.frame(auc=c(a1$auc, a2$auc, a3$auc, a4$auc),
                              model=c('pretrain + stability', 'stability only', 'pretrain only', 'transfer learning'),
                              seed=rep(fold, 4)))
}
p <- ggplot(to.plot, aes(x=model, y=auc, col=model)) + 
  geom_point(alpha=0.2) +
  stat_summary(data = to.plot,
               aes(x=as.numeric(factor(model)),
                   y = auc, col=model), 
               fun.data = mean_se, geom = "errorbar", width = 0.2) +
  stat_summary(data = to.plot, 
               aes(x=as.numeric(factor(model)),
                   y = auc, col=model), 
               fun.data = mean_se, geom = "point") +
  theme_bw() + ggtitle('PreMode on PTEN enzyme activity') + ggeasy::easy_center_title()
ggsave('figs/08.analysis.PTEN.stability.zeroshot.enzy.pdf', p, height = 4, width = 6)
# 
# original$aaChg <- paste0('p.', original$ref, original$pos.orig, original$alt)
# gene.result$aaChg <- paste0('p.', gene.result$ref, gene.result$pos.orig, gene.result$alt)
# gene.result$score.1 <- original$score.1[match(gene.result$aaChg, original$aaChg)]
# gene.result$score.2 <- original$score.2[match(gene.result$aaChg, original$aaChg)]
# source('~/Pipeline/AUROC.R')
# th.1 <- plot.AUC(gene.result$score.1[!is.na(gene.result$score.1)],
#                  gene.result$logits.0[!is.na(gene.result$score.1)])
# th.2 <- plot.AUC(gene.result$score.1[!is.na(gene.result$score.1)],
#                  gene.result$logits.1[!is.na(gene.result$score.1)])
# for (i in 1:dim(gene.result)[1]) {
#   if (gene.result$logits.0[i] >= th.1$cutoff & gene.result$logits.1[i] >= th.2$cutoff) {
#     gene.result$zero.shot[i] <- max(gene.result$logits.0[i], gene.result$logits.1[i])
#   } else if (gene.result$logits.0[i] >= th.1$cutoff & gene.result$logits.1[i] < th.2$cutoff) {
#     gene.result$zero.shot[i] <- gene.result$logits.0[i]
#   } else if (gene.result$logits.0[i] < th.1$cutoff & gene.result$logits.1[i] >= th.2$cutoff) {
#     gene.result$zero.shot[i] <- gene.result$logits.1[i]
#   } else {
#     gene.result$zero.shot[i] <- min(gene.result$logits.0[i], gene.result$logits.1[i])
#   }
# }
# 
# plot.AUC(gene.result$score.2[!is.na(gene.result$score.2)],
#          gene.result$zero.shot[!is.na(gene.result$score.2)])



