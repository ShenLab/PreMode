# source('utils.R')
library(ggplot2)
# args <- commandArgs(trailingOnly = T)
# base dir for transfer learning
base.dir <- "/share/terra/Users/gz2294/PreMode.final/scripts/CHPs.v4.esm.dssp.small.StarAttn.MSA.StarPool.1dim/"
task.dic <- list("PTEN"=c("score.1"="stability", "score.2"="enzyme.activity"), 
                      "NUDT15"=c("score.1"="stability", "score.2"="enzyme.activity"), 
                      "VKORC1"=c("score.1"="enzyme.activity", "score.2"="stability"), 
                      "CCR5"=c("score.1"="stability", "score.2"="binding Ab2D7", "score.3"="binding HIV-1"), 
                      "CXCR4"=c("score.1"="stability", "score.2"="binding CXCL12", "score.3"="binding Ab12G5"))
# base.dirs <- strsplit(base.dirs, split = ',')[[1]]
# results <- data.frame()
# for (base.dir in base.dirs) {
genes <- c("PTEN", "NUDT15", "VKORC1", "CCR5", "CXCR4")
# add baseline AUC
# esm alphabets
source('~/Pipeline/AUROC.R')
source('~/Pipeline/uniprot.table.add.annotation.R')
alphabet <- c('<cls>', '<pad>', '<eos>', '<unk>',
              'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
              'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
              'X', 'B', 'U', 'Z', 'O', '.', '-',
              '<null_1>', '<mask>')
for (i in 1:length(genes)) {
  # add ESM
  for (fold in 0:4) {
    test.result <- read.csv(paste0('PreMode.inference/', genes[i], '/',
                                   '/testing.fold.', fold, '.csv'))
    pretrain.logits <- read.csv(paste0('PreMode.inference/', genes[i], '/',
                                       '/testing.pretrain.fold.', fold, '.csv'))
    test.result$pretrain.logits <- pretrain.logits$logits
    esm.logits <- read.csv(paste0('esm2.inference/', genes[i], '/testing.fold.', fold, '.logits.csv'))
    esm.logits <- esm.logits[,2:34]
    colnames(esm.logits) <- alphabet
    score <- c()
    for (k in 1:dim(esm.logits)[1]) {
      score <- c(score, esm.logits[k, test.result$alt[k]] - esm.logits[k, test.result$ref[k]])
    }
    test.result$esm.logits <- score
    test.result <- uniprot.table.add.annotation.parallel(test.result, 'EVE')
    test.result <- uniprot.table.add.annotation.parallel(test.result, 'dbnsfp')
    test.result <- uniprot.table.add.annotation.parallel(test.result, 'Itan')
    test.result$itan.logits <- test.result$itan.gof / (1-test.result$itan.beni)
    test.result <- uniprot.table.add.annotation.parallel(test.result, 'gMVP')
    test.result <- uniprot.table.add.annotation.parallel(test.result, 'conservation')
    write.csv(test.result, paste0('PreMode.inference/', genes[i], '/',
                                  '/test.fold.', fold, '.annotated.csv'))
  }
}
result <- data.frame()
for (i in 1:length(genes)) {
  for (fold in 0:4) {
    # REVEL, PrimateAI, ESM AUC
    test.result <- read.csv(paste0('PreMode.inference/', genes[i], '/',
                                   '/test.fold.', fold, '.annotated.csv'), row.names = 1)
    task.length <- length(task.dic[[genes[i]]])
    PreMode.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], test.result[,paste0("logits.", 0:(task.length-1))])
    PreMode.pretrain.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], -test.result[,rep("pretrain.logits", task.length)])
    REVEL.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], -test.result[,rep("REVEL", task.length)])
    PrimateAI.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], -test.result[,rep("PrimateAI", task.length)])
    ESM.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], test.result[,rep("esm.logits", task.length)])
    EVE.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], -test.result[,rep("EVE", task.length)])
    gMVP.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], -test.result[,rep("gMVP", task.length)])
    to.append <- data.frame(min.val.R = c(PreMode.pretrain.auc$R2, PreMode.auc$R2, REVEL.auc$R2, PrimateAI.auc$R2, ESM.auc$R2, EVE.auc$R2, gMVP.auc$R2),
                            task.name = paste0(genes[i], ":", rep(task.dic[[genes[i]]], 7)))
    to.append$model <- rep(c("PreMode.pretrain", "PreMode.transfer", "REVEL", "PrimateAI", "ESM", 'EVE', 'gMVP'), each = task.length)
    result <- rbind(result, to.append)
  }
}
write.csv(result, 'figs/02.03.MAVE.PreMode.compare.csv')

num.models <- length(unique(result$model))
p <- ggplot(result, aes(y=min.val.R, x=task.name, col=model)) +
  geom_point(alpha=0.2) +
  stat_summary(data = result,
               aes(x=as.numeric(factor(task.name))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = min.val.R, col=model), 
               fun.data = mean_se, geom = "errorbar", width = 0.2) +
  stat_summary(data = result, 
               aes(x=as.numeric(factor(task.name))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = min.val.R, col=model), 
               fun.data = mean_se, geom = "point") +
  labs(x = "task", y = "min.val.R", fill = "model") +
  theme_bw() + 
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        legend.position="bottom", 
        legend.direction="horizontal") +
  # ylim(-1, 1) +
  xlab('task: (LoF/GoF)')
ggsave(paste0('figs/02.03.MAVE.PreMode.compare.pdf'), p, height = 6, width = 12)
