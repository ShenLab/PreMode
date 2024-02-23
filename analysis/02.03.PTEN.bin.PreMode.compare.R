# source('utils.R')
library(ggplot2)
# args <- commandArgs(trailingOnly = T)
# base dir for transfer learning
base.dir <- "/share/terra/Users/gz2294/PreMode.final/scripts/CHPs.v4.esm.dssp.small.StarAttn.MSA.StarPool.1dim/"
task.dic <- list("PTEN.bin"=c("score.1"="stability", "score.2"="enzyme.activity"))
# base.dirs <- strsplit(base.dirs, split = ',')[[1]]
# results <- data.frame()
# for (base.dir in base.dirs) {
genes <- c("PTEN.bin")
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
    pretrain.result <- read.csv(paste0('PreMode.inference/', genes[i], '/',
                                   '/testing.pretrain.fold.', fold, '.csv'))
    test.result$pretrain.logits <- pretrain.result$logits
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
    test.onehot.result <- read.csv(paste0('PreMode.onehot.inference/', genes[i], '/',
                                   '/testing.fold.', fold, '.csv'))
    test.pass.result <- read.csv(paste0('PreMode.pass.inference/', genes[i], '/',
                                   '/testing.fold.', fold, '.csv'))
    
    length.task <- length(task.dic[[genes[i]]])
    PreMode.auc <- plot.R2(test.result[,c("score.1", "score.2")], test.result[,paste0("logits.", 0:(length.task-1))], bin = T)
    PreMode.onehot.auc <- plot.R2(test.onehot.result[,c("score.1", "score.2")], test.onehot.result[,paste0("logits.", 0:(length.task-1))], bin = T)
    PreMode.pass.auc <- plot.R2(test.pass.result[,c("score.1", "score.2")], test.pass.result[,paste0("logits.", 0:(length.task-1))], bin = T)
    
    PreMode.pretrain.auc <- plot.R2(test.result[,c("score.1", "score.2")], -test.result[,rep("pretrain.logits", length.task)], bin = T)
    REVEL.auc <- plot.R2(test.result[,c("score.1", "score.2")], -test.result[,rep("REVEL", length.task)], bin = T)
    PrimateAI.auc <- plot.R2(test.result[,c("score.1", "score.2")], -test.result[,rep("PrimateAI", length.task)], bin = T)
    ESM.auc <- plot.R2(test.result[,c("score.1", "score.2")], test.result[,rep("esm.logits", length.task)], bin = T)
    EVE.auc <- plot.R2(test.result[,c("score.1", "score.2")], -test.result[,rep("EVE", length.task)], bin = T)
    gMVP.auc <- plot.R2(test.result[,c("score.1", "score.2")], -test.result[,rep("gMVP", length.task)], bin = T)
    to.append <- data.frame(min.val.auc = c(PreMode.pretrain.auc$R2, PreMode.auc$R2, PreMode.onehot.auc$R2, PreMode.pass.auc$R2, 
                                            REVEL.auc$R2, PrimateAI.auc$R2, ESM.auc$R2, EVE.auc$R2, gMVP.auc$R2),
                            task.name = paste0(genes[i], ":", rep(task.dic[[genes[i]]], 9)))
    to.append$model <- rep(c("PreMode.zero.shot", "PreMode", "Baseline (No ESM)", "Baseline (No Structure)", 
                             "REVEL", "PrimateAI", "ESM", 'EVE', 'gMVP'), each = length.task)
    result <- rbind(result, to.append)
  }
}
write.csv(result, 'figs/02.03.PTEN.bin.PreMode.compare.csv')

num.models <- length(unique(result$model))
p <- ggplot(result, aes(y=min.val.auc, x=task.name, col=model)) +
  geom_point(alpha=0.2) +
  stat_summary(data = result,
               aes(x=as.numeric(factor(task.name))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = min.val.auc, col=model), 
               fun.data = mean_se, geom = "errorbar", width = 0.2) +
  stat_summary(data = result, 
               aes(x=as.numeric(factor(task.name))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = min.val.auc, col=model), 
               fun.data = mean_se, geom = "point") +
  labs(x = "task", y = "min.val.auc", fill = "model") +
  theme_bw() + 
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        legend.position="bottom", 
        legend.direction="horizontal") +
  xlab('task: Molecular Level Mode of Action')
ggsave(paste0('figs/02.03.PTEN.bin.PreMode.compare.pdf'), p, height = 6, width = 2)

num.models <- 4
result.small <- result[startsWith(result$model, 'PreMode') | startsWith(result$model, 'Baseline'),]
p <- ggplot(result.small, aes(y=min.val.auc, x=task.name, col=model)) +
  geom_point(alpha=0.2) +
  stat_summary(data = result.small,
               aes(x=as.numeric(factor(task.name))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = min.val.auc, col=model), 
               fun.data = mean_se, geom = "errorbar", width = 0.2) +
  stat_summary(data = result.small, 
               aes(x=as.numeric(factor(task.name))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = min.val.auc, col=model), 
               fun.data = mean_se, geom = "point") +
  labs(x = "task", y = "min.val.auc", fill = "model") +
  theme_bw() + 
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        legend.position="bottom", 
        legend.direction="horizontal") +
  ylab('AUC') +
  xlab('task: Molecular Level Mode of Action')
ggsave(paste0('figs/02.03.PTEN.bin.PreMode.only.compare.pdf'), p, height = 5, width = 1.65)
