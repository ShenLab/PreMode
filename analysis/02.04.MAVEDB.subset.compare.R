# source('utils.R')
library(ggplot2)
# args <- commandArgs(trailingOnly = T)
# base dir for transfer learning
base.dir <- "/share/terra/Users/gz2294/PreMode.final/scripts/CHPs.v4.esm.dssp.small.StarAttn.MSA.StarPool.1dim/"
task.dic <- list("PTEN"=c("score.1"="stability", "score.2"="enzyme.activity"), 
                 "PTEN.bin"=c("score.1"="stability", "score.2"="enzyme.activity"), 
                 "NUDT15"=c("score.1"="stability", "score.2"="enzyme.activity"), 
                 "VKORC1"=c("score.1"="enzyme.activity", "score.2"="stability"), 
                 "CCR5"=c("score.1"="stability", "score.2"="binding Ab2D7", "score.3"="binding HIV-1"), 
                 "CXCR4"=c("score.1"="stability", "score.2"="binding CXCL12", "score.3"="binding Ab12G5"))
# base.dirs <- strsplit(base.dirs, split = ',')[[1]]
# results <- data.frame()
# for (base.dir in base.dirs) {
genes <- c("PTEN", "PTEN.bin", "NUDT15", "VKORC1", "CCR5", "CXCR4")
# add baseline AUC
# esm alphabets
source('~/Pipeline/AUROC.R')
source('~/Pipeline/uniprot.table.add.annotation.R')
alphabet <- c('<cls>', '<pad>', '<eos>', '<unk>',
              'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
              'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
              'X', 'B', 'U', 'Z', 'O', '.', '-',
              '<null_1>', '<mask>')
# prepare subset of datasets
for (i in 1:length(genes)) {
  for (subset in c(1,2,4,6)) {
    for (seed in 0:4) {
      train.file <- read.csv(paste0('/share/terra/Users/gz2294/Data/DMS/MAVEDB/', genes[i], 
                                    '/training.', subset, '.', seed, '.csv'))
      full.train.file <- read.csv(paste0('/share/terra/Users/gz2294/Data/DMS/MAVEDB/', genes[i], 
                                         '/train.seed.0.csv'))
      train.file$unique.id <- paste0(train.file$ref, train.file$pos.orig, train.file$alt)
      full.train.file$unique.id <- paste0(full.train.file$ref, full.train.file$pos.orig, full.train.file$alt)
      train.file.idx <- match(train.file$unique.id, full.train.file$unique.id)
      esm.logits <- read.csv(paste0('esm2.inference/', genes[i], '/training.fold.0.tokens.csv'))
      esm.logits <- esm.logits[,2:dim(esm.logits)[2]]
      colnames(esm.logits) <- alphabet
      write.csv(esm.logits[train.file.idx,], 
                paste0('esm2.inference/', genes[i], '/training.', subset, '.', seed, '.tokens.csv'))
    }
  }
}
result <- data.frame()
for (i in 1:length(genes)) {
  for (subset in c(1,2,4,6,8)) {
  for (fold in 0:4) {
    # REVEL, PrimateAI, ESM AUC
    if (subset == 8) {
      test.result <- read.csv(paste0('PreMode.inference/', genes[i], '/',
                                     '/testing.fold.', fold, '.csv'))
      # elastic.net.result <- read.csv(paste0('elastic.net.result/', genes[i], '/',
      #                                       '/prediction.test.subset.', subset, '.seed.', fold, '.csv'))
      baseline.result.1 <- read.csv(paste0('PreMode.onehot.inference/', genes[i], '/',
                                           '/testing.fold.', 0, '.csv'))
      baseline.result.2 <- read.csv(paste0('PreMode.pass.inference/', genes[i], '/',
                                           '/testing.fold.', fold, '.csv'))
    } else {
      test.result <- read.csv(paste0('PreMode.inference/', genes[i], '/',
                                     '/testing.subset.', subset, '.fold.', fold, '.csv'))
      # elastic.net.result <- read.csv(paste0('elastic.net.result/', genes[i], '/',
      #                                       '/prediction.test.subset.', subset, '.seed.', fold, '.csv'))
      baseline.result.1 <- read.csv(paste0('PreMode.onehot.inference/', genes[i], '/',
                                     '/testing.subset.', subset, '.fold.', fold, '.csv'))
      baseline.result.2 <- read.csv(paste0('PreMode.pass.inference/', genes[i], '/',
                                     '/testing.subset.', subset, '.fold.', fold, '.csv'))
    }
    
    task.length <- length(task.dic[[genes[i]]])
    
    PreMode.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], 
                           test.result[,paste0("logits.", 0:(task.length-1))],
                           bin = grepl("bin", genes[i]))
    # elas.net.auc <- plot.R2(elastic.net.result[,names(task.dic[[genes[i]]])], 
    #                         elastic.net.result[,paste0("prediction.score.", 1:(task.length))],
    #                        bin = grepl("bin", genes[i]))
    baseline.auc.1 <- plot.R2(baseline.result.1[,names(task.dic[[genes[i]]])],
                              baseline.result.1[,paste0("logits.", 0:(task.length-1))],
                           bin = grepl("bin", genes[i]))
    baseline.auc.2 <- plot.R2(baseline.result.2[,names(task.dic[[genes[i]]])],
                              baseline.result.2[,paste0("logits.", 0:(task.length-1))],
                           bin = grepl("bin", genes[i]))
    
    to.append <- data.frame(min.val.R = c(PreMode.auc$R2,
                                          # elas.net.auc$R2, 
                                          baseline.auc.1$R2, baseline.auc.2$R2
                                          ),
                            task.name = paste0(genes[i], ":", rep(task.dic[[genes[i]]], 3)))
    to.append$model <- rep(c("PreMode",
                             # "elastic.net", 
                             "Baseline (No ESM)", "Baseline (No Structure)"
                             ), each = task.length)
    to.append$subset <- subset
    to.append$seed <- fold
    result <- rbind(result, to.append)
  }
  }
}
write.csv(result, 'figs/02.03.MAVE.subset.PreMode.compare.csv')

num.models <- unique(result$model)
plots <- list()
library(patchwork)
for (i in 1:length(task.dic)) {
  task <- names(task.dic)[i]
  task.res <- result[startsWith(result$task.name, paste0(task, ":")),]
  task.res <- task.res[,!is.na(task.res[1,])]
  assays <- length(task.dic[[i]])
  data.points <- c()
  for (subset in c(1,2,4,6)) {
    data.points <- c(data.points,
                     as.numeric(
                       strsplit(system(paste0("wc -l ", "~/Data/DMS/MAVEDB/", task, ".", subset,".seed.0/training.csv"),
                                       intern = T), " ")[[1]][1])-1)
  }
  data.points <- c(data.points,
                   as.numeric(
                     strsplit(system(paste0("wc -l ", "~/Data/DMS/MAVEDB/", task, "/training.csv"),
                                     intern = T), " ")[[1]][1]))
  task.plots <- list()
  for (k in 1:length(num.models)) {
    model <- num.models[k]
    to.plot <- task.res[task.res$model==model,]
    p <- ggplot(to.plot, aes(x=subset, y=min.val.R, col=task.name)) + 
      geom_point() + 
      # geom_line(aes(y=zero.shot), linetype="dotted") + 
      geom_smooth() + scale_y_continuous(breaks=seq(0, 1, 0.2), limits = c(0, 1.05)) +
      scale_x_continuous(breaks=c(1, 2, 4, 6, 8),
                         labels=paste0(data.points,
                                       c(" (10%)", " (20%)", " (40%)", " (60%)", " (80%)"))) +
      theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      ggtitle(paste0(task, ":", model)) + ggeasy::easy_center_title() + xlab("training data size (%)")
    task.plots[[k]] <- p
  }
  plots[[i]] <- task.plots[[1]] + task.plots[[2]] + task.plots[[3]]
}
library(patchwork)
p <- plots[[1]] / plots[[2]] / plots[[3]] / plots[[4]] / plots[[5]] / plots[[6]]
ggsave(p, filename = paste0("figs/02.04.MAVE.subsets.pdf"), width = 15, height = 18)

# ggsave(paste0('figs/02.03.MAVE.subset.PreMode.compare.pdf'), p, height = 6, width = 12)
