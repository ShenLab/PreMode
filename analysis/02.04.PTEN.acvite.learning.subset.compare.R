# source('utils.R')
library(ggplot2)
# args <- commandArgs(trailingOnly = T)
# base dir for transfer learning
base.dir <- "/share/terra/Users/gz2294/PreMode.final/scripts/CHPs.v4.esm.dssp.small.StarAttn.MSA.StarPool.1dim/"
task.dic <- list("PTEN"=c("score.1"="stability", "score.2"="enzyme.activity"))
# base.dirs <- strsplit(base.dirs, split = ',')[[1]]
# results <- data.frame()
# for (base.dir in base.dirs) {
genes <- c("PTEN")
# add baseline AUC
# esm alphabets
source('~/Pipeline/AUROC.R')
source('/share/vault/Users/gz2294/Pipeline/uniprot.table.add.annotation.R')
alphabet <- c('<cls>', '<pad>', '<eos>', '<unk>',
              'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
              'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
              'X', 'B', 'U', 'Z', 'O', '.', '-',
              '<null_1>', '<mask>')
# prepare subset of datasets
# for (i in 1:length(genes)) {
#   for (seed in 0:4) {
#     testing.file <- read.csv(paste0(yaml::read_yaml(paste0('/share/vault/Users/gz2294/PreMode/scripts/CHPs.v4.adaptive.train/', 
#                                                            genes[i], '.5fold/', genes[i], '.fold.',
#                                                            seed, '.yaml'))$log_dir, '/testing.round.0.csv'))
#     testing.file <- uniprot.table.add.annotation.parallel(testing.file, 'dbnsfp')
#     testing.file <- uniprot.table.add.annotation.parallel(testing.file, 'EVE')
#     testing.file <- uniprot.table.add.annotation.parallel(testing.file, 'gMVP')
#     testing.file <- uniprot.table.add.annotation.parallel(testing.file, 'AlphaMissense')
#     write.csv(testing.file, 
#               paste0(yaml::read_yaml(paste0('/share/vault/Users/gz2294/PreMode/scripts/CHPs.v4.adaptive.train/', 
#                                             genes[i], '.5fold/', genes[i], '.fold.',
#                                             seed, '.yaml'))$log_dir, '/testing.round.0.annotated.csv'))
#   }
# }
result <- data.frame()
dash.base.line.models <- data.frame()
for (i in 1:length(genes)) {
  # test.result <- read.csv(paste0(yaml::read_yaml(paste0('/share/vault/Users/gz2294/PreMode/scripts/CHPs.v4.adaptive.train/', 
  #                                                       genes[i], '.5fold/', genes[i], '.fold.',
  #                                                       0, '.yaml'))$log_dir, '/testing.round.0.annotated.csv'))
  task.length <- length(task.dic[[genes[i]]])
  # PreMode.pretrain.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], -test.result[,rep("pretrain.logits", task.length)], bin = grepl('bin', genes[i]))
  # REVEL.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], -test.result[,rep("REVEL", task.length)], bin = grepl('bin', genes[i]))
  # PrimateAI.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], -test.result[,rep("PrimateAI", task.length)], bin = grepl('bin', genes[i]))
  # # ESM.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], test.result[,rep("esm.logits", task.length)], bin = grepl('bin', genes[i]))
  # EVE.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], -test.result[,rep("EVE", task.length)], bin = grepl('bin', genes[i]))
  # gMVP.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], -test.result[,rep("gMVP", task.length)], bin = grepl('bin', genes[i]))
  # best.other.models <- rbind(REVEL.auc$R2, PrimateAI.auc$R2, ESM.auc$R2, EVE.auc$R2, gMVP.auc$R2)
  # best.other.models <- apply(best.other.models, 2, max, na.rm=T)
  # dash.base.line <- data.frame(min.val.R = c(REVEL.auc$R2, EVE.auc$R2),
  #                              task.name = paste0(genes[i], ":", rep(task.dic[[genes[i]]], 2)))
  # dash.base.line$model <- rep(c("REVEL", "EVE"), each = task.length)
  # dash.base.line.models <- rbind(dash.base.line.models, dash.base.line)
  for (subset in 1:6) {
    for (fold in 0:3) {
      # REVEL, PrimateAI, ESM AUC
      if (!subset %in% c(1,2,4,6,8)) {
        test.result <- read.csv(paste0(yaml::read_yaml(paste0('/share/vault/Users/gz2294/PreMode/scripts/CHPs.v4.adaptive.train/', 
                                                              genes[i], '.5fold/', genes[i], '.fold.',
                                                              fold, '.yaml'))$log_dir, '/testing.round.', subset-1, '.csv'))
        baseline.auc.1 <- list(R2=rep(NA, task.length))
        baseline.auc.2 <- list(R2=rep(NA, task.length))
      } else {
        test.result <- read.csv(paste0(yaml::read_yaml(paste0('/share/vault/Users/gz2294/PreMode/scripts/CHPs.v4.adaptive.train/', 
                                                              genes[i], '.5fold/', genes[i], '.fold.',
                                                              fold, '.yaml'))$log_dir, '/testing.round.', subset-1, '.csv'))
        baseline.result.1 <- read.csv(paste0('PreMode.inference/', genes[i], '/',
                                             '/testing.subset.', subset, '.fold.', fold, '.csv'))
        baseline.result.2 <- read.csv(paste0('PreMode.pass.inference/', genes[i], '/',
                                             '/testing.subset.', subset, '.fold.', fold, '.csv'))
        baseline.auc.1 <- plot.R2(baseline.result.1[,names(task.dic[[genes[i]]])],
                                  baseline.result.1[,paste0("logits.", 0:(task.length-1))],
                                  bin = grepl("bin", genes[i]))
        baseline.auc.2 <- plot.R2(baseline.result.2[,names(task.dic[[genes[i]]])],
                                  baseline.result.2[,paste0("logits.", 0:(task.length-1))],
                                  bin = grepl("bin", genes[i]))
      }
      PreMode.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], 
                             test.result[,paste0("logits.", 0:(task.length-1))],
                             bin = grepl("bin", genes[i]))
      to.append <- data.frame(min.val.R = c(PreMode.auc$R2,
                                            baseline.auc.1$R2,
                                            baseline.auc.2$R2),
                              task.name = paste0(genes[i], ":", rep(task.dic[[genes[i]]], 3)))
      to.append$model <- rep(c("PreMode (Adaptive Learning)",
                               # "elastic.net", 
                               "PreMode", "Baseline (No Structure)"
      ), each = task.length)
      to.append$subset <- subset
      to.append$seed <- fold
      result <- rbind(result, to.append)
    }
  }
}
write.csv(result, 'figs/02.03.MAVE.adaptive.learning.compare.csv')

result <- read.csv('figs/02.03.MAVE.adaptive.learning.compare.csv', row.names = 1)
num.models <- unique(result$model)
plots <- list()
library(patchwork)
for (i in 1:length(task.dic)) {
  task <- names(task.dic)[i]
  task.res <- result[startsWith(result$task.name, paste0(task, ":")),]
  # dash.base.line <- dash.base.line.models[startsWith(dash.base.line.models$task.name, paste0(task, ":")) &
  #                                           dash.base.line.models$model == "Best Other Methods",]
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
    to.plot <- to.plot[!is.na(to.plot$min.val.R),]
    # only keep the mean and var
    to.plot.uniq <- to.plot[to.plot$seed==1,]
    for (j in 1:dim(to.plot.uniq)[1]) {
      rhos <- to.plot$min.val.R[to.plot$subset==to.plot.uniq$subset[j] & to.plot$task.name==to.plot.uniq$task.name[j]]
      rhos <- rhos[rhos>0]
      to.plot.uniq$rho[j] <- mean(rhos, na.rm = T)
      to.plot.uniq$rho.sd[j] <- sd(rhos, na.rm = T)
    }
    p <- ggplot(to.plot.uniq, aes(x=subset, y=rho, col=task.name)) + 
      geom_point() + 
      geom_line() +
      geom_errorbar(aes(ymin=rho-rho.sd, ymax=rho+rho.sd), width=.2) +
      scale_y_continuous(breaks=seq(0, 1, 0.2), limits = c(0, 1.05)) +
      scale_x_continuous(breaks=c(1, 2, 4, 6, 8),
                         labels=paste0(data.points,
                                       c(" (10%)", " (20%)", " (40%)", " (60%)", " (80%)"))) +
      # geom_abline(data = dash.base.line, aes(intercept=min.val.R, col=task.name, slope=0), linetype="dashed") +
      theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      ggtitle(paste0(task, ":", model)) + ggeasy::easy_center_title() + xlab("training data size (%)")
    task.plots[[k]] <- p
  }
  plots[[i]] <- task.plots[[1]] + task.plots[[2]]
}
library(patchwork)
p <- plots[[1]] 
ggsave(p, filename = paste0("figs/02.04.PTEN.active.learning.subsets.pdf"), width = 10, height = 4.5)

# ggsave(paste0('figs/02.03.MAVE.subset.PreMode.compare.pdf'), p, height = 6, width = 12)
