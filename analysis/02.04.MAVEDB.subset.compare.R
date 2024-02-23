# source('utils.R')
library(ggplot2)
# args <- commandArgs(trailingOnly = T)
# base dir for transfer learning
base.dir <- "/share/terra/Users/gz2294/PreMode.final/scripts/CHPs.v4.esm.dssp.small.StarAttn.MSA.StarPool.1dim/"
task.dic <- list("PTEN"=c("score.1"="stability", "score.2"="enzyme.activity"), 
                 "NUDT15"=c("score.1"="stability", "score.2"="enzyme.activity"), 
                 # "VKORC1"=c("score.1"="enzyme.activity", "score.2"="stability"), 
                 "CCR5"=c("score.1"="stability", "score.2"="binding Ab2D7", "score.3"="binding HIV-1"), 
                 "CXCR4"=c("score.1"="stability", "score.2"="binding CXCL12", "score.3"="binding Ab12G5"),
                 "SNCA"=c("score.1"="enzyme.activity", "score.2"="stability"),
                 "CYP2C9"=c("score.1"="enzyme.activity", "score.2"="stability"),
                 "GCK"=c("score.1"="enzyme.activity", "score.2"="stability"),
                 "ASPA"=c("score.1"="stability", "score.2"="enzyme.activity")
)
source('/share/vault/Users/gz2294/Pipeline/prepare.biochem.R')
# base.dirs <- strsplit(base.dirs, split = ',')[[1]]
# results <- data.frame()
# for (base.dir in base.dirs) {
genes <- c("PTEN", "NUDT15", "CCR5", "CXCR4", "SNCA", "CYP2C9", "GCK", "ASPA")
# add baseline AUC
# esm alphabets
source('~/Pipeline/AUROC.R')
source('/share/vault/Users/gz2294/Pipeline/uniprot.table.add.annotation.R')
source('/share/vault/Users/gz2294/Pipeline/prepare.biochem.R')
biochem.cols <- c('secondary_struc', 'rsa', 'conservation.entropy', 
                  'conservation.alt', 'conservation.ref', 'pLDDT')
alphabet <- c('<cls>', '<pad>', '<eos>', '<unk>',
              'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
              'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
              'X', 'B', 'U', 'Z', 'O', '.', '-',
              '<null_1>', '<mask>')
# prepare subset of datasets
for (i in 1:length(genes)) {
  # for (subset in c(1,2,4,6)) {
  #   for (seed in 0:4) {
  #     train.file <- read.csv(paste0('/share/terra/Users/gz2294/Data/DMS/MAVEDB/', genes[i], 
  #                                   '/training.', subset, '.', seed, '.csv'))
  #     full.train.file <- read.csv(paste0('/share/terra/Users/gz2294/Data/DMS/MAVEDB/', genes[i], 
  #                                        '/train.seed.0.csv'))
  #     train.file$unique.id <- paste0(train.file$ref, train.file$pos.orig, train.file$alt)
  #     full.train.file$unique.id <- paste0(full.train.file$ref, full.train.file$pos.orig, full.train.file$alt)
  #     train.file.idx <- match(train.file$unique.id, full.train.file$unique.id)
  #     esm.logits <- read.csv(paste0('esm2.inference/', genes[i], '/training.fold.0.tokens.csv'))
  #     esm.logits <- esm.logits[,2:dim(esm.logits)[2]]
  #     colnames(esm.logits) <- alphabet
  #     write.csv(esm.logits[train.file.idx,], 
  #               paste0('esm2.inference/', genes[i], '/training.', subset, '.', seed, '.tokens.csv'))
  #   }
  # }
}


# get test results
result <- data.frame()
for (i in 1:length(genes)) {
  test.result <- read.csv(paste0('PreMode.inference/', genes[i], '/test.fold.0.annotated.csv'))
  anno.all <- read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/MAVEDB/', genes[i], '/ALL.annotated.csv'))
  anno.all <- prepare.unique.id(anno.all)
  
  task.length <- length(task.dic[[genes[i]]])
  # PreMode.pretrain.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], -test.result[,rep("pretrain.logits", task.length)], bin = grepl('bin', genes[i]))
  # REVEL.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], -test.result[,rep("REVEL", task.length)], bin = grepl('bin', genes[i]))
  # PrimateAI.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], -test.result[,rep("PrimateAI", task.length)], bin = grepl('bin', genes[i]))
  # ESM.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], test.result[,rep("esm.logits", task.length)], bin = grepl('bin', genes[i]))
  # EVE.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], -test.result[,rep("EVE", task.length)], bin = grepl('bin', genes[i]))
  # gMVP.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], -test.result[,rep("gMVP", task.length)], bin = grepl('bin', genes[i]))
  # best.other.models <- rbind(REVEL.auc$R2, PrimateAI.auc$R2, ESM.auc$R2, EVE.auc$R2, gMVP.auc$R2)
  # best.other.models <- apply(best.other.models, 2, max, na.rm=T)
  # dash.base.line <- data.frame(min.val.R = c(PreMode.pretrain.auc$R2, best.other.models),
  #                              task.name = paste0(genes[i], ":", rep(task.dic[[genes[i]]], 2)))
  # dash.base.line$model <- rep(c("PreMode: Pretrain", "Best Other Methods"), each = task.length)
  # dash.base.line.models <- rbind(dash.base.line.models, dash.base.line)
  for (subset in c(1,2,4,6,8)) {
    for (fold in 0:4) {
      # REVEL, PrimateAI, ESM AUC
      if (subset == 8) {
        test.result <- read.csv(paste0('PreMode.inference/', genes[i], '/',
                                       '/testing.fold.', fold, '.csv'))
        gene.train <- read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/MAVEDB/', genes[i], '/',
                                      '/train.seed.', fold, '.csv'))
        # get train config
        train.config <- yaml::read_yaml(paste0('/share/vault/Users/gz2294/PreMode/scripts/CHPs.v4.esm.dssp.small.StarAttn.MSA.StarPool.1dim/',
                                               genes[i], '.5fold/', genes[i], '.fold.', fold, '.yaml'))
        # get train val split
        baseline.result.2 <- read.csv(paste0('PreMode.pass.inference/', genes[i], '/',
                                             '/testing.fold.', fold, '.csv'))
      } else {
        test.result <- read.csv(paste0('PreMode.inference/', genes[i], '/',
                                       '/testing.subset.', subset, '.fold.', fold, '.csv'))
        gene.train <- read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/MAVEDB/', genes[i], '/',
                                      '/training.', subset, '.', fold, '.csv'))
        train.config <- yaml::read_yaml(paste0('/share/vault/Users/gz2294/PreMode/scripts/CHPs.v4.esm.dssp.small.StarAttn.MSA.StarPool.1dim/',
                                               genes[i], '.subsets/subset.', subset, '/seed.', fold, '.yaml'))
        baseline.result.2 <- read.csv(paste0('PreMode.pass.inference/', genes[i], '/',
                                             '/testing.subset.', subset, '.fold.', fold, '.csv'))
      }
      np <- reticulate::import('numpy')
      train.val.split <- np$load(paste0(train.config$log_dir, 'splits.0.npz'))
      gene.train <- gene.train[train.val.split['idx_train']+1,]
      
      test.result <- prepare.unique.id(test.result)
      gene.train <- prepare.unique.id(gene.train)
      test.result[,biochem.cols] <- anno.all[match(test.result$unique.id, anno.all$unique.id), biochem.cols]
      gene.train[,biochem.cols] <- anno.all[match(gene.train$unique.id, anno.all$unique.id), biochem.cols]
      
      PreMode.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], 
                             test.result[,paste0("logits.", 0:(task.length-1))],
                             bin = grepl("bin", genes[i]))
      # baseline.auc.1 <- plot.R2(baseline.result.1[,names(task.dic[[genes[i]]])],
      #                           baseline.result.1[,paste0("logits.", 0:(task.length-1))],
      #                           bin = grepl("bin", genes[i]))
      baseline.auc.2 <- plot.R2(baseline.result.2[,names(task.dic[[genes[i]]])],
                                baseline.result.2[,paste0("logits.", 0:(task.length-1))],
                                bin = grepl("bin", genes[i]))
      # write train and test emb to files
      train.label.file <- tempfile()
      test.label.file <- tempfile()
      train.biochem.file <- tempfile()
      test.biochem.file <- tempfile()
      write.csv(gene.train, file = train.label.file)
      write.csv(test.result, file = test.label.file)
      write.csv(prepare.biochemical(gene.train), file = train.biochem.file)
      write.csv(prepare.biochemical(test.result), file = test.biochem.file)
      res <- system(paste0('/share/descartes/Users/gz2294/miniconda3/envs/RESCVE/bin/python ', 
                           '10.analysis.few.shot.elastic.net.dms.py ', 
                           train.biochem.file, ' ',
                           train.label.file, ' ',
                           test.biochem.file, ' ', 
                           test.label.file), intern = T)
      baseline.auc.3 <- list(R2=as.numeric(as.data.frame(strsplit(res, split = '='))[2,]))
      res <- system(paste0('/share/descartes/Users/gz2294/miniconda3/envs/RESCVE/bin/python ', 
                           '10.analysis.few.shot.random.forest.dms.py ', 
                           train.biochem.file, ' ',
                           train.label.file, ' ',
                           test.biochem.file, ' ', 
                           test.label.file), intern = T)
      baseline.auc.4 <- list(R2=as.numeric(as.data.frame(strsplit(res, split = '='))[2,]))
      to.append <- data.frame(min.val.R = c(PreMode.auc$R2,
                                            baseline.auc.3$R2,
                                            baseline.auc.4$R2,
                                            baseline.auc.2$R2
      ),
      task.name = paste0(genes[i], ":", rep(task.dic[[genes[i]]], 4)))
      to.append$model <- rep(c("PreMode",
                               "Elastic Net",
                               "Random Forest",
                               "ESM + SLR"
      ), each = task.length)
      to.append$subset <- subset
      to.append$seed <- fold
      result <- rbind(result, to.append)
    }
  }
}
write.csv(result, 'figs/02.03.MAVE.subset.PreMode.compare.csv')

result <- read.csv('figs/02.03.MAVE.subset.PreMode.compare.csv', row.names = 1)
result <- result[result$model != "Random Forest",]
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
                       strsplit(system(paste0("wc -l ", "/share/vault/Users/gz2294/Data/DMS/MAVEDB/", task, ".", subset,".seed.0/training.csv"),
                                       intern = T), " ")[[1]][1])-1)
  }
  data.points <- c(data.points,
                   as.numeric(
                     strsplit(system(paste0("wc -l ", "/share/vault/Users/gz2294/Data/DMS/MAVEDB/", task, "/training.csv"),
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
      ylab('Spearman rho') +
      theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      ggtitle(paste0(task, ":", model)) + ggeasy::easy_center_title() + xlab("training data size (%)")
    task.plots[[k]] <- p
  }
  plots[[i]] <- task.plots[[1]] + task.plots[[2]] + task.plots[[3]]
}
library(patchwork)
p <- plots[[1]] / plots[[2]] / plots[[3]] / plots[[4]] / plots[[5]] / plots[[6]] / plots[[7]] / plots[[8]] 
ggsave(p, filename = paste0("figs/02.04.MAVE.subsets.pdf"), width = 15, height = 28)

# ggsave(paste0('figs/02.03.MAVE.subset.PreMode.compare.pdf'), p, height = 6, width = 12)
# show weighted average
# plot the task weighted averages as well as task size weighted error bars
uniq.result.plot <- result[result$seed==0,]
for (i in 1:dim(uniq.result.plot)) {
  rhos <- result$min.val.R[result$model==uniq.result.plot$model[i] & 
                             result$task.name==uniq.result.plot$task.name[i] &
                             result$subset==uniq.result.plot$subset[i]]
  rhos <- rhos[rhos > 0]
  uniq.result.plot$rho[i] = mean(rhos, na.rm=T)
  uniq.result.plot$rho.sd[i] = sd(rhos, na.rm=T)
}

plots <- list()
library(patchwork)
for (i in 1:length(task.dic)) {
  task <- names(task.dic)[i]
  task.res <- uniq.result.plot[startsWith(uniq.result.plot$task.name, paste0(task, ":")),]
  task.res <- task.res[,!is.na(task.res[1,])]
  assays <- length(task.dic[[i]])
  data.points <- c()
  for (subset in c(1,2,4,6)) {
    data.points <- c(data.points,
                     as.numeric(
                       strsplit(system(paste0("wc -l ", "/share/vault/Users/gz2294/Data/DMS/MAVEDB/", task, ".", subset,".seed.0/training.csv"),
                                       intern = T), " ")[[1]][1])-1)
  }
  data.points <- c(data.points,
                   as.numeric(
                     strsplit(system(paste0("wc -l ", "/share/vault/Users/gz2294/Data/DMS/MAVEDB/", task, "/training.csv"),
                                     intern = T), " ")[[1]][1]))
  task.plots <- list()
  for (k in 1:length(num.models)) {
    model <- num.models[k]
    to.plot <- task.res[task.res$model==model,]
    p <- ggplot(to.plot, aes(x=subset, y=rho, col=task.name)) + 
      geom_point() + 
      geom_errorbar(aes(ymin=rho-rho.sd, ymax=rho+rho.sd), width=.4) +
      # geom_line(aes(y=zero.shot), linetype="dotted") + 
      geom_line() + 
      scale_y_continuous(breaks=seq(0, 1, 0.2), limits = c(-0.1, 1.05)) +
      scale_x_continuous(breaks=c(1, 2, 4, 6, 8),
                         labels=paste0(data.points,
                                       c(" (10%)", " (20%)", " (40%)", " (60%)", " (80%)"))) +
      ylab('Spearman rho') +
      theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      ggtitle(paste0(task, ":", model)) + ggeasy::easy_center_title() + xlab("training data size (%)")
    task.plots[[k]] <- p
  }
  plots[[i]] <- task.plots[[1]] + task.plots[[2]] + task.plots[[3]]
}
library(patchwork)
p <- plots[[1]] / plots[[2]] / plots[[3]] / plots[[4]] / plots[[5]] / plots[[6]] / plots[[7]] / plots[[8]] 
ggsave(p, filename = paste0("figs/02.04.MAVE.subsets.mean.sd.pdf"), width = 15, height = 28)

# aggregate across models
uniq.model.result.plot <- uniq.result.plot[!duplicated(uniq.result.plot[,c('model', "subset")]),]
for (i in 1:dim(uniq.model.result.plot)[1]) {
  rhos <- uniq.result.plot$rho[uniq.result.plot$model == uniq.model.result.plot$model[i] &
                                 uniq.result.plot$subset == uniq.model.result.plot$subset[i]]
  rho.sds <- uniq.result.plot$rho.sd[uniq.result.plot$model == uniq.model.result.plot$model[i] &
                                       uniq.result.plot$subset == uniq.model.result.plot$subset[i]]
  genes <- gsub(":.*", "", uniq.result.plot$task.name[uniq.result.plot$model == uniq.model.result.plot$model[i] &
                                                        uniq.result.plot$subset == uniq.model.result.plot$subset[i]])
  subsets <- uniq.result.plot$subset[uniq.result.plot$model == uniq.model.result.plot$model[i] &
                                       uniq.result.plot$subset == uniq.model.result.plot$subset[i]]
  # get data set sizes
  data.points <- c()
  for (k in 1:length(genes)) {
    if (subsets[k] != 8) {
      data.points <- c(data.points,
                       as.numeric(
                         strsplit(system(paste0("wc -l ", "/share/vault/Users/gz2294/Data/DMS/MAVEDB/", genes[k], ".", subsets[k],".seed.0/training.csv"),
                                         intern = T), " ")[[1]][1])-1)
    } else {
      data.points <- c(data.points,
                       as.numeric(
                         strsplit(system(paste0("wc -l ", "/share/vault/Users/gz2294/Data/DMS/MAVEDB/", genes[k], "/training.csv"),
                                         intern = T), " ")[[1]][1])-1)
    }
  }
  uniq.model.result.plot$rho[i] <- sum(rhos * data.points, na.rm = T) / sum(data.points)
  uniq.model.result.plot$rho.sd[i] <- sum(rho.sds * data.points, na.rm = T) / sum(data.points)
  # uniq.model.result.plot$rho[i] <- mean(rhos, na.rm = T)
  # uniq.model.result.plot$rho.sd[i] <- mean(rho.sds, na.rm = T)
}
uniq.model.result.plot$model[uniq.model.result.plot$model == "PreMode"] <- "PreMode (148k)"
uniq.model.result.plot$model[uniq.model.result.plot$model == "ESM + SLR"] <- "ESM emb + SLP"
p <- ggplot(uniq.model.result.plot, aes(x=subset, y=rho, col=model)) +
  geom_point() +
  geom_errorbar(aes(ymin=rho-rho.sd, ymax=rho+rho.sd), width=.2) +
  geom_line() + 
  scale_y_continuous(breaks=seq(0, 1, 0.2), limits = c(-0.1, 1.05)) +
  scale_x_continuous(breaks=c(1, 2, 4, 6, 8),
                     labels=paste0(c(" (10%)", " (20%)", " (40%)", " (60%)", " (80%)"))) +
  ylab('Spearman rho') +
  theme_bw() +
  ggtitle("Weighted Average of Model \nperformances on subsample of training") +
  ggeasy::easy_center_title() + xlab("training data size (% of full DMS dataset)")
ggsave('figs/02.04.MAVE.subsets.weighted.average.pdf', p, width = 5, height = 4)
