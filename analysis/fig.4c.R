library(ggplot2)
py.path = '/share/vault/Users/gz2294/miniconda3/envs/RESCVE/bin/python'
task.dic <- list("PTEN"=c("score.1"="stability", "score.2"="enzyme.activity"), 
                 "NUDT15"=c("score.1"="stability", "score.2"="enzyme.activity"), 
                 "CCR5"=c("score.1"="stability", "score.2"="binding Ab2D7", "score.3"="binding HIV-1"), 
                 "CXCR4"=c("score.1"="stability", "score.2"="binding CXCL12", "score.3"="binding Ab12G5"),
                 "SNCA"=c("score.1"="enzyme.activity", "score.2"="stability"),
                 "CYP2C9"=c("score.1"="enzyme.activity", "score.2"="stability"),
                 "GCK"=c("score.1"="enzyme.activity", "score.2"="stability"),
                 "ASPA"=c("score.1"="stability", "score.2"="enzyme.activity")
)
source('./prepare.biochem.R')
genes <- c("PTEN", "NUDT15", "CCR5", "CXCR4", "SNCA", "CYP2C9", "GCK", "ASPA")
# add baseline AUC
# esm alphabets
source('./AUROC.R')
biochem.cols <- c('secondary_struc', 'rsa', 'conservation.entropy', 
                  'conservation.alt', 'conservation.ref', 'pLDDT')
alphabet <- c('<cls>', '<pad>', '<eos>', '<unk>',
              'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
              'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
              'X', 'B', 'U', 'Z', 'O', '.', '-',
              '<null_1>', '<mask>')
# get test results
result <- data.frame()
for (i in 1:length(genes)) {
  test.result <- read.csv(paste0('PreMode/', genes[i], '/test.fold.0.annotated.csv'))
  anno.all <- read.csv(paste0('../data.files/', genes[i], '/ALL.annotated.csv'))
  anno.all <- prepare.unique.id(anno.all)
  task.length <- length(task.dic[[genes[i]]])
  for (subset in c(1,2,4,6,8)) {
    for (fold in 0:4) {
      # REVEL, PrimateAI, ESM AUC
      if (subset == 8) {
        test.result <- read.csv(paste0('PreMode/', genes[i], '/',
                                       '/testing.fold.', fold, '.csv'))
        gene.train <- read.csv(paste0('../data.files/', genes[i], '/',
                                      '/train.seed.', fold, '.csv'))
        # get train config
        train.config <- yaml::read_yaml(paste0('../scripts/PreMode/',
                                               genes[i], '.5fold/', genes[i], '.fold.', fold, '.yaml'))
        # get train val split
        baseline.result.2 <- read.csv(paste0('ESM.SLP/', genes[i], '/',
                                             '/testing.fold.', fold, '.csv'))
        # add hsu et al results
        hsu.unirep_onehot.auc <- list(R2=c())
        hsu.ev_onehot.auc <- list(R2=c())
        hsu.gesm_onehot.auc <- list(R2=c())
        hsu.eve_onehot.auc <- list(R2=c())
        for (s in 1:task.length) {
          test.result.hsu <- read.csv(paste0('./Hsu.et.al.git/results/', 
                                             genes[i], '.fold.', fold, '.score.', s, '/results.csv'))
          hsu.unirep_onehot.auc$R2 <- c(hsu.unirep_onehot.auc$R2, test.result.hsu$spearman[match('eunirep_ll+onehot', test.result.hsu$predictor)])
          hsu.ev_onehot.auc$R2 <- c(hsu.ev_onehot.auc$R2, test.result.hsu$spearman[match('ev+onehot', test.result.hsu$predictor)])
          hsu.gesm_onehot.auc$R2 <- c(hsu.gesm_onehot.auc$R2, test.result.hsu$spearman[match('gesm+onehot', test.result.hsu$predictor)])
          hsu.eve_onehot.auc$R2 <- c(hsu.eve_onehot.auc$R2, test.result.hsu$spearman[match('vae+onehot', test.result.hsu$predictor)])
        }
      } else {
        test.result <- read.csv(paste0('PreMode/', genes[i], '/',
                                       '/testing.subset.', subset, '.fold.', fold, '.csv'))
        gene.train <- read.csv(paste0('../data.files/', genes[i], '/',
                                      '/training.', subset, '.', fold, '.csv'))
        train.config <- yaml::read_yaml(paste0('../scripts/PreMode/',
                                               genes[i], '.subsets/subset.', subset, '/seed.', fold, '.yaml'))
        baseline.result.2 <- read.csv(paste0('ESM.SLP/', genes[i], '/',
                                             '/testing.subset.', subset, '.fold.', fold, '.csv'))
        # add hsu et al results
        hsu.unirep_onehot.auc <- list(R2=c())
        hsu.ev_onehot.auc <- list(R2=c())
        hsu.gesm_onehot.auc <- list(R2=c())
        hsu.eve_onehot.auc <- list(R2=c())
        for (s in 1:task.length) {
          test.result.hsu <- read.csv(paste0('./Hsu.et.al.git/results/', 
                                             genes[i], '.subset.', subset, '.fold.', fold, '.score.', s, '/results.csv'))
          hsu.unirep_onehot.auc$R2 <- c(hsu.unirep_onehot.auc$R2, test.result.hsu$spearman[match('eunirep_ll+onehot', test.result.hsu$predictor)])
          hsu.ev_onehot.auc$R2 <- c(hsu.ev_onehot.auc$R2, test.result.hsu$spearman[match('ev+onehot', test.result.hsu$predictor)])
          hsu.gesm_onehot.auc$R2 <- c(hsu.gesm_onehot.auc$R2, test.result.hsu$spearman[match('gesm+onehot', test.result.hsu$predictor)])
          hsu.eve_onehot.auc$R2 <- c(hsu.eve_onehot.auc$R2, test.result.hsu$spearman[match('vae+onehot', test.result.hsu$predictor)])
        }
      }
      np <- reticulate::import('numpy')
      train.val.split <- np$load(paste0('../', train.config$log_dir, 'splits.0.npz'))
      gene.train <- gene.train[train.val.split['idx_train']+1,]
      
      test.result <- prepare.unique.id(test.result)
      gene.train <- prepare.unique.id(gene.train)
      test.result[,biochem.cols] <- anno.all[match(test.result$unique.id, anno.all$unique.id), biochem.cols]
      gene.train[,biochem.cols] <- anno.all[match(gene.train$unique.id, anno.all$unique.id), biochem.cols]
      
      PreMode.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], 
                             test.result[,paste0("logits.", 0:(task.length-1))],
                             bin = grepl("bin", genes[i]))
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
      res <- system(paste0(py.path, ' ', 
                           'elastic.net.dms.py ', 
                           train.biochem.file, ' ',
                           train.label.file, ' ',
                           test.biochem.file, ' ', 
                           test.label.file), intern = T)
      baseline.auc.3 <- list(R2=as.numeric(as.data.frame(strsplit(res, split = '='))[2,]))
      to.append <- data.frame(min.val.R = c(PreMode.auc$R2,
                                            baseline.auc.3$R2,
                                            baseline.auc.2$R2,
                                            hsu.gesm_onehot.auc$R2,
                                            hsu.ev_onehot.auc$R2,
                                            hsu.unirep_onehot.auc$R2,
                                            hsu.eve_onehot.auc$R2
      ),
      task.name = paste0(genes[i], ":", rep(task.dic[[genes[i]]], 7)))
      to.append$model <- rep(c("PreMode",
                               "Elastic Net",
                               "ESM+SLP",
                               "Augmented ESM1b",
                               "Augmented EVmutation",
                               "Augmented Unirep",
                               "Augmented EVE"
                               
      ), each = task.length)
      to.append$subset <- subset
      to.append$seed <- fold
      result <- rbind(result, to.append)
    }
  }
}
num.models <- unique(result$model)
# show weighted average
# plot the task weighted averages as well as task size weighted error bars
uniq.result.plot <- result[result$seed==0,]
for (i in 1:dim(uniq.result.plot)[1]) {
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
                       strsplit(system(paste0("wc -l ", "../data.files/", task, "/training.", subset, ".0.csv"),
                                       intern = T), " ")[[1]][1])-1)
  }
  data.points <- c(data.points,
                   as.numeric(
                     strsplit(system(paste0("wc -l ", "../data.files/", task, "/training.csv"),
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
  plots[[i]] <- ggpubr::ggarrange(plotlist = task.plots, ncol = length(num.models), common.legend = T, legend = "bottom")
}
library(patchwork)
p <- plots[[1]] / plots[[2]] / plots[[3]] / plots[[4]] / plots[[5]] / plots[[6]] / plots[[7]] / plots[[8]] 
ggsave(p, filename = paste0("figs/fig.sup.4.pdf"), width = 20, height = 28)

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
                         strsplit(system(paste0("wc -l ", "../data.files/", genes[k], "/training.", subsets[k], ".0.csv"),
                                         intern = T), " ")[[1]][1])-1)
    } else {
      data.points <- c(data.points,
                       as.numeric(
                         strsplit(system(paste0("wc -l ", "../data.files/", genes[k], "/training.csv"),
                                         intern = T), " ")[[1]][1])-1)
    }
  }
  uniq.model.result.plot$rho[i] <- sum(rhos * data.points, na.rm = T) / sum(data.points)
  uniq.model.result.plot$rho.sd[i] <- sum(rho.sds * data.points, na.rm = T) / sum(data.points)
}
p <- ggplot(uniq.model.result.plot, aes(x=subset, y=rho, col=model)) +
  geom_point() +
  geom_errorbar(aes(ymin=rho-rho.sd, ymax=rho+rho.sd), width=.2) +
  geom_line() + 
  scale_y_continuous(breaks=seq(0, 1, 0.2), limits = c(-0.1, 1.05)) +
  scale_x_continuous(breaks=c(1, 2, 4, 6, 8),
                     labels=paste0(c(" (10%)", " (20%)", " (40%)", " (60%)", " (80%)"))) +
  ylab('Spearman rho') +
  theme_bw() +
  ggtitle("Weighted Average of Model \nPerformances on Subsample of Training") +
  ggeasy::easy_center_title() + xlab("training data size (% of full DMS dataset)")
ggsave('figs/fig.4c.pdf', p, width = 5, height = 4)
