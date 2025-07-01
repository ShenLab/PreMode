genes <- c("Q09428", "P15056", "O00555", "P21802",
           "Q14654", "P07949", "Q99250", "Q14524.clean", "P04637")
gene.names <- c("ABCC8", "BRAF", "CACNA1A", "FGFR2",
                "KCNJ11", "RET", "SCN2A", "SCN5A", "TP53")
py.path = '/share/descartes/Users/gz2294/miniconda3/envs/RESCVE/bin/python'
source('./AUROC.R')
summary.df <- data.frame()
plots <- list()
source('./prepare.biochem.R')
ALL <- read.csv('figs/ALL.csv', row.names = 1, na.strings = c('.', 'NA'))
ALL <- prepare.unique.id(ALL)
pick.cond <- 'auc'
for (i in 1:length(genes)) {
  gene <- genes[i]
  for (subset in c(1,2,4,6,8)) {
    for (fold in 0:4) {
      aucs <- c()
      if (subset == 8) {
        gene.test.res <- read.csv(paste0('PreMode/', gene, '/testing.fold.' ,fold, '.4fold.csv'))
        log.yaml <- yaml::read_yaml(paste0('../scripts/PreMode/',
                                           gene, '.5fold/', gene, '.fold.', fold, '.yaml'))
        # compare with large window
        gene.test.res.lw <- read.csv(paste0('PreMode/', gene, '.large.window/testing.fold.' ,fold, '.4fold.csv'))
        gene.train.res <- read.csv(paste0('PreMode/', gene, '/training.fold.' ,fold, '.4fold.csv'))
        gene.train.res.lw <- read.csv(paste0('PreMode/', gene, '.large.window/training.fold.' ,fold, '.4fold.csv'))
      } else {
        gene.test.res <- read.csv(paste0('PreMode/', gene, '/testing.subset.', subset, '.fold.' ,fold, '.4fold.csv'))
        log.yaml <- yaml::read_yaml(paste0('../scripts/PreMode/',
                                           gene, '.subset.', subset, '.5fold/', gene, '.subset.', subset, '.fold.', fold, '.yaml'))
        # compare with large window
        gene.test.res.lw <- read.csv(paste0('PreMode/', gene, '.large.window/testing.subset.', subset, '.fold.' ,fold, '.4fold.csv'))
        gene.train.res <- read.csv(paste0('PreMode/', gene, '/training.subset.', subset, '.fold.' ,fold, '.4fold.csv'))
        gene.train.res.lw <- read.csv(paste0('PreMode/', gene, '.large.window/training.subset.', subset, '.fold.' ,fold, '.4fold.csv'))
      }
      tr.auc <- plot.AUC(gene.train.res$score, rowMeans(gene.train.res[,paste0('logits.FOLD.', 0:3)]))$auc
      tr.auc.lw <- plot.AUC(gene.train.res.lw$score, rowMeans(gene.train.res.lw[,paste0('logits.FOLD.', 0:3)]))$auc
      tr.loss <- rowMeans(gene.train.res[,paste0('min_loss.FOLD.', 0:3)])[1]
      tr.loss.lw <- rowMeans(gene.train.res.lw[,paste0('min_loss.FOLD.', 0:3)])[1]
      if (pick.cond == 'auc') {
        cond <- tr.auc.lw > tr.auc
      } else if (pick.cond == 'loss') {
        cond <- tr.loss > tr.loss.lw
      } else if (pick.cond == 'auc+loss') {
        cond <- tr.auc.lw/tr.loss.lw > tr.auc/tr.loss
      } else {
        cond <- F
      }
      # do 4 fold auc
      if (cond) {
        auc <- plot.AUC(gene.test.res.lw$score, rowMeans(gene.test.res.lw[,paste0('logits.FOLD.', 0:3)]))
      } else {
        auc <- plot.AUC(gene.test.res$score, rowMeans(gene.test.res[,paste0('logits.FOLD.', 0:3)]))
      }
      aucs <- c(aucs, auc$auc)
      # do random forest
      gene.train <- read.csv(paste0('../', log.yaml$data_file_train))
      gene.test <- read.csv(paste0('../', log.yaml$data_file_test))
      # get the same training/val split
      fold.splits <- reticulate::py_load_object(paste0('../', log.yaml$log_dir, '/fold_split.pkl'))
      # prepare unique id
      gene.train <- prepare.unique.id(gene.train)
      gene.test <- prepare.unique.id(gene.test)
      train.biochem <- prepare.biochemical(ALL[match(gene.train$unique.id, ALL$unique.id),])
      test.biochem <- prepare.biochemical(ALL[match(gene.test$unique.id, ALL$unique.id),])
      rownames(train.biochem) <- gene.train[,1]
      rownames(test.biochem) <- gene.test[,1]
      rf.aucs <- c()
      for (f in 1:4) {
        # get split info
        val.gof.idx <- fold.splits[[1]][[f]]
        val.lof.idx <- fold.splits[[2]][[f]]
        train.idx <- !gene.train[,1] %in% c(val.gof.idx, val.lof.idx)
        # call python on elastic net
        train.biochem.file <- tempfile()
        test.biochem.file <- tempfile()
        train.label.file <- tempfile()
        test.label.file <- tempfile()
        output.file <- tempfile()
        write.csv(train.biochem[train.idx,],
                  file = train.biochem.file)
        write.csv(test.biochem, 
                  file = test.biochem.file)
        write.csv(gene.train[train.idx,], file = train.label.file)
        write.csv(gene.test, file = test.label.file)
        # call python on random forest
        res <- system(paste0(py.path, ' ', 
                             'random.forest.glof.py ', 
                             train.biochem.file, ' ',
                             train.label.file, ' ',
                             test.biochem.file, ' ', 
                             test.label.file), intern = T)
        rf.aucs <- c(rf.aucs, as.numeric(strsplit(res, split = '=')[[1]][2]))
      }
      aucs <- c(aucs, mean(el.aucs), mean(rf.aucs))
      summary.df <- rbind(summary.df, 
                          data.frame(auc=aucs,
                                     use.lw=c(cond, NA),
                                     model=c('PreMode.transfer', 'random.forest'),
                                     seed=fold,
                                     gene=gene.names[i],
                                     subset=subset,
                                     ngof.train=sum(gene.train$score==1),
                                     nlof.train=sum(gene.train$score==-1),
                                     ngof.test=sum(gene.test$score==1),
                                     nlof.test=sum(gene.test$score==-1)))
    }
  }
}
write.csv(summary.df, file = 'figs/fig.5e.prepare.csv')
library(ggplot2)

summary.df <- read.csv('figs/fig.5e.prepare.csv', row.names = 1)
plots <- list()
library(patchwork)
for (i in 1:length(genes)) {
  task <- gene.names[i]
  task.res <- summary.df[startsWith(summary.df$gene, task),]
  task.res <- task.res[,!is.na(task.res[1,])]
  task.plots <- list()
  data.points <- paste0(task.res$ngof.train[task.res$seed==0 & task.res$model=="PreMode.transfer"],
                        " | ",
                        task.res$nlof.train[task.res$seed==0 & task.res$model=="PreMode.transfer"])
  num.models <- length(unique(summary.df$model))
  p <- ggplot(task.res, aes(x=subset, y=auc, col=model)) + 
    geom_point(alpha=0.2) + 
    # geom_line(aes(y=zero.shot), linetype="dotted") + 
    stat_smooth(geom='line', span=0.3, se = FALSE, alpha=0.5) + scale_y_continuous(breaks=seq(0.4, 1, 0.2), limits = c(0.4, 1.0)) +
    scale_x_continuous(breaks=c(1, 2, 4, 6, 8),
                       labels=paste0(data.points,
                                     c(" (10%)", " (20%)", " (40%)", " (60%)", " (80%)"))) +
    stat_summary(data = task.res,
                 aes(x=as.numeric((subset))+0.4*(as.numeric((model)))/num.models-0.2*(num.models+1)/num.models,
                     y = auc, col=model), 
                 fun.data = mean_se, geom = "errorbar", width = 0.2) +
    stat_summary(data = task.res, 
                 aes(x=as.numeric((subset))+0.4*(as.numeric((model)))/num.models-0.2*(num.models+1)/num.models,
                     y = auc, col=model), 
                 fun.data = mean_se, geom = "point") +
    theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ggtitle(paste0(task)) + ggeasy::easy_center_title() + xlab("training data size, format: GoF | LoF (%)")
  plots[[i]] <- p
}
library(patchwork)
p <- plots[[1]] + plots[[2]] + plots[[3]] + plots[[4]] + plots[[5]] + plots[[6]] + plots[[7]] + plots[[8]] + plots[[9]] + plot_layout(ncol=3)

summary.df <- read.csv('figs/fig.5e.prepare.csv', row.names = 1)
summary.df <- summary.df[summary.df$model %in% c('PreMode.transfer', 'random.forest'),]
model.dic <- c("PreMode.transfer"="Supervised: PreMode", 
               "random.forest"="Supervised: Random Forest")
summary.df$model <- model.dic[summary.df$model]
summary.df$model <- factor(summary.df$model, levels = c("Supervised: PreMode", 
                                                        "Supervised: Random Forest"))
gene.names <- unique(summary.df$gene)

plots <- list()
library(patchwork)
for (i in 1:length(genes)) {
  task <- gene.names[i]
  task.res <- summary.df[startsWith(summary.df$gene, task),]
  task.res <- task.res[,!is.na(task.res[1,])]
  task.plots <- list()
  data.points <- paste0(task.res$ngof.train[task.res$seed==0 & task.res$model=="Supervised: PreMode"],
                        " | ",
                        task.res$nlof.train[task.res$seed==0 & task.res$model=="Supervised: PreMode"])
  num.models <- length(unique(summary.df$model))
  p <- ggplot(task.res, aes(x=subset, y=auc, col=model)) + 
    geom_point(alpha=0) + 
    # geom_line(aes(y=zero.shot), linetype="dotted") + 
    stat_smooth(geom='line', span=0.3, se = FALSE, alpha=0.5) + 
    scale_y_continuous(breaks=seq(0.4, 1, 0.2), limits = c(0.4, 1.0)) +
    scale_x_continuous(breaks=c(1, 2, 4, 6, 8),
                       labels=paste0(data.points,
                                     c(" (10%)", " (20%)", " (40%)", " (60%)", " (80%)"))) +
    stat_summary(data = task.res,
                 aes(x=as.numeric((subset))+0.4*(as.numeric((model)))/num.models-0.2*(num.models+1)/num.models,
                     y = auc, col=model), 
                 fun.data = mean_se, geom = "errorbar", width = 0.2) +
    stat_summary(data = task.res, 
                 aes(x=as.numeric((subset))+0.4*(as.numeric((model)))/num.models-0.2*(num.models+1)/num.models,
                     y = auc, col=model), 
                 fun.data = mean_se, geom = "point") +
    theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1),
                       legend.position="bottom", 
                       legend.direction="horizontal") +
    ggtitle(paste0(task)) + ggeasy::easy_center_title() + xlab("training data size, format: GoF | LoF (%)")
  if (i != 5) {
    p <- p + guides(color=FALSE)
  }
  plots[[i]] <- p
}
library(ggpubr)
p <- ggarrange(plots[[6]], plots[[5]], plots[[3]], 
               plots[[2]], plots[[8]], plots[[7]], 
               plots[[9]], plots[[1]], plots[[4]], 
          ncol=3, nrow=3, common.legend = TRUE, legend="bottom")

# plot weighted average
summary.df <- read.csv('figs/fig.5e.prepare.csv', row.names = 1)
summary.df <- summary.df[summary.df$model %in% c('PreMode.transfer', 'random.forest'),]
model.dic <- c("PreMode.transfer"="PreMode", 
               "random.forest"="Random Forest")
summary.df$model <- model.dic[summary.df$model]
summary.df$model <- factor(summary.df$model, levels = c("PreMode",  "Random Forest"))
# plot the task weighted averages as well as task size weighted error bars
uniq.result.plot <- summary.df[summary.df$seed==0,]
for (i in 1:dim(uniq.result.plot)[1]) {
  aucs <- summary.df$auc[summary.df$model==uniq.result.plot$model[i] & 
                           summary.df$gene==uniq.result.plot$gene[i] &
                           summary.df$subset==uniq.result.plot$subset[i]]
  # aucs <- aucs[aucs > 0]
  uniq.result.plot$auc[i] = mean(aucs, na.rm=T)
  uniq.result.plot$auc.se[i] = sd(aucs, na.rm=T) / sqrt(length(aucs))
}
task.dic <- unique(uniq.result.plot$gene)
plots <- list()
num.models <- unique(uniq.result.plot$model)
library(patchwork)
for (i in 1:length(task.dic)) {
  task <- (genes)[i]
  task.res <- uniq.result.plot[uniq.result.plot$gene == gene.names[i],]
  task.res <- task.res[,!is.na(task.res[1,])]
  data.points <- paste0(task.res$ngof.train[task.res$seed==0 & task.res$model=="PreMode"],
                        " | ",
                        task.res$nlof.train[task.res$seed==0 & task.res$model=="PreMode"])
  task.plots <- list()
  p <- ggplot(task.res, aes(x=subset, y=auc, col=model)) + 
    geom_point() + 
    geom_errorbar(aes(ymin=auc-auc.se, ymax=auc+auc.se), width=.4) +
    # geom_line(aes(y=zero.shot), linetype="dotted") + 
    geom_line() + 
    scale_y_continuous(breaks=seq(0.4, 1, 0.2), limits = c(0.4, 1.0)) +
    scale_x_continuous(breaks=c(1, 2, 4, 6, 8),
                       labels=paste0(data.points,
                                     c(" (10%)", " (20%)", " (40%)", " (60%)", " (80%)"))) +
    ylab('Spearman rho') +
    theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ggtitle(paste0(task)) + ggeasy::easy_center_title() + xlab("training data size (%)")
  plots[[i]] <- p
}
library(patchwork)
p <- ggarrange(plots[[6]], plots[[5]], plots[[3]], 
               plots[[2]], plots[[8]], plots[[7]], 
               plots[[9]], plots[[1]], plots[[4]], 
               ncol=3, nrow=3, common.legend = TRUE, legend="bottom")

# aggregate across models
uniq.model.result.plot <- uniq.result.plot[!duplicated(uniq.result.plot[,c('model', "subset")]),]
for (i in 1:dim(uniq.model.result.plot)[1]) {
  aucs <- uniq.result.plot$auc[uniq.result.plot$model == uniq.model.result.plot$model[i] &
                                 uniq.result.plot$subset == uniq.model.result.plot$subset[i]]
  auc.ses <- uniq.result.plot$auc.se[uniq.result.plot$model == uniq.model.result.plot$model[i] &
                                       uniq.result.plot$subset == uniq.model.result.plot$subset[i]]
  model.gene.names <- gsub(":.*", "", uniq.result.plot$gene[uniq.result.plot$model == uniq.model.result.plot$model[i] &
                                                        uniq.result.plot$subset == uniq.model.result.plot$subset[i]])
  subsets <- uniq.result.plot$subset[uniq.result.plot$model == uniq.model.result.plot$model[i] &
                                       uniq.result.plot$subset == uniq.model.result.plot$subset[i]]
  # get data set sizes
  ngof <- summary.df$ngof.train[summary.df$seed==0 & 
                                  summary.df$model=="PreMode" & 
                                  summary.df$subset == uniq.model.result.plot$subset[i]]
  nlof <- summary.df$nlof.train[summary.df$seed==0 & 
                                  summary.df$model=="PreMode" & 
                                  summary.df$subset == uniq.model.result.plot$subset[i]]
  data.points <- 1 / (1/ngof + 1/nlof)
  gene.ids <- genes[match(model.gene.names, gene.names)]
  # use harmonic prior of data points
  uniq.model.result.plot$auc[i] <- sum(aucs * data.points, na.rm = T) / sum(data.points)
  uniq.model.result.plot$auc.se[i] <- sum(auc.ses * data.points, na.rm = T) / sum(data.points)
}
p <- ggplot(uniq.model.result.plot, aes(x=subset, y=auc, col=model)) +
  geom_point() +
  geom_errorbar(aes(ymin=auc-auc.se, ymax=auc+auc.se), width=.2) +
  geom_line() + 
  scale_y_continuous(breaks=seq(0.4, 1, 0.2), limits = c(0.4, 1.0)) +
  scale_x_continuous(breaks=c(1, 2, 4, 6, 8),
                     labels=paste0(c(" (10%)", " (20%)", " (40%)", " (60%)", " (80%)"))) +
  ylab('AUC') +
  theme_bw() +
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        text = element_text(size = 16),
        plot.title = element_text(size=15),
        legend.text = element_text(size=10),
        axis.title.x = element_text(size=12),
        legend.position="bottom", 
        legend.direction="horizontal") +
  ggtitle("Weighted Average of Model AUC\non subsample of training") +
  ggeasy::easy_center_title() + xlab("training data size (% of full G/LoF dataset)")
ggsave('figs/fig.5e.pdf', p, width = 4, height = 5)

