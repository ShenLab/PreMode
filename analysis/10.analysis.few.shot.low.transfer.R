# idea:
# test the distances of training / testing points in each split at pre-train
# High AUC split should be related to low distances from training -> testing
# load pretrain logits
genes <- c("Q09428", "P15056", "O00555", "P21802",
           "Q14654", "P07949", "Q99250", "Q14524", "P04637")
gene.names <- c("ABCC8", "BRAF", "CACNA1A", "FGFR2",
                "KCNJ11", "RET", "SCN2A", "SCN5A", "TP53")
# low.transfer.method <- c('gp', 'elastic.net', 'spca')
source('~/Pipeline/AUROC.R')
summary.df <- data.frame()
plots <- list()
source('/share/vault/Users/gz2294/Pipeline/prepare.biochem.R')
ALL <- read.csv('figs/ALL.csv', row.names = 1, na.strings = c('.', 'NA'))
ALL$aaChg <- paste0('p.', ALL$ref, ALL$pos.orig, ALL$alt)
for (i in 1:length(genes)) {
  gene <- genes[i]
  for (subset in 1:6) {
    for (fold in 0:4) {
      aucs <- c()
      if (subset == 6) {
        gene.test.res <- read.csv(paste0('PreMode.inference/', gene, '/testing.fold.' ,fold, '.csv'))
      } else {
        gene.test.res <- read.csv(paste0('PreMode.inference/', gene, '/testing.subset.', subset, '.fold.' ,fold, '.csv'))
      }
      # gene.train.res <- read.csv(paste0('PreMode.inference/', gene, '/training.fold.' ,fold, '.csv'))
      source('~/Pipeline/AUROC.R')
      auc <- plot.AUC(gene.test.res$score, gene.test.res$logits)
      if (subset == 6) {
        gene.train <- read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', fold, '/',
                                      gene, '/training.csv'))
        gene.test <- read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', fold, '/',
                                     gene, '/testing.csv'))
        gene.train.emb.orig <- read.csv(paste0('PreMode.inference/', gene, '/training.pretrain.fold.' ,fold, '.csv'))
        gene.test.emb.orig <- read.csv(paste0('PreMode.inference/', gene, '/testing.pretrain.fold.' ,fold, '.csv'))
        gene.train.emb.orig$aaChg <- paste0('p.', gene.train.emb.orig$ref, gene.train.emb.orig$pos.orig, gene.train.emb.orig$alt)
        gene.test.emb.orig$aaChg <- paste0('p.', gene.test.emb.orig$ref, gene.test.emb.orig$pos.orig, gene.test.emb.orig$alt)
        train.config <- yaml::read_yaml(paste0('/share/vault/Users/gz2294/PreMode/scripts/CHPs.v4.esm.dssp.small.StarAttn.MSA.StarPool.1dim/',
                                               gene, '.5fold/', gene, '.fold.', fold, '.yaml'))
      } else {
        gene.train <- read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', fold, '/',
                                      gene, '.subset.', subset, '/training.csv'))
        gene.test <- read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.0/',
                                     gene, '.subset.', subset, '/testing.csv'))
        gene.train.emb.orig <- read.csv(paste0('PreMode.inference/', gene, '/training.pretrain.subset.', subset, '.fold.' ,fold, '.csv'))
        gene.test.emb.orig <- read.csv(paste0('PreMode.inference/', gene, '/testing.pretrain.subset.', subset, '.fold.', fold, '.csv'))
        gene.train.emb.orig$aaChg <- paste0('p.', gene.train.emb.orig$ref, gene.train.emb.orig$pos.orig, gene.train.emb.orig$alt)
        gene.test.emb.orig$aaChg <- paste0('p.', gene.test.emb.orig$ref, gene.test.emb.orig$pos.orig, gene.test.emb.orig$alt)
        train.config <- yaml::read_yaml(paste0('/share/vault/Users/gz2294/PreMode/scripts/CHPs.v4.esm.dssp.small.StarAttn.MSA.StarPool.1dim/',
                                               gene, '.subset.', subset, '.5fold/', gene, '.subset.', subset, '.fold.', fold, '.yaml'))
      }
      gene.train.emb <- gene.train.emb.orig[,colnames(gene.train.emb.orig)[startsWith(colnames(gene.train.emb.orig), 'X')]]
      gene.train.emb$X <- NULL
      gene.train.emb$X.1 <- NULL
      gene.test.emb <- gene.test.emb.orig[,colnames(gene.test.emb.orig)[startsWith(colnames(gene.test.emb.orig), 'X')]]
      gene.test.emb$X <- NULL
      gene.test.emb$X.1 <- NULL
      # gene.test.emb[is.na(gene.test.emb$X0),] <- 0.001
      # get train val split
      library(reticulate)
      np <- import('numpy')
      if (subset == 6) {
        train.val.split <- np$load(paste0(train.config$log_dir, 'splits.0.npz'))
        gene.val.emb <- gene.train.emb[train.val.split['idx_val']+1,]
        gene.train.emb <- gene.train.emb[train.val.split['idx_train']+1,]
        gene.train.emb.orig <- gene.train.emb.orig[train.val.split['idx_train']+1,]
        gene.train.label <- gene.train[train.val.split['idx_train']+1,]
        gene.val.label <- gene.train[train.val.split['idx_val']+1,]
      } else {
        gene.val.emb <- gene.train.emb[gene.train$split=='val',]
        gene.train.emb <- gene.train.emb[gene.train$split=='train',]
        gene.train.emb.orig <- gene.train.emb.orig[gene.train$split=='train',]
        gene.train.label <- gene.train[gene.train$split=='train',]
        gene.val.label <- gene.train[gene.train$split=='val',]
      }
      # write train and test emb to files
      train.emb.file <- tempfile()
      train.label.file <- tempfile()
      test.emb.file <- tempfile()
      test.label.file <- tempfile()
      
      write.csv(gene.train.emb, file = train.emb.file)
      write.csv(gene.test.emb, file = test.emb.file)
      write.csv(gene.train.label, file = train.label.file)
      write.csv(gene.test, file = test.label.file)
      # visualize embedding on umap
      labels <- data.frame(split=c(rep("train", dim(gene.train.emb)[1]),
                                   rep("val", dim(gene.val.emb)[1]),
                                   rep("test", dim(gene.test.emb)[1])),
                           pos=c(gene.train.label$pos.orig,
                                 gene.val.label$pos.orig,
                                 gene.test$pos.orig),
                           label=c(gene.train.label$score,
                                   gene.val.label$score,
                                   gene.test$score))
      aucs <- c(aucs, auc$auc)
      # call python on gp process
      res <- system(paste0('/share/descartes/Users/gz2294/miniconda3/envs/RESCVE/bin/python ', 
                           '10.analysis.few.shot.gp.classifier.py ', 
                           train.emb.file, ' ',
                           train.label.file, ' ',
                           test.emb.file, ' ', 
                           test.label.file), intern = T)
      aucs <- c(aucs, as.numeric(strsplit(res, split = '=')[[1]][2]))
      # call python on elastic net
      res <- system(paste0('/share/descartes/Users/gz2294/miniconda3/envs/RESCVE/bin/python ', 
                           '10.analysis.few.shot.elastic.net.py ', 
                           train.emb.file, ' ',
                           train.label.file, ' ',
                           test.emb.file, ' ', 
                           test.label.file), intern = T)
      aucs <- c(aucs, as.numeric(strsplit(res, split = '=')[[1]][2]))
      # call python on random forest
      res <- system(paste0('/share/descartes/Users/gz2294/miniconda3/envs/RESCVE/bin/python ', 
                           '10.analysis.few.shot.random.forest.py ', 
                           train.emb.file, ' ',
                           train.label.file, ' ',
                           test.emb.file, ' ', 
                           test.label.file), intern = T)
      aucs <- c(aucs, as.numeric(strsplit(res, split = '=')[[1]][2]))
      # call python on elastic net
      train.biochem.file <- tempfile()
      test.biochem.file <- tempfile()
      write.csv(prepare.biochemical(ALL[match(gene.train.emb.orig$aaChg, ALL$aaChg),]),
                file = train.biochem.file)
      write.csv(prepare.biochemical(ALL[match(gene.test.emb.orig$aaChg, ALL$aaChg),]), file = test.biochem.file)
      res <- system(paste0('/share/descartes/Users/gz2294/miniconda3/envs/RESCVE/bin/python ', 
                           '10.analysis.few.shot.elastic.net.py ', 
                           train.biochem.file, ' ',
                           train.label.file, ' ',
                           test.biochem.file, ' ', 
                           test.label.file), intern = T)
      aucs <- c(aucs, as.numeric(strsplit(res, split = '=')[[1]][2]))
      # call python on random forest
      res <- system(paste0('/share/descartes/Users/gz2294/miniconda3/envs/RESCVE/bin/python ', 
                           '10.analysis.few.shot.random.forest.py ', 
                           train.biochem.file, ' ',
                           train.label.file, ' ',
                           test.biochem.file, ' ', 
                           test.label.file), intern = T)
      aucs <- c(aucs, as.numeric(strsplit(res, split = '=')[[1]][2]))
      # call python on spca
      train.transform.file <- tempfile()
      test.transform.file <- tempfile()
      system(paste0('/share/descartes/Users/gz2294/miniconda3/envs/RESCVE/bin/python ', 
                    '10.analysis.few.shot.supervised.pca.py ', 
                    train.emb.file, ' ',
                    train.label.file, ' ',
                    test.emb.file, ' ', 
                    test.label.file, ' ',
                    train.transform.file, ' ',
                    test.transform.file))
      gene.train.spca.emb <- np$load(paste0(train.transform.file, '.npy'))
      gene.test.spca.emb <- np$load(paste0(test.transform.file, '.npy'))
      # calculate AUC
      aucs <- c(aucs, plot.AUC(labels$label[labels$split=='test'], gene.test.spca.emb[1,])$auc)
      source('~/Pipeline/cluster.leiden.R')
      emb.clust <- cluster.leiden(rbind(gene.train.emb, gene.val.emb, gene.test.emb))
      aucs <- c(aucs, plot.AUC(labels$label[labels$split=='test'], emb.clust$PC1[labels$split=='test'])$auc)
      # do mds on cosine similarity
      mds <- as.matrix(rbind(gene.train.emb, gene.val.emb, gene.test.emb))
      mds <- mds / sqrt(rowSums(mds * mds))
      mds <- mds %*% t(mds)
      mds <- as.dist(1 - mds)
      mds <- cmdscale(mds, eig=TRUE, k=2)
      aucs <- c(aucs, plot.AUC(labels$label[labels$split=='test'], mds$points[,1][labels$split=='test'])$auc)
      
      aucs[aucs <= 0.5] <- 1-aucs[aucs <= 0.5]
      summary.df <- rbind(summary.df, 
                          data.frame(auc=aucs,
                                     model=c('PreMode.transfer', 'gp', 'elastic.net.1', 'random.forest.1', 'elastic.net.2', 'random.forest.2', 'spca', 'pca', 'mds'),
                                     seed=fold,
                                     gene=gene.names[i],
                                     subset=subset,
                                     ngof.train=sum(labels$label[labels$split=='train']==1),
                                     nlof.train=sum(labels$label[labels$split=='train']==-1),
                                     ngof.test=sum(labels$label[labels$split=='test']==1),
                                     nlof.test=sum(labels$label[labels$split=='test']==-1)))
    }
  }
}
write.csv(summary.df, file = 'figs/10.few.shot.low.transfer.csv')
library(ggplot2)


summary.df <- read.csv('figs/10.few.shot.low.transfer.csv', row.names = 1)
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
    scale_x_continuous(breaks=c(1, 2, 3, 4, 5, 6),
                       labels=paste0(data.points,
                                     c(" (10%)", " (20%)", " (30%)", " (40%)", " (50%)", " (60%)"))) +
    stat_summary(data = task.res,
                 aes(x=as.numeric(factor(subset))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                     y = auc, col=model), 
                 fun.data = mean_se, geom = "errorbar", width = 0.2) +
    stat_summary(data = task.res, 
                 aes(x=as.numeric(factor(subset))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                     y = auc, col=model), 
                 fun.data = mean_se, geom = "point") +
    theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ggtitle(paste0(task)) + ggeasy::easy_center_title() + xlab("training data size, format: GoF | LoF (%)")
  plots[[i]] <- p
}
library(patchwork)
p <- plots[[1]] + plots[[2]] + plots[[3]] + plots[[4]] + plots[[5]] + plots[[6]] + plots[[7]] + plots[[8]] + plots[[9]] + plot_layout(ncol=3)
ggsave(paste0('figs/10.few.shot.low.transfer.pdf'), p, height = 15, width = 18)




summary.df <- read.csv('figs/10.few.shot.low.transfer.csv', row.names = 1)
summary.df <- summary.df[summary.df$model %in% c('PreMode.transfer', 'pca', 'spca'),]
model.dic <- c("PreMode.transfer"="Supervised: PreMode", 
               "pca"="Unsupervised: PCA", 
               "spca"="Semi-Supervised: sPCA")
summary.df$model <- model.dic[summary.df$model]
summary.df$model <- factor(summary.df$model, levels = c("Supervised: PreMode", 
                                                        "Semi-Supervised: sPCA", 
                                                        "Unsupervised: PCA"))
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
    geom_point(alpha=0.2) + 
    # geom_line(aes(y=zero.shot), linetype="dotted") + 
    stat_smooth(geom='line', span=0.3, se = FALSE, alpha=0.5) + scale_y_continuous(breaks=seq(0.4, 1, 0.2), limits = c(0.4, 1.0)) +
    scale_x_continuous(breaks=c(1, 2, 3, 4, 5, 6),
                       labels=paste0(data.points,
                                     c(" (10%)", " (20%)", " (30%)", " (40%)", " (50%)", " (60%)"))) +
    stat_summary(data = task.res,
                 aes(x=as.numeric(factor(subset))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                     y = auc, col=model), 
                 fun.data = mean_se, geom = "errorbar", width = 0.2) +
    stat_summary(data = task.res, 
                 aes(x=as.numeric(factor(subset))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                     y = auc, col=model), 
                 fun.data = mean_se, geom = "point") +
    theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ggtitle(paste0(task)) + ggeasy::easy_center_title() + xlab("training data size, format: GoF | LoF (%)")
  plots[[i]] <- p
}
library(patchwork)
p <- plots[[1]] + plots[[2]] + plots[[3]] + plots[[4]] + plots[[5]] + plots[[6]] + plots[[7]] + plots[[8]] + plots[[9]] + plot_layout(ncol=3)
ggsave(paste0('figs/10.few.shot.low.transfer.1.pdf'), p, height = 15, width = 18)




summary.df <- read.csv('figs/10.few.shot.low.transfer.csv', row.names = 1)
summary.df <- summary.df[summary.df$model %in% c('PreMode.transfer', 'elastic.net.1', 'random.forest.1'),]
model.dic <- c("PreMode.transfer"="Supervised: PreMode", 
               "elastic.net.1"="Supervised: Elastic Net",
               "random.forest.1"="Supervised: Random Forest")
summary.df$model <- model.dic[summary.df$model]
summary.df$model <- factor(summary.df$model, levels = c("Supervised: PreMode", 
                                                        "Supervised: Elastic Net",
                                                        "Supervised: Random Forest"))
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
    geom_point(alpha=0.2) + 
    # geom_line(aes(y=zero.shot), linetype="dotted") + 
    stat_smooth(geom='line', span=0.3, se = FALSE, alpha=0.5) + scale_y_continuous(breaks=seq(0.4, 1, 0.2), limits = c(0.4, 1.0)) +
    scale_x_continuous(breaks=c(1, 2, 3, 4, 5, 6),
                       labels=paste0(data.points,
                                     c(" (10%)", " (20%)", " (30%)", " (40%)", " (50%)", " (60%)"))) +
    stat_summary(data = task.res,
                 aes(x=as.numeric(factor(subset))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                     y = auc, col=model), 
                 fun.data = mean_se, geom = "errorbar", width = 0.2) +
    stat_summary(data = task.res, 
                 aes(x=as.numeric(factor(subset))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                     y = auc, col=model), 
                 fun.data = mean_se, geom = "point") +
    theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ggtitle(paste0(task)) + ggeasy::easy_center_title() + xlab("training data size, format: GoF | LoF (%)")
  plots[[i]] <- p
}
library(patchwork)
p <- plots[[1]] + plots[[2]] + plots[[3]] + plots[[4]] + plots[[5]] + plots[[6]] + plots[[7]] + plots[[8]] + plots[[9]] + plot_layout(ncol=3)
ggsave(paste0('figs/10.few.shot.low.transfer.2.pdf'), p, height = 15, width = 18)


summary.df <- read.csv('figs/10.few.shot.low.transfer.csv', row.names = 1)
summary.df <- summary.df[summary.df$model %in% c('PreMode.transfer', 'random.forest.2'),]
model.dic <- c("PreMode.transfer"="Supervised: PreMode", 
               "random.forest.2"="Supervised: Random Forest")
summary.df$model <- model.dic[summary.df$model]
summary.df$model <- factor(summary.df$model, levels = c("Supervised: PreMode", 
                                                        "Supervised: Random Forest"))
summary.df <- summary.df[!summary.df$gene %in% c('SCN2A', "FGFR2", "SCN5A"),]
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
    geom_point(alpha=0.2) + 
    # geom_line(aes(y=zero.shot), linetype="dotted") + 
    stat_smooth(geom='line', span=0.3, se = FALSE, alpha=0.5) + scale_y_continuous(breaks=seq(0.4, 1, 0.2), limits = c(0.4, 1.0)) +
    scale_x_continuous(breaks=c(1, 2, 3, 4, 5, 6),
                       labels=paste0(data.points,
                                     c(" (10%)", " (20%)", " (30%)", " (40%)", " (50%)", " (60%)"))) +
    stat_summary(data = task.res,
                 aes(x=as.numeric(factor(subset))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                     y = auc, col=model), 
                 fun.data = mean_se, geom = "errorbar", width = 0.2) +
    stat_summary(data = task.res, 
                 aes(x=as.numeric(factor(subset))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
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
p <- ggarrange(plots[[1]], plots[[2]], plots[[3]], plots[[4]], plots[[5]], plots[[6]],
          ncol=3, nrow=2, common.legend = TRUE, legend="bottom")
# p <- plots[[1]] + plots[[2]] + plots[[3]] + plots[[4]] + plots[[5]] + plots[[6]] + plot_layout(ncol=3)
ggsave(paste0('figs/10.few.shot.low.transfer.3.pdf'), p, height = 6, width = 10)


