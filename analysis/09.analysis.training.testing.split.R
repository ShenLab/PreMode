# idea:
# test the distances of training / testing points in each split at pre-train
# High AUC split should be related to low distances from training -> testing
# load pretrain logits
genes <- c("P15056", "P07949", "P04637", 
           "Q14654", "Q14524", "Q99250", 
           "O00555",
           "P21802")
gene.names <- c("BRAF", "RET", "TP53", 
                "KCNJ11", "SCN5A", "SCN2A",
                "CACNA1A", 
                "FGFR2")
summary.df <- data.frame()
for (i in 1:length(genes)) {
  gene <- genes[i]
  gene.emb <- read.csv(paste0('5genes.all.mut/CHPs.v4.esm.torchmdnet.small.TriAttn.StarPool.1dim/', 
                              gene, '.pretrain.csv'))
  for (fold in 0:4) {
    train.config <- yaml::read_yaml(paste0('/share/pascal/Users/gz2294/PreMode/scripts/CHPs.v4.esm.dssp.small.StarAttn.MSA.StarPool.1dim/',
                                           gene, '.5fold/', gene, '.fold.', fold, '.yaml'))
    gene.test.res <- read.csv(paste0('PreMode.inference/', gene, '/testing.fold.' ,fold, '.csv'))
    gene.train <- read.csv(paste0('/share/pascal/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', fold, '/',
                                  gene, '/training.csv'))
    gene.test <- read.csv(paste0('/share/pascal/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', fold, '/',
                                 gene, '/testing.csv'))
    gene.train.emb <- gene.emb[match(gene.train$aaChg, gene.emb$aaChg), colnames(gene.emb)[startsWith(colnames(gene.emb), 'X')]]
    gene.test.emb <- gene.emb[match(gene.test$aaChg, gene.emb$aaChg), colnames(gene.emb)[startsWith(colnames(gene.emb), 'X')]]
    # get train val split
    library(reticulate)
    np <- import('numpy')
    train.val.split <- np$load(paste0(train.config$log_dir, 'splits.0.npz'))
    gene.val.emb <- gene.train.emb[train.val.split['idx_val']+1,]
    gene.train.emb <- gene.train.emb[train.val.split['idx_train']+1,]
    # visualize embedding on umap
    labels <- data.frame(split=c(rep("train", dim(gene.train.emb)[1]),
                                 rep("test", dim(gene.test.emb)[1]),
                                 rep("val", dim(gene.val.emb)[1])),
                         pos=c(gene.train$pos.orig[train.val.split['idx_train']+1],
                               gene.test$pos.orig,
                               gene.train$pos.orig[train.val.split['idx_val']+1]),
                         label=c(gene.train$score[train.val.split['idx_train']+1],
                                 gene.test$score,
                                 gene.train$score[train.val.split['idx_val']+1]))
    source('~/Pipeline/cluster.leiden.R')
    source('~/Pipeline/AUROC.R')
    emb.clust <- cluster.leiden(rbind(gene.train.emb, gene.test.emb, gene.val.emb))
    emb.clust$split <- labels$split
    emb.clust$label <- labels$label
    emb.clust$pos <- labels$pos
    # do mds on cosine similarity
    mds <- as.matrix(rbind(gene.train.emb, gene.test.emb, gene.val.emb))
    mds <- mds / sqrt(rowSums(mds * mds))
    mds <- mds %*% t(mds)
    mds <- as.dist(1 - mds)
    mds <- cmdscale(mds, eig=TRUE, k=2)
    emb.clust$MDS1 <- mds$points[,1]
    emb.clust$MDS2 <- mds$points[,2]
    # get domain and zn fing information
    prot_data <- drawProteins::get_features(gene)
    prot_data <- drawProteins::feature_to_dataframe(prot_data)
    prot_data <- prot_data[prot_data$type %in% c("DOMAIN", "ZN_FING"),]
    emb.clust$domain <- NA
    for (k in 1:dim(prot_data)[1]) {
      emb.clust$domain[emb.clust$pos >= prot_data$begin[k] & emb.clust$pos <= prot_data$end[k]] <- prot_data$description[k]
    }
    # visualize
    auc <- plot.AUC(gene.test.res$score, gene.test.res$logits)
    library(cluster)
    s1 <- silhouette(as.numeric(as.factor(paste0(labels$split[labels$split!='test'], labels$label[labels$split!='test']))), 
                     dist(as.matrix(emb.clust[labels$split!='test',paste0('PC', 1:5)])))
    s2 <- silhouette(as.numeric(as.factor(paste0(labels$split[labels$split!='train'], labels$label[labels$split!='train']))), 
                     dist(as.matrix(emb.clust[labels$split!='train',paste0('PC', 1:5)])))
    s3 <- silhouette(as.numeric(as.factor(paste0(labels$split[labels$split!='val'], labels$label[labels$split!='val']))), 
                     dist(as.matrix(emb.clust[labels$split!='val',paste0('PC', 1:5)])))
    library(ggplot2)
    library(patchwork)
    p1 <- ggplot(emb.clust, aes(x=PC1, y=PC2, col=paste0(split, label))) +
      geom_point() + theme_bw() + ggtitle(paste0('AUC = ', round(auc$auc, 2)))
    p2 <- ggplot(emb.clust, aes(x=PC1, y=PC2, col=as.character(domain))) +
      geom_point() + theme_bw() + ggtitle(paste0('Sihouttee score = ',
                                                 round(mean(s1[,3]), 2), ',',
                                                 round(mean(s2[,3]), 2), ',',
                                                 round(mean(s3[,3]), 2)))
    plots[[fold+1]] <- p1 + p2 + plot_layout(ncol = 2)
    summary.df <- rbind(summary.df,
                        data.frame(AUC=auc$auc, s1=mean(s1[,3]), s2=mean(s2[,3]), s3=mean(s3[,3]), 
                                   fold=fold, gene=gene, gene.name=gene.names[i]))
  }
}
# p <- plots[[1]] / plots[[2]] / plots[[3]] / plots[[4]] / plots[[5]] 
# ggsave(filename = 'figs/09.training.testing.split.pdf', p, height = 25, width = 12)
for (gene in genes) {
  print(cor.test(summary.df$AUC[summary.df$gene==gene], summary.df$s2[summary.df$gene==gene], method = 'spearman'))
}
