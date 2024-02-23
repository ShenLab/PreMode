# idea:
# test the distances of training / testing points in each split at pre-train
# High AUC split should be related to low distances from training -> testing
# load pretrain logits
genes <- c("Q09428", "P15056", "O00555", "P21802",
           "Q14654", "P07949", "Q99250", "Q14524", "P04637")
gene.names <- c("ABCC8", "BRAF", "CACNA1A", "FGFR2",
                "KCNJ11", "RET", "SCN2A", "SCN5A", "TP53")
args <- commandArgs(trailingOnly = T)
result.dir <- args[1]
summary.df <- data.frame()
plots <- list()
plots.conn <- list()
for (i in 1:length(genes)) {
  gene <- genes[i]
  # gene.emb <- read.csv(paste0('5genes.all.mut/CHPs.v4.esm.torchmdnet.small.TriAttn.StarPool.1dim/', 
  #                             gene, '.pretrain.csv'))
  aucs <- c()
  for (fold in 0:4) {
    gene.test.res <- read.csv(paste0(result.dir, '/', gene, '/testing.fold.' ,fold, '.csv'))
    source('~/Pipeline/AUROC.R')
    auc <- plot.AUC(gene.test.res$score, gene.test.res$logits)
    aucs <- c(aucs, auc$auc)
  }
  gene.train <- read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', fold, '/',
                                gene, '/training.csv'))
  gene.test <- read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', fold, '/',
                               gene, '/testing.csv'))
  gene.train.emb.orig <- read.csv(paste0(result.dir, '/', gene, '/training.pretrain.fold.' ,fold, '.csv'))
  gene.test.emb.orig <- read.csv(paste0(result.dir, '/', gene, '/testing.pretrain.fold.' ,fold, '.csv'))
  
  gene.train.emb.orig$aaChg <- paste0('p.', gene.train.emb.orig$ref, gene.train.emb.orig$pos.orig, gene.train.emb.orig$alt)
  gene.test.emb.orig$aaChg <- paste0('p.', gene.test.emb.orig$ref, gene.test.emb.orig$pos.orig, gene.test.emb.orig$alt)
  train.config <- yaml::read_yaml(paste0('/share/vault/Users/gz2294/PreMode/scripts/CHPs.v4.esm.dssp.small.StarAttn.MSA.StarPool.1dim/',
                                         gene, '.5fold/', gene, '.fold.', fold, '.yaml'))
  
  gene.train.emb <- gene.train.emb.orig[,colnames(gene.train.emb.orig)[startsWith(colnames(gene.train.emb.orig), 'X')]]
  gene.train.emb$X <- NULL
  gene.train.emb$X.1 <- NULL
  gene.test.emb <- gene.test.emb.orig[,colnames(gene.test.emb.orig)[startsWith(colnames(gene.test.emb.orig), 'X')]]
  gene.test.emb$X <- NULL
  gene.test.emb$X.1 <- NULL
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
                       aaChg=c(gene.train.emb.orig$aaChg[train.val.split['idx_train']+1],
                               gene.test.emb.orig$aaChg,
                               gene.train.emb.orig$aaChg[train.val.split['idx_val']+1]),
                       pos=c(gene.train$pos.orig[train.val.split['idx_train']+1],
                             gene.test$pos.orig,
                             gene.train$pos.orig[train.val.split['idx_val']+1]),
                       label=c(gene.train$score[train.val.split['idx_train']+1],
                               gene.test$score,
                               gene.train$score[train.val.split['idx_val']+1]))
  source('/share/vault/Users/gz2294/Pipeline/cluster.leiden.R')
  
  # emb.clust <- cluster.leiden(rbind(gene.train.emb, gene.test.emb, gene.val.emb))
  rownames(gene.all.emb) <- paste0('X', 1:dim(gene.all.emb)[1])
  emb.clust <- cluster.leiden(gene.all.emb)
  emb.clust$split <- labels$split
  emb.clust$label <- 'GoF'
  emb.clust$label[labels$label==-1] <- 'LoF'
  emb.clust$pos <- labels$pos
  # do mds on cosine similarity
  # mds <- as.matrix(rbind(gene.train.emb, gene.test.emb, gene.val.emb))
  mds <- as.matrix(gene.all.emb)
  mds <- mds / sqrt(rowSums(mds * mds))
  mds <- mds %*% t(mds)
  mds <- as.dist(1 - mds)
  mds <- cmdscale(mds, eig=TRUE, k=2)
  emb.clust$MDS1 <- mds$points[,1]
  emb.clust$MDS2 <- mds$points[,2]
  # get connectivity
  # emb.clust$connectivity <- calculate.connectivity(rbind(gene.train.emb, gene.test.emb, gene.val.emb))
  emb.clust$connectivity <- calculate.connectivity(gene.all.emb)
  # get domain and zn fing information
  prot_data <- drawProteins::get_features(gene)
  prot_data <- drawProteins::feature_to_dataframe(prot_data)
  if (gene %in% c("Q14654", "Q14524", "Q99250", "O00555")) {
    prot_data <- prot_data[prot_data$type %in% c("TOPO_DOM", "TRANSMEM"),]
    prot_data$description <- gsub("\\s+of\\s+.*$", "", prot_data$description)
  } else {
    prot_data$description[prot_data$type=="DNA_BIND"] <- "DNA_BIND"
    prot_data <- prot_data[prot_data$type %in% c("DOMAIN", "ZN_FING", "CROSSLNK", "DNA_BIND"),]
    prot_data$description[prot_data$description == "Glycyl lysine isopeptide (Lys-Gly) (interchain with G-Cter in ubiquitin)"] <- "crosslink: ubiquitin"
  }
  emb.clust$domain <- NA
  for (k in 1:dim(prot_data)[1]) {
    emb.clust$domain[emb.clust$pos >= prot_data$begin[k] & emb.clust$pos <= prot_data$end[k]] <- prot_data$description[k]
  }
  if (sum(is.na(emb.clust$domain)) == dim(emb.clust)[1]) {
    emb.clust$domain <- ""
  }
  # visualize
  library(cluster)
  s0 <- silhouette(as.numeric(as.factor(labels$label)), 
                   dist(as.matrix(emb.clust[,paste0('PC', 1:2)])))
  s1 <- silhouette(as.numeric(as.factor(paste0(labels$split[labels$split!='test'], labels$label[labels$split!='test']))), 
                   dist(as.matrix(emb.clust[labels$split!='test',paste0('PC', 1:5)])))
  s2 <- silhouette(as.numeric(as.factor(paste0(labels$split[labels$split!='train'], labels$label[labels$split!='train']))), 
                   dist(as.matrix(emb.clust[labels$split!='train',paste0('PC', 1:5)])))
  s3 <- silhouette(as.numeric(as.factor(paste0(labels$split[labels$split!='val'], labels$label[labels$split!='val']))), 
                   dist(as.matrix(emb.clust[labels$split!='val',paste0('PC', 1:5)])))
  library(ggplot2)
  library(patchwork)
  p1 <- ggplot(emb.clust, aes(x=PC1, y=PC2, col=label, shape=split)) +
    geom_point() + theme_bw() + ggtitle(paste0(gene.names[i])) + 
    scale_shape_manual(values=c(1, 15, 12))+
    ggeasy::easy_center_title()
  p2 <- ggplot(emb.clust, aes(x=PC1, y=PC2, col=domain)) +
    geom_point() + theme_bw() + 
    xlab(paste0("PC1: ", round(emb.clust$PC1.importance[1], 3))) + 
    ylab(paste0("PC2: ", round(emb.clust$PC2.importance[1], 3))) +
    ggtitle(paste0('AUC = ', round(mean(aucs), 2), ', SD = ', round(sd(aucs), 2))) + 
    ggeasy::easy_center_title()
  p3 <- ggplot(emb.clust, aes(x=PC1, y=PC2, col=connectivity/10, shape=split)) +
    geom_point() + theme_bw() + ggtitle(paste0(gene.names[i])) + 
    scale_shape_manual(values=c(1, 15, 12))+
    scale_color_gradient2(c('yellow', 'purple', 'black')) +
    ggeasy::easy_center_title()
  plots[[i]] <- p1 + p2 + plot_layout(ncol = 2)
  plots.conn[[i]] <- p3 + p2 + plot_layout(ncol = 2)
}
dir.create(paste0(result.dir, '/figs/'))
p <- plots[[1]] / plots[[2]] / plots[[3]] / plots[[4]] / plots[[5]] / plots[[6]] / plots[[7]] / plots[[8]]
ggsave(filename = paste0(result.dir, '/figs/10.01.analysis.zero.shot.pdf'), p, height = 30, width = 12)
p <- plots[[1]] / plots[[2]] / plots[[3]] / plots[[5]] / plots[[6]] 
ggsave(filename = 'figs/10.02.analysis.zero.shot.pdf', p, height = 18, width = 12)
p <- plots.conn[[1]] / plots.conn[[2]] / plots.conn[[3]] / plots.conn[[4]] / plots.conn[[5]] / plots.conn[[6]] / plots.conn[[7]] / plots.conn[[8]]
ggsave(filename = paste0(result.dir, '/figs/10.03.analysis.zero.shot.pdf'), p, height = 30, width = 12)

