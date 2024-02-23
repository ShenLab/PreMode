# idea:
# test the distances of training / testing points in each split at pre-train
# High AUC split should be related to low distances from training -> testing
# load pretrain logits
genes <- c("Q09428", "P15056", "O00555", "P21802",
           "Q14654", "P07949", "Q99250", "Q14524", "P04637")
gene.names <- c("ABCC8", "BRAF", "CACNA1A", "FGFR2",
                "KCNJ11", "RET", "SCN2A", "SCN5A", "TP53")
summary.df <- data.frame()
plots <- list()
for (i in 1:length(genes)) {
  gene <- genes[i]
  gene.emb <- read.csv(paste0('5genes.all.mut/CHPs.v4.esm.torchmdnet.small.TriAttn.StarPool.1dim/', 
                              gene, '.pretrain.csv'))
  aucs <- c()
  for (fold in 0:4) {
    gene.test.res <- read.csv(paste0('PreMode.inference/', gene, '/testing.fold.' ,fold, '.csv'))
    source('~/Pipeline/AUROC.R')
    auc <- plot.AUC(gene.test.res$score, gene.test.res$logits)
    aucs <- c(aucs, auc$auc)
  }
  train.config <- yaml::read_yaml(paste0('/share/pascal/Users/gz2294/PreMode/scripts/CHPs.v4.esm.dssp.small.StarAttn.MSA.StarPool.1dim/',
                                         gene, '.5fold/', gene, '.fold.', fold, '.yaml'))
  gene.train <- read.csv(paste0('/share/pascal/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', fold, '/',
                                gene, '/training.csv'))
  gene.test <- read.csv(paste0('/share/pascal/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', fold, '/',
                               gene, '/testing.csv'))
  gene.train.emb <- gene.emb[match(gene.train$aaChg, gene.emb$aaChg), colnames(gene.emb)[startsWith(colnames(gene.emb), 'X')]]
  gene.test.emb <- read.csv(paste0('PreMode.inference/', gene, '/testing.pretrain.fold.' ,fold, '.csv'))
  gene.test.emb <- gene.test.emb[, colnames(gene.test.emb)[startsWith(colnames(gene.test.emb), 'X')]]
  gene.test.emb$X <- NULL
  # gene.test.emb[is.na(gene.test.emb$X0),] <- 0.001
  # get train val split
  library(reticulate)
  np <- import('numpy')
  train.val.split <- np$load(paste0(train.config$log_dir, 'splits.0.npz'))
  gene.val.emb <- gene.train.emb[train.val.split['idx_val']+1,]
  gene.train.emb <- gene.train.emb[train.val.split['idx_train']+1,]
  gene.train.label <- gene.train[train.val.split['idx_train']+1,]
  gene.val.label <- gene.train[train.val.split['idx_val']+1,]
  # write train and test emb to files
  train.emb.file <- tempfile()
  train.label.file <- tempfile()
  test.emb.file <- tempfile()
  test.label.file <- tempfile()
  
  write.csv(gene.train.emb[!is.na(gene.train.emb$X0),], file = train.emb.file)
  write.csv(gene.test.emb, file = test.emb.file)
  write.csv(gene.train.label[!is.na(gene.train.emb$X0),], file = train.label.file)
  write.csv(gene.test, file = test.label.file)
  
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
  # visualize embedding on umap
  labels <- data.frame(split=c(rep("train", sum(!is.na(gene.train.emb$X0))),
                               rep("test", dim(gene.test.emb)[1])),
                       pos=c(gene.train$pos.orig[train.val.split['idx_train']+1][!is.na(gene.train.emb$X0)],
                             gene.test$pos.orig),
                       label=c(gene.train$score[train.val.split['idx_train']+1][!is.na(gene.train.emb$X0)],
                               gene.test$score))
  source('~/Pipeline/cluster.leiden.R')
  gene.train.emb <- np$load(paste0(train.transform.file, '.npy'))
  gene.test.emb <- np$load(paste0(test.transform.file, '.npy'))
  emb.clust <- as.data.frame(rbind(t(gene.train.emb[1:2,]), t(gene.test.emb[1:2,])))
  colnames(emb.clust) <- c('PC1', 'PC2')
  # perform k-means clustering
  knn.clust <- class::knn(t(gene.train.emb), t(gene.test.emb), cl=labels$label[labels$split=='train'], k=5)
  # calculate precision
  tab <- table(knn.clust, labels$label[labels$split=='test'])
  accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
  accuracy(tab)
  # emb.clust <- cluster.leiden(rbind(gene.train.emb, gene.test.emb, gene.val.emb))
  emb.clust$split <- labels$split
  emb.clust$label <- 'GoF'
  emb.clust$label[labels$label==-1] <- 'LoF'
  emb.clust$pos <- labels$pos
  # do mds on cosine similarity
  # mds <- as.matrix(rbind(gene.train.emb, gene.test.emb, gene.val.emb))
  # mds <- mds / sqrt(rowSums(mds * mds))
  # mds <- mds %*% t(mds)
  # mds <- as.dist(1 - mds)
  # mds <- cmdscale(mds, eig=TRUE, k=2)
  # emb.clust$MDS1 <- mds$points[,1]
  # emb.clust$MDS2 <- mds$points[,2]
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
                   dist(as.matrix(emb.clust[labels$split!='test',paste0('PC', 1:2)])))
  s2 <- silhouette(as.numeric(as.factor(paste0(labels$split[labels$split!='train'], labels$label[labels$split!='train']))), 
                   dist(as.matrix(emb.clust[labels$split!='train',paste0('PC', 1:2)])))
  s3 <- silhouette(as.numeric(as.factor(paste0(labels$split[labels$split!='val'], labels$label[labels$split!='val']))), 
                   dist(as.matrix(emb.clust[labels$split!='val',paste0('PC', 1:2)])))
  library(ggplot2)
  library(patchwork)
  p1 <- ggplot(emb.clust, aes(x=PC1, y=PC2, col=label, shape=split)) +
    geom_point() + theme_bw() + ggtitle(paste0(gene.names[i])) + 
    scale_shape_manual(values=c(1, 15))+
    xlab(paste0("supervised PC1")) + 
    ylab(paste0("supervised PC2")) +
    ggeasy::easy_center_title()
  if (gene=='P15056') {
    emb.clust$domain[emb.clust$pos==600] <- '600'
    emb.clust$domain[emb.clust$pos%in%c(597)] <- '597'
    emb.clust$domain[emb.clust$pos%in%c(598)] <- '598'
    emb.clust$domain[emb.clust$pos%in%c(599)] <- '599'
  }
  p2 <- ggplot(emb.clust, aes(x=PC1, y=PC2, col=domain)) +
    geom_point() + theme_bw() + 
    xlab(paste0("supervised PC1")) + 
    ylab(paste0("supervised PC2")) +
    ggtitle(paste0('PreMode AUC = ', round(mean(aucs), 2), ', SD = ', round(sd(aucs), 2))) + 
    ggeasy::easy_center_title()
  plots[[i]] <- p1 + p2 + plot_layout(ncol = 2)
  # summary.df <- rbind(summary.df,
  #                     data.frame(AUC=auc$auc, s1=mean(s1[,3]), s2=mean(s2[,3]), s3=mean(s3[,3]), 
  #                                fold=fold, gene=gene, gene.name=gene.names[i]))
}
p <- plots[[1]] / plots[[2]] / plots[[3]] / plots[[4]] / plots[[5]] / plots[[6]] / plots[[7]] / plots[[8]]
ggsave(filename = 'figs/10.01.analysis.zero.shot.spca.pdf', p, height = 30, width = 12)
p <- plots[[1]] / plots[[2]] / plots[[3]] / plots[[5]] / plots[[6]] 
ggsave(filename = 'figs/10.02.analysis.zero.shot.spca.pdf', p, height = 18, width = 12)
# for (gene in genes) {
#   print(cor.test(summary.df$AUC[summary.df$gene==gene], summary.df$s2[summary.df$gene==gene], method = 'spearman'))
# }
