setwd('/share/terra/Users/gz2294/RESCVE/')
# args <- commandArgs(trailingOnly = T)
# configs <- yaml::read_yaml(args[1])
# # configs <- yaml::read_yaml('scripts/CHPs.v1.good.batch.ct.yaml')
# log.dir <- configs$log_dir
# 
# embeddings.raw <- read.csv(paste0(log.dir, 'x_embeds.csv'), row.names = 1)
embeddings.raw <- read.csv(paste0('~/Data/DMS/ClinVar/testing.csv.inference.radius.csv'), 
                           row.names = 1)

embeddings <- embeddings.raw[, startsWith(colnames(embeddings.raw), "X")]
source('/share/terra/Users/gz2294/Pipeline/cluster.leiden.R')

source('/share/terra/Users/gz2294/Pipeline/AUROC.R')
if (0 %in% embeddings.raw$score & -1 %in% embeddings.raw$score) {
  score.tmp <- embeddings.raw$score
  AUC <- plot.AUC(score.tmp[score.tmp != -1], embeddings.raw$logits[score.tmp != -1])
  J_stats <- AUC$curve[,2] - AUC$curve[,1]
  optimal.cutoff.1 <- max(AUC$curve[which(J_stats==max(J_stats)),3])
  score.tmp <- embeddings.raw$score
  logits <- embeddings.raw$logits
  logits <- logits[score.tmp != 1]
  score.tmp <- score.tmp[score.tmp != 1]
  score.tmp[score.tmp == 0] <- 1
  score.tmp[score.tmp == -1] <- 0
  AUC <- plot.AUC(score.tmp, logits)
  J_stats <- AUC$curve[,2] - AUC$curve[,1]
  optimal.cutoff.2 <- min(AUC$curve[which(J_stats==max(J_stats)),3])
  score.tmp <- embeddings.raw$score
  AUC <- plot.AUC(score.tmp[score.tmp!=0], embeddings.raw$logits[score.tmp!=0])
  J_stats <- AUC$curve[,2] - AUC$curve[,1]
  optimal.cutoff.3 <- median(AUC$curve[which(J_stats==max(J_stats)),3])
} else {
  AUC <- plot.AUC(embeddings.raw$score, embeddings.raw$logits)
  J_stats <- AUC$curve[,2] - AUC$curve[,1]
  optimal.cutoff <- max(AUC$curve[which(J_stats==max(J_stats)),3])
}

if (0 %in% embeddings.raw$score & ! -1 %in% embeddings.raw$score) {
  neg.score <- "0"
} else if (-1 %in% embeddings.raw$score) {
  neg.score <- "-1"
}
if (0 %in% embeddings.raw$score & -1 %in% embeddings.raw$score) {
  embeddings.raw <- embeddings.raw[embeddings.raw$score != 0,]
  embeddings <- embeddings.raw[, startsWith(colnames(embeddings.raw), "X")]
  embeddings.pca <- cluster.leiden(embeddings)
  embeddings.pca$label <- as.character(embeddings.raw$score)
  embeddings.pca$uniprotID <- as.character(embeddings.raw$uniprotID)
  embeddings.pca$y <- as.numeric(embeddings.raw$logits)
  embeddings.pca$label.pred <- "1"
  # embeddings.pca$label.pred[embeddings.pca$y <= optimal.cutoff.1] <- "0"
  # embeddings.pca$label.pred[embeddings.pca$y <= optimal.cutoff.2] <- "-1"
  # optimal.cutoff <- c(optimal.cutoff.1, optimal.cutoff.2)
  embeddings.pca$label.pred[embeddings.pca$y <= optimal.cutoff.3] <- neg.score
  optimal.cutoff <- optimal.cutoff.3
} else {
  embeddings.pca <- cluster.leiden(embeddings)
  embeddings.pca$label <- as.character(embeddings.raw$score)
  embeddings.pca$uniprotID <- as.character(embeddings.raw$uniprotID)
  embeddings.pca$y <- as.numeric(embeddings.raw$logits)
  embeddings.pca$label.pred <- "1"
  embeddings.pca$label.pred[embeddings.pca$y <= optimal.cutoff] <- neg.score
}
# print sihoutte score if label by uniprotID
library(cluster)
dis.sim.mat <- as.matrix(dist(embeddings.pca[,startsWith(colnames(embeddings.pca), "PC")]))
library(ggplot2)
sil.score.label <- silhouette(as.numeric(as.factor(embeddings.pca$label)), dmatrix = dis.sim.mat)
avg.sil.label <- mean(sil.score.label[,3])
p1 <- ggplot(embeddings.pca, aes(x=UMAP1, y=UMAP2, col=label)) +
  geom_point(size=0.5) + ggtitle(paste0('Average silhoutte score on label: ', round(avg.sil.label, 2))) +
  theme_bw() + ggeasy::easy_center_title()
sil.score.uniprotID <- silhouette(as.numeric(as.factor(embeddings.pca$uniprotID)), dmatrix = dis.sim.mat)
avg.sil.uniprotID <- mean(sil.score.uniprotID[,3])
if (length(unique(embeddings.pca$uniprotID)) < 20) {
  p2 <- ggplot(embeddings.pca, aes(x=UMAP1, y=UMAP2, col=uniprotID)) +
    geom_point(size=0.5) + ggtitle(paste0('Average silhoutte score on uniprotID: ', round(avg.sil.uniprotID, 2))) +
    theme_bw() + ggeasy::easy_center_title()
} else {
  p2 <- ggplot(embeddings.pca, aes(x=UMAP1, y=UMAP2, col=uniprotID)) +
    geom_point(size=0.5) + ggtitle(paste0('Average silhoutte score on uniprotID: ', round(avg.sil.uniprotID, 2))) +
    theme_bw() + theme(legend.position = "none") + ggeasy::easy_center_title()
}
p3 <- ggplot(embeddings.pca, aes(x=UMAP1, y=UMAP2, col=y)) +
  geom_point(size=0.5) + ggtitle(paste0("optimal threshold: ", 
                                        paste(round(optimal.cutoff, 2), collapse = ","), 
                                        " Accuracy = ",
                                        round(sum(embeddings.pca$label.pred == embeddings.pca$label) / dim(embeddings.pca)[1], 2))) + 
  theme_bw() + ggeasy::easy_center_title()
sil.score.label.pred <- silhouette(as.numeric(as.factor(embeddings.pca$label.pred)), dmatrix = dis.sim.mat)
avg.sil.label.pred <- mean(sil.score.label.pred[,3])
p4 <- ggplot(embeddings.pca, aes(x=UMAP1, y=UMAP2, col=label.pred)) +
  geom_point(size=0.5) + ggtitle(paste0('Average silhoutte score on label.pred: ', round(avg.sil.label.pred, 2))) +
  theme_bw() + ggeasy::easy_center_title()
print(paste0("Accuracy at threshold ", optimal.cutoff, " : ",
             sum(embeddings.pca$label.pred == embeddings.pca$label) / dim(embeddings.pca)[1]))
# print(paste0("Baseline: ", 124/dim(embeddings.pca)[1]))
library(patchwork)
ggsave('test.umap.pdf', (p1 + p2) / (p3 + p4), width = 12, height = 10)
# ggsave('test.umap.pdf', p1 + p2 + p3, width = 21, height = 6)
# setwd(log.dir)

