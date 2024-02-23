# visualize with dssp secondary structure 
library(ggplot2)
library(bio3d)
library(patchwork)
genes <- c("Q99250", "Q14524", "O00555")
gene.names <- c("SCN2A", "SCN5A", "CACNA1A")

af2.seqs <- read.csv('~/Data/af2_uniprot/swissprot_and_human.csv', row.names = 1)
aa.dict <- c('L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
             'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C')
log.dir <- 'PreMode.inference/'
# auc.dir <- '../scripts/CHPs.v4.esm.dssp.small.StarAttn.MSA.StarPool.1dim/'
# use.logits <- 'meta.logits'
folds <- c(0:4)
# source('~/Pipeline/plot.genes.scores.heatmap.R')
source('~/Pipeline/AUROC.R')
# for three genes, first only visualize seed 0
result.plot <- data.frame()
for (o in 1:length(genes)) {
  gene <- genes[o]
  print(gene)
  gene.testing.result <- read.csv(paste0(log.dir, gene, '/testing.fold.heyne.', 0, '.csv'))
  # heyne results
  heyne.testing.result <- readxl::read_excel('figs/ICC.FunCion.xlsx', sheet = paste0('Sheet', o))
  # remove heyne training points
  heyne.testing.result <- heyne.testing.result[heyne.testing.result$`FuNCion Training`==F,]
  premode.auc <- plot.AUC(gene.testing.result$score, gene.testing.result$logits)
  heyne.auc <- plot.AUC(heyne.testing.result$score, as.numeric(heyne.testing.result$FuNCion))
  result.plot <- rbind(result.plot, data.frame(PreMode.auc=premode.auc$auc,
                                               FunCion.auc=heyne.auc$auc,
                                               HGNC=gene.names[o]))
}
p <- ggplot(result.plot, aes(x=FunCion.auc, y=PreMode.auc, label=HGNC, col=HGNC)) + 
  xlim(0.5, 0.8) + ylim(0.5, 0.8) +
  theme_bw() +
  geom_abline(intercept = 0, slope = 1, linetype='dashed') +
  geom_point() + ggrepel::geom_text_repel()
ggsave('figs/02.02.ICC.tasks.compare.heyne.pdf', p, height = 4, width = 5)



for (o in 1:length(genes)) {
  gene <- genes[o]
  print(gene)
  assemble.logits <- 0
  patch.plot <- list()
  all.training.logits <- NULL
  all.testing.logits <- NULL
  for (fold in folds) {
    gene.result <- read.csv(paste0(log.dir, gene, '/training.fold.heyne.', fold, '.csv'))
    gene.testing.result <- read.csv(paste0(log.dir, gene, '/testing.fold.heyne.', fold, '.csv'))
    if (is.null(all.training.logits)) {
      all.training.logits <- matrix(NA, nrow = dim(gene.result)[1], ncol = 5)
      all.testing.logits <- matrix(NA, nrow = dim(gene.testing.result)[1], ncol = 5)
    }
    all.training.logits[,fold+1] <- gene.result$logits
    all.testing.logits[,fold+1] <- gene.testing.result$logits
    assemble.logits <- assemble.logits + gene.result$logits
    train.score <- gene.result$score
    all.training <- gene.result
    all.testing <- gene.testing.result
  }
  assemble.logits <- assemble.logits / (length(folds) - 1)
  library(caret)
  set.seed(0)
  colnames(all.training.logits) <- paste0('logits', 0:4)
  colnames(all.testing.logits) <- paste0('logits', 0:4)
  meta_model_fit <- train(all.training.logits, 
                          as.factor(train.score),
                          # weights = as.array(sum(table(as.factor(train.score)))/table(as.factor(train.score)))[as.factor(train.score)],
                          method='ada')
  saveRDS(meta_model_fit, file = paste0(log.dir, gene, '/heyne.meta.RDS'))
  meta.logits <- predict(meta_model_fit, all.training.logits, type = 'prob')[,2]
  all.training$meta.logits <- meta.logits
  # add colnames
  gene.result$meta.logits <- meta.logits
  meta.auc <- plot.AUC(all.training$score[all.training$score %in% c(-1, 1)],
                       meta.logits[all.training$score %in% c(-1, 1)])
  print(meta.auc$auc)
  # do for testing
  meta.testing.logits <- predict(meta_model_fit, all.testing.logits, type = 'prob')[,2]
  meta.auc <- plot.AUC(all.testing$score[all.testing$score %in% c(-1, 1)],
                       meta.testing.logits[all.testing$score %in% c(-1, 1)])
  print(meta.auc$auc)
}
