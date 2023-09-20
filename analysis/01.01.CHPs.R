library(ggplot2)
CHPs <- read.csv('/share/pascal/Users/gz2294/Data/DMS/ClinVar.HGMD.PrimateAI.syn/all.clinvar.4.splits/training.csv', row.names = 1)
CHPs.test <- read.csv('/share/vault/Users/gz2294/PreMode.final/CHPs.v4.esm.dssp.small.StarAttn.MSA.StarPool.1dim.seed.0/test_result.step.30000.csv', row.names = 1)
pos.genes <- length(unique(CHPs$uniprotID[CHPs$score==1]))
neg.genes <- length(unique(CHPs$uniprotID[CHPs$score==0]))

# histogram of genes / variants
gene.variants <- as.data.frame(table(CHPs$uniprotID))
pos.gene.variants <- as.data.frame(table(CHPs$uniprotID[CHPs$score==1]))
neg.gene.variants <- as.data.frame(table(CHPs$uniprotID[CHPs$score==0]))
colnames(gene.variants) <- c("Gene", "variant.counts")
p <- ggplot(gene.variants, aes(x=variant.counts)) +
  geom_histogram(binwidth=10) + theme_bw() + scale_y_log10()
p

gene.variants$pos.variant.counts <- pos.gene.variants$Freq[match(gene.variants$Gene, pos.gene.variants$Var1)]
gene.variants$neg.variant.counts <- neg.gene.variants$Freq[match(gene.variants$Gene, neg.gene.variants$Var1)]
gene.variants[is.na(gene.variants)] <- 0
p <- ggplot(gene.variants, aes(x=pos.variant.counts, y=neg.variant.counts)) +
  # geom_point() +
  stat_density2d(aes(fill = ..density..), geom = "polygon", bins = 100,
                 alpha = 0.5, contour = FALSE, show.legend = FALSE,
                 data = gene.variants,
                 h = c(1, 1)) +
  scale_fill_gradient(low = "white", high = "blue") +
  theme_bw()
p

# add EVE score
source('/share/pascal/Users/gz2294/Pipeline/uniprot.table.add.annotation.R')
CHPs.test <- uniprot.table.add.annotation.parallel(CHPs.test, 'EVE')
CHPs.test <- uniprot.table.add.annotation.parallel(CHPs.test, 'dbnsfp')
CHPs.test <- uniprot.table.add.annotation.parallel(CHPs.test, 'gMVP')
CHPs.test <- uniprot.table.add.annotation.parallel(CHPs.test, 'conservation')

source('/share/pascal/Users/gz2294/Pipeline/AUROC.R')
auc.list <- list()
auc.list[[1]] <- plot.AUC(CHPs.test$score, CHPs.test$y.0)
# for NA value, impute with mean
# for (c in c("EVE", "REVEL", "gMVP", "PrimateAI")) {
#   CHPs.test[is.na(CHPs.test[,c]),c] <- median(CHPs.test[!is.na(CHPs.test[,c]),c])
# }
auc.list[[2]] <- plot.AUC(CHPs.test$score, CHPs.test$EVE)
# plot.AUC(CHPs.test$score[!is.na(CHPs.test$EVE)], CHPs.test$y.0[!is.na(CHPs.test$EVE)])

auc.list[[3]] <- plot.AUC(CHPs.test$score, CHPs.test$REVEL)
# plot.AUC(CHPs.test$score[!is.na(CHPs.test$REVEL)], CHPs.test$y.0[!is.na(CHPs.test$REVEL)])

auc.list[[4]] <- plot.AUC(CHPs.test$score, CHPs.test$PrimateAI)
# plot.AUC(CHPs.test$score[!is.na(CHPs.test$PrimateAI)], CHPs.test$y.0[!is.na(CHPs.test$PrimateAI)])

auc.list[[5]] <- plot.AUC(CHPs.test$score, CHPs.test$gMVP)
# plot.AUC(CHPs.test$score[!is.na(CHPs.test$gMVP)], CHPs.test$y.0[!is.na(CHPs.test$gMVP)])

esm.logits <- read.csv('esm2.inference/CHPs.v4/testing.logits.csv')
alphabet <- c('<cls>', '<pad>', '<eos>', '<unk>',
              'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
              'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
              'X', 'B', 'U', 'Z', 'O', '.', '-',
              '<null_1>', '<mask>')
esm.logits <- esm.logits[,2:34]
colnames(esm.logits) <- alphabet
score <- c()
for (k in 1:dim(esm.logits)[1]) {
  score <- c(score, esm.logits[k, CHPs.test$alt[k]] - esm.logits[k, CHPs.test$ref[k]])
}
CHPs.test$esm.logits <- score
auc.list[[6]] <- plot.AUC(CHPs.test$score, CHPs.test$esm.logits)

auc.list[[7]] <-conservation.auc <- plot.AUC(CHPs.test$score, CHPs.test$conservation)

model.names <- c("PreMode", "EVE", "REVEL", "PrimateAI", "gMVP", "ESM", "conservation")
to.plot <- data.frame()
model.rank <- c()
model.name <- c()
for (i in 1:length(auc.list)) {
  model.auc <- as.data.frame(auc.list[[i]]$curve)
  model.auc$model <- paste0(model.names[i], "(", round(auc.list[[i]]$auc, 3), ")")
  to.plot <- rbind(to.plot, model.auc)
  model.rank <- c(model.rank, auc.list[[i]]$auc)
  model.name <- c(model.name, paste0(model.names[i], "(", round(auc.list[[i]]$auc, 3), ")"))
}
colnames(to.plot)[1:3] <- c("FPR", "TPR", "cutoff")
ggplot(to.plot, aes(x=FPR, y=TPR, col=factor(model, levels = model.name[order(model.rank, decreasing = T)]))) + 
  geom_line() + 
  ggtitle("ROC curve on pathogenicity task") + 
  xlab("False Positive Rates") +
  ylab("True Positive Rates (Sensitivities)") +
  theme_bw() + labs(colour="Model") + ggeasy::easy_center_title()
ggsave('figs/01.01.CHPs.AUROC.pdf', height = 4, width = 6)

# plot PR curve
pr.list <- list()
pr.list[[1]] <- plot.PR(CHPs.test$score, CHPs.test$y.0)
pr.list[[2]] <- plot.PR(CHPs.test$score, CHPs.test$EVE)
pr.list[[3]] <- plot.PR(CHPs.test$score, CHPs.test$REVEL)
pr.list[[4]] <- plot.PR(CHPs.test$score, CHPs.test$PrimateAI)
pr.list[[5]] <- plot.PR(CHPs.test$score, CHPs.test$gMVP)
pr.list[[6]] <- plot.PR(CHPs.test$score, CHPs.test$esm.logits)
pr.list[[7]] <- plot.PR(CHPs.test$score, CHPs.test$conservation)
to.plot <- data.frame()
model.rank <- c()
model.name <- c()
for (i in 1:length(pr.list)) {
  model.auc <- as.data.frame(pr.list[[i]]$curve)
  model.auc$model <- paste0(model.names[i], "(", round(pr.list[[i]]$auc, 3), ")")
  to.plot <- rbind(to.plot, model.auc)
  model.rank <- c(model.rank, pr.list[[i]]$auc)
  model.name <- c(model.name, paste0(model.names[i], "(", round(pr.list[[i]]$auc, 3), ")"))
}
colnames(to.plot)[1:3] <- c("recall", "precision", "cutoff")
ggplot(to.plot, aes(x=recall, y=precision, col=factor(model, levels = model.name[order(model.rank, decreasing = T)]))) + 
  geom_line() + 
  ggtitle("PR curve on pathogenicity task") + 
  xlab("recall") +
  ylab("precision") +
  theme_bw() + labs(colour="Model") + ggeasy::easy_center_title()
ggsave('figs/01.01.CHPs.AUPR.pdf', height = 4, width = 6)

# plot density of logits
ggplot(CHPs.test, aes(x=y.0, col=as.factor(score))) + geom_density()
