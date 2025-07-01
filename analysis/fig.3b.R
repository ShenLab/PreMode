library(ggplot2)
CHPs.test <- read.csv('PreMode/cancer.hotspots.csv', row.names = 1)
source('./AUROC.R')
auc.list <- list()
auc.list[[1]] <- plot.AUC(CHPs.test$score, CHPs.test$logits, rev.ok = T)
auc.list[[2]] <- plot.AUC(CHPs.test$score, CHPs.test$EVE, rev.ok = T)
auc.list[[3]] <- plot.AUC(CHPs.test$score, CHPs.test$REVEL, rev.ok = T)
auc.list[[4]] <- plot.AUC(CHPs.test$score, CHPs.test$PrimateAI, rev.ok = T)
auc.list[[5]] <- plot.AUC(CHPs.test$score, CHPs.test$gMVP, rev.ok = T)
esm.logits <- read.csv('esm2.inference/testing.logits.csv')
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
auc.list[[6]] <- plot.AUC(CHPs.test$score, CHPs.test$esm.logits, rev.ok = T)
auc.list[[7]] <- plot.AUC(CHPs.test$score, CHPs.test$conservation.entropy, rev.ok = T)
auc.list[[8]] <- plot.AUC(CHPs.test$score, CHPs.test$AlphaMissense, rev.ok = T)
model.names <- c("PreMode", "EVE", "REVEL", "PrimateAI", "gMVP", "ESM", "conservation", "AlphaMissense")
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
ggsave('figs/fig.3b.pdf', height = 4, width = 6)

# plot PR curve
pr.list <- list()
pr.list[[1]] <- plot.PR(CHPs.test$score, CHPs.test$logits)
pr.list[[2]] <- plot.PR(CHPs.test$score, CHPs.test$EVE)
pr.list[[3]] <- plot.PR(CHPs.test$score, CHPs.test$REVEL)
pr.list[[4]] <- plot.PR(CHPs.test$score, CHPs.test$PrimateAI)
pr.list[[5]] <- plot.PR(CHPs.test$score, CHPs.test$gMVP)
pr.list[[6]] <- plot.PR(CHPs.test$score, CHPs.test$esm.logits)
pr.list[[7]] <- plot.PR(CHPs.test$score, CHPs.test$conservation.entropy)
pr.list[[8]] <- plot.PR(CHPs.test$score, CHPs.test$AlphaMissense)
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
# ggsave('figs/fig.3b.pdf', height = 4, width = 6)

# plot density of logits
ggplot(CHPs.test, aes(x=y.0, col=as.factor(score))) + geom_density()
