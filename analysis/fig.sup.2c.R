library(ggplot2)
CHPs.test <- read.csv('PreMode/cancer.hotspots.csv', row.names = 1)
CHPs.test.no.esm <- read.csv('PreMode.noESM/cancer.hotspots.csv', row.names = 1)
CHPs.test.no.se3 <- read.csv('ESM.SLP/cancer.hotspots.csv', row.names = 1)
CHPs.test.no.structure <- read.csv('PreMode.noStructure/cancer.hotspots.csv', row.names = 1)
CHPs.test.no.MSA <- read.csv('PreMode.noMSA/cancer.hotspots.csv', row.names = 1)
CHPs.test.no.paralog <- read.csv('PreMode.revision.noparalog/cancer.hotspots.csv', row.names = 1)

CHPs.test$unique.id <- paste0(CHPs.test$uniprotID, ":", CHPs.test$aaChg)
CHPs.test.no.esm$unique.id <- paste0(CHPs.test.no.esm$uniprotID, ":", CHPs.test.no.esm$aaChg)
CHPs.test.no.structure$unique.id <- paste0(CHPs.test.no.structure$uniprotID, ":", CHPs.test.no.structure$aaChg)
CHPs.test.no.se3$unique.id <- paste0(CHPs.test.no.se3$uniprotID, ":", CHPs.test.no.se3$aaChg)
CHPs.test.no.MSA$unique.id <- paste0(CHPs.test.no.MSA$uniprotID, ":", CHPs.test.no.MSA$aaChg)
CHPs.test.no.paralog$unique.id <- paste0(CHPs.test.no.paralog$uniprotID, ":", CHPs.test.no.paralog$aaChg)

CHPs.test.no.se3 <- CHPs.test.no.se3[match(CHPs.test$unique.id, CHPs.test.no.se3$unique.id),]
CHPs.test.no.esm <- CHPs.test.no.esm[match(CHPs.test$unique.id, CHPs.test.no.esm$unique.id),]
CHPs.test.no.structure <- CHPs.test.no.structure[match(CHPs.test$unique.id, CHPs.test.no.structure$unique.id),]
CHPs.test.no.MSA <- CHPs.test.no.MSA[match(CHPs.test$unique.id, CHPs.test.no.MSA$unique.id),]
CHPs.test.no.paralog <- CHPs.test.no.paralog[match(CHPs.test$unique.id, CHPs.test.no.paralog$unique.id),]

source('./AUROC.R')
auc.list <- list()
auc.list[[1]] <- plot.AUC(CHPs.test$score, CHPs.test$logits, rev.ok = T)
auc.list[[2]] <- plot.AUC(CHPs.test.no.esm$score, CHPs.test.no.esm$logits, rev.ok = T)
auc.list[[3]] <- plot.AUC(CHPs.test.no.structure$score, CHPs.test.no.structure$logits, rev.ok = T)
auc.list[[4]] <- plot.AUC(CHPs.test.no.se3$score, CHPs.test.no.se3$logits, rev.ok = T)
auc.list[[5]] <- plot.AUC(CHPs.test.no.MSA$score, CHPs.test.no.MSA$logits, rev.ok = T)
auc.list[[6]] <- plot.AUC(CHPs.test.no.paralog$score, CHPs.test.no.paralog$logits, rev.ok = T)


model.names <- c("PreMode", "PreMode:\nNo ESM", "PreMode:\nNo Structure", "ESM + SLP",  "PreMode:\nNo MSA", "PreMode.noparalog")
to.plot <- data.frame()
model.rank <- c()
model.name <- c()
for (i in 1:length(auc.list)) {
  model.auc <- as.data.frame(auc.list[[i]]$curve)
  model.rank <- c(model.rank, auc.list[[i]]$auc)
  if (model.names[i] == "PreMode.noparalog") {
    model.auc$model <- paste0("PreMode: No paralog\nin training ", " (", round(auc.list[[i]]$auc, 3), ")")
    model.name <- c(model.name, paste0("PreMode: No paralog\nin training ", " (", round(auc.list[[i]]$auc, 3), ")"))
  } else {
    model.auc$model <- paste0(model.names[i], " (", round(auc.list[[i]]$auc, 3), ")")
    model.name <- c(model.name, paste0(model.names[i], " (", round(auc.list[[i]]$auc, 3), ")"))
  }
  to.plot <- rbind(to.plot, model.auc)
}
colnames(to.plot)[1:3] <- c("FPR", "TPR", "cutoff")
ggplot(to.plot, aes(x=FPR, y=TPR, col=factor(model, levels = model.name[order(model.rank, decreasing = T)]))) + 
  geom_line() + 
  ggtitle("Ablation Analysis: ROC on pathogenicity") + 
  xlab("False Positive Rates") +
  ylab("True Positive Rates (Sensitivities)") +
  theme_bw() + labs(colour="PreMode Models") + ggeasy::easy_center_title()
ggsave('figs/fig.sup.2c.pdf', height = 4, width = 6)

