library(ggplot2)
CHPs <- read.csv('~/Data/DMS/ClinVar.HGMD.PrimateAI.syn/training.csv', row.names = 1)
CHPs.test <- read.csv('~/Data/DMS/ClinVar.HGMD.PrimateAI.syn/testing.csv', row.names = 1)
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
source('~/Pipeline/uniprot.table.add.annotation.R')
CHPs.test <- uniprot.table.add.annotation.parallel(CHPs.test, 'EVE')
CHPs.test <- uniprot.table.add.annotation.parallel(CHPs.test, 'dbnsfp')
CHPs.test <- uniprot.table.add.annotation.parallel(CHPs.test, 'gMVP')
CHPs.test <- uniprot.table.add.annotation.parallel(CHPs.test, 'conservation')

source('~/Pipeline/AUROC.R')
# for NA value, impute with mean
for (c in c("EVE", "REVEL", "gMVP", "PrimateAI")) {
  CHPs.test[is.na(CHPs.test[,c]),c] <- median(CHPs.test[!is.na(CHPs.test[,c]),c])
}
plot.AUC(CHPs.test$score, CHPs.test$EVE)
plot.AUC(CHPs.test$score, CHPs.test$REVEL)
plot.AUC(CHPs.test$score, CHPs.test$PrimateAI)
plot.AUC(CHPs.test$score, CHPs.test$gMVP)
plot.AUC(CHPs.test$score, CHPs.test$conservation)

ClinVar.test.1 <- read.csv('~/Data/DMS/ClinVar/testing.ClinVar+.DiscovEHR-.csv.inference.gRESCVE.1D.csv')
ClinVar.test <- read.csv('~/Data/DMS/ClinVar/testing.ClinVar+.DiscovEHR-.csv.inference.gRESCVE.csv')
ClinVar.test$logits.1D <- ClinVar.test.1$logits

ClinVar.test <- uniprot.table.add.annotation.parallel(ClinVar.test, 'EVE')
ClinVar.test <- uniprot.table.add.annotation.parallel(ClinVar.test, 'dbnsfp')
ClinVar.test <- uniprot.table.add.annotation.parallel(ClinVar.test, 'gMVP')

plot.AUC(ClinVar.test$score, ClinVar.test$logits)
plot.AUC(ClinVar.test$score, ClinVar.test$EVE)
plot.AUC(ClinVar.test$score, ClinVar.test$REVEL)
plot.AUC(ClinVar.test$score, ClinVar.test$PrimateAI)
plot.AUC(ClinVar.test$score, ClinVar.test$gMVP)

ggplot(ClinVar.test)


