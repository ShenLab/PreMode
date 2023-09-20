args <- commandArgs(trailingOnly = T)
file.test <- args[1]
file.test <- paste0('~/Data/DMS/ClinVar/testing.ClinVar+.DiscovEHR-.csv.inference.PreMode.v2.onehot.tiny.dssp.StarAttn.step.3600.csv')
ClinVar.test <- read.csv(file.test)
source('~/Pipeline/AUROC.R')
res <- plot.AUC(ClinVar.test$score, ClinVar.test$logits, 'tmp.pdf')
print(res$auc)
pos.label <- 1
if (-1 %in% ClinVar.test$score) {
  neg.label <- -1
} else {
  neg.label <- 0
}
source('~/Pipeline/uniprot.table.add.annotation.R')

ClinVar.test <- uniprot.table.add.annotation.parallel(ClinVar.test, 'pLDDT.all')
ClinVar.test <- uniprot.table.add.annotation.parallel(ClinVar.test, 'pLDDT')
plddt <- ClinVar.test$pLDDT
plddt.all <- ClinVar.test$pLDDT.all

cutoff <- 0.5
# cutoff <- res$cutoff
res.by.id <- plot.AUC.by.uniprotID(ClinVar.test$score, ClinVar.test$logits, ClinVar.test$uniprotID)

mis.pos.points <- ClinVar.test[ClinVar.test$score == pos.label & ClinVar.test$logits <= cutoff,]
mis.neg.points <- ClinVar.test[ClinVar.test$score == neg.label & ClinVar.test$logits >= cutoff,]
hist(mis.pos.points$pLDDT)
hist(mis.neg.points$pLDDT)
hist(ClinVar.test$pLDDT)

file.train <- paste0('~/Data/DMS/ClinVar.HGMD.PrimateAI.syn/add.clinvar.neg.4.splits/training.csv')
ClinVar.train <- read.csv(file.train, row.names = 1)

for (i in 1:dim(res.by.id)) {
  res.by.id$pos.train[i] <- sum(ClinVar.train$uniprotID == res.by.id$uniprotID[i] & ClinVar.train$score==1)
  res.by.id$neg.train[i] <- sum(ClinVar.train$uniprotID == res.by.id$uniprotID[i] & ClinVar.train$score==0)
  res.by.id$pos.test[i] <- sum(ClinVar.test$uniprotID == res.by.id$uniprotID[i] & ClinVar.test$score==1)
  res.by.id$neg.test[i] <- sum(ClinVar.test$uniprotID == res.by.id$uniprotID[i] & ClinVar.test$score==0)
  
  cutoff <- res.by.id$optimal.cutoff[i]
  if (is.na(cutoff)) {
    cutoff <- res$cutoff
  }
  mis.pos.points <- ClinVar.test[ClinVar.test$uniprotID == res.by.id$uniprotID[i] & ClinVar.test$score == pos.label & ClinVar.test$logits <= cutoff,]
  mis.neg.points <- ClinVar.test[ClinVar.test$uniprotID == res.by.id$uniprotID[i] & ClinVar.test$score == neg.label & ClinVar.test$logits >= cutoff,]
  res.by.id$mis.pos[i] <- sum(mis.pos.points$uniprotID == res.by.id$uniprotID[i])
  res.by.id$mis.neg[i] <- sum(mis.neg.points$uniprotID == res.by.id$uniprotID[i])
}


