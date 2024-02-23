# setwd('/share/terra/Users/gz2294/PreMode.final/')
source('/share/pascal/Users/gz2294/Pipeline/AUROC.R')
args <- commandArgs(trailingOnly = T)
file <- read.csv(args[1])
# configs <- yaml::read_yaml('scripts/CHPs.v1.SAGPool.add.pos.ct/PF07714/PF07714.seed.0.yaml')
# res <- get.auc.by.step.split.pLDDT(configs, base.line=F)
res <- plot.AUC(file$score, file$logits)
print(res$auc)
# get min test file
# test.file <- read.csv(paste0(configs$log_dir, "test_result.step.86800.csv"))
# # annotate with pLDDT
# 
# hist(test.file$pLDDT.region)
