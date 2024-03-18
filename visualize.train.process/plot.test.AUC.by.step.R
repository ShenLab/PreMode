# setwd('/share/terra/Users/gz2294/PreMode.final/')
source('utils.R')
args <- commandArgs(trailingOnly = T)
configs <- yaml::read_yaml(args[1])
# configs <- yaml::read_yaml('scripts/CHPs.v1.SAGPool.add.pos.ct/PF07714/PF07714.seed.0.yaml')
# res <- get.auc.by.step.split.pLDDT(configs, base.line=F)
res <- get.auc.by.step(configs, base.line=F)

# get min test file
# test.file <- read.csv(paste0(configs$log_dir, "test_result.step.86800.csv"))
# # annotate with pLDDT
# 
# hist(test.file$pLDDT.region)
