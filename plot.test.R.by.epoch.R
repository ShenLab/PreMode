setwd('/share/terra/Users/gz2294/RESCVE.final/')
source('utils.R')
args <- commandArgs(trailingOnly = T)
configs <- yaml::read_yaml(args[1])
# configs <- yaml::read_yaml('scripts/CHPs.v1.SAGPool.add.pos.ct/PF07714/PF07714.seed.0.yaml')
res <- get.R.by.epoch(configs)

