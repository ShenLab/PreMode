setwd('/share/terra/Users/gz2294/RESCVE.final/')
source('utils.R')
args <- commandArgs(trailingOnly = T)
configs <- yaml::read_yaml(args[1])
# configs <- yaml::read_yaml('scripts/CHPs.v1.ct/B_LAC/B_LAC.seed.0.yaml')
res <- get.R.by.step(configs)

