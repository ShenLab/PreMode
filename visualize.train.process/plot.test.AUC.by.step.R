source('utils.R')
args <- commandArgs(trailingOnly = T)
configs <- yaml::read_yaml(args[1])
res <- get.auc.by.step(configs, base.line=F)

