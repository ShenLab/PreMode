source('/share/pascal/Users/gz2294/Pipeline/AUROC.R')
args <- commandArgs(trailingOnly = T)
configs <- read.csv(args[1])
# configs <- yaml::read_yaml('scripts/CHPs.v1.ct/B_LAC/B_LAC.seed.0.yaml')
res <- plot.R2(configs[,colnames(configs)[startsWith(colnames(configs), 'score')]],
	       configs[,colnames(configs)[startsWith(colnames(configs), 'logits')]], bin = grepl('bin', args[1]))
print(res$R2)
