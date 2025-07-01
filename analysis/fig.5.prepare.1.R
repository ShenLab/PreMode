# create a folder called PreMode.all to store the result from ALL
genes <- read.csv('../scripts/gene.txt', header = F)$V1
dir.create('PreMode.all')
for (f in 0:4) {
  result.all <- read.csv(paste0('PreMode/ALL/testing.fold.', f, '.4fold.csv'))
  train.all <- read.csv(paste0('PreMode/ALL/training.fold.', f, '.4fold.csv'))
  for (gene in genes) {
    result.gene <- result.all[result.all$uniprotID==gsub('.clean', '', gene),]
    dir.create(paste0('PreMode.all/', gene, '/'))
    write.csv(result.gene, paste0('PreMode.all/', gene, '/testing.fold.', f, '.4fold.csv'))
    train.gene <- train.all[train.all$uniprotID==gsub('.clean', '', gene),]
    write.csv(result.gene, paste0('PreMode.all/', gene, '/training.fold.', f, '.4fold.csv'))
  }
}
