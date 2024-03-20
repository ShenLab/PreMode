seqs <- c(); for (gene in c('PTEN', 'NUDT15', 'CCR5', 'CXCR4', 'GCK', 'CYP2C9', 'ASPA', 'SNCA')) {tmp <- read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/MAVEDB/', gene, '/testing.csv'), row.names = 1); seqs <- c(seqs, tmp$wt.orig[1])}
seq.df <- data.frame(hgnc=c('PTEN', 'NUDT15', 'CCR5', 'CXCR4', 'GCK', 'CYP2C9', 'ASPA', 'SNCA'), seqs)
write.csv(seq.df, 'Hsu.et.al.git/fasta/all.fasta', quote = F, row.names = F)
# process for each data
genes <- c('PTEN', 'NUDT15', 'CCR5', 'CXCR4', 'GCK', 'CYP2C9', 'ASPA', 'SNCA')
for (gene in genes) {
  for (fold in 0:4) {
    training <- read.csv(paste0('../data.files/', gene, '/train.seed.', fold, '.csv'), 
                         row.names = 1)
    nscores <- sum(startsWith(colnames(training), 'score'))
    for (s in 1:nscores) {
      target.dir <- paste0('Hsu.et.al.git/data/', gene, '.fold.', fold, '.score.', s)
      dir.create(target.dir)
      dat <- training[,c('sequence.orig', paste0('score.', s))]
      colnames(dat) <- c('seq', 'log_fitness')
      dat$n_mut <- 1
      dat$mutant <- paste0(training$ref, training$pos.orig, training$alt)
      write.csv(dat, paste0(target.dir, '/data.csv'), row.names = F, quote = F)
      write.csv(training$wt.orig[1], paste0(target.dir, '/wt.fasta'), row.names = F, quote = F)
      system(paste0('sed -i "s|^x|>', gene, '.score.', s, '|g" ', target.dir, '/wt.fasta'))
    }
  }
}