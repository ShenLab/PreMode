# revision, split by domains, but only use variants from other proteins to train the model
gene.pfams <- read.csv('../scripts/gene.pfams.txt', header = F)$V1
gene.pfams <- gene.pfams[grep('self', gene.pfams, invert = T)]
# for gene.pfams, remove the gene itself
for (uid.pfam in gene.pfams) {
  uid <- strsplit(uid.pfam, '\\.')[[1]][1]
  for (seed in 0:4) {
    uid.pfam.trn <-read.csv(paste0('ICC.seed.', seed, '/', uid.pfam, '/training.csv'), row.names = 1)
    uid.pfam.tst <- read.csv(paste0('ICC.seed.', seed, '/', uid.pfam, '/testing.csv'), row.names = 1)
    # remove uid in uid.pfam.trn
    uid.pfam.trn <- uid.pfam.trn[uid.pfam.trn$uniprotID!=uid,]
    # make directory
    dir.create(paste0('ICC.seed.', seed, '/', uid.pfam, '.noself/'), showWarnings = F)
    write.csv(uid.pfam.trn, paste0('ICC.seed.', seed, '/', uid.pfam, '.noself/training.csv'))
    system(paste0('ln -s ', 
                  'ICC.seed.', seed, '/', uid.pfam, '/testing.csv ',
                  'ICC.seed.', seed, '/', uid.pfam, '.noself/testing.csv'))
  }
}