genes <- c("P15056", "P07949", "P04637",
           "Q09428", "P60484")
af2.seqs <- read.csv('genes.full.seq.csv', row.names = 1)
af2.seqs <- af2.seqs[af2.seqs$uniprotID %in% genes,]
ICC <- read.csv('figs/ALL.csv', row.names = 1)
aa.dict <- c('L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
             'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C')
source('./dnv.table.to.uniprot.R')
# get all possible mutants for 5 genes
for (gene in genes) {
  gene.seq <- af2.seqs$seq[af2.seqs$uniprotID==gene]
  all.variants <- c()
  for (i in 1:nchar(gene.seq)) {
    ref <- substr(gene.seq, i, i)
    alts <- aa.dict[aa.dict != ref]
    for (alt in alts) {
      all.variants <- c(all.variants, paste0('p.', ref, i, alt))
    }
  }
  all.variants.df <- data.frame(VarID=paste0(gene, all.variants),
                                score=NA,
                                uniprotID=gene,
                                aaChg=all.variants)
  all.variants.df <- dnv.table.to.uniprot.by.af2.uniprotID.parallel(all.variants.df, 'VarID', 'score', 'uniprotID', 'aaChg')
  write.csv(all.variants.df$result.noNA, file = paste0('5genes.all.mut/', gene, '.csv'))
}

for (gene in genes) {
  gene.variants.df <- read.csv(paste0('5genes.all.mut/', gene, '.csv'), row.names = 1)
  gene.variants.df$unique.id <- paste0(gene.variants.df$uniprotID, ":", gene.variants.df$ref, gene.variants.df$pos.orig, gene.variants.df$alt)
  ICC$unique.id <- paste0(ICC$uniprotID, ":", ICC$ref, ICC$pos.orig, ICC$alt)
  gene.variants.df$score <- ICC$score[match(gene.variants.df$unique.id, ICC$unique.id)]
  gene.variants.df$ENST <- ICC$ENST[match(gene, ICC$uniprotID)]
  write.csv(gene.variants.df, paste0('5genes.all.mut/', gene, '.csv'))
}
