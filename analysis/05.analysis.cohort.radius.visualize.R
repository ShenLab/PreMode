# visualize with dssp secondary structure 
library(ggplot2)
library(patchwork)

genes <- c("Q99250", "O00555", "Q13936", "Q9UQD0",
           "P15056", "Q02156", "P04049", 
           "P21802", "P22607")
# genes <- c("P15056", "P21802", "P07949", "P04637", "Q09428", "Q14654", "Q14524", "P60484")
pfams <- c("IonChannel", "IonChannel", "IonChannel", "IonChannel", 
           "PF00130", "PF00130", "PF00130",
           "PF07679", "PF07679")
gene.name <- c("SCN2A", "CACNA1A", "CACNA1C", "SCN8A",
               "BRAF", "PRKCE", "RAF1",
               "FGFR2", "FGFR3")
af2.seqs <- read.csv('~/Data/Protein/alphafold2_v4/swissprot_and_human.full.seq.csv', row.names = 1)
aa.dict <- c('L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
             'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C')
log.dir <- 'cohort.CHPs.v4.esm.StarAttn.MSA.StarPool/'
folds <- c(0:4)
source('~/Pipeline/plot.genes.scores.heatmap.R')
for (k in 1:length(genes)) {
  gene <- genes[k]
  pfam <- pfams[k]
  
  assemble.logits <- 0
  all.training <- data.frame()
  patch.plot <- list()
  pretrain.result <- read.csv(paste0(log.dir, pfam, '.pretrain.csv'))
  pretrain.result <- pretrain.result[pretrain.result$uniprotID==gene,]
  for (fold in folds) {
    gene.result <- read.csv(paste0(log.dir, pfam, '.fold.', fold, '.csv'))
    gene.result <- gene.result[gene.result$uniprotID==gene,]
    
    theta <- gene.result$logits
    
    gene.result$logits.x <- pretrain.result$logits * cos(theta*pi/2)
    gene.result$logits.y <- pretrain.result$logits * sin(theta*pi/2)
    
    assemble.logits <- assemble.logits + theta
    
    p1 <- ggplot(gene.result, aes(x=logits.x, y=logits.y, col=cohort)) +
      geom_point() +
      theme_bw() + 
      ggtitle(gene.result$HGNC[!is.na(gene.result$HGNC)][1]) + ggeasy::easy_center_title()
    p1 <- ggExtra::ggMarginal(p1, type="density", groupColour=T)
    patch.plot[[fold+1]] <- p1
  }
  assemble.logits <- assemble.logits / (length(folds) - 1)
  
  gene.result$logits.x <- pretrain.result$logits * cos(assemble.logits*pi/2)
  gene.result$logits.y <- pretrain.result$logits * sin(assemble.logits*pi/2)
  gene.result$assemble.logits <- assemble.logits
  
  gene.result$aaChg[gene.result$cohort=="NDD"] <- NA
  
  p1 <- ggplot(gene.result, aes(x=logits.x, y=logits.y, col=cohort, label=aaChg)) +
    geom_point() +
    ggrepel::geom_text_repel() +
    theme_bw() + 
    ggtitle(gene.name[k]) + ggeasy::easy_center_title()
  p1 <- ggExtra::ggMarginal(p1, type="density", groupColour=T)
  p2 <- ggplot(gene.result, aes(x=assemble.logits, col=cohort, label=aaChg)) +
    geom_density() +
    theme_bw() + 
    ggtitle(gene.name[k]) + ggeasy::easy_center_title()
  p <- patchwork::wrap_elements(p1) + p2
  ggsave(paste0(log.dir, pfam, '.', gene, '.radius.pdf'), p, width = 12, height = 5)
  
  p <- patch.plot[[1]] / patch.plot[[2]] / patch.plot[[3]] / patch.plot[[4]] / patch.plot[[5]]
  ggsave(paste0(log.dir, pfam, '.', gene, '.5folds.radius.pdf'), p, width = 5, height = 4*5)
}
