# visualize with dssp secondary structure 
library(ggplot2)
library(patchwork)
genes <- c("P15056", "P21802", "P07949", "P04637", "Q09428", "Q14654", "Q14524")
# genes <- c("P15056", "P21802", "P07949", "P04637", "Q09428", "Q14654", "Q14524", "P60484")
aa.dict <- c('L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
             'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C')
log.dir <- '5genes.all.mut/CHPs.v4.esm.torchmdnet.TriAttn.StarPool.1dim/'
folds <- c(0:4)
source('~/Pipeline/plot.genes.scores.heatmap.R')
for (gene in genes) {
  assemble.logits <- 0
  all.training <- data.frame()
  patch.plot <- list()
  pretrain.result <- read.csv(paste0(log.dir, gene, '.pretrain.csv'), row.names = 1)
  for (fold in folds) {
    gene.result <- read.csv(paste0(log.dir, gene, '.fold.', fold, '.csv'), row.names = 1)
    training.file <- read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', fold, '/', gene, '.chps/training.csv'))[,c("HGNC", "uniprotID", "pos.orig", "ref", "alt", "score", "data_source")]
    testing.file <- read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', fold, '/', gene, '.chps/testing.csv'))[,c("HGNC", "uniprotID", "pos.orig", "ref", "alt", "score", "data_source")]

    theta <- gene.result$logits
    
    gene.result$logits.x <- pretrain.result$logits * cos(theta*pi/2)
    gene.result$logits.y <- pretrain.result$logits * sin(theta*pi/2)
    
    assemble.logits <- assemble.logits + theta
      
    training.file$split <- "train"
    testing.file$split <- "test"
    all.file <- dplyr::bind_rows(training.file, testing.file)
    all.file$unique.id <- paste(all.file$uniprotID, paste(all.file$ref, all.file$pos.orig, all.file$alt, sep = ""), sep = ":")
    all.file$BP.split[all.file$score==0] <- paste0(all.file$split[all.file$score==0], ": Benign")
    all.file$BP.split[all.file$score!=0] <- paste0(all.file$split[all.file$score!=0], ": Patho")
    
    all.file$GL.split[all.file$score==0] <- paste0(all.file$split[all.file$score==0], ": Benign")
    all.file$GL.split[all.file$score==1] <- paste0(all.file$split[all.file$score==1], ": GoF")
    all.file$GL.split[all.file$score==-1] <- paste0(all.file$split[all.file$score==-1], ": LoF")
      
    all.file$logits.x <- gene.result$logits.x[match(all.file$unique.id, gene.result$unique.id)]
    all.file$logits.y <- gene.result$logits.y[match(all.file$unique.id, gene.result$unique.id)]
      
    p1 <- ggplot(all.file[all.file$score %in% c(-1,1),], aes(x=logits.x, y=logits.y, col=GL.split)) +
      geom_point() +
      theme_bw() + 
      ggtitle(training.file$HGNC[!is.na(training.file$HGNC)][1]) + ggeasy::easy_center_title()
    p1 <- ggExtra::ggMarginal(p1, type="density", groupColour=T)
    p2 <- ggplot(all.file, aes(x=logits.x, y=logits.y, col=BP.split)) +
      geom_point() +
      theme_bw() + 
      ggtitle(training.file$HGNC[!is.na(training.file$HGNC)][1]) + ggeasy::easy_center_title()
    p2 <- ggExtra::ggMarginal(p2, type="density", groupColour=T)
    p <- patchwork::wrap_elements(p1) + patchwork::wrap_elements(p2) + plot_layout(nrow = 1)
    patch.plot[[fold+1]] <- p
  }
  assemble.logits <- assemble.logits / (length(folds) - 1)
  
  gene.result$logits.x <- pretrain.result$logits * cos(assemble.logits*pi/2)
  gene.result$logits.y <- pretrain.result$logits * sin(assemble.logits*pi/2)
  
  # all.file <- all.file
  all.file$unique.id <- paste(all.file$uniprotID, paste(all.file$ref, all.file$pos.orig, all.file$alt, sep = ""), sep = ":")
  
  all.file$logits.x <- gene.result$logits.x[match(all.file$unique.id, gene.result$unique.id)]
  all.file$logits.y <- gene.result$logits.y[match(all.file$unique.id, gene.result$unique.id)]
  
  all.file$BP.split[all.file$score==0] <- paste0("Benign")
  all.file$BP.split[all.file$score!=0] <- paste0("Patho")
  
  all.file$GL.split[all.file$score==0] <- paste0("Benign")
  all.file$GL.split[all.file$score==1] <- paste0("GoF")
  all.file$GL.split[all.file$score==-1] <- paste0("LoF")
  
  p1 <- ggplot(all.file[all.file$score %in% c(-1,1),], aes(x=logits.x, y=logits.y, col=GL.split)) +
    geom_point() +
    theme_bw() + 
    ggtitle(all.file$HGNC[!is.na(all.file$HGNC)][1]) + ggeasy::easy_center_title()
  p1 <- ggExtra::ggMarginal(p1, type="density", groupColour=T)
  p2 <- ggplot(all.file, aes(x=logits.x, y=logits.y, col=BP.split)) +
    geom_point() +
    theme_bw() + 
    ggtitle(all.file$HGNC[!is.na(all.file$HGNC)][1]) + ggeasy::easy_center_title()
  p2 <- ggExtra::ggMarginal(p2, type="density", groupColour=T)
  p <- patchwork::wrap_elements(p1) + patchwork::wrap_elements(p2) + plot_layout(nrow = 1)
  # ggsave(paste0(log.dir, gene, '.', xlower, '-', xupper, '.pdf'), p, width = min(nchar(gene.seq)/70*2, 49.9), height = 10)
  ggsave(paste0(log.dir, gene, '.radius.pdf'), p, width = 10, height = 5)
  
  # save 5 folds
  p <- patch.plot[[1]] / patch.plot[[2]] / patch.plot[[3]] / patch.plot[[4]] / patch.plot[[5]]
  ggsave(paste0(log.dir, gene, '.5folds.radius.pdf'), p, width = 20, height = 5*5)
}
