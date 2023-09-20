# visualize with dssp secondary structure 
library(ggplot2)
library(patchwork)
genes <- c("P15056", "P07949", "P04637", "Q14654", "Q14524")
# genes <- c("P15056", "P21802", "P07949", "P04637", "Q09428", "Q14654", "Q14524", "P60484")
aa.dict <- c('L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
             'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C')
log.dir <- '5genes.all.mut/CHPs.v4.esm.torchmdnet.small.TriAttn.StarPool.1dim/'
folds <- c(-1, 0:4)
source('~/Pipeline/plot.genes.scores.heatmap.R')
for (gene in genes) {
  assemble.logits <- 0
  all.training <- data.frame()
  patch.plot <- list()
  for (fold in folds) {
    if (fold == -1) {
      gene.result <- read.csv(paste0(log.dir, gene, '.pretrain.csv'), row.names = 1)
      pretrain.result <- gene.result
      training.file <- read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', 0, '/', gene, '.chps/training.csv'))[,c("HGNC", "uniprotID", "pos.orig", "ref", "alt", "score", "data_source")]
      testing.file <- read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', 0, '/', gene, '.chps/testing.csv'))[,c("HGNC", "uniprotID", "pos.orig", "ref", "alt", "score", "data_source")]
      training.file$score[training.file$score!=0] <- 1
      testing.file$score[testing.file$score!=0] <- 1
    } else {
      gene.result <- read.csv(paste0(log.dir, gene, '.fold.', fold, '.csv'), row.names = 1)
      training.file <- read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', fold, '/', gene, '.chps/training.csv'))[,c("HGNC", "uniprotID", "pos.orig", "ref", "alt", "score", "data_source")]
      testing.file <- read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', fold, '/', gene, '.chps/testing.csv'))[,c("HGNC", "uniprotID", "pos.orig", "ref", "alt", "score", "data_source")]
      training.file <- training.file[training.file$score %in% c(-1, 0, 1),]
      testing.file <- testing.file[testing.file$score %in% c(-1, 0, 1),]
    }
    if (!"logits" %in% colnames(gene.result) | fold != -1) {
      source('~/Pipeline/AUROC.R')
      if ("logits.2" %in% colnames(gene.result)) {
        logits <- gene.result[, c('logits.0', 'logits.1', 'logits.2')]
        logits <- t(apply(as.matrix(logits), 1, soft.max))
        gene.result$logits.0 <- logits[,1]
        gene.result$logits.1 <- logits[,2]
        gene.result$logits.2 <- logits[,3]
      } else if ("logits.1" %in% colnames(gene.result)) {
        logits.gof.lof <- gene.result$logits.1
        logits.gof <- (1 - logits.gof.lof)
        logits.lof <- logits.gof.lof
        logits <- cbind(pretrain.result$logits, logits.lof, logits.gof)
        gene.result$logits.0 <- pretrain.result$logits
        gene.result$logits.1 <- logits.lof
        gene.result$logits.2 <- logits.gof
      } else {
        logits.gof.lof <- gene.result$logits
        logits.gof <- (1 - logits.gof.lof)
        logits.lof <- logits.gof.lof
        logits <- cbind(pretrain.result$logits, logits.lof, logits.gof)
        gene.result$logits.0 <- pretrain.result$logits
        gene.result$logits.1 <- logits.lof
        gene.result$logits.2 <- logits.gof
      }
      assemble.logits <- assemble.logits + logits
      
      training.file$split <- "train"
      testing.file$split <- "test"
      all.file <- dplyr::bind_rows(training.file, testing.file)
      all.file$unique.id <- paste(all.file$uniprotID, paste(all.file$ref, all.file$pos.orig, all.file$alt, sep = ""), sep = ":")
      all.file$BP.split[all.file$score==0] <- paste0(all.file$split[all.file$score==0], ": Benign")
      all.file$BP.split[all.file$score!=0] <- paste0(all.file$split[all.file$score!=0], ": Patho")
      
      all.file$GL.split[all.file$score==0] <- paste0(all.file$split[all.file$score==0], ": Benign")
      all.file$GL.split[all.file$score==1] <- paste0(all.file$split[all.file$score==1], ": GoF")
      all.file$GL.split[all.file$score==-1] <- paste0(all.file$split[all.file$score==-1], ": LoF")
      
      all.file$logits.0 <- gene.result$logits.0[match(all.file$unique.id, gene.result$unique.id)]
      all.file$logits.1 <- gene.result$logits.1[match(all.file$unique.id, gene.result$unique.id)]
      all.file$logits.2 <- gene.result$logits.2[match(all.file$unique.id, gene.result$unique.id)]
      
      p1 <- ggplot(all.file[all.file$score %in% c(-1, 1),], aes(x=logits.1, col=GL.split)) +
        geom_density() +
        theme_bw() + 
        ggtitle(training.file$HGNC[!is.na(training.file$HGNC)][1]) + ggeasy::easy_center_title()
      p2 <- ggplot(all.file, aes(x=logits.0, col=BP.split)) +
        geom_density() +
        theme_bw() + 
        ggtitle(training.file$HGNC[!is.na(training.file$HGNC)][1]) + ggeasy::easy_center_title()
      p <- p1 + p2 + plot_layout(nrow = 1)
    } else {
      training.file$split <- "train"
      testing.file$split <- "test"
      
      all.file <- dplyr::bind_rows(training.file, testing.file)
      all.file$unique.id <- paste(all.file$uniprotID, paste(all.file$ref, all.file$pos.orig, all.file$alt, sep = ""), sep = ":")
      all.file$BP.split[all.file$score==0] <- paste0(all.file$split[all.file$score==0], ": Benign")
      all.file$BP.split[all.file$score!=0] <- paste0(all.file$split[all.file$score!=0], ": Patho")
      all.file$logits <- gene.result$logits[match(all.file$unique.id, gene.result$unique.id)]
      
      p <- ggplot(all.file, aes(x=logits, col=BP.split)) +
        geom_density() +
        theme_bw() + 
        ggtitle(training.file$HGNC[!is.na(training.file$HGNC)][1]) + ggeasy::easy_center_title()
    }
    if (fold != -1) {
      patch.plot[[fold+1]] <- p
      all.training <- dplyr::bind_rows(all.training, training.file, testing.file)
    } else {
      all.pretrain <- dplyr::bind_rows(training.file, testing.file)
      ggsave(paste0(log.dir, gene, '.pretrain.hist.pdf'), p, width = 10, height = 5)
    }
  }
  assemble.logits <- assemble.logits / (length(folds) - 1)
  if (!is.null(dim(assemble.logits))) {
    gene.result$logits.0 <- assemble.logits[,1]
    gene.result$logits.1 <- assemble.logits[,2]
    gene.result$logits.2 <- assemble.logits[,3]
    gene.result$logits <- NULL
  } else {
    gene.result$logits <- assemble.logits
  }
  if (!"logits" %in% colnames(gene.result)) {
    all.file <- all.training
    all.file$unique.id <- paste(all.file$uniprotID, paste(all.file$ref, all.file$pos.orig, all.file$alt, sep = ""), sep = ":")
    
    all.file$logits.0 <- gene.result$logits.0[match(all.file$unique.id, gene.result$unique.id)]
    all.file$logits.1 <- gene.result$logits.1[match(all.file$unique.id, gene.result$unique.id)]
    all.file$logits.2 <- gene.result$logits.2[match(all.file$unique.id, gene.result$unique.id)]
    
    all.file$BP.split[all.file$score==0] <- paste0("Benign")
    all.file$BP.split[all.file$score!=0] <- paste0("Patho")
    
    all.file$GL.split[all.file$score==0] <- paste0("Benign")
    all.file$GL.split[all.file$score==1] <- paste0("GoF")
    all.file$GL.split[all.file$score==-1] <- paste0("LoF")
    
    p1 <- ggplot(all.file[all.file$score %in% c(-1, 1),], aes(x=logits.1, col=GL.split)) +
      geom_density() +
      theme_bw() + 
      ggtitle(all.file$HGNC[!is.na(all.file$HGNC)][1]) + ggeasy::easy_center_title()
    p2 <- ggplot(all.file, aes(x=logits.0, col=BP.split)) +
      geom_density() +
      theme_bw() + 
      ggtitle(all.file$HGNC[!is.na(all.file$HGNC)][1]) + ggeasy::easy_center_title()
    p <- p1 + p2 + plot_layout(nrow = 1)
    # ggsave(paste0(log.dir, gene, '.', xlower, '-', xupper, '.pdf'), p, width = min(nchar(gene.seq)/70*2, 49.9), height = 10)
    ggsave(paste0(log.dir, gene, '.hist.pdf'), p, width = 10, height = 5)
  } 
  p <- patch.plot[[1]] / patch.plot[[2]] / patch.plot[[3]] / patch.plot[[4]] / patch.plot[[5]]
  ggsave(paste0(log.dir, gene, '.5folds.hist.pdf'), p, width = 20, height = 5*5)
}
