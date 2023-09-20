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
folds <- c(-1, 0:4)
source('~/Pipeline/plot.genes.scores.heatmap.R')
for (k in 1:length(genes)) {
  gene <- genes[k]
  pfam <- pfams[k]
  prot_data <- drawProteins::get_features(gene)
  prot_data <- drawProteins::feature_to_dataframe(prot_data)
  secondary <- prot_data[prot_data$type %in% c("HELIX", "STRAND", "TURN"),]
  secondary.df <- data.frame()
  if (dim(secondary)[1] > 0) {
    for (i in 1:dim(secondary)[1]) {
      sec.df <- data.frame(pos.orig = secondary$begin[i]:secondary$end[i],
                           alt = ".anno_secondary",
                           ANNO_secondary = secondary$type[i])
      secondary.df <- dplyr::bind_rows(secondary.df, sec.df)
    }
  } else {
    secondary.df <- data.frame(pos.orig = 1,
                               alt = '.anno_secondary',
                               ANNO_secondary = 'none')
  }
  #plot the domain types that only have one row of description
  others <- prot_data[prot_data$description != "NONE",]
  others <- others[!others$type %in% c("VARIANT", "MUTAGEN", "CONFLICT", "VAR_SEQ", "CHAIN"),]
  others$type[others$type=="MOD_RES"] <- "Phosphotyrosine"
  others$type[others$type=="DOMAIN"] <- others$description[others$type=="DOMAIN"]
  others$type <- tolower(others$type)
  unique.df <- data.frame()
  for (i in 1:dim(others)[1]) {
    if(i==1){
      if(!identical(others$type[i],others$type[i+1])){
        unq.df <- data.frame(pos.orig = others$begin[i]:others$end[i],
                             alt = paste0(".", others$type[i]),
                             ANNO_domain_type = others$type[i])
        unique.df <- dplyr::bind_rows(unique.df, unq.df)
      }
    }else{
      if(!identical(others$type[i],others$type[i+1]) && !identical(others$type[i],others$type[i-1])){
        unq.df <- data.frame(pos.orig = others$begin[i]:others$end[i],
                             alt = paste0(".", others$type[i]),
                             ANNO_domain_type = others$type[i])
        unique.df <- dplyr::bind_rows(unique.df, unq.df)
      }
    }
  }
  #plot the other domain types that have multiple kinds of descriptions
  multiple.df <- data.frame()
  for (i in 1:dim(others)[1]) {
    if(identical(others$type[i],others$type[i+1]) | identical(others$type[i],others$type[i-1])){
      mult.df <- data.frame(pos.orig = others$begin[i]:others$end[i],
                            alt = paste0(".", others$type[i]),
                            ANNO_domain_type = others$description[i])
      multiple.df <- dplyr::bind_rows(multiple.df, mult.df)
    }
  }
  
  gene.seq <- af2.seqs$seq[af2.seqs$uniprotID==gene]
  xlabs <- strsplit(gene.seq, "")[[1]]
  xlabs <- paste0(1:nchar(gene.seq), ":", xlabs)
  assemble.logits <- 0
  all.training <- data.frame()
  patch.plot <- list()
  for (fold in folds) {
    if (fold == -1) {
      gene.result <- read.csv(paste0(log.dir, pfam, '.pretrain.csv'))
      training.file <- read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', 0, '/', pfam, '.chps.even.uniprotID/training.csv'))[,c("uniprotID", "pos.orig", "ref", "alt", "score", "data_source")]
      testing.file <- read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', 0, '/', pfam, '.chps.even.uniprotID/testing.csv'))[,c("uniprotID", "pos.orig", "ref", "alt", "score")]
      training.file$score[training.file$score!=0] <- 1
      testing.file$score[testing.file$score!=0] <- 1
      training.file <- training.file[training.file$uniprotID == gene,]
      testing.file <- testing.file[testing.file$uniprotID == gene,]
      gene.result <- gene.result[gene.result$uniprotID == gene,]
      pretrain.result <- gene.result
    } else {
      gene.result <- read.csv(paste0(log.dir, pfam, '.fold.', fold, '.csv'))
      training.file <- read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', fold, '/', pfam, '.chps.even.uniprotID/training.csv'))[,c("uniprotID", "pos.orig", "ref", "alt", "score", "data_source")]
      testing.file <- read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', fold, '/', pfam, '.chps.even.uniprotID/testing.csv'))[,c("uniprotID", "pos.orig", "ref", "alt", "score")]
      training.file <- training.file[training.file$score %in% c(-1, 0, 1),]
      testing.file <- testing.file[testing.file$score %in% c(-1, 0, 1),]
      training.file <- training.file[training.file$uniprotID == gene,]
      testing.file <- testing.file[testing.file$uniprotID == gene,]
      gene.result <- gene.result[gene.result$uniprotID == gene,]
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
      gene.result[,"logits.2/logits.1*logits.0"] <- gene.result$logits.2 / gene.result$logits.1 * gene.result$logits.0
      ps <- list()
      col.to.plot <- paste0("logits.", c(0:2, "2/logits.1*logits.0"))
      for (j in 1:4) {
        ps[[j]] <- ggplot() +
          geom_tile(data=gene.result, aes_string(x="pos.orig", y="alt", fill=col.to.plot[j], width=1)) + 
          ggrepel::geom_text_repel(data=gene.result[!duplicated(gene.result$aaChg),], aes(label=aaChg, x=pos.orig, y=alt), col='grey', size=0.5) +
          scale_fill_gradientn(colors = c("light blue", "white", "pink"), limits=c(0,1), na.value = 'grey') + labs(fill=col.to.plot[j]) +
          scale_x_continuous(breaks=seq(0, nchar(gene.seq), 50)) +
          ggnewscale::new_scale_fill() +
          geom_tile(data=training.file, aes(x=pos.orig, y=alt, fill=score)) +
          scale_fill_gradientn(colors = c("blue", "white", "red"), limits=c(-1,1)) +
          ggnewscale::new_scale_fill() +
          geom_tile(data=secondary.df, aes(x=pos.orig, y=alt, fill=ANNO_secondary, width=1)) +
          ggnewscale::new_scale_fill() +
          geom_tile(data=unique.df, aes(x=pos.orig, y=alt, fill=ANNO_domain_type, width=1),show.legend = F) +
          ggnewscale::new_scale_fill() +
          geom_tile(data=multiple.df, aes(x=pos.orig, y=alt, fill=ANNO_domain_type, width=1),show.legend = F) +
          theme_bw() + 
          ggtitle(gene.name[k]) + ggeasy::easy_center_title()
      }
      p <- ps[[1]] + ps[[2]] + ps[[3]] + ps[[4]] + plot_layout(nrow = 1)
    } else {
      p <- ggplot() +
        geom_tile(data=gene.result, aes(x=pos.orig, y=alt, fill=logits, width=1)) + 
        ggrepel::geom_text_repel(data=gene.result[!duplicated(gene.result$aaChg),], aes(label=aaChg, x=pos.orig, y=alt), col='grey', size=0.5) +
        scale_fill_gradientn(colors = c("light blue", "white", "pink"), limits=c(0,1), na.value = 'grey') +
        scale_x_continuous(breaks=seq(0, nchar(gene.seq), 50)) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=training.file, aes(x=pos.orig, y=alt, fill=score)) +
        scale_fill_gradientn(colors = c("blue", "white", "red"), limits=c(0,1)) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=secondary.df, aes(x=pos.orig, y=alt, fill=ANNO_secondary, width=1)) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=unique.df, aes(x=pos.orig, y=alt, fill=ANNO_domain_type, width=1),show.legend = F) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=multiple.df, aes(x=pos.orig, y=alt, fill=ANNO_domain_type, width=1),show.legend = F) +
        theme_bw() + 
        ggtitle(gene.name[k]) + ggeasy::easy_center_title()
    }
    if (fold != -1) {
      patch.plot[[fold+1]] <- p
      all.training <- dplyr::bind_rows(all.training, training.file, testing.file)
    } else {
      all.pretrain <- dplyr::bind_rows(training.file, testing.file)
      ggsave(paste0(log.dir, pfam, '.', gene, '.pretrain.pdf'), p, width = 25, height = 4)
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
    gene.result[,"logits.2/logits.1*logits.0"] <- gene.result$logits.2 / gene.result$logits.1 * gene.result$logits.0
    gene.result.to.plot <- gene.result
    all.training.to.plot <- all.training
    secondary.df.to.plot <- secondary.df
    unique.df.to.plot <- unique.df
    multiple.df.to.plot <- multiple.df
    # xlower <- 800
    # xupper <- 1300
    # gene.result.to.plot <- gene.result.to.plot[gene.result.to.plot$pos.orig <= xupper &
    #                                              gene.result.to.plot$pos.orig >= xlower, ]
    ps <- list()
    col.to.plot <- paste0("logits.", c(0:2, "2/logits.1*logits.0"))
    for (j in 1:4) {
      if (j == 1) {
        all.training.to.plot.plot <- all.pretrain
      } else {
        all.training.to.plot.plot <- all.training.to.plot
      }
      ps[[j]] <- ggplot() +
        geom_tile(data=gene.result, aes_string(x="pos.orig", y="alt", fill=col.to.plot[j], width=1)) + labs(fill=col.to.plot[j]) +
        ggrepel::geom_text_repel(data=gene.result[!duplicated(gene.result$aaChg),], aes(label=aaChg, x=pos.orig, y=alt), alpha=0.5, size=2) +
        scale_fill_gradientn(colors = c("light blue", "white", "pink"), na.value = 'grey') +
        scale_x_continuous(breaks=seq(0, nchar(gene.seq), 50)) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=all.training.to.plot.plot, aes(x=pos.orig, y=alt, fill=score, width=1, height=1)) +
        scale_fill_gradientn(colors = c("blue", "white", "red")) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=secondary.df.to.plot, aes(x=pos.orig, y=alt, fill=ANNO_secondary, width=1, height=1)) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=unique.df.to.plot, aes(x=pos.orig, y=alt, fill=ANNO_domain_type, width=1, height=1),show.legend = F) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=multiple.df.to.plot, aes(x=pos.orig, y=alt, fill=ANNO_domain_type, width=1, height=1),show.legend = F) +
        theme_bw() + 
        ggtitle(gene.name[k]) + ggeasy::easy_center_title()
    }
    p <- ps[[1]] + ps[[2]] + ps[[3]] + ps[[4]] + plot_layout(nrow=4)
    # ggsave(paste0(log.dir, gene, '.', xlower, '-', xupper, '.pdf'), p, width = min(nchar(gene.seq)/70*2, 49.9), height = 10)
    ggsave(paste0(log.dir, pfam, '.', gene, '.pdf'), p, width = max(25, min(nchar(gene.seq)/70, 49.9)), height = 20)
  } else {
    p <- ggplot() +
      geom_tile(data=gene.result, aes(x=pos.orig, y=alt, fill=logits, width=1)) +
      ggrepel::geom_text_repel(data=gene.result[!duplicated(gene.result$aaChg),], aes(label=aaChg, x=pos.orig, y=alt), col='grey', size=0.5) +
      scale_fill_gradientn(colors = c("light blue", "white", "pink"), limits=c(0,1), na.value = 'grey') +
      scale_x_continuous(breaks=seq(0, nchar(gene.seq), 50)) +
      ggnewscale::new_scale_fill() +
      geom_tile(data=all.training, aes(x=pos.orig, y=alt, fill=score)) +
      scale_fill_gradientn(colors = c("blue", "white", "red"), limits=c(-1,1)) +
      ggnewscale::new_scale_fill() +
      geom_tile(data=secondary.df, aes(x=pos.orig, y=alt, fill=ANNO_secondary)) +
      theme_bw() + 
      ggtitle(gene.name[k]) + ggeasy::easy_center_title()
    ggsave(paste0(log.dir, pfam, '.', gene, '.pdf'), p, width = nchar(gene.seq)/50, height = 4)
  }
  p <- patch.plot[[1]] / patch.plot[[2]] / patch.plot[[3]] / patch.plot[[4]] / patch.plot[[5]]
  ggsave(paste0(log.dir, pfam, '.', gene, '.5folds.pdf'), p, width = max(25, min(nchar(gene.seq)/70*4, 49.9)), height = 4*5)
}
