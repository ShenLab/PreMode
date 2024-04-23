# visualize with dssp secondary structure 
library(ggplot2)
library(patchwork)
library(bio3d)
genes <- c("P60484")
# plot original scores
original <- rbind(read.csv('../data.files/PTEN.bin/train.seed.0.csv'),
                  read.csv('../data.files/PTEN.bin/test.seed.0.csv'))

af2.seqs <- read.csv('genes.full.seq.csv', row.names = 1)
aa.dict <- c('L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
             'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C')
log.dir <- '5genes.all.mut/PreMode/'
folds <- c(-1, 0:4)
source('~/Pipeline/plot.genes.scores.heatmap.R')
for (gene in genes) {
  prot_data <- drawProteins::get_features(gene)
  prot_data <- drawProteins::feature_to_dataframe(prot_data)
  secondary <- prot_data[prot_data$type %in% c("HELIX", "STRAND", "TURN"),]
  secondary.df <- data.frame()
  for (i in 1:dim(secondary)[1]) {
    sec.df <- data.frame(pos.orig = secondary$begin[i]:secondary$end[i],
                         alt = ".anno_secondary",
                         ANNO_secondary = secondary$type[i])
    secondary.df <- dplyr::bind_rows(secondary.df, sec.df)
  }
  #plot the AF2 predicted secondary.df and rsa
  gene.af2.file <- paste0("../data.files/af2.files/AF-",
                          gene, '-F', 1,
                          '-model_v4.pdb.gz')
  dssp.res <- dssp(read.pdb(gene.af2.file), 
                   exefile='/share/vault/Users/gz2294/miniconda3/bin/mkdssp')
  pdb.res <- read.pdb(gene.af2.file)
  plddt.res <- pdb.res$atom$b[pdb.res$calpha]
  af2.secondary <- rbind(cbind(as.data.frame(dssp.res$helix)[,1:4], type="HELIX"), 
                         cbind(as.data.frame(dssp.res$sheet), type="STRAND"), 
                         cbind(as.data.frame(dssp.res$turn), type="TURN"))
  for (i in 1:dim(af2.secondary)[1]) {
    sec.df <- data.frame(pos.orig = af2.secondary$start[i]:af2.secondary$end[i],
                         alt = ".anno_af2_secondary",
                         ANNO_secondary = af2.secondary$type[i])
    secondary.df <- dplyr::bind_rows(secondary.df, sec.df)
  }
  rsa.df <- data.frame(pos.orig=1:length(dssp.res$acc), alt = ".anno_af2_rsa", 
                       ANNO_RSA=(dssp.res$acc)/max(dssp.res$acc))
  plddt.df <- data.frame(pos.orig=1:length(plddt.res), alt = ".anno_af2_pLDDT", 
                         ANNO_pLDDT=plddt.res)
  #plot the domain types that only have one row of description
  others <- prot_data[prot_data$description != "NONE",]
  others <- others[!others$type %in% c("VARIANT", "MUTAGEN", "CONFLICT", "VAR_SEQ", "CHAIN"),]
  others$type[others$type=="MOD_RES"] <- "post transl. mod."
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
  all.pretrain <- data.frame()
  patch.plot <- list()
  for (fold in folds) {
    if (fold == -1) {
      gene.result <- read.csv(paste0(log.dir, gene, '.pretrain.csv'), row.names = 1)
      pretrain.result <- gene.result
      training.file <- read.csv(paste0('../data.files/pretrain/training.csv'))[,c("HGNC", "uniprotID", "pos.orig", "ref", "alt", "score", "data_source")]
      training.file$score[training.file$score!=0] <- 1
      training.file <- training.file[training.file$uniprotID == gene,]
      pretrain.training.file <- training.file
    } else {
      gene.result <- read.csv(paste0(log.dir, gene, '.fold.', fold, '.csv'), row.names = 1)
      training.file <- read.csv(paste0('../data.files/PTEN.bin/train.seed.', fold, '.csv'))[,c("HGNC", "pos.orig", "ref", "alt", "score.1", "score.2")]
      training.file$score <- NA
      testing.file <- read.csv(paste0('../data.files/PTEN.bin/test.seed.', fold, '.csv'))[,c("HGNC", "pos.orig", "ref", "alt", "score.1", "score.2")]
      testing.file$score <- NA
    }
    if (!"logits" %in% colnames(gene.result) | fold != -1) {
      logits <- cbind(pretrain.result$logits, gene.result$logits.0, gene.result$logits.1)
      gene.result$logits.2 <- gene.result$logits.1
      gene.result$logits.1 <- gene.result$logits.0
      gene.result$logits.0 <- pretrain.result$logits
      assemble.logits <- assemble.logits + logits
      ps <- list()
      col.to.plot <- paste0("logits.", c(0:2))
      score.to.plot <- c('score', 'score.1', 'score.2')
      data.train <- list(pretrain.training.file, training.file, training.file)
      for (j in 1:3) {
        ps[[j]] <- ggplot() +
          geom_tile(data=gene.result, aes_string(x="pos.orig", y="alt", fill=col.to.plot[j])) + 
          scale_fill_gradientn(colors = c("light blue", "white", "pink"), na.value = 'grey') + labs(fill=col.to.plot[j]) +
          scale_x_continuous(breaks=seq(0, nchar(gene.seq), 50)) +
          ggnewscale::new_scale_fill() +
          geom_tile(data=data.train[[j]], aes_string(x="pos.orig", y="alt", fill=score.to.plot[j])) +
          scale_fill_gradientn(colors = c("blue", "white", "red")) +
          ggnewscale::new_scale_fill() +
          geom_tile(data=secondary.df, aes(x=pos.orig, y=alt, fill=ANNO_secondary, width=1)) +
          ggnewscale::new_scale_fill() +
          geom_tile(data=unique.df, aes(x=pos.orig, y=alt, fill=ANNO_domain_type, width=1),show.legend = F) +
          ggnewscale::new_scale_fill() +
          geom_tile(data=multiple.df, aes(x=pos.orig, y=alt, fill=ANNO_domain_type, width=1),show.legend = F) +
          theme_bw() + 
          ggtitle("PTEN") + ggeasy::easy_center_title()
      }
      p <- ps[[1]] + ps[[2]] + ps[[3]] + plot_layout(nrow = 1)
    } else {
      p <- ggplot() +
        geom_tile(data=gene.result, aes(x=pos.orig, y=alt, fill=logits)) + 
        scale_fill_gradientn(colors = c("light blue", "white", "pink"), na.value = 'grey') +
        scale_x_continuous(breaks=seq(0, nchar(gene.seq), 50)) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=training.file, aes(x=pos.orig, y=alt, fill=score)) +
        scale_fill_gradientn(colors = c("blue", "white", "red")) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=secondary.df, aes(x=pos.orig, y=alt, fill=ANNO_secondary, width=1)) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=unique.df, aes(x=pos.orig, y=alt, fill=ANNO_domain_type, width=1),show.legend = F) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=multiple.df, aes(x=pos.orig, y=alt, fill=ANNO_domain_type, width=1),show.legend = F) +
        theme_bw() + 
        ggtitle("PTEN") + ggeasy::easy_center_title()
    }
    if (fold != -1) {
      patch.plot[[fold+1]] <- p
      all.training <- dplyr::bind_rows(all.training, training.file, testing.file)
    } else {
      all.pretrain <- dplyr::bind_rows(all.pretrain, pretrain.training.file)
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
    gene.result$logits.diff <- gene.result$logits.2 - gene.result$logits.1
    gene.result.to.plot <- gene.result
    all.training.to.plot <- all.training
    secondary.df.to.plot <- secondary.df
    unique.df.to.plot <- unique.df
    multiple.df.to.plot <- multiple.df
    ps <- list()
    col.to.plot <- c(paste0("logits.", c(0:2)), 'logits.diff')
    all.training.to.plot$score.diff <- 0
    all.training.to.plot$score.diff[all.training.to.plot$score.1==0 & all.training.to.plot$score.2==1] <- 1
    all.training.to.plot$score.diff[all.training.to.plot$score.1==1 & all.training.to.plot$score.2==0] <- -1
    all.training.to.plot$score.diff[all.training.to.plot$score.1==1 & all.training.to.plot$score.2==1] <- NA
    score.to.plot <- c('score', 'score.1', 'score.2', 'score.diff')
    score.name <- c('Patho', 'Stability', 'Enzyme', 'Enzyme-Stability')
    for (j in 1:4) {
      if (j %in% c(1)) {
        all.training.to.plot.plot <- all.pretrain
      } else {
        all.training.to.plot.plot <- all.training.to.plot
      }
      ps[[j]] <- ggplot() +
        geom_tile(data=gene.result, aes_string(x="pos.orig", y="alt", fill=col.to.plot[j])) + labs(fill=col.to.plot[j]) +
        scale_fill_gradientn(colors = c("light blue", "white", "pink"), na.value = 'grey') +
        scale_x_continuous(breaks=seq(0, nchar(gene.seq), 50), minor_breaks = seq(0, nchar(gene.seq), 10)) +
        labs(fill=score.name[j]) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=all.training.to.plot.plot, aes_string(x="pos.orig", y="alt", fill=score.to.plot[j], width=1, height=1)) +
        scale_fill_gradientn(colors = c("blue", "white", "red"), limits = c(0,1)) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=secondary.df.to.plot, aes(x=pos.orig, y=alt, fill=ANNO_secondary, width=1, height=1)) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=rsa.df, aes(x=pos.orig, y=alt, fill=ANNO_RSA, width=1, height=1)) +
        scale_fill_gradientn(colors = c("grey", "blue")) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=plddt.df, aes(x=pos.orig, y=alt, fill=ANNO_pLDDT, width=1, height=1)) +
        scale_fill_gradientn(colors = c("orange", "yellow", "lightblue", "blue")) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=unique.df.to.plot, aes(x=pos.orig, y=alt, fill=ANNO_domain_type, width=1, height=1),show.legend = F) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=multiple.df.to.plot, aes(x=pos.orig, y=alt, fill=ANNO_domain_type, width=1, height=1),show.legend = F) +
        theme_bw() + theme(legend.position="bottom") +
        ggtitle("PTEN") + ggeasy::easy_center_title()
    }
    p <- ps[[1]] + ps[[2]] + ps[[3]] + ps[[4]] + plot_layout(nrow=4)
    p <- ps[[1]] + ps[[4]] + plot_layout(nrow=2)
    ggsave(paste0(log.dir, gene, '.part.pdf'), p, width = max(25, min(nchar(gene.seq)/70, 49.9)), height = 10)
  } else {
    p <- ggplot() +
      geom_tile(data=gene.result, aes(x=pos.orig, y=alt, fill=logits)) +
      scale_x_continuous(breaks=seq(0, nchar(gene.seq), 50), minor_breaks = seq(0, nchar(gene.seq), 10)) +
      scale_fill_gradientn(colors = c("light blue", "white", "pink"), na.value = 'grey') +
      ggnewscale::new_scale_fill() +
      geom_tile(data=all.training, aes(x=pos.orig, y=alt, fill=score)) +
      scale_fill_gradientn(colors = c("blue", "white", "red")) +
      ggnewscale::new_scale_fill() +
      geom_tile(data=secondary.df, aes(x=pos.orig, y=alt, fill=ANNO_secondary)) +
      theme_bw() + 
      scale_x_continuous(breaks=seq(0, nchar(gene.seq), 100)) +
      ggtitle("PTEN") + ggeasy::easy_center_title()
    ggsave(paste0(log.dir, gene, '.pdf'), p, width = nchar(gene.seq)/50, height = 4)
  }
  p <- patch.plot[[1]] / patch.plot[[2]] / patch.plot[[3]] / patch.plot[[4]] / patch.plot[[5]]
}
system('mv 5genes.all.mut/PreMode/P60484.part.pdf figs/fig.sup.12.pdf')
