# visualize with dssp secondary structure 
library(ggplot2)
library(patchwork)
library(bio3d)
genes <- c("P60484")
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
  fold <- 0
  for (subset in c(1,2,4,6)) {
    gene.result <- read.csv(paste0(log.dir, gene, '.subset.', subset, '.fold.', fold, '.csv'), row.names = 1)
    pretrain.result <- read.csv(paste0(log.dir, gene, '.pretrain.csv'), row.names = 1)
    training.file <- read.csv(paste0('../data.files/PTEN.bin/training.', subset, '.', fold, '.csv'))[,c("HGNC", "pos.orig", "ref", "alt", "score.1", "score.2")]
    training.file$score <- NA
    testing.file <- read.csv(paste0('../data.files/PTEN.bin/test.seed.0.csv'))[,c("HGNC", "pos.orig", "ref", "alt", "score.1", "score.2")]
    testing.file$score <- NA
    
    logits <- cbind(pretrain.result$logits, gene.result$logits.0, gene.result$logits.1)
    gene.result$logits.2 <- gene.result$logits.1
    gene.result$logits.1 <- gene.result$logits.0
    gene.result$logits.0 <- pretrain.result$logits
    ps <- list()
    col.to.plot <- paste0("logits.", c(0:2))
    score.to.plot <- c('score', 'score.1', 'score.2')
    pretrain.training.file <- read.csv(paste0('../data.files/pretrain/training.csv'))[,c("HGNC", "uniprotID", "pos.orig", "ref", "alt", "score", "data_source")]
    pretrain.training.file$score[pretrain.training.file$score!=0] <- 1
    pretrain.training.file <- pretrain.training.file[pretrain.training.file$uniprotID == gene,]
    data.train <- list(pretrain.training.file, training.file, training.file)
    for (j in 1:3) {
      ps[[j]] <- ggplot() +
        geom_tile(data=gene.result, aes_string(x="pos.orig", y="alt", fill=col.to.plot[j])) + 
        scale_fill_gradientn(colors = c("light blue", "white", "pink"), na.value = 'grey') + labs(fill=col.to.plot[j]) +
        scale_x_continuous(breaks=seq(0, nchar(gene.seq), 50), minor_breaks = seq(0, nchar(gene.seq), 10)) +
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
    
    gene.result$logits.diff <- gene.result$logits.2 - gene.result$logits.1
    gene.result.to.plot <- gene.result
    all.training.to.plot <- training.file
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
        all.training.to.plot.plot <- pretrain.training.file
      } else {
        all.training.to.plot.plot <- all.training.to.plot
      }
      ps[[j]] <- ggplot() +
        geom_tile(data=gene.result, aes_string(x="pos.orig", y="alt", fill=col.to.plot[j])) + labs(fill=col.to.plot[j]) +
        scale_fill_gradientn(colors = c("light blue", "white", "pink"), na.value = 'grey', 
                             values = c(0, (0-min(gene.result[,col.to.plot[j]], na.rm = T))/(max(gene.result[,col.to.plot[j]], na.rm = T)-min(gene.result[,col.to.plot[j]], na.rm = T)), 1)) +
        scale_x_continuous(breaks=seq(0, nchar(gene.seq), 50), minor_breaks = seq(0, nchar(gene.seq), 10)) + labs(fill=score.name[j]) +
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
    p <- ps[[4]] 
    ggsave(paste0(log.dir, gene, '.subset.', subset, '.fold.', fold, '.part.pdf'), p, width = max(25, min(nchar(gene.seq)/70, 49.9)), height = 5)
  }
}
system('mv 5genes.all.mut/PreMode/P60484.subset.1.fold.0.part.pdf figs/fig.sup.13a.pdf')
system('mv 5genes.all.mut/PreMode/P60484.subset.2.fold.0.part.pdf figs/fig.sup.13b.pdf')
system('mv 5genes.all.mut/PreMode/P60484.subset.4.fold.0.part.pdf figs/fig.sup.13c.pdf')
system('mv 5genes.all.mut/PreMode/P60484.subset.6.fold.0.part.pdf figs/fig.sup.13d.pdf')
