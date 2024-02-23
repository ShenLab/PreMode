# visualize with dssp secondary structure 
library(ggplot2)
library(patchwork)
library(bio3d)
gene <- "P60484"
data.idxes <- c(255,279,257,265,252,268,231,235)
label <- c('domiN', 'domiN', 'beni', 'beni', 'loss', 'loss', 'struc_loss', 'struc_loss')
log.dir <- '5genes.all.mut/CHPs.v4.esm.torchmdnet.small.TriAttn.StarPool.1dim/'

library(reticulate)

af2.seqs <- read.csv('~/Data/Protein/alphafold2_v4/swissprot_and_human.full.seq.csv', row.names = 1)

source('~/Pipeline/plot.genes.scores.heatmap.R')
prot_data <- drawProteins::get_features(gene)
prot_data <- drawProteins::feature_to_dataframe(prot_data)
secondary <- prot_data[prot_data$type %in% c("HELIX", "STRAND", "TURN"),]
secondary.df <- data.frame()
for (i in 1:dim(secondary)[1]) {
  sec.df <- data.frame(pos.orig = secondary$begin[i]:secondary$end[i],
                       anno = ".anno_secondary",
                       ANNO_secondary = secondary$type[i])
  secondary.df <- dplyr::bind_rows(secondary.df, sec.df)
}
#plot the AF2 predicted secondary.df and rsa
gene.af2.file <- paste0("~/Data/Protein/alphafold2_v4/swissprot/AF-",
                        gene, '-F', 1,
                        '-model_v4.pdb.gz')
dssp.res <- dssp(read.pdb(gene.af2.file), 
                 exefile='/share/vault/Users/gz2294/miniconda3/bin/mkdssp')
af2.secondary <- rbind(cbind(as.data.frame(dssp.res$helix)[,1:4], type="HELIX"), 
                       cbind(as.data.frame(dssp.res$sheet), type="STRAND"), 
                       cbind(as.data.frame(dssp.res$turn), type="TURN"))
for (i in 1:dim(af2.secondary)[1]) {
  sec.df <- data.frame(pos.orig = af2.secondary$start[i]:af2.secondary$end[i],
                       anno = ".anno_af2_secondary",
                       ANNO_secondary = af2.secondary$type[i])
  secondary.df <- dplyr::bind_rows(secondary.df, sec.df)
}
rsa.df <- data.frame(pos.orig=1:length(dssp.res$acc), anno = ".anno_af2_rsa", 
                     ANNO_RSA=(dssp.res$acc)/max(dssp.res$acc))
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
                           anno = paste0(".", others$type[i]),
                           ANNO_domain_type = others$type[i])
      unique.df <- dplyr::bind_rows(unique.df, unq.df)
    }
  }else{
    if(!identical(others$type[i],others$type[i+1]) && !identical(others$type[i],others$type[i-1])){
      unq.df <- data.frame(pos.orig = others$begin[i]:others$end[i],
                           anno = paste0(".", others$type[i]),
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
                          anno = paste0(".", others$type[i]),
                          ANNO_domain_type = others$description[i])
    multiple.df <- dplyr::bind_rows(multiple.df, mult.df)
  }
}

for (data.idx in data.idxes) {
  attn.pickle.file <- paste0(log.dir, gene, '.attn.', data.idx, '.pkl')
  attn.file <- reticulate::py_load_object(attn.pickle.file)
  gene.seq <- af2.seqs$seq[af2.seqs$uniprotID==gene]
  node.freq <- table(attn.file[[2]])
  center.node <- as.numeric(names(node.freq)[which.max(node.freq)])
  attn <- list()
  attn[[1]] <- attn.file[[1]][[1]][attn.file[[1]][[1]][,1]==center.node,3:dim(attn.file[[1]][[1]])[2]]
  attn[[2]] <- attn.file[[1]][[1]][attn.file[[1]][[1]][,2]==center.node,3:dim(attn.file[[1]][[1]])[2]]
  attn[[3]] <- attn.file[[1]][[3]]
  gradient.1 <- as.matrix(rowSums(attn.file[[6]][[1]]))
  gradient.2 <- as.matrix(rowSums(attn.file[[6]][[2]]))
  library(reshape2)
  ps <- list()
  n.annos <- length(unique(unique.df$anno)) + length(unique(multiple.df$anno)) + 1
  for (j in 1:3) {
    gene.result <- melt(attn[[j]])
    gradient.1.result <- melt(gradient.1)
    gradient.2.result <- melt(gradient.2)
    gradient.1.result$Var2 <- "Gradient-Stability"
    gradient.2.result$Var2 <- "Gradient-Enzyme"
    gradient.result <- rbind(gradient.1.result, gradient.2.result)
    colnames(gradient.result) <- c("pos.orig", "anno", "grad.Value")
    colnames(gene.result) <- c("pos.orig", "anno", "attn.Value")
    gene.result$anno <- sprintf("Head-%02d", gene.result$anno)
    gradient.result$pos.orig <- gradient.result$pos.orig + attn.file[[4]][[1]] + attn.file[[4]][[3]] - 2
    gene.result$pos.orig <- gene.result$pos.orig + attn.file[[4]][[1]] + attn.file[[4]][[3]] - 2
    gene.result <- gene.result[gene.result$pos.orig != attn.file[[4]][[1]] + attn.file[[4]][[3]] + center.node - 1,]
    ps[[j]] <- ggplot() +
      geom_tile(data=gene.result, aes(x=pos.orig, y=anno, fill=attn.Value)) +
      scale_fill_gradientn(colors = c("light blue", "white", "pink"), na.value = 'grey') +
      scale_x_continuous(breaks=seq(0, nchar(gene.seq), 50)) +
      geom_segment(aes(x = attn.file[[3]]$pos.orig , y = 1,
                       xend = attn.file[[3]]$pos.orig, yend = n.annos), col='grey') +
      # geom_vline(xintercept = attn.file[[3]]$pos.orig, col='grey') +
      ggnewscale::new_scale_fill() +
      geom_tile(data=gradient.result, aes(x=pos.orig, y=anno, fill=grad.Value, width=1)) +
      scale_fill_gradientn(colors = c("blue", "white", "red"), na.value = 'grey') +
      ggnewscale::new_scale_fill() +
      geom_tile(data=secondary.df, aes(x=pos.orig, y=anno, fill=ANNO_secondary, width=1)) +
      ggnewscale::new_scale_fill() +
      geom_tile(data=rsa.df, aes(x=pos.orig, y=anno, fill=ANNO_RSA, width=1, height=1)) +
      scale_fill_gradientn(colors = c("grey", "white", "blue")) +
      ggnewscale::new_scale_fill() +
      geom_tile(data=unique.df, aes(x=pos.orig, y=anno, fill=ANNO_domain_type, width=1), show.legend = F) +
      ggnewscale::new_scale_fill() +
      geom_tile(data=multiple.df, aes(x=pos.orig, y=anno, fill=ANNO_domain_type, width=1), show.legend = F) +
      theme_bw() + 
      ggtitle(paste0(gene, ":", attn.file[[3]]$aaChg, ":", label[which(data.idxes==data.idx)])) + 
      ggeasy::easy_center_title()
  }
  p <- ps[[1]] / ps[[2]] / ps[[3]] 
  ggsave(paste0(log.dir, gene, '.attn.', attn.file[[3]]$aaChg, '.', label[which(data.idxes==data.idx)], '.pdf'), p, width = max(25, min(nchar(gene.seq)/70*4, 49.9)), height = 4*5)
}
