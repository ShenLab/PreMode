# visualize with dssp secondary structure 
library(ggplot2)
library(bio3d)
library(patchwork)
genes <- c("P15056", "P07949", "P04637", 
           "Q14654", "Q14524", "Q99250", 
           "O00555",
           # "P22607", 
           "P21802")
gene.names <- c("BRAF", "RET", "TP53", 
                "KCNJ11", "SCN5A", "SCN2A",
                "CACNA1A", 
                # "FGFR3",
                "FGFR2")
# genes <- c("P22607", "P21802")
# genes <- c("P15056", "P21802", "P07949", "P04637", "Q09428", "Q14654", "Q14524")

af2.seqs <- read.csv('~/Data/Protein/alphafold2_v4/swissprot_and_human.full.seq.csv', row.names = 1)
aa.dict <- c('L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
             'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C')
log.dir <- '5genes.all.mut/CHPs.v4.esm.torchmdnet.small.TriAttn.StarPool.1dim/'
auc.dir <- '../scripts/CHPs.v4.esm.dssp.small.StarAttn.MSA.StarPool.1dim/'
use.logits <- 'meta.logits'
folds <- c(-1, 0:4)
source('~/Pipeline/plot.genes.scores.heatmap.R')
source('~/Pipeline/AUROC.R')

for (o in 1:length(genes)) {
  gene <- genes[o]
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
                         alt = ".anno_af2_secondary",
                         ANNO_secondary = af2.secondary$type[i])
    secondary.df <- dplyr::bind_rows(secondary.df, sec.df)
  }
  rsa.df <- data.frame(pos.orig=1:length(dssp.res$acc), alt = ".anno_af2_rsa", 
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
  weighted.assemble.logits <- 0
  auc.weights <- NULL
  all.training <- data.frame()
  patch.plot <- list()
  for (fold in folds) {
    if (fold == -1) {
      gene.result <- read.csv(paste0(log.dir, gene, '.pretrain.csv'), row.names = 1)
      pretrain.result <- gene.result
      training.file <- read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', 0, '/', gene, '/training.csv'))[,c("HGNC", "pos.orig", "ref", "alt", "score", "data_source")]
      testing.file <- read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', 0, '/', gene, '/testing.csv'))[,c("HGNC", "pos.orig", "ref", "alt", "score", "data_source")]
      training.file$score[training.file$score!=0] <- 1
      testing.file$score[testing.file$score!=0] <- 1
      all.logits <- matrix(NA, nrow = dim(gene.result)[1], ncol = 5)
      colnames(all.logits) <- paste0('model.', 0:4)
    } else {
      gene.result <- read.csv(paste0(log.dir, gene, '.fold.', fold, '.csv'), row.names = 1)
      training.file <- read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', fold, '/', gene, '/training.csv'))[,c("HGNC", "pos.orig", "ref", "alt", "score", "data_source")]
      testing.file <- read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', fold, '/', gene, '/testing.csv'))[,c("HGNC", "pos.orig", "ref", "alt", "score", "data_source")]
      training.file <- training.file[training.file$score %in% c(-1, 0, 1),]
      testing.file <- testing.file[testing.file$score %in% c(-1, 0, 1),]
      auc.file <- read.csv(paste0(auc.dir, 'ICC.5fold.csv'))
      auc <- auc.file$min.val.auc[startsWith(auc.file$task, paste0(gene, ':')) & auc.file$fold==fold]
      auc.weights <- c(auc.weights, auc)
      all.logits[,fold+1] <- gene.result$logits
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
      weighted.assemble.logits <- weighted.assemble.logits + logits * auc
      gene.result[,"logits.2/logits.1*logits.0"] <- (gene.result$logits.2 - gene.result$logits.1) * gene.result$logits.0
      ps <- list()
      col.to.plot <- paste0("logits.", c(0:2, "2/logits.1*logits.0"))
      for (j in 1:4) {
        ps[[j]] <- ggplot() +
          geom_tile(data=gene.result, aes_string(x="pos.orig", y="alt", fill=col.to.plot[j])) + 
          scale_fill_gradientn(colors = c("light blue", "white", "pink"), na.value = 'grey') + labs(fill=col.to.plot[j]) +
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
          theme_bw() + theme(legend.position="bottom", legend.direction="vertical") +
          ggtitle(gene.names[o]) + ggeasy::easy_center_title()
      }
      p <- ps[[2]] + ps[[3]] + ps[[4]] + plot_layout(nrow = 1)
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
        theme_bw() + theme(legend.position="bottom", legend.direction="vertical") +
        ggtitle(gene.names[o]) + ggeasy::easy_center_title()
    }
    if (fold != -1) {
      patch.plot[[fold+1]] <- p
      all.training <- dplyr::bind_rows(all.training, training.file, testing.file)
    } else {
      all.pretrain <- dplyr::bind_rows(training.file, testing.file)
      ggsave(paste0(log.dir, gene, '.pretrain.pdf'), p, width = 25, height = 4)
    }
  }
  assemble.logits <- assemble.logits / (length(folds) - 1)
  weighted.assemble.logits <- weighted.assemble.logits / sum(auc.weights)
  # plot assemble.logits auc and weighted.assembl.logits auc
  all.training$unique.id <- paste0(gene, ":", all.training$ref, all.training$pos.orig, all.training$alt)
  all.training <- all.training[!duplicated(all.training$unique.id),]
  all.training$assemble.logits <- assemble.logits[match(all.training$unique.id, gene.result$unique.id)]
  if (length(weighted.assemble.logits) != 0) {
    all.training$weighted.assemble.logits <- weighted.assemble.logits[match(all.training$unique.id, gene.result$unique.id)]
  }
  all.training.logits <- all.logits[match(all.training$unique.id, gene.result$unique.id),]
  library(caret)
  train.score <- all.training$score[all.training$score%in%c(-1,1)] * 0.5 + 0.5
  table(as.factor(train.score))
  if (!gene %in% c("P22607", "P21802")) {
    set.seed(0)
    meta_model_fit <- train(all.training.logits[all.training$score %in% c(-1,1),], 
                            as.factor(train.score),
                            # weights = as.array(sum(table(as.factor(train.score)))/table(as.factor(train.score)))[as.factor(train.score)],
                            method='ada')
    saveRDS(meta_model_fit, file = paste0(log.dir, gene, '.meta.RDS'))
  } else {
    meta_model_fit <- readRDS(paste0(log.dir, 'PF00130.meta.RDS'))
  }
  meta.logits <- predict(meta_model_fit, all.logits, type = 'prob')[,2]
  all.training$meta.logits <- meta.logits[match(all.training$unique.id, gene.result$unique.id)]
  # add colnames
  gene.result$assemble.logits <- assemble.logits
  if (length(weighted.assemble.logits) != 0) {
    gene.result$weighted.assemble.logits <- weighted.assemble.logits
  }
  gene.result$meta.logits <- meta.logits
  gene.result$pretrain.logits <- pretrain.result$logits
  for (fold in 0:4) {
    gene.result[,paste0('fold.', fold, '.logits')] <- all.logits[,fold+1]
  }
  if (use.logits=="assemble.logits") {
    assemble.auc <- plot.AUC(all.training$score[all.training$score %in% c(-1, 1)], 
                             all.training$assemble.logits[all.training$score %in% c(-1, 1)])
    print(assemble.auc$auc)
    if (!is.null(dim(assemble.logits))) {
      gene.result$logits.0 <- assemble.logits[,1]
      gene.result$logits.1 <- assemble.logits[,2]
      gene.result$logits.2 <- assemble.logits[,3]
      gene.result$logits <- NULL
    } else {
      gene.result$logits <- assemble.logits
    }
  } else if (use.logits=="meta.logits") {
    meta.auc <- plot.AUC(all.training$score[all.training$score %in% c(-1, 1)],
                         all.training$meta.logits[all.training$score %in% c(-1, 1)])
    print(meta.auc$auc)
    gene.result$logits <- NULL
    gene.result$logits.1 <- 1-meta.logits
    gene.result$logits.0 <- pretrain.result$logits
    gene.result$logits.2 <- meta.logits
  } else if (use.logits=="weighted.assemble.logits") {
    weighted.assemble.auc <- plot.AUC(all.training$score[all.training$score %in% c(-1, 1)], 
                                      all.training$weighted.assemble.logits[all.training$score %in% c(-1, 1)])
    print(weighted.assemble.auc$auc)
    gene.result$logits <- NULL
    gene.result$logits.1 <- 1-weighted.assemble.logits
    gene.result$logits.0 <- pretrain.result$logits
    gene.result$logits.2 <- weighted.assemble.logits
  } else if (use.logits=="best.logits") {
    best.logits <- all.logits[,which.max(auc.weights)]
    all.training$best.logits <- best.logits[match(all.training$unique.id, gene.result$unique.id)]
    best.auc <- plot.AUC(all.training$score[all.training$score %in% c(-1, 1)], 
                         all.training$best.logits[all.training$score %in% c(-1, 1)])
    print(best.auc$auc)
    gene.result$logits <- NULL
    gene.result$logits.1 <- 1-best.logits
    gene.result$logits.0 <- pretrain.result$logits
    gene.result$logits.2 <- best.logits
  }
  if (!"logits" %in% colnames(gene.result)) {
    gene.result[,"(logits.2-logits.1)*logits.0"] <- (gene.result$logits.2 - gene.result$logits.1) * gene.result$logits.0
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
    col.to.plot <- c(paste0("logits.", c(0:2)), "(logits.2-logits.1)*logits.0")
    fill.name <- c("Patho", "GoF", "LoF", "GoF-LoF")
    for (j in 1:4) {
      if (j == 1) {
        all.training.to.plot.plot <- all.pretrain
      } else {
        all.training.to.plot.plot <- all.training.to.plot
      }
      ps[[j]] <- ggplot() +
        geom_tile(data=gene.result, aes_string(x="pos.orig", y="alt", fill=col.to.plot[j])) + labs(fill=col.to.plot[j]) +
        scale_fill_gradientn(colors = c("light blue", "white", "pink"), na.value = 'grey') +
        scale_x_continuous(breaks=seq(0, nchar(gene.seq), 50)) + labs(fill=fill.name[j]) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=all.training.to.plot.plot, aes(x=pos.orig, y=alt, fill=score, width=1, height=1)) +
        scale_fill_gradientn(colors = c("blue", "white", "red")) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=secondary.df.to.plot, aes(x=pos.orig, y=alt, fill=ANNO_secondary, width=1, height=1)) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=rsa.df, aes(x=pos.orig, y=alt, fill=ANNO_RSA, width=1, height=1)) +
        scale_fill_gradientn(colors = c("grey", "white", "blue")) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=unique.df.to.plot, aes(x=pos.orig, y=alt, fill=ANNO_domain_type, width=1, height=1),show.legend = F) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=multiple.df.to.plot, aes(x=pos.orig, y=alt, fill=ANNO_domain_type, width=1, height=1),show.legend = F) +
        theme_bw() + theme(legend.position="bottom") +
        ggtitle(gene.names[o]) + ggeasy::easy_center_title()
    }
    p <- ps[[1]] + ps[[2]] + ps[[3]] + ps[[4]] + plot_layout(nrow=4)
    # ggsave(paste0(log.dir, gene, '.', xlower, '-', xupper, '.pdf'), p, width = min(nchar(gene.seq)/70*2, 49.9), height = 10)
    ggsave(paste0(log.dir, gene, '.pdf'), p, width = max(25, min(nchar(gene.seq)/70, 49.9)), height = 20)
    p <- ps[[1]] + ps[[4]] + plot_layout(nrow=2)
    # ggsave(paste0(log.dir, gene, '.', xlower, '-', xupper, '.pdf'), p, width = min(nchar(gene.seq)/70*2, 49.9), height = 10)
    ggsave(paste0(log.dir, gene, '.part.pdf'), p, width = max(25, min(nchar(gene.seq)/70, 49.9)), height = 10)
  } else {
    p <- ggplot() +
      geom_tile(data=gene.result, aes(x=pos.orig, y=alt, fill=logits)) +
      scale_fill_gradientn(colors = c("light blue", "white", "pink"), na.value = 'grey') +
      ggnewscale::new_scale_fill() +
      geom_tile(data=all.training, aes(x=pos.orig, y=alt, fill=score)) +
      scale_fill_gradientn(colors = c("blue", "white", "red")) +
      ggnewscale::new_scale_fill() +
      geom_tile(data=secondary.df, aes(x=pos.orig, y=alt, fill=ANNO_secondary)) +
      theme_bw() + theme(legend.position="bottom") +
      scale_x_continuous(breaks=seq(0, nchar(gene.seq), 100)) +
      ggtitle(gene.names[o]) + ggeasy::easy_center_title()
    ggsave(paste0(log.dir, gene, '.pdf'), p, width = nchar(gene.seq)/50, height = 4)
  }
  p <- patch.plot[[1]] / patch.plot[[2]] / patch.plot[[3]] / patch.plot[[4]] / patch.plot[[5]]
  write.csv(gene.result, paste0(log.dir, gene, '.logits.csv'))
  ggsave(paste0(log.dir, gene, '.5folds.pdf'), p, width = max(25, min(nchar(gene.seq)/70*4, 49.9)), height = 4*5)
}
