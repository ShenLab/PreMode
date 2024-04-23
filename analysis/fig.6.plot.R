# visualize with dssp secondary structure 
library(ggplot2)
library(bio3d)
library(patchwork)
dssp.exec <- '/share/vault/Users/gz2294/miniconda3/bin/mkdssp'
genes <- c("P15056", "P07949", "P04637", "Q14654")
gene.names <- c("BRAF", "RET", "TP53","KCNJ11")
use.lw.df <- readRDS('figs/fig.5a.plot.RDS')
use.lw.df <- use.lw.df[use.lw.df$model == '1: PreMode',]

af2.seqs <- read.csv('genes.full.seq.csv', row.names = 1)
aa.dict <- c('L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
             'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C')
log.dir <- '5genes.all.mut/PreMode/'
auc.dir <- './'
use.logits <- 'assemble.logits'
folds <- c(-1, 0:4)
source('./AUROC.R')
for (o in 1:length(genes)) {
  gene <- genes[o]
  use.lw <- c(F, use.lw.df$use.lw[use.lw.df$task.id==gene])
  names(use.lw) <- as.character(folds)
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
                   exefile=dssp.exec)
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
  others$type[others$type=="MOD_RES"] <- 'Post Transl. Mod.'
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
      training.file <- read.csv(paste0('../data.files/ICC.seed.0/', gene, '/training.csv'))[,c("HGNC", "pos.orig", "ref", "alt", "score", "data_source")]
      testing.file <- read.csv(paste0('../data.files/ICC.seed.0/', gene, '/testing.csv'))[,c("HGNC", "pos.orig", "ref", "alt", "score", "data_source")]
      training.file$score[training.file$score!=0] <- 1
      testing.file$score[testing.file$score!=0] <- 1
      all.logits <- matrix(NA, nrow = dim(gene.result)[1], ncol = 0)
      all.mean.logits <- matrix(NA, nrow = dim(gene.result)[1], ncol = 5)
      colnames(all.mean.logits) <- paste0('model.', 0:4)
    } else {
      if (use.lw[as.character(fold)]) {
        gene.result <- read.csv(paste0(log.dir, gene, '.large.window.fold.', fold, '.4fold.csv'), row.names = 1)
      } else {
        gene.result <- read.csv(paste0(log.dir, gene, '.fold.', fold, '.4fold.csv'), row.names = 1)
      }
      training.file <- read.csv(paste0('../data.files/ICC.seed.', fold, '/', gene, '/training.csv'))[,c("HGNC", "pos.orig", "ref", "alt", "score", "data_source")]
      testing.file <- read.csv(paste0('../data.files/ICC.seed.', fold, '/', gene, '/testing.csv'))[,c("HGNC", "pos.orig", "ref", "alt", "score", "data_source")]
      training.file <- training.file[training.file$score %in% c(-1, 0, 1),]
      testing.file <- testing.file[testing.file$score %in% c(-1, 0, 1),]
      auc <- use.lw.df$tr.auc[use.lw.df$fold == fold & use.lw.df$task.id == gene]
      auc.weights <- c(auc.weights, auc)
      all.logits <- cbind(all.logits, gene.result[,paste0('logits.FOLD.', 0:3)])
      all.mean.logits[,fold+1] <- rowMeans(gene.result[,paste0('logits.FOLD.', 0:3)])
    }
    if (!"logits" %in% colnames(gene.result) | fold != -1) {
      source('~/Pipeline/AUROC.R')
      logits.gof.lof <- rowMeans(gene.result[,paste0('logits.FOLD.', 0:3)])
      logits.gof <- (1 - logits.gof.lof)
      logits.lof <- logits.gof.lof
      logits <- cbind(pretrain.result$logits, logits.lof, logits.gof)
      gene.result$logits.0 <- pretrain.result$logits
      gene.result$logits.1 <- logits.lof
      gene.result$logits.2 <- logits.gof
      # average logits
      assemble.logits <- assemble.logits + logits
      weighted.assemble.logits <- weighted.assemble.logits + logits * auc
      gene.result[,"logits.2/logits.1*logits.0"] <- (gene.result$logits.2 - gene.result$logits.1) * gene.result$logits.0
      ps <- list()
      col.to.plot <- paste0("logits.", c(0:2, "2/logits.1*logits.0"))
      for (j in 1:4) {
        ps[[j]] <- ggplot() +
          geom_tile(data=gene.result, aes_string(x="pos.orig", y="alt", fill=col.to.plot[j])) + 
          scale_fill_gradientn(colors = c("light blue", "white", "pink"), na.value = 'grey') + labs(fill=col.to.plot[j]) +
          scale_x_continuous(breaks=seq(0, nchar(gene.seq), 50), minor_breaks = seq(0, nchar(gene.seq), 10)) +
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
        scale_x_continuous(breaks=seq(0, nchar(gene.seq), 50), minor_breaks = seq(0, nchar(gene.seq), 10)) +
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
    }
  }
  assemble.logits <- assemble.logits / (length(folds) - 1)
  weighted.assemble.logits <- weighted.assemble.logits / sum(auc.weights)
  # plot assemble.logits auc and weighted.assembl.logits auc
  all.training$unique.id <- paste0(gene, ":", all.training$ref, all.training$pos.orig, all.training$alt)
  all.training <- all.training[!duplicated(all.training$unique.id),]
  all.training$assemble.logits <- rowMeans(all.logits[match(all.training$unique.id, gene.result$unique.id),])
  if (length(weighted.assemble.logits) != 0) {
    all.training$weighted.assemble.logits <- weighted.assemble.logits[match(all.training$unique.id, gene.result$unique.id)]
  }
  all.training.logits <- all.logits[match(all.training$unique.id, gene.result$unique.id),]
  library(caret)
  train.score <- all.training$score[all.training$score%in%c(-1,1)] * 0.5 + 0.5
  table(as.factor(train.score))
  set.seed(0)
  unregister_dopar <- function() {
    env <- foreach:::.foreachGlobals
    rm(list=ls(name=env), pos=env)
  }
  unregister_dopar()
  # only fit model on pathogenic variants, remove benign
  meta_model_fit <- train(all.training.logits[all.training$score %in% c(-1,1),], 
                          as.factor(train.score))
  saveRDS(meta_model_fit, file = paste0(log.dir, gene, '.meta.RDS'))
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
    gene.result[,paste0('fold.', fold, '.logits')] <- all.mean.logits[,fold+1]
  }
  gene.result$all.logits <- all.logits
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
                         1-all.training$meta.logits[all.training$score %in% c(-1, 1)])
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
    write.csv(gene.result, paste0(log.dir, gene, '.logits.csv'))
    gene.result.to.plot <- gene.result
    all.training.to.plot <- all.training
    secondary.df.to.plot <- secondary.df
    unique.df.to.plot <- unique.df
    multiple.df.to.plot <- multiple.df
    ps <- list()
    col.to.plot <- c(paste0("logits.", c(0:2)), "(logits.2-logits.1)*logits.0")
    fill.name <- c("Patho", "GoF", "LoF", "GoF-LoF")
    for (j in 1:4) {
      if (j == 1) {
        all.training.to.plot.plot <- all.pretrain
        col.fill.limits <- c(0, 1)
      } else {
        all.training.to.plot.plot <- all.training.to.plot
        col.fill.limits <- c(-1, 1)
      }
      all.training.to.plot.plot$label <- all.training.to.plot.plot$score
      ps[[j]] <- ggplot() +
        geom_tile(data=gene.result, aes_string(x="pos.orig", y="alt", fill=col.to.plot[j])) + labs(fill=col.to.plot[j]) +
        scale_fill_gradientn(colors = c("light blue", "white", "pink"), na.value = 'grey') +
        scale_x_continuous(breaks=seq(0, nchar(gene.seq), 50), minor_breaks = seq(0, nchar(gene.seq), 10)) + labs(fill=fill.name[j]) +
        ggnewscale::new_scale_fill() +
        geom_tile(data=all.training.to.plot.plot, aes(x=pos.orig, y=alt, fill=label, width=1, height=1)) +
        scale_fill_gradientn(colors = c("blue", "white", "red"), limits=col.fill.limits) +
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
        ggtitle(gene.names[o]) + ggeasy::easy_center_title()
    }
    p <- ps[[1]] + ps[[4]] + plot_layout(nrow=2)
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
  }
  p <- patch.plot[[1]] / patch.plot[[2]] / patch.plot[[3]] / patch.plot[[4]] / patch.plot[[5]]
}

system('mv 5genes.all.mut/PreMode/P15056.part.pdf figs/fig.6a.pdf')
system('mv 5genes.all.mut/PreMode/P04637.part.pdf figs/fig.sup.10a.pdf')
system('mv 5genes.all.mut/PreMode/P07949.part.pdf figs/fig.sup.10b.pdf')
system('mv 5genes.all.mut/PreMode/Q14654.part.pdf figs/fig.sup.10c.pdf')
