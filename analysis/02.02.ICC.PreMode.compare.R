# source('utils.R')
library(ggplot2)
ALL.gof.lof <- read.csv('figs/ALL.csv', row.names = 1, na.strings = c('.', 'NA'))
source('/share/vault/Users/gz2294/Pipeline/prepare.biochem.R')
ALL.gof.lof <- prepare.unique.id(ALL.gof.lof)
biochem.cols <- c('secondary_struc', 'rsa', 'conservation.entropy', 'conservation.alt', 'conservation.ref', 'FoldXddG', 'pLDDT')
# args <- commandArgs(trailingOnly = T)
# base dir for transfer learning
# base.dir <- "/share/pascal/Users/gz2294/PreMode/scripts/CHPs.v4.esm.dssp.small.StarAttn.MSA.StarPool.1dim/"
uniprotID.dic <- c("P21802"="FGFR2", "P15056"="BRAF", "P07949"="RET", "P04637"="TP53", 
                   "Q09428"="ABCC8",
                   "O00555"="CACNA1A", "Q14654"="KCNJ11", 
                   "Q99250"="SCN2A", "Q14524"="SCN5A", 
                   # "IonChannel.split.uniprotID"="Na+/Ca2+ Channel",
                   "IonChannel.chps"="Na+/Ca2+ Channel",
                   "IonChannel"="Na+/Ca2+ Channel",
                   "IPR000719"="Protein Kinase Domain",
                   "IPR001806"="Small GTPase",
                   "IPR001245"="Protein Kinase Catalytic Domain",
                   "IPR016248"="Fibroblast Growth Factor Receptor Family",
                   "IPR005821"="Ion Transport Domain",
                   "IPR027359"="Voltage-dependent Channel Domain"
)
task.ids <- rep(c(read.csv('../scripts/pfams.txt', header = F)$V1,
                  read.csv('../scripts/gene.pfams.txt', header = F)$V1), each=5)
task.ids <- task.ids[-grep('heyne.split', task.ids)]
result <- data.frame(task.id = task.ids)
result$fold <- rep(0:4, dim(result)[1]/5)
result$model <- "PreMode"
result$task.name <- result$task.id
result$task.size.lof <- NA
result$task.size.gof <- NA
result$task.type <- NA
for (i in 1:dim(result)[1]) {
  result$task.name[i] <- gsub("\\.chps\\.even\\.uniprotID", "", result$task.id[i])
  result$task.name[i] <- gsub("\\.even\\.uniprotID", "", result$task.id[i])
  if (!startsWith(result$task.name[i], "PF") & !startsWith(result$task.name[i], "IonChannel")) {
    # tmp <- strsplit(result$task.name[i], ".chps")[[1]]
    tmp <- strsplit(result$task.name[i], ".chps")[[1]]
    if (grepl('.PF', result$task.name[i]) | grepl('.IonChannel', result$task.name[i])) {
      result$task.type[i] <- "Gene.Pfam"
    } else {
      result$task.type[i] <- "Gene"
    }
    tmp[1] <- uniprotID.dic[tmp[1]]
    if (startsWith(result$task.name[i], "IPR")) {
      result$task.name[i] <- paste("Domain: ", tmp, collapse = "")
      result$task.type[i] <- "Domain"
    } else {
      if (grepl('.PF', result$task.name[i]) | grepl('.IonChannel', result$task.name[i]) | grepl('.IPR', result$task.name[i])) {
        tmp <- strsplit(result$task.name[i], "\\.")[[1]]
        tmp[1] <- uniprotID.dic[tmp[1]]
        if (tmp[2] %in% names(uniprotID.dic)) {
          tmp[2] <- uniprotID.dic[tmp[2]]  
        }
        if (length(tmp) >= 3) {
          if (tmp[3] == 'self') {
            tmp[3] <- "(Gene Only)"
          } else {
            tmp[3] <- "(Family)"
          }
        }
      }
      result$task.name[i] <- paste0("Gene: ", paste(tmp, collapse = "."))
    }
  } else {
    result$task.type <- "Pfam"
    if (result$task.name[i] %in% names(uniprotID.dic)) {
      result$task.name[i] <- uniprotID.dic[result$task.name[i]]
    }
    result$task.name[i] <- paste("Pfam: ", result$task.name[i], collapse = "")
  }
  task.train <- read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.0/', result$task.id[i], '/training.csv'))
  task.test <- read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.0/', result$task.id[i], '/testing.csv'))
  result$task.size.lof <- sum(task.train$score==-1) + sum(task.test$score==-1)
  result$task.size.gof <- sum(task.train$score==1) + sum(task.test$score==1)
}
# add baseline AUC
# esm alphabets
source('~/Pipeline/AUROC.R')
source('~/Pipeline/uniprot.table.add.annotation.R')
alphabet <- c('<cls>', '<pad>', '<eos>', '<unk>',
              'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
              'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
              'X', 'B', 'U', 'Z', 'O', '.', '-',
              '<null_1>', '<mask>')
result <- result[result$task.type %in% c("Pfam", "Gene"),]
source('~/Pipeline/bind_rows.R')
for (i in 1:dim(result)[1]) {
  # add ESM
  if (!file.exists(paste0('PreMode.inference/', result$task.id[i], '/testing.fold.', result$fold[i], '.annotated.csv')) &
      file.exists(paste0('PreMode.inference/', result$task.id[i], '/testing.fold.', result$fold[i], '.csv'))) {
    print(paste0("Begin ", i))
    test.result <- read.csv(paste0('PreMode.inference/', result$task.id[i], '/testing.fold.',
                                   result$fold[i], '.csv'))
    # baseline.result <- read.csv(paste0('PreMode.pass.inference/', result$task.id[i], '/testing.fold.',
    #                                    result$fold[i], '.csv'))
    test.result$pretrain.logits <- read.csv(paste0('PreMode.inference/', result$task.id[i], '/testing.pretrain.fold.',
                                                   result$fold[i], '.csv'))$logits
    if (file.exists(paste0('esm2.inference/', result$task.id[i], '/testing.fold.', result$fold[i], '.logits.csv'))) {
      esm.logits <- read.csv(paste0('esm2.inference/', result$task.id[i], '/testing.fold.', result$fold[i], '.logits.csv'))
      if (file.exists(paste0('PreMode.inference/', result$task.id[i], '/beni.fold.',
                             result$fold[i], '.csv'))) {
        beni.result <- read.csv(paste0('PreMode.inference/', result$task.id[i], '/beni.fold.',
                                       result$fold[i], '.csv'))
        if (beni.result$alt[1]==TRUE) {
          beni.result$alt[1] <- 'T'
        }
        beni.result$pretrain.logits <- read.csv(paste0('PreMode.inference/', result$task.id[i], '/beni.pretrain.csv'))$logits
        test.result <- my.bind.rows(test.result, beni.result)
        esm.beni.logits <- read.csv(paste0('esm2.inference/', result$task.id[i], '/beni.logits.csv'))
        esm.logits <- my.bind.rows(esm.logits, esm.beni.logits)
      }
      esm.logits <- esm.logits[,2:34]
      colnames(esm.logits) <- alphabet
      score <- c()
      for (k in 1:dim(esm.logits)[1]) {
        score <- c(score, esm.logits[k, test.result$alt[k]] - esm.logits[k, test.result$ref[k]])
      }
    } else {
      score <- NA
    }
    test.result$gof.logits <- test.result$pretrain.logits * (1-test.result$logits)
    test.result$lof.logits <- test.result$pretrain.logits * (test.result$logits)
    test.result$esm.logits <- score
    # save original score
    test.result <- uniprot.table.add.annotation.parallel(test.result, 'EVE')
    test.result <- uniprot.table.add.annotation.parallel(test.result, 'dbnsfp')
    test.result <- uniprot.table.add.annotation.parallel(test.result, 'Itan')
    test.result$itan.logits <- test.result$itan.gof / (1-test.result$itan.beni)
    test.result <- uniprot.table.add.annotation.parallel(test.result, 'gMVP')
    test.result <- uniprot.table.add.annotation.parallel(test.result, 'conservation')
    write.csv(test.result, paste0('PreMode.inference/', result$task.id[i], '/testing.fold.',
                                  result$fold[i], '.annotated.csv'))
  }
}
result.all <- list()
for (j in c("gof", "lof", "glof", 'beni')) {
  result.gof <- data.frame()
  for (i in 1:dim(result)[1]) {
    # AUC on gof variants
    test.result <- read.csv(paste0('PreMode.inference/', result$task.id[i], '/testing.fold.',
                                   result$fold[i], '.annotated.csv'), row.names = 1)
    test.result <- test.result[test.result$score != 0,]
    if (file.exists(paste0('elastic.net.result/', result$task.id[i], '/prediction.test.fold.',
                           result$fold[i], '.csv'))) {
      elastic.net.result <- read.csv(paste0('elastic.net.result/', result$task.id[i], '/prediction.test.fold.',
                                            result$fold[i], '.csv'), row.names = 1)
      elastic.net.auc <- plot.AUC(elastic.net.result$score, elastic.net.result$prediction_prob)
    } else {
      elastic.net.auc <- list(auc=NA)
    }
    if (file.exists(paste0('PreMode.pass.inference/', result$task.id[i], '/testing.fold.',
                           result$fold[i], '.csv'))) {
      baseline.result <- read.csv(paste0('PreMode.pass.inference/', result$task.id[i], '/testing.fold.',
                                         result$fold[i], '.csv'))
      baseline.auc.1 <- plot.AUC(baseline.result$score, baseline.result$logits)
    } else {
      baseline.auc.1 <- list(auc=NA)
    }
    if (file.exists(paste0('PreMode.onehot.inference/', result$task.id[i], '/testing.fold.',
                           result$fold[i], '.csv'))) {
      baseline.result.2 <- read.csv(paste0('PreMode.onehot.inference/', result$task.id[i], '/testing.fold.',
                                           result$fold[i], '.csv'))
      baseline.auc.2 <- plot.AUC(baseline.result.2$score, baseline.result.2$logits)
    } else {
      baseline.auc.2 <- list(auc=NA)
    }
    if (file.exists(paste0('PreMode.radius.inference/', result$task.id[i], '/testing.fold.',
                           result$fold[i], '.csv'))) {
      baseline.result.3 <- read.csv(paste0('PreMode.radius.inference/', result$task.id[i], '/testing.fold.',
                                           result$fold[i], '.csv'))
      baseline.auc.3 <- plot.AUC(baseline.result.3$score, baseline.result.3$logits)
    } else {
      baseline.auc.3 <- list(auc=NA)
    }
    # add baseline.auc.4, 5, which is auc from human curated labels
    # Including secondary structures, rsa, conservation, pLDDT, ddG, biochemical properties of ref, alt, 
    # Not sure how to add post-translational modifications, PPI information
    if (file.exists(paste0('PreMode.inference/', result$task.id[i], '/training.fold.',
                           result$fold[i], '.csv'))) {
      train.config <- yaml::read_yaml(paste0('/share/vault/Users/gz2294/PreMode/scripts/CHPs.v4.esm.dssp.small.StarAttn.MSA.StarPool.1dim/',
                                             result$task.id[i], '.5fold/', result$task.id[i], '.fold.', result$fold[i], '.yaml'))
      gene.train <- read.csv(paste0('PreMode.inference/', result$task.id[i], '/training.fold.',
                                    result$fold[i], '.csv'))
      test.result <- prepare.unique.id(test.result)
      gene.train <- prepare.unique.id(gene.train)
      gene.train[,biochem.cols] <- ALL.gof.lof[match(gene.train$unique.id, ALL.gof.lof$unique.id), biochem.cols]
      test.result[,biochem.cols] <- ALL.gof.lof[match(test.result$unique.id, ALL.gof.lof$unique.id), biochem.cols]
      np <- reticulate::import('numpy')
      train.val.split <- np$load(paste0(train.config$log_dir, 'splits.0.npz'))
      gene.train <- gene.train[train.val.split['idx_train']+1,]
      
      # write train and test emb to files
      train.label.file <- tempfile()
      test.label.file <- tempfile()
      train.biochem.file <- tempfile()
      test.biochem.file <- tempfile()
      write.csv(gene.train, file = train.label.file)
      write.csv(test.result, file = test.label.file)
      write.csv(prepare.biochemical(gene.train), file = train.biochem.file)
      write.csv(prepare.biochemical(test.result), file = test.biochem.file)
      res <- system(paste0('/share/descartes/Users/gz2294/miniconda3/envs/RESCVE/bin/python ', 
                           '10.analysis.few.shot.elastic.net.py ', 
                           train.biochem.file, ' ',
                           train.label.file, ' ',
                           test.biochem.file, ' ', 
                           test.label.file), intern = T)
      baseline.auc.4 <- list(auc=as.numeric(strsplit(res, split = '=')[[1]][2]))
      res <- system(paste0('/share/descartes/Users/gz2294/miniconda3/envs/RESCVE/bin/python ', 
                           '10.analysis.few.shot.random.forest.py ', 
                           train.biochem.file, ' ',
                           train.label.file, ' ',
                           test.biochem.file, ' ', 
                           test.label.file), intern = T)
      baseline.auc.5 <- list(auc=as.numeric(strsplit(res, split = '=')[[1]][2]))
    } else {
      baseline.auc.4 <- list(auc=NA)
      baseline.auc.5 <- list(auc=NA)
    }
    if (file.exists(paste0('PreMode.confidence.inference/', result$task.id[i], '/testing.fold.',
                           result$fold[i], '.csv'))) {
      baseline.result.6 <- read.csv(paste0('PreMode.confidence.inference/', result$task.id[i], '/testing.fold.',
                                           result$fold[i], '.csv'))
      baseline.auc.6 <- plot.AUC(baseline.result.6$score, baseline.result.6$logits)
    } else {
      baseline.auc.6 <- list(auc=NA)
    }
    original.score <- test.result$score
    # set beni to half of lof
    beni.idx <- which(test.result$score==0)
    if (length(beni.idx) > sum(test.result$score%in%c(1, -1))) {
      set.seed(0)
      to.keep <- sample(beni.idx, floor(min(sum(test.result$score==1), sum(test.result$score==-1))/2))
      test.result <- test.result[-beni.idx[!beni.idx%in%to.keep],]
    }
    if (j == "gof") {
      test.result$score[test.result$score==-1] <- 0
      baseline.result$score[baseline.result$score==-1] <- 0
      baseline.result.2$score[baseline.result$score==-1] <- 0
      PreMode.auc <- plot.AUC(test.result$score, test.result$gof.logits)
      # baseline.auc <- plot.AUC(baseline.result$score, baseline.result$logits)
      # baseline.auc.2 <- plot.AUC(baseline.result.2$score, baseline.result.2$logits)
      itan.gof.auc <- plot.AUC(test.result$score, test.result$itan.gof)
      if (sum(is.na(test.result$itan.gof)) >= 1) {itan.gof.auc$auc <- NA}
    } else if (j == "lof") {
      test.result$score <- -test.result$score
      test.result$score[test.result$score==1] <- 0
      baseline.result$score[baseline.result$score==1] <- 0
      baseline.result.2$score[baseline.result$score==1] <- 0
      PreMode.auc <- plot.AUC(test.result$score, test.result$lof.logits)
      # baseline.auc <- plot.AUC(baseline.result$score, baseline.result$logits)
      # baseline.auc.2 <- plot.AUC(baseline.result.2$score, baseline.result.2$logits)
      itan.gof.auc <- plot.AUC(test.result$score, test.result$itan.lof)
      if (sum(is.na(test.result$itan.lof)) >= 1) {itan.gof.auc$auc <- NA}
    } else if (j == "glof") {
      test.result <- test.result[test.result$score!=0,]
      baseline.result <- baseline.result[baseline.result$score!=0,]
      baseline.result.2 <- baseline.result.2[baseline.result.2$score!=0,]
      PreMode.auc <- plot.AUC(test.result$score, test.result$gof.logits/test.result$lof.logits)
      # baseline.auc <- plot.AUC(baseline.result$score, baseline.result$logits)
      # baseline.auc.2 <- plot.AUC(baseline.result.2$score, baseline.result.2$logits)
      itan.gof.auc <- plot.AUC(test.result$score, test.result$itan.logits)
      if (sum(is.na(test.result$itan.logits)) >= 1) {itan.gof.auc$auc <- NA}
    } else if (j == "beni") {
      test.result$score[test.result$score==-1] <- 1
      PreMode.auc <- plot.AUC(test.result$score, test.result$pretrain.logits)
      # baseline.auc <- list(auc=NA)
      # baseline.auc.2 <- list(auc=NA)
      itan.gof.auc <- plot.AUC(test.result$score, test.result$itan.beni)
      if (sum(is.na(test.result$itan.beni)) >= 1) {itan.gof.auc$auc <- NA}
    }
    REVEL.auc <- plot.AUC(test.result$score, test.result$REVEL)
    # if (sum(is.na(test.result$REVEL)) >= 1) {REVEL.auc$auc <- NA}
    PrimateAI.auc <- plot.AUC(test.result$score, test.result$PrimateAI)
    # if (sum(is.na(test.result$PrimateAI)) >= 1) {PrimateAI.auc$auc <- NA}
    ESM.auc <- plot.AUC(test.result$score, test.result$esm.logits)
    # if (sum(is.na(test.result$esm.logits)) >= 1) {ESM.auc$auc <- NA}
    EVE.auc <- plot.AUC(test.result$score, test.result$EVE)
    # if (sum(is.na(test.result$EVE)) >= 1) {EVE.auc$auc <- NA}
    gMVP.auc <- plot.AUC(test.result$score, test.result$gMVP)
    # if (sum(is.na(test.result$gMVP)) >= 1) {gMVP.auc$auc <- NA}
    to.append <- result[rep(i, 13), ]
    to.append$min.val.auc <- c(PreMode.auc$auc, baseline.auc.1$auc,
                               baseline.auc.2$auc, 
                               baseline.auc.4$auc, baseline.auc.5$auc, baseline.auc.6$auc,
                               REVEL.auc$auc, PrimateAI.auc$auc, 
                               ESM.auc$auc, EVE.auc$auc, gMVP.auc$auc, 
                               elastic.net.auc$auc, itan.gof.auc$auc)
    to.append$model <- c("PreMode", "Baseline (No structure)",
                         "Baseline (No ESM)", 
                         "BioChem (Elastic Net)", "BioChem (Random Forest)", "PreMode (Add PAE)",
                         "REVEL",
                         "PrimateAI", "ESM", 'EVE', 'gMVP', 
                         'elastic.net', 'Itan')
    result.gof <- rbind(result.gof, to.append)
  }
  result.all[[j]] <- result.gof
}
saveRDS(result.all, 'figs/02.02.ICC.PreMode.compare.RDS')
result.plot <- result.all[[3]]
result.plot <- result.plot[result.plot$model %in% c("PreMode", "Baseline (No structure)", 
                                                    "Baseline (No ESM)", 
                                                    "BioChem (Elastic Net)", "BioChem (Random Forest)"),]
num.models <- length(unique(result.plot$model))
p <- ggplot(result.plot, aes(y=min.val.auc, x=task.name, col=model)) +
  geom_point(alpha=0.2) +
  stat_summary(data = result.plot,
               aes(x=as.numeric(factor(task.name))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = min.val.auc, col=model), 
               fun.data = mean_se, geom = "errorbar", width = 0.2) +
  stat_summary(data = result.plot, 
               aes(x=as.numeric(factor(task.name))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = min.val.auc, col=model), 
               fun.data = mean_se, geom = "point") +
  labs(x = "task", y = "min.val.auc", fill = "model") +
  theme_bw() + 
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        legend.position="bottom", 
        legend.direction="horizontal") +
  ylim(0.5, 1) + xlab('task: Genetics Level Mode of Action') 
ggsave(paste0('figs/02.02.ICC.PreMode.compare.pdf'), p, height = 4, width = 10)


result.plot <- result.all[[4]]
num.models <- length(unique(result.plot$model))
p <- ggplot(result.plot, aes(y=min.val.auc, x=task.name, col=model)) +
  geom_point(alpha=0.2) +
  stat_summary(data = result.plot,
               aes(x=as.numeric(factor(task.name))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = min.val.auc, col=model), 
               fun.data = mean_se, geom = "errorbar", width = 0.2) +
  stat_summary(data = result.plot, 
               aes(x=as.numeric(factor(task.name))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = min.val.auc, col=model), 
               fun.data = mean_se, geom = "point") +
  labs(x = "task", y = "min.val.auc", fill = "model") +
  theme_bw() + 
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        legend.position="bottom", 
        legend.direction="horizontal") +
  ylim(0.5, 1) + xlab('task: (LoF/GoF)')
ggsave(paste0('figs/02.02.ICC.PreMode.patho.compare.pdf'), p, height = 6, width = 15)

