library(ggplot2)
python.path = "/share/vault/Users/gz2294/miniconda3/envs/RESCVE/bin/python"
args <- commandArgs(trailingOnly = T)
# base dir for transfer learning
base.dirs <- args[1]
base.dirs <- strsplit(base.dirs, split = ',')[[1]]
ALL.gof.lof <- read.csv('figs/ALL.csv', row.names = 1, na.strings = c('.', 'NA'))
source('./prepare.biochem.R')
ALL.gof.lof <- prepare.unique.id(ALL.gof.lof)
biochem.cols <- c('secondary_struc', 'rsa', 'conservation.entropy', 
                  'conservation.alt', 'conservation.ref', 'pLDDT',
                  'itan.gof', 'itan.lof', 'data_source')
# base dir for transfer learning
uniprotID.dic <- c("P21802"="FGFR2", "P15056"="BRAF", 
                   "P07949"="RET", "P04637"="TP53", 
                   "Q09428"="ABCC8", "O00555"="CACNA1A", 
                   "Q14654"="KCNJ11", "Q99250"="SCN2A", 
                   "Q14524"="SCN5A",
                   "Q14524.clean"="SCN5A",
                   "P21802.itan.split"="FGFR2", "P15056.itan.split"="BRAF", 
                   "P07949.itan.split"="RET", "P04637.itan.split"="TP53", 
                   "Q09428.itan.split"="ABCC8", "O00555.itan.split"="CACNA1A", "Q14654.itan.split"="KCNJ11", 
                   "Q99250.itan.split"="SCN2A", "Q14524.clean.itan.split"="SCN5A",
                   "IPR016248"="Fibroblast Growth Factor Receptor Family",
                   "IPR005821"="Ion Transport Domain",
                   "IPR013518"="Potassium Channel Inwardly Rectifying Kir Cytoplasmic Domain",
                   "IPR020635"="Tyrosine Protein Kinase Catalytic Domain"
)
task.ids <- rep(
  c(read.csv('../scripts/gene.txt', header = F)$V1,
    read.csv('../scripts/gene.itan.txt', header = F)$V1,
    read.csv('../scripts/gene.pfams.txt', header = F)$V1,
    read.csv('../scripts/gene.split.by.pos.txt', header = F)$V1,
    read.csv('../scripts/gene.split.by.pos.itan.txt', header = F)$V1
    ),
  each=5)
result <- data.frame(task.id = task.ids)
result$fold <- rep(0:4, dim(result)[1]/5)
result$model <- "PreMode"
result$task.name <- result$task.id
result$task.size.lof <- NA
result$task.size.gof <- NA
result$task.type <- NA
for (i in 1:dim(result)[1]) {
  result$task.type[i] <- "Gene"
  tmp <- strsplit(result$task.name[i], "\\.")[[1]]
  tmp[1] <- uniprotID.dic[tmp[1]]
  if (startsWith(result$task.name[i], "IPR")) {
    result$task.name[i] <- paste("Domain: ", tmp, collapse = "")
    result$task.type[i] <- "Domain"
  } else {
    if (grepl('.IPR', result$task.name[i])) {
      tmp <- strsplit(result$task.name[i], "\\.")[[1]]
      tmp[1] <- uniprotID.dic[tmp[1]]
      if (tmp[2] %in% names(uniprotID.dic)) {
        tmp[2] <- uniprotID.dic[tmp[2]]  
      }
      if (length(tmp) >= 3) {
        if (tmp[3] == 'self') {
          tmp[3] <- "(Gene Only)"
          result$task.type[i] <- "Gene.Gene"
        } else {
          tmp[3] <- "(Family)"
          result$task.type[i] <- "Gene.Domain"
        }
      }
      result$task.name[i] <- paste0("Gene: ", paste(tmp, collapse = "."))
    } else {
      result$task.name[i] <- paste0("Gene: ", tmp[1])
    }
  }
  task.train <- read.csv(paste0('../data.files/ICC.seed.0/', result$task.id[i], '/training.csv'))
  task.test <- read.csv(paste0('../data.files/ICC.seed.0/', result$task.id[i], '/testing.csv'))
  result$task.size.lof[i] <- sum(task.train$score==-1) + sum(task.test$score==-1)
  result$task.size.gof[i] <- sum(task.train$score==1) + sum(task.test$score==1)
}
# add baseline AUC
# esm alphabets
source('./AUROC.R')
alphabet <- c('<cls>', '<pad>', '<eos>', '<unk>',
              'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
              'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
              'X', 'B', 'U', 'Z', 'O', '.', '-',
              '<null_1>', '<mask>')
alphabet_2 <- c('L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
                'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C')

source('./bind_rows.R')
result.gof <- data.frame()
for (i in 1:dim(result)[1]) {
  # AUC on gof variants
  baseline.PreModes <- c()
  baseline.PreModes.glm <- c()
  baseline.PreModes.bst <- c()
  baseline.PreModes.lw <- c()
  baseline.PreModes.lw.glm <- c()
  baseline.PreModes.lw.bst <- c()
  PreMode.tr.aucs <- c()
  PreMode.tr.lw.aucs <- c()
  PreMode.min.loss <- c()
  PreMode.min.loss.lw <- c()
  for (mod in base.dirs) {
    print(paste0(result$task.id[i], ":", mod))
    if (file.exists(paste0(mod, result$task.id[i], '/testing.fold.', result$fold[i], '.4fold.csv'))) {
      baseline.result <- read.csv(paste0(mod, result$task.id[i], '/testing.fold.', result$fold[i], '.4fold.csv'))
      baseline.auc <- plot.AUC(baseline.result$score, rowMeans(baseline.result[,paste0('logits.FOLD.', 0:3)]))
      if (file.exists(paste0(mod, result$task.id[i], '/training.fold.', result$fold[i], '.4fold.csv'))) {
        set.seed(0)
        baseline.train <- read.csv(paste0(mod, result$task.id[i], '/training.fold.', result$fold[i], '.4fold.csv'))
        baseline.train$score[baseline.train$score==-1] <- 0
        myglm <- glm(score ~ logits.FOLD.0 + logits.FOLD.1 + logits.FOLD.2 + logits.FOLD.3, data=baseline.train, family = "binomial")
        baseline.result$logits.glm <- predict(myglm, newdata = baseline.result, type = "response")
        baseline.auc.glm <- plot.AUC(baseline.result$score, baseline.result$logits.glm)
        tr.aucs <- c(plot.AUC(baseline.train$score, baseline.train$logits.FOLD.0)$auc,
                     plot.AUC(baseline.train$score, baseline.train$logits.FOLD.1)$auc,
                     plot.AUC(baseline.train$score, baseline.train$logits.FOLD.2)$auc,
                     plot.AUC(baseline.train$score, baseline.train$logits.FOLD.3)$auc)
        best.auc <- which.max(tr.aucs) - 1
        baseline.auc.bst <- plot.AUC(baseline.result$score, baseline.result[,paste0('logits.FOLD.', best.auc)])
        mean.tr.aucs <- plot.AUC(baseline.train$score, rowMeans(baseline.train[,paste0('logits.FOLD.', 0:3)]))$auc
        min.val.loss <- rowMeans(baseline.train[,paste0('min_loss.FOLD.', 0:3)])[1]
      }
      # compare with large.window models, if exist
      if (file.exists(paste0(mod, result$task.id[i], '.large.window/testing.fold.', result$fold[i], '.4fold.csv'))) {
        baseline.result <- read.csv(paste0(mod, result$task.id[i], '.large.window/testing.fold.', result$fold[i], '.4fold.csv'))
        baseline.auc.lw <- plot.AUC(baseline.result$score, rowMeans(baseline.result[,paste0('logits.FOLD.', 0:3)]))
        if (file.exists(paste0(mod, result$task.id[i], '.large.window/training.fold.', result$fold[i], '.4fold.csv'))) {
          set.seed(0)
          baseline.train <- read.csv(paste0(mod, result$task.id[i], '.large.window/training.fold.', result$fold[i], '.4fold.csv'))
          baseline.train$score[baseline.train$score==-1] <- 0
          myglm <- glm(score ~ logits.FOLD.0 + logits.FOLD.1 + logits.FOLD.2 + logits.FOLD.3, data=baseline.train, family = "binomial")
          baseline.result$logits.glm <- predict(myglm, newdata = baseline.result, type = "response")
          baseline.auc.lw.glm <- plot.AUC(baseline.result$score, baseline.result$logits.glm)
          tr.aucs <- c(plot.AUC(baseline.train$score, baseline.train$logits.FOLD.0)$auc,
                       plot.AUC(baseline.train$score, baseline.train$logits.FOLD.1)$auc,
                       plot.AUC(baseline.train$score, baseline.train$logits.FOLD.2)$auc,
                       plot.AUC(baseline.train$score, baseline.train$logits.FOLD.3)$auc)
          best.auc <- which.max(tr.aucs) - 1
          baseline.auc.lw.bst <- plot.AUC(baseline.result$score, baseline.result[,paste0('logits.FOLD.', best.auc)])
          mean.tr.aucs.lw <- plot.AUC(baseline.train$score, rowMeans(baseline.train[,paste0('logits.FOLD.', 0:3)]))$auc
          min.val.loss.lw <- rowMeans(baseline.train[,paste0('min_loss.FOLD.', 0:3)])[1]
        }
      } else {
        baseline.auc.lw <- list(auc=NA)
        baseline.auc.lw.glm <- list(auc=NA)
        baseline.auc.lw.bst <- list(auc=NA)
        mean.tr.aucs.lw <- NA
      }
    } else {
      use.large <- NA
      baseline.auc <- list(auc=NA)
      baseline.auc.glm <- list(auc=NA)
      baseline.auc.bst <- list(auc=NA)
      baseline.auc.lw <- list(auc=NA)
      baseline.auc.lw.glm <- list(auc=NA)
      baseline.auc.lw.bst <- list(auc=NA)
      mean.tr.aucs <- NA
      mean.tr.aucs.lw <- NA
      min.val.loss <- NA
      min.val.loss.lw <- NA
    }
    baseline.PreModes <- c(baseline.PreModes, baseline.auc$auc)
    baseline.PreModes.glm <- c(baseline.PreModes.glm, baseline.auc.glm$auc)
    baseline.PreModes.bst <- c(baseline.PreModes.bst, baseline.auc.bst$auc)
    baseline.PreModes.lw <- c(baseline.PreModes.lw, baseline.auc.lw$auc)
    baseline.PreModes.lw.glm <- c(baseline.PreModes.lw.glm, baseline.auc.lw.glm$auc)
    baseline.PreModes.lw.bst <- c(baseline.PreModes.lw.bst, baseline.auc.lw.bst$auc)
    PreMode.tr.aucs <- c(PreMode.tr.aucs, mean.tr.aucs)
    PreMode.tr.lw.aucs <- c(PreMode.tr.lw.aucs, mean.tr.aucs.lw)
    PreMode.min.loss <- c(PreMode.min.loss, min.val.loss)
    PreMode.min.loss.lw <- c(PreMode.min.loss.lw, min.val.loss.lw)
  }
  # add itan
  baseline.result <- read.csv(paste0('PreMode/', result$task.id[i], '/testing.fold.', result$fold[i], '.4fold.csv'))
  test.result <- prepare.unique.id(baseline.result)
  test.result[,biochem.cols] <- ALL.gof.lof[match(test.result$unique.id, ALL.gof.lof$unique.id), biochem.cols]
  
  baseline.auc.6 <- plot.AUC(test.result$score[!grepl("Itan", test.result$data_source)],
                             1-test.result$itan.gof[!grepl("Itan", test.result$data_source)])
  # add random forest, elastic net
  # add alphamissense, esm1b_LLR
  if (file.exists(paste0('PreMode/', result$task.id[i],
                         '/training.fold.', result$fold[i], '.4fold.csv')) &
      file.exists(paste0('PreMode/', result$task.id[i],
                         '/testing.fold.', result$fold[i], '.4fold.csv'))) {
    gene.train <- read.csv(paste0('PreMode/', result$task.id[i],
                                  '/training.fold.', result$fold[i], '.4fold.csv'))
    gene.test <- read.csv(paste0('PreMode/', result$task.id[i],
                                 '/testing.fold.', result$fold[i], '.4fold.csv'))
    gene.train <- prepare.unique.id(gene.train)
    gene.test <- prepare.unique.id(gene.test)
    # write train and test emb to files
    train.label.file <- tempfile()
    test.label.file <- tempfile()
    train.biochem.file <- tempfile()
    test.biochem.file <- tempfile()
    write.csv(gene.train, file = train.label.file)
    write.csv(gene.test, file = test.label.file)
    gene.train.biochem <- prepare.biochemical(ALL.gof.lof[match(gene.train$unique.id, ALL.gof.lof$unique.id),])
    gene.test.biochem <- prepare.biochemical(ALL.gof.lof[match(gene.test$unique.id, ALL.gof.lof$unique.id),])
    write.csv(gene.train.biochem,
              file = train.biochem.file)
    write.csv(gene.test.biochem,
              file = test.biochem.file)
    res <- system(paste0(python.path, ' ',
                         'random.forest.glof.py ', 
                         train.biochem.file, ' ',
                         train.label.file, ' ',
                         test.biochem.file, ' ', 
                         test.label.file), intern = T)
    baseline.auc.5 <- list(auc=as.numeric(strsplit(res, split = '=')[[1]][2]))
    # add alphamissense, esm1b_LLR
    gene.test$AlphaMissense <- ALL.gof.lof$AlphaMissense[match(gene.test$unique.id, ALL.gof.lof$unique.id)]
    gene.test$ESM1b.LLR <- ALL.gof.lof$ESM1b.LLR[match(gene.test$unique.id, ALL.gof.lof$unique.id)]
    baseline.auc.7 <- plot.AUC(gene.test$score, gene.test$AlphaMissense)
    baseline.auc.8 <- plot.AUC(gene.test$score, -gene.test$ESM1b.LLR)
  } else {
    baseline.auc.5 <- list(auc=NA)
    baseline.auc.7 <- list(auc=NA)
    baseline.auc.8 <- list(auc=NA)
  }
  
  # aggregate results
  to.append <- result[rep(i, length(base.dirs)*6+4), ]
  to.append$auc <- c(baseline.PreModes, baseline.PreModes.lw, 
                     baseline.PreModes.glm, baseline.PreModes.lw.glm, 
                     baseline.PreModes.bst, baseline.PreModes.lw.bst, 
                     baseline.auc.5$auc, baseline.auc.6$auc, baseline.auc.7$auc, baseline.auc.8$auc
                     )
  to.append$model <- c(base.dirs, paste0(base.dirs, '.lw'),
                       paste0(base.dirs, '.glm'), paste0(base.dirs, '.lw.glm'),
                       paste0(base.dirs, '.bst'), paste0(base.dirs, '.lw.bst'),
                       "random.forest", "Itan.1", 'AlphaMissense', 'ESM1b.LLR')
  to.append$noitan.gof <- sum(test.result$score==1 & baseline.result$data_source != "Itan")
  to.append$noitan.lof <- sum(test.result$score==-1 & baseline.result$data_source != "Itan")
  to.append$tr.auc <- c(PreMode.tr.aucs, PreMode.tr.lw.aucs, 
                           PreMode.tr.aucs, PreMode.tr.lw.aucs, 
                           PreMode.tr.aucs, PreMode.tr.lw.aucs,
                           rep(NA, 4))
  to.append$val.loss <- c(PreMode.min.loss, PreMode.min.loss.lw, 
                          PreMode.min.loss, PreMode.min.loss.lw, 
                          PreMode.min.loss, PreMode.min.loss.lw,
                          rep(NA, 4))
  result.gof <- rbind(result.gof, to.append)
}
saveRDS(result.gof, 'figs/fig.5.prepare.RDS')


