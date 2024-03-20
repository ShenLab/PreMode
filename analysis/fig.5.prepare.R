# source('utils.R')
library(ggplot2)
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
uniprotID.dic <- c("P21802"="FGFR2", "P15056"="BRAF", "P07949"="RET", "P04637"="TP53", 
                   "Q09428"="ABCC8",
                   "O00555"="CACNA1A", "Q14654"="KCNJ11", 
                   "Q99250"="SCN2A", "Q14524"="SCN5A", 
                   "IonChannel.chps"="Na+/Ca2+ Channel",
                   "IonChannel"="Na+/Ca2+ Channel",
                   "IPR000719"="Protein Kinase Domain",
                   "IPR001806"="Small GTPase",
                   "IPR001245"="Protein Kinase Catalytic Domain",
                   "IPR016248"="Fibroblast Growth Factor Receptor Family",
                   "IPR005821"="Ion Transport Domain",
                   "IPR027359"="Voltage-dependent Channel Domain"
)
task.ids <- rep(c(read.csv('../scripts/gene.txt', header = F)$V1,
                  read.csv('../scripts/gene.pfams.txt', header = F)$V1), each=5)
task.ids <- task.ids[-grep('Heyne', task.ids)]
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
            result$task.type[i] <- "Gene.Gene"
          } else {
            tmp[3] <- "(Family)"
            result$task.type[i] <- "Gene.Domain"
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
  result$task.size.lof[i] <- sum(task.train$score==-1) + sum(task.test$score==-1)
  result$task.size.gof[i] <- sum(task.train$score==1) + sum(task.test$score==1)
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
alphabet_2 <- c('L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
              'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C')
source('./bind_rows.R')
result.gof <- data.frame()
for (i in 1:dim(result)[1]) {
  # AUC on gof variants
  baseline.PreModes <- c()
  for (mod in base.dirs) {
    if (file.exists(paste0(mod, result$task.id[i], '/testing.fold.',
                           result$fold[i], '.csv'))) {
      baseline.result <- read.csv(paste0(mod, result$task.id[i], '/testing.fold.',
                                         result$fold[i], '.csv'))
      # baseline.result <- prepare.unique.id(baseline.result)
      # baseline.result[,'data_source'] <- ALL.gof.lof[match(baseline.result$unique.id, ALL.gof.lof$unique.id), 'data_source']
      if (!'logits' %in% colnames(baseline.result) & 'logits.19' %in% colnames(baseline.result)) {
        baseline.result$logits <- NA
        for (k in 1:dim(baseline.result)[1]) {
          baseline.result$logits[k] <- baseline.result[k, paste0('logits.', match(baseline.result$alt[k], alphabet_2)-1)]
        }
      }
      # baseline.result <- baseline.result[baseline.result$data_source != "Itan",]
      baseline.auc <- plot.AUC(baseline.result$score, baseline.result$logits)
    } else {
      baseline.auc <- list(auc=NA)
    }
    baseline.PreModes <- c(baseline.PreModes, baseline.auc$auc)
  }
  if (file.exists(paste0('PreMode.inference/', result$task.id[i], '/training.fold.',
                         result$fold[i], '.csv')) &
      file.exists(paste0('PreMode.inference/', result$task.id[i], '/testing.fold.',
                         result$fold[i], '.csv'))) {
    train.config <- yaml::read_yaml(paste0('../scripts/PreMode/',
                                           result$task.id[i], '.5fold/', result$task.id[i], '.fold.', result$fold[i], '.yaml'))
    gene.train <- read.csv(paste0('PreMode.inference/', result$task.id[i], '/training.fold.',
                                  result$fold[i], '.csv'))
    baseline.result <- read.csv(paste0('PreMode.inference/', result$task.id[i], '/testing.fold.',
                                       result$fold[i], '.csv'))
    baseline.auc.noItan <- plot.AUC(baseline.result$score[baseline.result$data_source != "Itan"],
                                    baseline.result$logits[baseline.result$data_source != "Itan"])
    if (file.exists(paste0('PreMode.CHPs.v4.large.window/', result$task.id[i], '/testing.fold.',
                           result$fold[i], '.csv'))) {
      baseline.result.large.window <- read.csv(paste0('PreMode.CHPs.v4.large.window/', result$task.id[i], '/testing.fold.',
                                                      result$fold[i], '.csv'))
      baseline.auc.large.window.noItan <- plot.AUC(baseline.result.large.window$score[baseline.result.large.window$data_source != "Itan"],
                                                   baseline.result.large.window$logits[baseline.result.large.window$data_source != "Itan"])
    } else {
      baseline.auc.large.window.noItan <- list(auc=NA)
    }
    
    test.result <- prepare.unique.id(baseline.result)
    gene.train <- prepare.unique.id(gene.train)
    gene.train[,biochem.cols] <- ALL.gof.lof[match(gene.train$unique.id, ALL.gof.lof$unique.id), biochem.cols]
    test.result[,biochem.cols] <- ALL.gof.lof[match(test.result$unique.id, ALL.gof.lof$unique.id), biochem.cols]
    np <- reticulate::import('numpy')
    train.val.split <- np$load(paste0('../', train.config$log_dir, 'splits.0.npz'))
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
                         '10.analysis.few.shot.random.forest.py ', 
                         train.biochem.file, ' ',
                         train.label.file, ' ',
                         test.biochem.file, ' ', 
                         test.label.file), intern = T)
    baseline.auc.5 <- list(auc=as.numeric(strsplit(res, split = '=')[[1]][2]))
  } else {
    baseline.auc.5 <- list(auc=NA)
  }
  # add itan
  baseline.auc.6 <- plot.AUC(test.result$score[baseline.result$data_source != "Itan"],
                             test.result$itan.gof[baseline.result$data_source != "Itan"])
  to.append <- result[rep(i, length(base.dirs)+4), ]
  to.append$auc <- c(baseline.PreModes, baseline.auc.noItan$auc, baseline.auc.large.window.noItan$auc, 
                     baseline.auc.5$auc,
                     baseline.auc.6$auc)
  to.append$model <- c(base.dirs, "PreMode.noItan", "PreMode.large.window.noItan",
                       "BioChem (Random Forest)",
                       "Itan")
  result.gof <- rbind(result.gof, to.append)
}
saveRDS(result.gof, 'figs/fig.5.prepare.RDS')


