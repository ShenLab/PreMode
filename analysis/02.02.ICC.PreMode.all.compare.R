# source('utils.R')
library(ggplot2)
# args <- commandArgs(trailingOnly = T)
# base dir for transfer learning
base.dir <- "/share/terra/Users/gz2294/PreMode.final/scripts/CHPs.v4.esm.dssp.small.StarAttn.MSA.StarPool.1dim/"
uniprotID.dic <- c("P21802"="FGFR2", "P15056"="BRAF", "P07949"="RET", "P04637"="TP53", 
                   # "Q09428"="ABCC8", 
                   "O00555"="CACNA1A", "Q14654"="KCNJ11", 
                   "Q99250"="SCN2A", "Q14524"="SCN5A", 
                   "IonChannel.split.uniprotID"="Na+/Ca2+ Channel*",
                   "IonChannel"="Na+/Ca2+ Channel"
                   # "IPR000719"="Protein Kinase Domain",
                   # "IPR001806"="Small GTPase"
)
# base.dirs <- strsplit(base.dirs, split = ',')[[1]]
# results <- data.frame()
# for (base.dir in base.dirs) {
result <- read.csv(paste0(base.dir, 'ICC.5fold.csv'))
result$model <- "PreMode"
result$task.name <- result$task
result$task.size <- result$task
result$task.type <- result$task
for (i in 1:dim(result)[1]) {
  result$task.id[i] <- strsplit(result$task[i], ": ")[[1]][1]
  result$task.size[i] <- strsplit(result$task[i], ": ")[[1]][2]
  result$task.name[i] <- gsub("\\.chps\\.even\\.uniprotID", "", result$task.id[i])
  if (!startsWith(result$task.name[i], "PF")) {
    tmp <- strsplit(result$task.name[i], ".chps")[[1]]
    if (length(tmp)==1) {
      result$task.type[i] <- "Gene"
    } else {
      result$task.type[i] <- "Gene/Pfam"
    }
    tmp[1] <- uniprotID.dic[tmp[1]]
    if (startsWith(result$task.name[i], "IPR")) {
      result$task.type[i] <- "Domain"
      result$task.name[i] <- paste(c("Domain: ", tmp), collapse = "")
    } else {
      result$task.name[i] <- paste(c("Gene: ", tmp), collapse = "")
    }
  } else {
    result$task.type[i] <- "Pfam"
    result$task.name[i] <- paste("Pfam: ", result$task.name[i], collapse = "")
  }
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
# result <- result[!duplicated(result$task.id),]
source('~/Pipeline/bind_rows.R')

result <- result[result$task.type %in% c("Gene", "Pfam"),]
for (i in 1:dim(result)[1]) {
  print(paste0('Begin ', i))
  # add ESM
  test.result <- read.csv(paste0('PreMode.inference/', result$task.id[i], '/all.fold.', result$fold[i], '.csv'))
  if (result$fold[i] == 0) {
    esm.logits <- my.bind.rows(read.csv(paste0('esm2.inference/', result$task.id[i], '/testing.fold.', 0, '.logits.csv')),
                               read.csv(paste0('esm2.inference/', result$task.id[i], '/training.fold.', 0, '.logits.csv')))
    esm.logits <- esm.logits[,2:34]
    colnames(esm.logits) <- alphabet
    score <- c()
    for (k in 1:dim(esm.logits)[1]) {
      score <- c(score, esm.logits[k, test.result$alt[k]] - esm.logits[k, test.result$ref[k]])
    }
    test.result$esm.logits <- score
    test.result <- uniprot.table.add.annotation.parallel(test.result, 'EVE')
    test.result <- uniprot.table.add.annotation.parallel(test.result, 'dbnsfp')
    test.result <- uniprot.table.add.annotation.parallel(test.result, 'Itan')
    test.result$itan.logits <- test.result$itan.gof / (1-test.result$itan.beni)
    test.result <- uniprot.table.add.annotation.parallel(test.result, 'gMVP')
    test.result <- uniprot.table.add.annotation.parallel(test.result, 'conservation')
  }
  write.csv(test.result, paste0('PreMode.inference/', result$task.id[i], '/all.fold.',
                                result$fold[i], '.annotated.csv'))
}
result.all <- data.frame()
for (i in 1:dim(result)[1]) {
  print(paste0('Begin ', i))
  test.result <- read.csv(paste0('PreMode.inference/', result$task.id[i], '/all.fold.',
                                result$fold[i], '.annotated.csv'), row.names = 1)
  # REVEL, PrimateAI, ESM AUC
  PreMode.auc <- plot.AUC(test.result$score, test.result$logits)
  if (result$fold[i] == 0) {
    REVEL.auc <- plot.AUC(test.result$score, test.result$REVEL)
    if (sum(is.na(test.result$REVEL)) >= 1) {REVEL.auc$auc <- NA}
    PrimateAI.auc <- plot.AUC(test.result$score, test.result$PrimateAI)
    if (sum(is.na(test.result$PrimateAI)) >= 1) {PrimateAI.auc$auc <- NA}
    ESM.auc <- plot.AUC(test.result$score, test.result$esm.logits)
    if (sum(is.na(test.result$esm.logits)) >= 1) {ESM.auc$auc <- NA}
    EVE.auc <- plot.AUC(test.result$score, test.result$EVE)
    if (sum(is.na(test.result$EVE)) >= 1) {EVE.auc$auc <- NA}
    gMVP.auc <- plot.AUC(test.result$score, test.result$gMVP)
    if (sum(is.na(test.result$gMVP)) >= 1) {gMVP.auc$auc <- NA}
    itan.auc <- plot.AUC(test.result$score, test.result$itan.logits)
    if (sum(is.na(test.result$itan.logits)) >= 1) {itan.auc$auc <- NA}
    itan.gof.auc <- plot.AUC(test.result$score, test.result$itan.gof)
    if (sum(is.na(test.result$itan.gof)) >= 1) {itan.gof.auc$auc <- NA}
    to.append <- result[rep(i, 8), ]
    to.append$min.val.auc <- c(PreMode.auc$auc, REVEL.auc$auc, PrimateAI.auc$auc, ESM.auc$auc, EVE.auc$auc, gMVP.auc$auc, itan.auc$auc, itan.gof.auc$auc)
    to.append$model <- c("PreMode", "REVEL", "PrimateAI", "ESM", 'EVE', 'gMVP', 'Itan', 'Itan.GOF')
  } else {
    to.append <- result[rep(i, 1), ]
    to.append$min.val.auc <- PreMode.auc$auc
    to.append$model <- c("PreMode")
  }
  result.all <- rbind(result.all, to.append)
}
write.csv(result.all, 'figs/02.02.ICC.PreMode.all.compare.csv')
result <- result.all
# result <- result[result$model != "Itan.GOF",]
result <- result[!grepl("\\.", result$task.name) & !grepl("NA$", result$task.name),]
num.models <- length(unique(result$model))
p <- ggplot(result, aes(y=min.val.auc, x=task.name, col=model)) +
  geom_point(alpha=0.2) +
  # geom_boxplot(width = 0.5, coef = 0) +
  # geom_segment(aes(x = as.numeric(factor(task))-0.4, xend = as.numeric(factor(task))+0.4, 
  #                  y = baseline.auc, yend = baseline.auc), color = "black") +
  stat_summary(data = result,
               aes(x=as.numeric(factor(task.name))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = min.val.auc, col=model), 
               fun.data = mean_se, geom = "errorbar", width = 0.2) +
  stat_summary(data = result, 
               aes(x=as.numeric(factor(task.name))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = min.val.auc, col=model), 
               fun.data = mean_se, geom = "point") +
  labs(x = "task", y = "min.val.auc", fill = "model") +
  theme_bw() + 
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        legend.position="bottom", 
        legend.direction="horizontal") +
  ylim(0.5, 1) + xlab('task: (LoF/GoF)')
ggsave(paste0('figs/02.02.ICC.PreMode.all.compare.pdf'), p, height = 6, width = 15)
