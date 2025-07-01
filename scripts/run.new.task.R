source('./analysis/AUROC.R')
# check task type
args <- commandArgs(trailingOnly = T)
scripts.prefix <- args[1]
output.dir <- args[2]
task.0 <- yaml::read_yaml(paste0(scripts.prefix, '.seed.0.yaml'))
gene.name <- strsplit(scripts.prefix, '/')[[1]]
gene.name <- gene.name[length(gene.name)]
task.type <- task.0$data_type
if (task.type == 'DMS') {
  # check if files are here
  completed <- T
  for (s in 0:4) {
    if (!file.exists(paste0(output.dir, '/', gene.name, '.testing.seed.', s, '.csv'))) {
      completed <- F
    }
    if (!file.exists(paste0(output.dir, '/', gene.name, '.training.seed.', s, '.csv'))) {
      completed <- F
    }
  }
} else if (task.type == 'GLOF') {
  # check if files are here
  completed <- T
  for (s in 0:4) {
    if (!file.exists(paste0(output.dir, '/', gene.name, '.testing.seed.', s, '.csv'))) {
      completed <- F
    }
    if (!file.exists(paste0(output.dir, '/', gene.name, '.training.seed.', s, '.csv'))) {
      completed <- F
    }
    if (!file.exists(paste0(output.dir, '/', gene.name, '.testing.seed.', s, '.large.window.csv'))) {
      completed <- F
    }
    if (!file.exists(paste0(output.dir, '/', gene.name, '.training.seed.', s, '.large.window.csv'))) {
      completed <- F
    }
  }
}
source('./analysis/AUROC.R')
if (completed) {
  if (task.type == 'GLOF') {
    test.file <- read.csv(paste0(output.dir, '/', gene.name, '.testing.seed.', 0, '.csv'))
    test.file <- test.file[,!colnames(test.file) %in% paste0('logits.FOLD.', 0:3)]
    for (s in 0:4) {
      tr.res <- read.csv(paste0(output.dir, '/', gene.name, '.training.seed.', s, '.csv'))
      tr.lw.res <- read.csv(paste0(output.dir, '/', gene.name, '.training.seed.', s, '.large.window.csv'))
      tr.auc <- plot.AUC(tr.res$score, rowMeans(tr.res[,paste0('logits.FOLD.', 0:3)]))$auc
      tr.lw.auc <- plot.AUC(tr.lw.res$score, rowMeans(tr.lw.res[,paste0('logits.FOLD.', 0:3)]))$auc
      if (tr.lw.auc > tr.auc) {
        test.res <- read.csv(paste0(output.dir, '/', gene.name, '.testing.seed.', s, '.csv'))
      } else {
        test.res <- read.csv(paste0(output.dir, '/', gene.name, '.testing.seed.', s, '.large.window.csv'))
      }
      test.logits <- rowMeans(test.res[,paste0('logits.FOLD.', 0:3)])
      test.file[,paste0('logits.seed.', s)] <- test.logits
    }
    test.file$logits <- rowMeans(test.file[,paste0('logits.seed.', 0:4)])
    write.csv(test.file, paste0(output.dir, '/', gene.name, '.inference.result.csv'))
  } else {
    test.file <- read.csv(paste0(output.dir, '/', gene.name, '.testing.seed.', 0, '.csv'))
    logits.cols <- colnames(test.file)[startsWith(colnames(test.file), 'logits')]
    for (s in 1:4) {
      test.res <- read.csv(paste0(output.dir, '/', gene.name, '.testing.seed.', s, '.csv'))
      for (i in logits.cols) {
        test.file[,i] <- test.file[,i] + test.res[,i]
      }
    }
    for (i in logits.cols) {
      test.file[,i] <- test.file[,i]/5
    }
    write.csv(test.file, paste0(output.dir, '/', gene.name, '.inference.result.csv'))
  }
}
