source('utils.R')
np <- reticulate::import("numpy")
args <- commandArgs(trailingOnly = T)
# base dir for transfer learning
base.dir <- args[1]
# base.dir <- 'scripts/CHPs.v3.onehot.small.dssp.TriStarAttn.Drop.MSA.GRU/'
tasks <- read.csv('scripts/pfams.txt', header = F)$V1
result <- data.frame()
# tasks <- c("IonChannel.split.uniprotID", tasks)
# tasks <- tasks[!startsWith(tasks, "IPR000719")]
# tasks <- c("IonChannel")
for (task in tasks) {
  neg.points <- 0
  pos.points <- 0
  print(task)
  for (fold in 0:4) {
    if (file.exists(paste0(base.dir, task, '.5fold/', task, '.fold.', fold, '.yaml'))) {
      configs <- yaml::read_yaml(paste0(base.dir, task, '.5fold/', task, '.fold.', fold, '.yaml'))
      data.train <- as.numeric(strsplit(system(paste0("wc -l ", configs$data_file_train), intern = T), split = " ")[[1]][1])
      num_saved_batches = floor(ceiling(data.train * configs$train_size / configs$ngpus / configs$batch_size)
                                * configs$num_epochs / configs$num_save_batches) + 1
      if (num_saved_batches >= configs$num_epochs) {
        res <- get.auc.by.step(configs, base.line="")
      } else {
        res <- get.auc.by.epoch(configs, base.line="")
      }
      data.test <- read.csv(configs$data_file_test, row.names = 1)
      data.train <- read.csv(configs$data_file_train, row.names = 1)
      # idx <- np$load(paste0(configs$log_dir, 'splits.0.npz'))
      # data.train <- data.train[idx['idx_train']+1,]
      # data.train <- data.train[data.train$score %in% c(-1, 1),]
      # data.val <- data.train[idx['idx_val']+1,]
      # data.val <- data.val[data.val$score %in% c(-1, 1),]
      neg.points <- sum(data.test$score==-1) + neg.points
      pos.points <- sum(data.test$score==1) + pos.points
      if (dim(res)[1] > 0) {
        if (is.null(res$baseline.auc)) {
          baseline.auc <- 0.5
        } else {
          baseline.auc <- res$baseline.auc[1]
        }
        df <- data.frame(task = task,
                         fold = fold,
                         min.val.loss = min(res$val),
                         min.train.loss = res$train[which(res$val==min(res$val))][1],
                         min.test.loss = res$test[which(res$val==min(res$val))][1],
                         min.val.auc = res$aucs[which(res$val==min(res$val))][1],
                         min.val.step = which(res$val==min(res$val))[1],
                         end.auc = res$aucs[dim(res)[1]][1],
                         max.auc = res$aucs[which(res$aucs==max(res$aucs))][1],
                         max.auc.step = which(res$aucs==max(res$aucs))[1],
                         baseline.auc = baseline.auc
        )
      } else {
        df <- data.frame(task = task,
                         fold = fold
        )
      }
    } else {
      df <- data.frame(task = task,
                       fold = fold
      )
    }
    result <- dplyr::bind_rows(result, df)
  }
  result$task[result$task==task] <- paste0(task, ": (", floor(neg.points/5), "/", floor(pos.points/5), ")")
}
p <- ggplot(result, aes(y=min.val.auc, x=task)) + geom_point() + geom_boxplot() +
  geom_segment(aes(x = as.numeric(factor(task))-0.4, xend = as.numeric(factor(task))+0.4, 
                   y = baseline.auc, yend = baseline.auc), color = "red") +
  theme_bw() + theme(axis.text.x = element_text(angle=45, vjust = 1, hjust = 1)) + ylim(0.5, 1) + xlab('task: (LoF/GoF)')
ggsave(paste0(base.dir, 'ICC.5fold.pdf'), p, height = 6, width = 8)
write.csv(result, paste0(base.dir, 'ICC.5fold.csv'))
