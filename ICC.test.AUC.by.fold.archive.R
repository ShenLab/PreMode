source('utils.R')
args <- commandArgs(trailingOnly = T)
# base dir for transfer learning
base.dir <- args[1]
# base.dir <- 'scripts/CHPs.v3.onehot.small.dssp.TriStarAttn.Drop.MSA.GRU/'
tasks <- read.csv('scripts/pfams.txt.archive', header = F)
result <- data.frame()
tasks <- c("IonChannel.split.uniprotID", tasks$V1)
tasks <- tasks[!startsWith(tasks, "PF_IPR000719")]
tasks <- tasks[!startsWith(tasks, "ALL")]
# tasks <- c("IonChannel")
for (task in tasks) {
  neg.points <- 0
  pos.points <- 0
  print(task)
  for (fold in 0:4) {
    configs <- yaml::read_yaml(paste0(base.dir, task, '.5fold/', task, '.fold.', fold, '.yaml'))
    res <- get.auc.by.epoch(configs)
    data.test <- read.csv(configs$data_file_test)
    neg.points <- sum(data.test$score==-1) + neg.points
    pos.points <- sum(data.test$score==1) + pos.points
    if (dim(res) > 0) {
      df <- data.frame(task = task,
                       fold = fold,
                       min.val.auc = res$aucs[which(res$val==min(res$val))][1],
                       min.val.step = which(res$val==min(res$val))[1],
                       end.auc = res$aucs[dim(res)[1]][1],
                       max.auc = res$aucs[which(res$aucs==max(res$aucs))][1],
                       max.auc.step = which(res$aucs==max(res$aucs))[1],
                       baseline.auc = res$baseline.auc[1]
      )
    } else {
      df <- data.frame(task = task,
                       fold = fold)
    }
    result <- dplyr::bind_rows(result, df)
  }
  result$task[result$task==task] <- paste0(task, ": (", floor(neg.points/5), "/", floor(pos.points/5), ")")
}
p <- ggplot(result, aes(y=end.auc, x=task)) + geom_point() + geom_boxplot() +
  geom_segment(aes(x = as.numeric(factor(task))-0.4, xend = as.numeric(factor(task))+0.4, 
                   y = baseline.auc, yend = baseline.auc), color = "red") +
  theme_bw() + theme(axis.text.x = element_text(angle=45, vjust = 1, hjust = 1)) + ylim(0.5, 1) + xlab('task: (LoF/GoF)')
ggsave(paste0(base.dir, 'ICC.5fold.archive.pdf'), p, height = 6, width = 8)
write.csv(result, paste0(base.dir, 'ICC.5fold.archive.csv'))
