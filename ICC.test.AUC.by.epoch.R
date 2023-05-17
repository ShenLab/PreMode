source('utils.R')
args <- commandArgs(trailingOnly = T)
# base dir for transfer learning
base.dir <- args[1]
tasks <- read.csv('scripts/pfams.txt', header = F)
result <- data.frame()
tasks <- c("IonChannel", tasks$V1)
for (task in tasks) {
  for (seed in 0:2) {
    configs <- yaml::read_yaml(paste0(base.dir, task, '/', task, '.seed.', seed, '.yaml'))
    res <- get.auc.by.epoch(configs)
    df <- data.frame(task = task,
                     seed = seed,
                     min.val.auc = res$aucs[which(res$val==min(res$val))][1],
                     end.auc = res$aucs[dim(res)[1]][1],
                     max.auc = res$aucs[which(res$aucs==max(res$aucs))][1]
    )
    result <- rbind(result, df)
  }
}
p <- ggplot(result, aes(y=end.auc, x=task)) + geom_point() + geom_boxplot() + theme_bw() + ylim(0.5, 1)
ggsave(paste0(base.dir, 'ICC.pdf'), p, height = 6, width = 8)
write.csv(result, paste0(base.dir, 'ICC.csv'))
