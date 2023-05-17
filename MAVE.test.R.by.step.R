source('utils.R')
args <- commandArgs(trailingOnly = T)
# base dir for transfer learning
base.dir <- args[1]
result <- data.frame()
tasks <- c("PTEN", "NUDT15", "CCR5", "CXCR4", "VKORC1")
for (task in tasks) {
  for (seed in 0:2) {
    configs <- yaml::read_yaml(paste0(base.dir, task, '/', task, '.seed.', seed, '.yaml'))
    res <- get.R.by.step(configs)
    R2s <- as.data.frame(res[,startsWith(colnames(res), "R2")])
    df <- data.frame(task = task,
                     seed = seed,
                     min.val.R = R2s[which(res$val==min(res$val))[1],],
                     end.R = R2s[dim(res)[1],],
                     max.R = R2s[which(rowMeans(R2s)==max(rowMeans(R2s)))[1],]
    )
    result <- dplyr::bind_rows(result, df)
  }
}
# p <- ggplot(result, aes(y=end.R, x=task)) + geom_point() + geom_boxplot() + theme_bw() + ylim(0.5, 1)
# ggsave(paste0(base.dir, 'MAVE.pdf'), p, height = 6, width = 8)
write.csv(result, paste0(base.dir, 'MAVE.csv'))
