source('utils.R')
args <- commandArgs(trailingOnly = T)
# base dir for transfer learning
base.dir <- args[1]
tasks <- read.csv('/share/terra/Users/gz2294/ld1/Data/DMS/Hsu_NBT/useful.data.csv', header = F)
result <- data.frame()
tasks <- c(tasks$V1)
# tasks <- tasks[tasks != "PF_IPR000719"]
# tasks <- c("IonChannel")
for (task in tasks) {
  for (fold in 0:4) {
    print(paste(task, ":", fold))
    configs <- yaml::read_yaml(paste0(base.dir, task, '.5fold/', task, '.fold.', fold, '.yaml'))
    res <- get.R.by.epoch(configs)
    # print(res)
    if (dim(res)[1] > 0) {
      df <- data.frame(task = task,
                       fold = fold,
                       min.val.R = res$R2s[which(res$val==min(res$val))][1],
                       end.R = res$R2s[dim(res)[1]][1],
                       max.R = res$R2s[which(res$R2s==max(res$R2s))][1]
      )
    } else {
      df <- NA
    }
    result <- rbind(result, df)
  }
}
p <- ggplot(result, aes(y=min.val.R, x=task)) + 
  geom_point() + geom_boxplot() + theme_bw() + ylim(0.5, 1)
ggsave(paste0(base.dir, 'Hsu.5fold.pdf'), p, height = 6, width = 8)
write.csv(result, paste0(base.dir, 'Hsu.5fold.csv'))
