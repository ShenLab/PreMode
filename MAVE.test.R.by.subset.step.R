source('utils.R')
args <- commandArgs(trailingOnly = T)
# base dir for transfer learning
base.dir <- args[1]
result <- data.frame()
# tasks <- c("PTEN", "NUDT15", "CCR5", "CXCR4", "VKORC1")
tasks <- c("PTEN.bin", "PTEN", "NUDT15", "CCR5", "CXCR4")
for (task in tasks) {
  for (subset in c(1,2,4,6,8)) {
    for (seed in 0:2) {
      if (subset == 8) {
        configs <- yaml::read_yaml(paste0(base.dir, task, '.5fold/', 
                                          task, '.fold.', seed, '.yaml'))
      } else {
        configs <- yaml::read_yaml(paste0(base.dir, task, '.subsets/subset.', 
                                          subset, '/seed.', seed, '.yaml'))
      }
      bin = task == "PTEN.bin"
      res <- get.R.by.epoch(configs, bin=bin)
      R2s <- as.data.frame(res[,startsWith(colnames(res), "R2s")])
      if (dim(res)[1] == 0) {
        df <- NA
      } else {
        df <- data.frame(task = task,
                         subset = subset*10,
                         seed = seed,
                         min.val.R = R2s[which(res$val==min(res$val))[1],],
                         end.R = R2s[dim(res)[1],],
                         max.R = R2s[which(rowMeans(R2s)==max(rowMeans(R2s)))[1],]
        )
      }
      result <- dplyr::bind_rows(result, df)
    }
  }
}
# for (task in "IonChannel") {
#   for (subset in c(1,2,4,6,8)) {
#     for (seed in 0:2) {
#       if (subset == 8) {
#         configs <- yaml::read_yaml(paste0(base.dir, task, '/', 
#                                           task, '.seed.', seed, '.yaml'))
#       } else {
#         configs <- yaml::read_yaml(paste0(base.dir, task, '.subsets/subset.', 
#                                           subset, '/seed.', seed, '.yaml'))
#       }
#       res <- get.auc.by.epoch(configs)
#       df <- data.frame(task = task,
#                        subset = subset*10,
#                        seed = seed,
#                        min.val.R.R2s.1 = res$aucs[which(res$val==min(res$val))[1]],
#                        end.R = res$aucs[dim(res)[1]],
#                        max.R = res$aucs[which(res$aucs==max(res$aucs))[1]]
#       )
#       result <- dplyr::bind_rows(result, df)
#     }
#   }
# }
# p <- ggplot(result, aes(y=end.R, x=task)) + geom_point() + geom_boxplot() + theme_bw() + ylim(0.5, 1)
# ggsave(paste0(base.dir, 'MAVE.pdf'), p, height = 6, width = 8)
write.csv(result, paste0(base.dir, 'MAVE.subsets.csv'))
assay.ref <- list(c("stability", "enzyme activity"),
                  c("stability", "enzyme activity"),
                  c("stability", "bind AB-2D7", "bind HIV-1"),
                  c("stability", "bind CXCL12", "bind AB-12G5"),
                  c("enzyme activity", "stability"))
                  # c("GoF / LoF"))
baseline.ref <- list(c(0.464, 0.491),
                     c(0.389, 0.518),
                     c(0.424, 0.458, 0.411),
                     c(0.281, 0.171, 0.127),
                     c(NA, NA))
                     # c(0.328, 0.564),
                     # c(0.570))
# baseline.ref <- list(c(0.462, 0.491),
#                      c(0.421, 0.589),
#                      c(0.443, 0.424, 0.404),
#                      c(0.329, 0.193, 0.218),
#                      c(0.353, 0.561),
#                      c(0.570))
tasks <- c("PTEN", "NUDT15", "CCR5", "CXCR4", "PTEN.bin")
# tasks <- c("PTEN", "NUDT15", "CCR5", "CXCR4", "VKORC1", "IonChannel")
data.dir <- c("~/Data/DMS/MAVEDB/", "~/Data/DMS/MAVEDB/", 
              "~/Data/DMS/MAVEDB/", 
              "~/Data/DMS/MAVEDB/", "~/Data/DMS/MAVEDB/", 
              "~/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8/")
plots <- list()
for (i in 1:length(tasks)) {
  task <- tasks[i]
  task.res <- result[result$task==task,]
  task.res <- task.res[,!is.na(task.res[1,])]
  assays <- sum(startsWith(colnames(task.res), "min.val.R.R2s"))
  data.points <- c()
  for (subset in c(1,2,4,6)) {
    data.points <- c(data.points,
                     as.numeric(
                       strsplit(system(paste0("wc -l ", data.dir[i], task, ".", subset,".seed.0/training.csv"),
                                       intern = T), " ")[[1]][1])-1)
  }
  data.points <- c(data.points,
                   as.numeric(
                     strsplit(system(paste0("wc -l ", data.dir[i], task, "/training.csv"),
                                     intern = T), " ")[[1]][1]))
  to.plot <- data.frame()
  for (assay in 1:assays) {
    assay.res <- task.res[,c('task', 'subset', 'seed', paste0('min.val.R.R2s.', assay))]
    colnames(assay.res)[4] <- "assay"
    assay.res$zero.shot <- baseline.ref[[i]][assay]
    assay.res$assay.name <- assay.ref[[i]][assay]
    to.plot <- rbind(to.plot, assay.res)
  }
  p <- ggplot(to.plot, aes(x=subset, y=assay, col=assay.name)) + 
    geom_point() + geom_line(aes(y=zero.shot), linetype="dotted") + 
    geom_smooth() + scale_y_continuous(breaks=seq(0, 1, 0.2), limits = c(0, 1.05)) +
    scale_x_continuous(breaks=c(10, 20, 40, 60, 80),
                       labels=paste0(data.points,
                                     c(" (10%)", " (20%)", " (40%)", " (60%)", " (80%)"))) +
     theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ggtitle(task) + ggeasy::easy_center_title() + xlab("training data size (%)")
  plots[[i]] <- p
}
library(patchwork)
p <- (plots[[1]] + plots[[2]]) / (plots[[3]] + plots[[4]]) 
ggsave(p, filename = paste0("MAVE.subsets.pdf"), width = 10, height = 12)

