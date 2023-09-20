# source('utils.R')
library(ggplot2)
args <- commandArgs(trailingOnly = T)
# base dir for transfer learning
base.dirs <- args[1]
base.dirs <- strsplit(base.dirs, split = ',')[[1]]
results <- data.frame()
for (base.dir in base.dirs) {
  result <- read.csv(paste0(base.dir, 'Hsu.5fold.csv'))
  result$model <- strsplit(base.dir, "/")[[1]][2]
  results <- dplyr::bind_rows(results, result)
}
good.tasks <- read.csv('/share/terra/Users/gz2294/ld1/Data/DMS/Hsu_NBT/useful.data.human.csv', header = F)$V1

hsu.baseline <- read.csv('~/Data/DMS/Hsu_NBT/results_all.csv')
colnames(hsu.baseline)[2] <- "task"

hsu.baseline$min.val.R <- hsu.baseline$spearman
hsu.baseline$end.R <- hsu.baseline$spearman
hsu.baseline$max.R <- hsu.baseline$spearman
hsu.baseline$model <- hsu.baseline$predictor
hsu.baseline <- hsu.baseline[hsu.baseline$n_train==-1,]
hsu.baseline$fold <- hsu.baseline$seed

results <- rbind(results, hsu.baseline[,colnames(hsu.baseline)[colnames(hsu.baseline) %in% colnames(results)]])
results <- results[results$task %in% good.tasks,]
results <- results[results$fold %in% 0:4,]

tasks <- as.character(unique(results$task[!is.na(results$task)]))
n_data <- tasks
names(n_data) <- tasks
for (i in 1:length(tasks)) {
  n_train <- dim(read.csv(paste0('~/Data/DMS/Hsu_NBT/', tasks[i], '/train.seed.0.csv')))[1]
  n_test <- dim(read.csv(paste0('~/Data/DMS/Hsu_NBT/', tasks[i], '/test.seed.0.csv')))[1]
  n_data[tasks[i]] <- paste0(tasks[i], ": (", n_train, "/", n_test, ")")
}
results$task <- n_data[results$task]

summary_data <- aggregate(min.val.R ~ task+model, results, 
                          function(x) c(mean = mean(x), se = sd(x) / sqrt(length(x))))
num.models <- length(unique(results$model))
p <- ggplot(results, aes(y=min.val.R, x=task, col=model)) +
  geom_point(alpha=0.1) +
  stat_summary(data = results, aes(x=as.numeric(factor(task))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                                   y = min.val.R, col=model), 
               fun.data = mean_se, geom = "errorbar", width = 0.2) +
  stat_summary(data = results, aes(x=as.numeric(factor(task))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                                   y = min.val.R, col=model), 
               fun.data = mean_se, geom = "point") +
  labs(x = "task", y = "min.val.R", fill = "model") +
  theme_bw() + 
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        legend.position="bottom", 
        legend.direction="vertical") +
  xlab('task: (train/test)')
ggsave(paste0('Hsu.5fold.compare.pdf'), p, height = 15, width = 15)
