# source('utils.R')
library(ggplot2)
args <- commandArgs(trailingOnly = T)
# base dir for transfer learning
base.dirs <- args[1]
base.dirs <- strsplit(base.dirs, split = ',')[[1]]
results <- data.frame()
for (base.dir in base.dirs) {
  result <- read.csv(paste0(base.dir, 'ICC.5fold.csv'))
  result$model <- strsplit(base.dir, "/")[[1]][2]
  results <- dplyr::bind_rows(results, result)
}
summary_data <- aggregate(min.val.auc ~ task+model, results, 
                          function(x) c(mean = mean(x), se = sd(x) / sqrt(length(x))))
num.models <- length(unique(results$model))
p <- ggplot(results, aes(y=min.val.auc, x=task, col=model)) +
  geom_point(alpha=0.2) +
  # geom_boxplot(width = 0.5, coef = 0) +
  # geom_segment(aes(x = as.numeric(factor(task))-0.4, xend = as.numeric(factor(task))+0.4, 
  #                  y = baseline.auc, yend = baseline.auc), color = "black") +
  stat_summary(data = results, aes(x=as.numeric(factor(task))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                                   y = min.val.auc, col=model), 
               fun.data = mean_se, geom = "errorbar", width = 0.2) +
  stat_summary(data = results, aes(x=as.numeric(factor(task))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                                   y = min.val.auc, col=model), 
               fun.data = mean_se, geom = "point") +
  labs(x = "task", y = "min.val.auc", fill = "model") +
  theme_bw() + 
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        legend.position="bottom", 
        legend.direction="vertical") +
  ylim(0.5, 1) + xlab('task: (LoF/GoF)')
ggsave(paste0('ICC.5fold.compare.pdf'), p, height = 6, width = 15)
