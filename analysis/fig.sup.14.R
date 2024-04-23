library(ggplot2)
task.dic <- list("fluorescence"=c("score"="fluorescence"))
genes <- c("fluorescence")
# add baseline AUC
# esm alphabets
source('./AUROC.R')
alphabet <- c('<cls>', '<pad>', '<eos>', '<unk>',
              'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
              'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
              'X', 'B', 'U', 'Z', 'O', '.', '-',
              '<null_1>', '<mask>')
result <- data.frame()
dash.base.line.models <- data.frame()
for (i in 1:length(genes)) {
  task.length <- length(task.dic[[genes[i]]])
  for (subset in 1:6) {
    for (fold in 1:4) {
      # REVEL, PrimateAI, ESM AUC
      if (!subset %in% c(1,2,4,6,8)) {
        test.result <- read.csv(paste0('../',
                                       yaml::read_yaml(paste0('../scripts/PreMode.mean.var/', 
                                                              genes[i], '/', genes[i], '.seed.',
                                                              fold, '.yaml'))$log_dir, '/testing.round.', subset-1, '.csv'))
        baseline.auc.1 <- list(R2=rep(NA, task.length))
      } else {
      test.result <- read.csv(paste0('../', 
                                     yaml::read_yaml(paste0('../scripts/PreMode.mean.var/', 
                                                            genes[i], '/', genes[i], '.seed.',
                                                            fold, '.yaml'))$log_dir, '/testing.round.', subset-1, '.csv'))
      baseline.result.1 <- read.csv(paste0('PreMode/', genes[i], '/',
                                           '/testing.subset.', subset, '.fold.', fold, '.csv'))
      baseline.auc.1 <- plot.R2(baseline.result.1[,names(task.dic[[genes[i]]])],
                                baseline.result.1[,paste0("logits")],
                                bin = grepl("bin", genes[i]))
      }
      PreMode.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], 
                             test.result[,paste0("logits")],
                             bin = grepl("bin", genes[i]))
      to.append <- data.frame(min.val.R = c(PreMode.auc$R2,
                                            baseline.auc.1$R2),
                              task.name = paste0(genes[i], ":", rep(task.dic[[genes[i]]], 2)))
      to.append$model <- rep(c("PreMode (Adaptive Learning)",
                               "PreMode"
      ), each = task.length)
      to.append$subset <- subset
      to.append$seed <- fold
      result <- rbind(result, to.append)
    }
  }
}
num.models <- unique(result$model)
plots <- list()
library(patchwork)
for (i in 1:length(task.dic)) {
  task <- names(task.dic)[i]
  task.res <- result[startsWith(result$task.name, paste0(task, ":")),]
  task.res <- task.res[,!is.na(task.res[1,])]
  assays <- length(task.dic[[i]])
  data.points <- c()
  for (subset in 1:6) {
    train.file <- read.csv(paste0('../', yaml::read_yaml(paste0('../scripts/PreMode.mean.var/', 
                                                         genes[i], '/', genes[i], '.seed.',
                                                         fold, '.yaml'))$log_dir, '/data_file_train.round.', subset-1, '.csv'))
    data.points <- c(data.points, sum(train.file$split=='train'))
  }
  task.plots <- list()
  for (k in 1:length(num.models)) {
    model <- num.models[k]
    to.plot <- task.res[task.res$model==model,]
    to.plot <- to.plot[!is.na(to.plot$min.val.R),]
    # only keep the mean and var
    to.plot.uniq <- to.plot[to.plot$seed==1,]
    for (j in 1:dim(to.plot.uniq)[1]) {
      rhos <- to.plot$min.val.R[to.plot$subset==to.plot.uniq$subset[j]]
      rhos <- rhos[rhos>0]
      to.plot.uniq$rho[j] <- mean(rhos, na.rm = T)
      to.plot.uniq$rho.sd[j] <- sd(rhos, na.rm = T)
    }
    to.plot.uniq$task.name <- 'fluorescence'
    p <- ggplot(to.plot.uniq, aes(x=subset, y=rho, col=task.name)) + 
      geom_point() + 
      geom_line() +
      geom_errorbar(aes(ymin=rho-rho.sd, ymax=rho+rho.sd), width=.2) +
      scale_y_continuous(breaks=seq(0.4, 0.8, 0.2), limits = c(0.4, 0.8)) +
      scale_x_continuous(breaks=1:6,
                         labels=paste0(data.points,
                                       paste0(" (", 1:6, "0%)"))) +
      labs(col = "Fluorescence") +
      theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      ggtitle(paste0(task, ":", model)) + ggeasy::easy_center_title() + xlab("training data size (%)")
    p <- p + geom_abline(slope=0, intercept=0.69, linetype='dashed') +
      geom_text(x=2, y=0.72, label='rho=0.69', col='black')
    task.plots[[k]] <- p
  }
  plots[[i]] <- task.plots[[1]] + task.plots[[2]] 
}
library(patchwork)
p <- plots[[1]] 
ggsave(p, filename = paste0("figs/fig.sup.14.pdf"), width = 10, height = 4.5)