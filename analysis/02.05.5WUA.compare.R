# source('utils.R')
library(ggplot2)
# args <- commandArgs(trailingOnly = T)
# base dir for transfer learning
# base.dir <- "/share/pascal/Users/gz2294/PreMode/scripts/CHPs.v4.esm.dssp.small.StarAttn.MSA.StarPool.1dim/"
uniprotID.dic <- c("Q14654"="KCNJ11",  "5WUA"="KCNJ11")
# result.all <- list()
result.gof <- data.frame()
for (fold in 0:4) {
  test.result <- read.csv(paste0('PreMode.inference/', 'Q14654/testing.fold.',
                                 fold, '.annotated.csv'), row.names = 1)
  baseline.result <- read.csv(paste0('PreMode.pass.inference/', 'Q14654', '/testing.fold.',
                                     fold, '.csv'))
  baseline.result.2 <- read.csv(paste0('PreMode.onehot.inference/', 'Q14654', '/testing.fold.',
                                       fold, '.csv'))
  baseline.result.3 <- read.csv(paste0('PreMode.radius.inference/', '5WUA', '/testing.fold.',
                                       fold, '.csv'))
  
  test.result <- test.result[test.result$score!=0,]
  baseline.result <- baseline.result[baseline.result$score!=0,]
  baseline.result.2 <- baseline.result.2[baseline.result.2$score!=0,]
  PreMode.auc <- plot.AUC(test.result$score, test.result$gof.logits/test.result$lof.logits)
  baseline.auc <- plot.AUC(baseline.result$score, baseline.result$logits)
  baseline.auc.2 <- plot.AUC(baseline.result.2$score, baseline.result.2$logits)
  baseline.auc.3 <- plot.AUC(baseline.result.3$score, baseline.result.3$logits)
  
  to.append <- data.frame(fold=rep(fold, 4))
  to.append$min.val.auc <- c(PreMode.auc$auc, baseline.auc$auc,
                             baseline.auc.2$auc, baseline.auc.3$auc)
  to.append$model <- c("PreMode", "Baseline (No structure)",
                       "Baseline (No ESM)", "PreMode (radius)")
  result.gof <- rbind(result.gof, to.append)
}

result.plot <- result.gof
num.models <- length(unique(result.plot$model))
p <- ggplot(result.plot, aes(y=min.val.auc, x=model, col=model)) +
  geom_point(alpha=0.2) +
  stat_summary(data = result.plot,
               aes(x=as.numeric(factor(model))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = min.val.auc, col=model), 
               fun.data = mean_se, geom = "errorbar", width = 0.2) +
  stat_summary(data = result.plot, 
               aes(x=as.numeric(factor(model))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = min.val.auc, col=model), 
               fun.data = mean_se, geom = "point") +
  labs(x = "task", y = "min.val.auc", fill = "model") +
  theme_bw() + 
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        legend.position="right", 
        legend.direction="vertical") +
  ylim(0.5, 1) + xlab('task: (LoF/GoF)') 
ggsave(paste0('figs/02.05.5WUA.PreMode.compare.pdf'), p, height = 4, width = 6)

