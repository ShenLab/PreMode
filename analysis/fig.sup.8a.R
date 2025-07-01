library(ggplot2)
result.plot <- readRDS('figs/fig.5.prepare.RDS')
result.plot <- result.plot[result.plot$task.type=='Gene',]
result.plot$use.lw <- F
# remove itan tasks
result.plot <- result.plot[!grepl('.itan.split', result.plot$task.id),]
pick.cond <- 'auc'
# get unique models
uniq.models <- unique(gsub('.lw', '', result.plot$model))
# only keep the original models
uniq.models <- uniq.models[grepl('/$', uniq.models)]
# get unique genes, remove Q14524
uniq.genes <- unique(result.plot$task.id)
uniq.genes <- uniq.genes[uniq.genes != "Q14524"]
# for each gene and each fold, decide weather to use large window
for (g in uniq.genes) {
  for (m in uniq.models) {
    for (f in 0:4) {
      lw.loss <- result.plot$val.loss[result.plot$model == paste0(m, '.lw') & result.plot$task.id == g & result.plot$fold==f]
      loss <- result.plot$val.loss[result.plot$model == m & result.plot$task.id == g & result.plot$fold==f]
      lw.tr.auc <- result.plot$tr.auc[result.plot$model == paste0(m, '.lw') & result.plot$task.id == g & result.plot$fold==f]
      tr.auc <- result.plot$tr.auc[result.plot$model == m & result.plot$task.id == g & result.plot$fold==f]
      if (pick.cond == 'auc') {
        cond <- !is.na(mean(lw.tr.auc)) & lw.tr.auc > tr.auc
      } else if (pick.cond == 'loss') {
        cond <- !is.na(mean(lw.loss)) & loss > lw.loss
      } else if (pick.cond == 'auc+loss') {
        cond <- !is.na(lw.loss) & !is.na(lw.tr.auc) & (tr.auc/loss > lw.tr.auc/lw.loss)
      } else {
        cond <- F
      }
      if (cond) {
        # use lw
        to.remove <- which(result.plot$model == m & result.plot$task.id == g & result.plot$fold==f)
        to.anno <- which(result.plot$model == paste0(m, '.lw') & result.plot$task.id == g & result.plot$fold==f)
        result.plot$model[to.anno] <- m
        result.plot$use.lw[to.anno] <- T
        result.plot <- result.plot[-to.remove,]
      } else {
        to.remove <- which(result.plot$model == paste0(m, '.lw') & result.plot$task.id == g & result.plot$fold==f)
        result.plot <- result.plot[-to.remove,]
      }
    }
  }
}

result.plot <- result.plot[!result.plot$task.id %in% c('Q14524'),]
result.plot$task.name[result.plot$task.id == "Q14524.clean"] <- "Gene: SCN5A"
result.plot <- result.plot[result.plot$model %in% c("PreMode/",
                                                    "PreMode.noStructure/"),]
model.dic <- c("PreMode/"="1: PreMode",
               "PreMode.noStructure/"="7: PreMode: no Structure")
result.plot$model <- model.dic[result.plot$model]
num.models <- length(unique(result.plot$model))
p1 <- ggplot(result.plot, aes(y=auc, x=task.name, col=model)) +
  geom_point(alpha=0) + 
  stat_summary(data = result.plot,
               aes(x=as.numeric(factor(task.name))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = auc, col=model), 
               fun.data = mean_se, geom = "errorbar", width = 0.2) +
  stat_summary(data = result.plot, 
               aes(x=as.numeric(factor(task.name))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = auc, col=model), 
               fun.data = mean_se, geom = "point") +
  labs(x = "task", y = "AUC", fill = "model") +
  theme_bw() + 
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        text = element_text(size = 16),
        plot.title = element_text(size=15),
        legend.text = element_text(size=10),
        legend.position="bottom", 
        legend.direction="horizontal") +
  ggtitle('PreMode Ablation Analysis') +
  ggeasy::easy_center_title() +
  coord_flip() + guides(col=guide_legend(nrow=2),
                        shape=guide_legend(nrow=2)) +
  ylim(0.25, 1) + xlab('task: Genetics Level Mode of Action') 
ggsave(paste0('figs/fig.sup.8a.pdf'), p1, height = 5, width = 6)


