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
uniq.genes <- unique(result.plot$task.id)
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

result.plot$task.name[result.plot$task.id == "Q14524.clean"] <- "Gene: SCN5A"
result.plot <- result.plot[result.plot$model %in% c("PreMode/",
                                                    "PreMode.noESM/",
                                                    "PreMode.noMSA/",
                                                    "PreMode.noStructure/",
                                                    "PreMode.ptm/",
                                                    "ESM.SLP/",
                                                    "PreMode.noPretrain/",
                                                    "random.forest"
                                                    ),]
model.dic <- c("PreMode/"="1: PreMode",
               "PreMode.noPretrain/"="2: PreMode: no Pretrain",
               "random.forest"="3: Random Forest",
               "ESM.SLP/"="4: ESM + SLP",
               "PreMode.noESM/"="5: PreMode: no ESM",
               "PreMode.noMSA/"="6: PreMode: no MSA",
               "PreMode.noStructure/"="7: PreMode: no Structure",
               "PreMode.ptm/"="8: PreMode: add ptm")
result.plot$model <- model.dic[result.plot$model]

# plot the task weighted averages as well as task size weighted error bars
uniq.result.plot <- result.plot[result.plot$fold==0,]
for (i in 1:dim(uniq.result.plot)[1]) {
  aucs <- result.plot$auc[result.plot$model==uniq.result.plot$model[i] & 
                            result.plot$task.name==uniq.result.plot$task.name[i]]
  uniq.result.plot$auc[i] = mean(aucs, na.rm=T)
  uniq.result.plot$auc.se[i] = sd(aucs, na.rm=T) / sqrt(length(aucs))
}
# aggregate across models
uniq.model.result.plot <- uniq.result.plot[!duplicated(uniq.result.plot$model),]
for (i in 1:dim(uniq.model.result.plot)[1]) {
  task.sizes.lof <- uniq.result.plot$task.size.lof[uniq.result.plot$model==uniq.model.result.plot$model[i]] 
  task.sizes.gof <- uniq.result.plot$task.size.gof[uniq.result.plot$model==uniq.model.result.plot$model[i]]
  # change to harmonic average of task size
  task.sizes <- task.sizes.lof * task.sizes.gof / (task.sizes.lof + task.sizes.gof)
  aucs <- uniq.result.plot$auc[uniq.result.plot$model==uniq.model.result.plot$model[i]]
  auc.ses <- uniq.result.plot$auc.se[uniq.result.plot$model==uniq.model.result.plot$model[i]]
  # remove NA values
  task.sizes <- task.sizes[!is.na(aucs)]
  aucs <- aucs[!is.na(aucs)]
  auc.ses <- auc.ses[!is.na(auc.ses)]
  uniq.model.result.plot$auc[i] <- sum(aucs * task.sizes / sum(task.sizes), na.rm=T)
  uniq.model.result.plot$auc.se[i] <- sum(auc.ses * task.sizes / sum(task.sizes), na.rm=T)
}

uniq.model.result.plot$model.type <- 'PreMode: Ablation'
uniq.model.result.plot$model.type[uniq.model.result.plot$model == "8: PreMode: add ptm"] <- 'PreMode: add ptm'
uniq.model.result.plot$model.type[uniq.model.result.plot$model == "4: ESM + SLP"] <- 'Baselines'
uniq.model.result.plot$model.type[uniq.model.result.plot$model == "3: Random Forest"] <- 'Baselines'
uniq.model.result.plot$model.type[uniq.model.result.plot$model == "1: PreMode"] <- 'PreMode'
uniq.model.result.plot$model.type <- factor(uniq.model.result.plot$model.type, 
                                            levels = c('PreMode', 'PreMode: add ptm', 'PreMode: Ablation', 'Baselines'))
# chose other color scale
p <- ggplot(uniq.model.result.plot, aes(x=model, y=auc, col=model.type)) +
  geom_point() + scale_color_manual(values = c("#F8766D", "#CD9600", "#999999", "#619CFF")) + 
  geom_errorbar(aes(ymin=auc-auc.se, ymax=auc+auc.se), width=.2) +
  coord_flip() + guides(col=guide_legend(ncol=2)) +
  labs(x = "models", y = "auc", fill = "model") +
  theme_bw() + ylim(0.5, 0.9) + ggtitle('PreMode ablation analysis') +
  ggeasy::easy_center_title() +
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        text = element_text(size = 16),
        plot.title = element_text(size=15),
        legend.text = element_text(size=10),
        legend.title = element_blank(),
        legend.position="bottom", 
        legend.direction="horizontal") + 
  ggeasy::easy_center_title()
ggsave('figs/fig.5b.pdf', p, height=5, width=6)


