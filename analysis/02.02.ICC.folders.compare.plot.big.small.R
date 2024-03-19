# args <- commandArgs(trailingOnly = T)
out.dir.1 <- 'figs/02.02.ICC.tasks.compare.big.small.pdf'

result.plot <- readRDS('figs/02.02.ICC.folders.compare.RDS')
result.plot <- result.plot[result.plot$task.type %in% c("Gene"),]
result.plot <- result.plot[result.plot$model %in% c("PreMode.inference/", 
                                                    # "PreMode.PRE.v8/",
                                                    # "PreMode.CHPs.v4.large.window/",
                                                    # "PreMode.CHPs.v4.af2.rep/",
                                                    "PRE.v12/",
                                                    "PRE.v13/",
                                                    "PRE.v15/"
                                                    # "PreMode.CHPs.v4.esm_mask/"
                                                    # "PreMode.CHPs.v4.new.coevol/",
                                                    # "Itan.1"
                                                    ),]
model.dic <- c("PreMode.inference/"="1: PreMode (148k)",
               "PreMode.PRE.v8/"="2: PreMode (4.7M)",
               # "PreMode.CHPs.v4.esm_mask/"="3: ESM + LM Head",
               "PreMode.CHPs.v4.large.window/"="2: PreMode (148k, large window)",
               "PreMode.CHPs.v4.af2.rep/"="02: PreMode (af2.rep)",
               "PRE.v12/"="02: PreMode (v12)",
               "PRE.v13/"="02: PreMode (v13)",
               "PRE.v15/"="02: PreMode (v15)"
               # "PreMode.CHPs.v4.new.coevol/"="3: PreMode (148k new evol)",
               # "Itan.1"="2: LoGoFunc"
               )

result.plot$model <- model.dic[result.plot$model]

num.models <- length(unique(result.plot$model))
p <- ggplot(result.plot, aes(y=auc, x=task.name, col=model)) +
  geom_point(alpha=0) +
  stat_summary(data = result.plot,
               aes(x=as.numeric(factor(task.name))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = auc, col=model), 
               fun.data = mean_se, geom = "errorbar", width = 0.2) +
  stat_summary(data = result.plot, 
               aes(x=as.numeric(factor(task.name))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = auc, col=model), 
               fun.data = mean_se, geom = "point") +
  labs(x = "task", y = "auc", fill = "model") +
  theme_bw() + 
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        legend.position="bottom", 
        legend.direction="horizontal") +
  coord_flip() + guides(col=guide_legend(ncol=1)) +
  ylim(0.5, 1) + xlab('task: Genetics Level Mode of Action') 
ggsave(paste0(out.dir.1), p, height = 6, width = 6)



# plot the task weighted averages as well as task size weighted error bars
uniq.result.plot <- result.plot[result.plot$fold==0,]
for (i in 1:dim(uniq.result.plot)[1]) {
  uniq.result.plot$auc[i] = mean(result.plot$auc[result.plot$model==uniq.result.plot$model[i] & 
                                                   result.plot$task.name==uniq.result.plot$task.name[i]], na.rm=T)
  uniq.result.plot$auc.sd[i] = sd(result.plot$auc[result.plot$model==uniq.result.plot$model[i] & 
                                                    result.plot$task.name==uniq.result.plot$task.name[i]], na.rm=T)
}
# aggregate across models
uniq.model.result.plot <- uniq.result.plot[!duplicated(uniq.result.plot$model),]
for (i in 1:dim(uniq.model.result.plot)[1]) {
  task.sizes.lof <- uniq.result.plot$task.size.lof[uniq.result.plot$model==uniq.model.result.plot$model[i]] 
  task.sizes.gof <- uniq.result.plot$task.size.gof[uniq.result.plot$model==uniq.model.result.plot$model[i]]
  task.sizes <- task.sizes.lof + task.sizes.gof
  aucs <- uniq.result.plot$auc[uniq.result.plot$model==uniq.model.result.plot$model[i]]
  auc.sds <- uniq.result.plot$auc.sd[uniq.result.plot$model==uniq.model.result.plot$model[i]]
  # remove NA values
  task.sizes <- task.sizes[!is.na(aucs)]
  aucs <- aucs[!is.na(aucs)]
  auc.sds <- auc.sds[!is.na(auc.sds)]
  uniq.model.result.plot$auc[i] <- sum(aucs * task.sizes / sum(task.sizes), na.rm=T)
  uniq.model.result.plot$auc.sd[i] <- sum(auc.sds * task.sizes / sum(task.sizes), na.rm=T)
}
# chose other color scale
p <- ggplot(uniq.model.result.plot, aes(x=model, y=auc, col=model)) +
  geom_point() + 
  # scale_color_manual(values = c("#F8766D", "#CD9600", "#999999", "#619CFF")) + 
  geom_errorbar(aes(ymin=auc-auc.sd, ymax=auc+auc.sd), width=.2) +
  coord_flip() + guides(col=guide_legend(ncol=2)) +
  labs(x = "models", y = "auc", fill = "model") +
  theme_bw() + ylim(0.5, 0.9) + ggtitle('PreMode ablation analysis') +
  ggeasy::easy_center_title() +
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        text = element_text(size = 13),
        legend.position="bottom", 
        legend.direction="horizontal") + 
  ggeasy::easy_center_title()
ggsave('figs/02.02.ICC.tasks.compare.big.small.models.pdf', p, height=6, width=6)

