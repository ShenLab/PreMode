out.dir.2 <- 'figs/fig.5b.pdf'
result.plot <- readRDS('figs/fig.5.prepare.RDS')
result.plot <- result.plot[result.plot$task.type %in% c("Gene"),]
result.plot <- result.plot[result.plot$task.id != "Heyne",]

result.plot <- result.plot[result.plot$model %in% c("PreMode.inference/",
                                                    "PreMode.noPretrain/",
                                                    "PreMode.pass.inference/",
                                                    "PreMode.noMSA.inference/",
                                                    "PreMode.noStructure/",
                                                    "PreMode.ptm/",
                                                    "PreMode.onehot.inference/",
                                                    "Itan.1",
                                                    "BioChem (Random Forest)"
                                                    ),]
model.dic <- c("PreMode.inference/"="1: PreMode",
               "PreMode.pass.inference/"="8: ESM + SLP",
               "PreMode.noPretrain/"="6: PreMode: no pretrain",
               "BioChem (Random Forest)"="7: Random Forest",
               "PreMode.ptm/"="5: PreMode: add ptm",
               "Itan.1"="9: LoGoFunc",
               "PreMode.noMSA.inference/"="2: PreMode: no MSA",
               "PreMode.onehot.inference/"="3: PreMode: no ESM",
               "PreMode.noStructure/"="4: PreMode: no Structure")
result.plot$model <- model.dic[result.plot$model]

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

uniq.model.result.plot$model.type <- 'PreMode: Ablation'
uniq.model.result.plot$model.type[uniq.model.result.plot$model == "5: PreMode: add ptm"] <- 'PreMode: add ptm'
uniq.model.result.plot$model.type[uniq.model.result.plot$model == "8: ESM + SLP"] <- 'Baselines'
uniq.model.result.plot$model.type[uniq.model.result.plot$model == "7: Random Forest"] <- 'Baselines'
uniq.model.result.plot$model.type[uniq.model.result.plot$model == "9: LoGoFunc"] <- 'Baselines'
uniq.model.result.plot$model.type[uniq.model.result.plot$model == "1: PreMode"] <- 'PreMode'
uniq.model.result.plot$model.type <- factor(uniq.model.result.plot$model.type, 
                                            levels = c('PreMode', 'PreMode: add ptm', 'PreMode: Ablation', 'Baselines'))
# chose other color scale
p <- ggplot(uniq.model.result.plot, aes(x=model, y=auc, col=model.type)) +
  geom_point() + scale_color_manual(values = c("#F8766D", "#CD9600", "#999999", "#619CFF")) + 
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
ggsave(out.dir.2, p, height=6, width=6)

