args <- commandArgs(trailingOnly = T)
out.dir.2 <- 'figs/02.02.ICC.models.compare.pdf'

result.plot <- readRDS('figs/02.02.ICC.folders.compare.RDS')
result.plot <- result.plot[result.plot$task.type %in% c("Gene"),]

result.plot <- result.plot[result.plot$model %in% c("PreMode.inference/",
                                                    # "PreMode.PRE.v4/",
                                                    # "PRE.v10/",
                                                    # "PRE.v10.noconfidence/",
                                                    # "PRE.v11/",
                                                    # "PRE.v9/",
                                                    # "PRE.v9.noconfidence/",
                                                    "PreMode.PRE.v8/",
                                                    "PreMode.noPretrain/",
                                                    "PreMode.pass.inference/",
                                                    "PreMode.noMSA.inference/",
                                                    "PreMode.noStructure/",
                                                    "PreMode.ptm/",
                                                    "PreMode.onehot.inference/",
                                                    "Itan.1",
                                                    "BioChem (Random Forest)"
                                                    ),]
model.dic <- c("PreMode.inference/"="01: PreMode (148k)",
               "PreMode.PRE.v4/"="02: PreMode (83M)",
               "PreMode.PRE.v8/"="02: PreMode (4.7M)",
               "PRE.v10/"="02: PreMode (v10)",
               "PRE.v10.noconfidence/"="02: PreMode (v10.noconfidence)",
               "PRE.v11/"="02: PreMode (v11)",
               "PRE.v9/"="02: PreMode (v9)",
               "PRE.v9.noconfidence/"="02: PreMode (v9.noconfidence)",
               "PreMode.pass.inference/"="09: ESM + SLP",
               "PreMode.noPretrain/"="07: PreMode: no pretrain",
               "BioChem (Random Forest)"="08: Random Forest",
               "PreMode.ptm/"="06: PreMode: add ptm",
               "Itan.1"="10: LoGoFunc",
               "PreMode.noMSA.inference/"="03: PreMode: no MSA",
               "PreMode.onehot.inference/"="04: PreMode: no ESM",
               "PreMode.noStructure/"="05: PreMode: no Structure")
result.plot$model <- model.dic[result.plot$model]

# plot the task weighted averages as well as task size weighted error bars
uniq.result.plot <- result.plot[result.plot$fold==0,]
for (i in 1:dim(uniq.result.plot)) {
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
  uniq.model.result.plot$auc[i] <- sum(uniq.result.plot$auc[uniq.result.plot$model==uniq.model.result.plot$model[i]] * task.sizes / sum(task.sizes), na.rm=T)
  uniq.model.result.plot$auc.sd[i] <- sum(uniq.result.plot$auc.sd[uniq.result.plot$model==uniq.model.result.plot$model[i]] * task.sizes / sum(task.sizes), na.rm=T)
}
p <- ggplot(uniq.model.result.plot, aes(x=model, y=auc, col=model)) +
  geom_point() +
  geom_errorbar(aes(ymin=auc-auc.sd, ymax=auc+auc.sd), width=.2) +
  coord_flip() + guides(col=guide_legend(ncol=2)) +
  labs(x = "task", y = "auc", fill = "model") +
  theme_bw() + ylim(0.5, 0.9) + ggtitle('Weighted Sum of AUC across genes') +
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        legend.position="bottom", 
        legend.direction="horizontal") + 
  ggeasy::easy_center_title()
ggsave(out.dir.2, p, height=6, width=6)

