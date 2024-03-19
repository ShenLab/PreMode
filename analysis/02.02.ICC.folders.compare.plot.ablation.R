args <- commandArgs(trailingOnly = T)
out.dir.1 <- 'figs/02.02.ICC.tasks.compare.ablation.pdf'

result.plot <- readRDS('figs/02.02.ICC.folders.compare.RDS')
result.plot <- result.plot[result.plot$task.type %in% c("Gene"),]

result.plot <- result.plot[result.plot$model %in% c("PreMode.inference/",
                                                    # "PreMode.PRE.v8/",
                                                    # "PRE.v10/",
                                                    # "PRE.v10.noconfidence/",
                                                    # "PRE.v11/",
                                                    # "PRE.v9/",
                                                    # "PRE.v9.noconfidence/",
                                                    # "PreMode.CHPs.v4.af2.rep/",
                                                    "PreMode.noMSA.inference/",
                                                    "PreMode.noStructure/",
                                                    "PreMode.ptm/",
                                                    "PreMode.onehot.inference/"
                                                    ),]
model.dic <- c("PreMode.inference/"="1: PreMode",
               "PreMode.PRE.v8/"="2: PreMode (big.2)",
               "PRE.v10/"="02: PreMode (v10)",
               "PRE.v10.noconfidence/"="02: PreMode (v10.noconfidence)",
               "PRE.v11/"="02: PreMode (v11)",
               "PRE.v9/"="02: PreMode (v9)",
               "PRE.v9.noconfidence/"="02: PreMode (v9.noconfidence)",
               "PreMode.CHPs.v4.af2.rep/"="02: PreMode (af2.rep)",
               "PreMode.ptm/"="2: PreMode: add ptm",
               "PreMode.noMSA.inference/"="3: PreMode: no MSA",
               "PreMode.onehot.inference/"="4: PreMode: no ESM",
               "PreMode.noStructure/"="5: PreMode:\nno Structure")
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
  theme_bw() + ggtitle('PreMode ablation analysis') +
  ggeasy::easy_center_title() +
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        text = element_text(size = 13),
        legend.position="bottom", 
        legend.direction="horizontal") +
  coord_flip() + guides(col=guide_legend(ncol=2)) +
  ylim(0.5, 1) + xlab('task: Genetics Level Mode of Action') 
ggsave(paste0(out.dir.1), p, height = 6, width = 6)

