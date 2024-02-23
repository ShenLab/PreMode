# args <- commandArgs(trailingOnly = T)
out.dir.1 <- 'figs/02.02.ICC.tasks.compare.big.small.pdf'

result.plot <- readRDS('figs/02.02.ICC.folders.compare.RDS')
result.plot <- result.plot[result.plot$task.type %in% c("Gene"),]
result.plot <- result.plot[result.plot$model %in% c("PreMode.inference/", 
                                                    "PreMode.PRE.v8/"
                                                    # "PreMode.CHPs.v4.large.window/"
                                                    # "PreMode.CHPs.v4.esm_mask/"
                                                    # "PreMode.CHPs.v4.new.coevol/",
                                                    # "Itan.1"
                                                    ),]
model.dic <- c("PreMode.inference/"="1: PreMode (148k)",
               "PreMode.PRE.v8/"="2: PreMode (4.7M)"
               # "PreMode.CHPs.v4.esm_mask/"="3: ESM + LM Head",
               # "PreMode.CHPs.v4.large.window/"="2: PreMode (148k, large window)"
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

