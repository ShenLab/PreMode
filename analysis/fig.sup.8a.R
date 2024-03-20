out.dir.1 <- 'figs/fig.sup.8a.pdf'

result.plot <- readRDS('figs/fig.5.prepare.RDS')
result.plot <- result.plot[result.plot$task.type %in% c("Gene"),]
result.plot <- result.plot[result.plot$model %in% c("PreMode.noItan", 
                                                    "PreMode.large.window.noItan", 
                                                    "Itan"),]
model.dic <- c("PreMode.noItan"="1: PreMode",
               "PreMode.large.window.noItan"="2: PreMode (large window)",
               "Itan"="3: LoGoFunc"
               )

result.plot$model <- model.dic[result.plot$model]

num.models <- length(unique(result.plot$model))
p <- ggplot(result.plot, aes(y=auc, x=task.name, col=model)) +
  geom_point(alpha=0) + 
  scale_color_manual(values = c("#F8766D", "#C99800", "#00B0F6")) + 
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
        text = element_text(size = 13),
        legend.position="bottom", 
        legend.direction="horizontal") +
  ggtitle('PreMode compared to LoGoFunc method') +
  ggeasy::easy_center_title() +
  coord_flip() + guides(col=guide_legend(ncol=2)) +
  ylim(0.5, 1) + xlab('task: Genetics Level Mode of Action') 
ggsave(paste0(out.dir.1), p, height = 6, width = 6)

