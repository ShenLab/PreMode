# args <- commandArgs(trailingOnly = T)
out.dir.1 <- 'figs/fig.sup.10.pdf'

result.plot <- readRDS('figs/fig.5.prepare.RDS')
# result.plot <- result.plot[result.plot$model %in% c(),]
result.plot <- result.plot[result.plot$task.type %in% c("Gene.Domain", "Gene.Gene"),]
result.plot <- result.plot[result.plot$model %in% c("PreMode.inference/"),]
result.plot <- result.plot[!grepl('Heyne', result.plot$task.name),]
model.dic <- c("PreMode.inference/"="1: PreMode")

result.plot$HGNC <- NA
for (i in 1:dim(result.plot)[1]) {
  result.plot$HGNC[i] <- gsub('Gene: ', '', strsplit(result.plot$task.name[i], "\\.")[[1]][1])
}
result.plot$model <- model.dic[result.plot$model]

num.models <- length(unique(result.plot$HGNC))
p <- ggplot(result.plot, aes(y=auc, x=task.name, col=HGNC)) +
  geom_point(alpha=0) +
  stat_summary(data = result.plot,
               aes(x=as.numeric(factor(task.name)),
                   y = auc, col=HGNC), 
               fun.data = mean_se, geom = "errorbar", width = 0.2) +
  stat_summary(data = result.plot, 
               aes(x=as.numeric(factor(task.name)),
                   y = auc, col=HGNC), 
               fun.data = mean_se, geom = "point") +
  labs(x = "task", y = "auc", fill = "HGNC") +
  theme_bw() + 
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        legend.position="bottom", 
        legend.direction="horizontal") +
  coord_flip() + guides(col=guide_legend(ncol=3)) +
  ylim(0.5, 1) + xlab('task: Genetics Level Mode of Action') 
ggsave(paste0(out.dir.1), p, height = 6, width = 6)
