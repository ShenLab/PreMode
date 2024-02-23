# args <- commandArgs(trailingOnly = T)
out.dir.1 <- 'figs/02.02.ICC.gene.pfams.compare.pdf'
out.dir.2 <- 'figs/02.02.ICC.models.compare.pdf'

result.plot <- readRDS('figs/02.02.ICC.folders.compare.with.ddG.RDS')
# result.plot <- result.plot[result.plot$model %in% c(),]
result.plot <- result.plot[result.plot$task.type %in% c("Gene.Domain", "Gene.Gene"),]
result.plot <- result.plot[result.plot$model %in% c("PreMode.inference/"),]
result.plot <- result.plot[!grepl('heyne', result.plot$task.name),]
model.dic <- c("PreMode.inference/"="1: PreMode",
               "PreMode.pass.inference/"="2: ESM + SLP",
               "PreMode.noPretrain/"="3: PreMode: no pretrain",
               "BioChem (Random Forest)"="4: Random Forest")


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

# get task specific df
result.df.gene <- result.plot[result.plot$fold==0 & endsWith(result.plot$task.name, '(Gene Only)'),]
for (i in 1:dim(result.df.gene)) {
  task.prefix <- gsub('\\(Gene Only\\)', '', result.df.gene$task.name[i])
  result.df.gene.only <- result.plot$auc[result.plot$task.name==paste0(task.prefix, '(Gene Only)')]
  result.df.gene$auc.gene[i] <- mean(result.df.gene.only, na.rm = T)
  result.df.gene$auc.sd.gene[i] <- sd(result.df.gene.only, na.rm = T)
  result.df.gene.pfam <- result.plot$auc[result.plot$task.name==paste0(task.prefix, '(Family)')]
  result.df.gene$auc.pfam[i] <- mean(result.df.gene.pfam, na.rm = T)
  result.df.gene$auc.sd.pfam[i] <- sd(result.df.gene.pfam, na.rm = T)
  result.df.gene$task.name[i] <- task.prefix
}

p <- ggplot(result.df.gene, aes(y=auc.pfam, x=auc.gene, col=task.name)) +
  geom_point() +
  geom_errorbar(aes(ymin=auc.pfam-auc.sd.pfam, ymax=auc.pfam+auc.sd.pfam)) + 
  geom_errorbarh(aes(xmin=auc.gene-auc.sd.gene, xmax=auc.gene+auc.sd.gene)) +
  labs(x = "auc (gene only)", y = "auc (pfam)", col = "HGNC") +
  geom_abline(intercept=0, slope=1, linetype='dashed') +
  theme_bw() + 
  guides(col=guide_legend(ncol=1)) +
  ylim(0.5, 1) + xlim(0.5, 1)
# ggsave(paste0(out.dir.1), p, height = 6, width = 6)