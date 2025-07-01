result.plot <- readRDS('figs/fig.5.prepare.RDS')
result.plot <- result.plot[result.plot$task.type %in% c("Gene.Domain", "Gene.Gene"),]
result.plot <- result.plot[result.plot$model == 'PreMode/' | result.plot$model == 'PreMode/.lw',]
result.plot$use.lw <- F
pick.cond <- 'auc'
# get unique models
uniq.models <- unique(gsub('.lw', '', result.plot$model))
# only keep the original models
uniq.models <- uniq.models[grepl('/$', uniq.models)]
# get unique genes, remove Q14524
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
        cond <- !is.na(lw.loss) & !is.na(lw.tr.auc) & (lw.tr.auc/lw.loss > tr.auc/loss)
      } else if (pick.cond == 'lw') {
        cond <- T
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
result.plot <- result.plot[result.plot$model %in% c("PreMode/"),]
model.dic <- c("PreMode/"="1: PreMode")

result.plot$HGNC <- NA
for (i in 1:dim(result.plot)[1]) {
  result.plot$HGNC[i] <- gsub('Gene: ', '', strsplit(result.plot$task.name[i], "\\.")[[1]][1])
}
result.plot$model <- model.dic[result.plot$model]
# rename result plot model
result.plot$model <- "1: PreMode\n(Gene Only)"
result.plot$model[grepl('\\(Family\\)$', result.plot$task.name)] <- "2: PreMode\n(Protein Family)"
result.plot$model[grepl('\\(Family\\).noself$', result.plot$task.name)] <- "3: PreMode\n(Protein Family,\nexclude gene)"
# rename result task name
result.plot$task.name <- gsub('Gene: ', '', result.plot$task.name)
result.plot$task.name <- gsub('\\.\\(Gene Only\\)', '', result.plot$task.name)
result.plot$task.name <- gsub('\\.\\(Family\\)$', '', result.plot$task.name)
result.plot$task.name <- gsub('\\.\\(Family\\).noself$', '', result.plot$task.name)
result.plot$task.name <- gsub('\\.', ': ', result.plot$task.name)
rename.dict <- c('KCNJ11: Potassium Channel Inwardly Rectifying Kir Cytoplasmic Domain'='KCNJ11: Potassium Channel Inwardly\nRectifying Kir Cytoplasmic Domain',
                 'FGFR2: Fibroblast Growth Factor Receptor Family'='FGFR2: Fibroblast Growth Factor\nReceptor Family',
                 'RET: Tyrosine Protein Kinase Catalytic Domain'='RET: Tyrosine Protein Kinase\nCatalytic Domain')
result.plot$task.name[result.plot$task.name %in% names(rename.dict)] <- rename.dict[result.plot$task.name[result.plot$task.name %in% names(rename.dict)]]

result.plot <- result.plot[result.plot$task.id != "Q14654.IPR013518.IPR013518.noself",]
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
  labs(x = "task", y = "AUC") +
  ggtitle('PreMode trained on Gene/Protein Family data') +
  theme_bw() + 
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        text = element_text(size = 16),
        plot.title = element_text(size=15),
        legend.title = element_blank(),
        legend.text = element_text(size=10),
        legend.position="bottom", 
        legend.direction="horizontal") +
  coord_flip() + guides(col=guide_legend(ncol=3)) + 
  ggeasy::easy_center_title() +
  ylim(0.2, 1) + xlab('task: Genetics Level Mode of Action') 
ggsave(paste0('figs/fig.5f.revision.pdf'), p,  width = 8, height = 5)
