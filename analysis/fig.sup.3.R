genes <- c('PTEN', 'NUDT15', 'SNCA', 'CYP2C9', 'GCK', 'ASPA', 'CCR5', 'CXCR4')
stab.assay <- c(1, 1, 2, 2, 2, 1, 1, 1)
task.dic <- list("PTEN"=c("score.1"="stability", "score.2"="enzyme.activity"), 
                 "NUDT15"=c("score.1"="stability", "score.2"="enzyme.activity"), 
                 "VKORC1"=c("score.1"="enzyme.activity", "score.2"="stability"), 
                 "CCR5"=c("score.1"="stability", "score.2"="binding Ab2D7", "score.3"="binding HIV-1"), 
                 "CXCR4"=c("score.1"="stability", "score.2"="binding CXCL12", "score.3"="binding Ab12G5"),
                 "SNCA"=c("score.1"="enzyme.activity", "score.2"="stability"),
                 "CYP2C9"=c("score.1"="enzyme.activity", "score.2"="stability"),
                 "GCK"=c("score.1"="enzyme.activity", "score.2"="stability"),
                 "ASPA"=c("score.1"="stability", "score.2"="enzyme.activity")
)
result <- NULL
sp.stats <- NULL
pr.stats <- NULL
all.plots <- list()
k = 1
for (i in 1:length(genes)) {
  assay <- read.csv(paste0('../data.files/', genes[i], '/ALL.annotated.csv'))
  # test the correlation between stab and foldx_ddG
  stab.score.columns <- paste0('score.', stab.assay[i])
  stab.corr <- abs(cor.test(assay$FoldXddG, assay[,stab.score.columns])$estimate)
  other.score.columns <- colnames(assay)[startsWith(colnames(assay), 'score')]
  other.score.columns <- other.score.columns[!other.score.columns %in% stab.score.columns]
  other.corr <- NULL
  for (c in other.score.columns) {
    other.corr <- c(other.corr, abs(cor.test(assay$RosettaddG, assay[,c])$estimate))
  }
  other.corr <- mean(other.corr, na.rm = T)
  result <- rbind(result,
                  data.frame(HGNC=genes[i],
                             stab.corr=stab.corr,
                             other.corr=other.corr))
  if (genes[i] == 'ASPA') {
    assay[,other.score.columns] <- -assay[,other.score.columns]
    x.pos <- 'right'
    y.pos <- 'bottom'
  } else {
    x.pos <- 'left'
    y.pos <- 'top'
  }
  # plot scatter plot of stability and other assay
  for (c in other.score.columns) {
    sp.stats[k] <- cor.test(assay[,stab.score.columns],
                            assay[,c], method = 'spearman')$estimate
    pr.stats[k] <- cor.test(assay[,stab.score.columns],
                            assay[,c], method = 'pearson')$estimate
    p <- ggplot(assay, aes_string(x=stab.score.columns, y=c)) + 
      geom_point(alpha=0.2, color='grey') +
      geom_density_2d(color='gray1') +
      stat_smooth(method = "lm", formula = y~x, color='blue') +
      ggpp::geom_text_npc(data=data.frame(x=x.pos, y=y.pos,
                                          label=paste0("Pearson r=", signif(pr.stats[k], digits = 2),
                                                       "\nSpearman rho=", signif(sp.stats[k], digits = 2))),
                          aes(npcx=x, npcy=y, label=label),
                          col='black') +
      ggtitle(genes[i]) +
      xlab(task.dic[[genes[i]]][stab.score.columns]) +
      ylab(task.dic[[genes[i]]][c]) + 
      theme_bw() + ggeasy::easy_center_title()
    all.plots[[k]] <- p
    k <- k + 1
  }
}
# make plot
library(patchwork)
p <- (all.plots[[1]] + all.plots[[2]] + all.plots[[3]]) /
  (all.plots[[4]] + all.plots[[5]] + all.plots[[6]]) /
  (all.plots[[7]] + all.plots[[8]] + all.plots[[9]] + all.plots[[10]] + plot_layout(ncol = 4))
ggsave('figs/fig.sup.3.pdf', p, height = 10, width = 10)



