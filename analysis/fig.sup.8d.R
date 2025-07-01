ALL <- read.csv('figs/ALL.csv', row.names = 1, na.strings = c("NA", "."))
ALL$score[ALL$score==0] <- -1
# remove glazer
ALL <- ALL[ALL$data_source != "glazer",]
library(ggplot2)
good.genes <- c('P21802', "Q14524", "P04637")
bad.genes <- c("Q99250", "Q14654", 'P07949', "Q09428", 'P15056', 'O00555')
ALL$plot.label <- ALL$uniprotID
ALL$plot.label[!ALL$uniprotID %in% c(good.genes, bad.genes)] <- NA
ALL$plot.label[ALL$uniprotID %in% c(bad.genes)] <- 'Bad Genes'
ALL$plot.label[ALL$uniprotID %in% c(good.genes)] <- 'Good Genes'
ALL$score.label <- ALL$score
ALL$score.label[ALL$score == -1] <- 'LoF'
ALL$score.label[ALL$score == 1] <- 'GoF'
wil.stat <- wilcox.test(ALL$PTM_dist_3d[!is.na(ALL$plot.label) & ALL$plot.label=='Good Genes' & ALL$score==-1],
                        ALL$PTM_dist_3d[!is.na(ALL$plot.label) & ALL$plot.label=='Good Genes' & ALL$score==1])
p1 <- ggplot(ALL[!is.na(ALL$plot.label) & ALL$plot.label=='Good Genes',],
             aes(x=PTM_dist_3d, col=score.label)) + geom_density() + 
  ggpp::geom_text_npc(data=data.frame(x="right", y="middle",
                            label=paste0("Mann-Whitney test p=", signif(wil.stat$p.value, digits = 2))),
            aes(npcx=x, npcy=y, label=label),
            col='black') + 
  theme_bw() + xlim(0, 100) +
  ggtitle('PTM increase prediction: PTM distance') + ggeasy::easy_center_title()
wil.stat <- wilcox.test(ALL$PTM_dist_3d[!is.na(ALL$plot.label) & ALL$plot.label=='Bad Genes' & ALL$score==-1],
                        ALL$PTM_dist_3d[!is.na(ALL$plot.label) & ALL$plot.label=='Bad Genes' & ALL$score==1])
p5 <- ggplot(ALL[!is.na(ALL$plot.label) & ALL$plot.label=='Bad Genes',],
             aes(x=PTM_dist_3d, col=score.label)) + geom_density() + 
  ggpp::geom_text_npc(data=data.frame(x="right", y="middle",
                                      label=paste0("Mann-Whitney test p=", signif(wil.stat$p.value, digits = 2))),
                      aes(npcx=x, npcy=y, label=label),
                      col='black') + 
  theme_bw() + xlim(0, 100) +
  ggtitle('PTM worsen prediction: PTM distance') + ggeasy::easy_center_title()

library(patchwork)
p <- (p1 + p5) + plot_layout(ncol = 2)
ggsave(filename = 'figs/fig.sup.8d.pdf', p, width = 12, height = 2)
