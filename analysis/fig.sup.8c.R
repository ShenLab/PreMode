ALL <- read.csv('figs/ALL.csv', row.names = 1, na.strings = c("NA", "."))
ALL$score[ALL$score==0] <- -1
library(ggplot2)
good.genes <- c("P15056", "P04637", "Q09428", "Q14654", "O00555",  "Q99250", "Q14524", 'P07949')
bad.genes <- c('P21802')
ALL$plot.label <- ALL$uniprotID
ALL$plot.label[!ALL$uniprotID %in% c(good.genes, bad.genes)] <- NA
ALL$plot.label[ALL$uniprotID %in% c(bad.genes)] <- 'Bad Genes'
# ALL$plot.label[ALL$uniprotID %in% c(med.genes)] <- 'Med Genes'
ALL$plot.label[ALL$uniprotID %in% c(good.genes)] <- 'Good Genes'
ALL$score.label <- ALL$score
ALL$score.label[ALL$score == -1] <- 'LoF'
ALL$score.label[ALL$score == 1] <- 'GoF'
wil.stat <- wilcox.test(ALL$pLDDT[!is.na(ALL$plot.label) & ALL$plot.label=='Good Genes' & ALL$score==-1],
                        ALL$pLDDT[!is.na(ALL$plot.label) & ALL$plot.label=='Good Genes' & ALL$score==1])
p1 <- ggplot(ALL[!is.na(ALL$plot.label) & ALL$plot.label=='Good Genes',],
             aes(x=pLDDT, col=score.label)) + geom_density() + 
  ggpp::geom_text_npc(data=data.frame(x="left", y="middle",
                            label=paste0("Mann-Whitney test p=", signif(wil.stat$p.value, digits = 2))),
            aes(npcx=x, npcy=y, label=label),
            col='black') + 
  theme_bw() + xlim(0, 100) +
  ggtitle('Structure increase prediction: site pLDDT') + ggeasy::easy_center_title()
wil.stat <- wilcox.test(ALL$pLDDT[!is.na(ALL$plot.label) & ALL$plot.label=='Bad Genes' & ALL$score==-1],
                        ALL$pLDDT[!is.na(ALL$plot.label) & ALL$plot.label=='Bad Genes' & ALL$score==1])
p5 <- ggplot(ALL[!is.na(ALL$plot.label) & ALL$plot.label=='Bad Genes',],
             aes(x=pLDDT, col=score.label)) + geom_density() + 
  ggpp::geom_text_npc(data=data.frame(x="left", y="middle",
                                      label=paste0("Mann-Whitney test p=", signif(wil.stat$p.value, digits = 2))),
                      aes(npcx=x, npcy=y, label=label),
                      col='black') + 
  theme_bw() + xlim(0, 100) +
  ggtitle('Structure worsen prediction: site pLDDT') + ggeasy::easy_center_title()

library(patchwork)
p <- (p1 + p5) + plot_layout(ncol = 2)
ggsave(filename = 'figs/fig.sup.8c.pdf', p, width = 12, height = 2)
