source('/share/pascal/Users/gz2294/Pipeline/uniprot.table.add.annotation.R')
ALL <- read.csv('figs/ALL.csv', row.names = 1, na.strings = c("NA", "."))
ALL$score[ALL$score==0] <- -1
library(ggplot2)
# get distribution of scn2a and fgfr2
good.genes <- c("P15056", "P04637", "Q09428", "Q14654", "O00555")
bad.genes <- c('Q99250', 'P21802', "P07949", "Q14524")
ALL$plot.label <- ALL$uniprotID
ALL$plot.label[!ALL$uniprotID %in% c(good.genes, bad.genes)] <- NA
ALL$plot.label[ALL$uniprotID %in% c(bad.genes)] <- 'Bad Genes'
# ALL$plot.label[ALL$uniprotID %in% c(med.genes)] <- 'Med Genes'
ALL$plot.label[ALL$uniprotID %in% c(good.genes)] <- 'Good Genes'
ALL$score.label <- ALL$score
ALL$score.label[ALL$score == -1] <- 'LoF'
ALL$score.label[ALL$score == 1] <- 'GoF'
# p1 <- ggplot(ALL[!is.na(ALL$plot.label),], aes(x=pLDDT, col=plot.label)) + geom_density() + theme_bw() + ggtitle('All variants: site pLDDT') + ggeasy::easy_center_title()
# p2 <- ggplot(ALL[!is.na(ALL$plot.label),], aes(x=pLDDT.region, col=plot.label)) + geom_density() + theme_bw() + ggtitle('All variants: region pLDDT') + ggeasy::easy_center_title()
# p3 <- ggplot(ALL[!is.na(ALL$plot.label) & ALL$score==1,], aes(x=pLDDT, col=plot.label)) + geom_density() + theme_bw() + ggtitle('GoF variants: site pLDDT') + ggeasy::easy_center_title()
# p4 <- ggplot(ALL[!is.na(ALL$plot.label) & ALL$score==-1,], aes(x=pLDDT, col=plot.label)) + geom_density() + theme_bw() + ggtitle('LoF variants: site pLDDT') + ggeasy::easy_center_title()
# p5 <- ggplot(ALL[!is.na(ALL$plot.label) & ALL$score==1,], aes(x=pLDDT.region, col=plot.label)) + geom_density() + theme_bw() + ggtitle('GoF variants: region pLDDT') + ggeasy::easy_center_title()
# p6 <- ggplot(ALL[!is.na(ALL$plot.label) & ALL$score==-1,], aes(x=pLDDT.region, col=plot.label)) + geom_density() + theme_bw() + ggtitle('LoF variants: region pLDDT') + ggeasy::easy_center_title()
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
ggsave(filename = 'figs/02.02.explain.Structure.pdf', p, width = 12, height = 2)
