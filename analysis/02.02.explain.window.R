source('/share/vault/Users/gz2294/Pipeline/uniprot.table.add.aKNN.medianotation.R')
ALL <- read.csv('figs/ALL.csv', row.names = 1, na.strings = c("NA", "."))
ALL <- uniprot.table.add.aKNN.medianotation.parallel(ALL, 'KKNN.median')
ALL$score[ALL$score==0] <- -1
library(ggplot2)
# get distribution of scn2a and fgfr2
# good.genes <- c('P15056', 'O00555', "P07949", "Q14524", "P04637", "Q09428", "Q14654")
good.genes <- c('Q99250', 'Q09428')
bad.genes <- c('P04637', 'Q14524', 'P07949', 'Q14654', 'O00555', 'P15056')
ALL$plot.label <- ALL$uniprotID
ALL$plot.label[!ALL$uniprotID %in% c(good.genes, bad.genes)] <- NA
ALL$plot.label[ALL$uniprotID %in% c(bad.genes)] <- 'Bad Genes'
# ALL$plot.label[ALL$uniprotID %in% c(med.genes)] <- 'Med Genes'
ALL$plot.label[ALL$uniprotID %in% c(good.genes)] <- 'Good Genes'
ALL$score.label <- ALL$score
ALL$score.label[ALL$score == -1] <- 'LoF'
ALL$score.label[ALL$score == 1] <- 'GoF'
wil.stat <- wilcox.test(ALL$KNN.median[!is.na(ALL$plot.label) & ALL$plot.label=='Good Genes' & ALL$score==-1],
                        ALL$KNN.median[!is.na(ALL$plot.label) & ALL$plot.label=='Good Genes' & ALL$score==1])
p1 <- ggplot(ALL[!is.na(ALL$plot.label) & ALL$plot.label=='Good Genes',],
             aes(x=KNN.median, col=score.label)) + geom_density() + 
  # ggpp::geom_text_npc(data=data.frame(x="right", y="middle",
  #                           label=paste0("MaKNN.median-Whitney test p=", signif(wil.stat$p.value, digits = 2))),
  #           aes(npcx=x, npcy=y, label=label),
  #           col='black') + 
  theme_bw() + 
  scale_x_continuous(trans = ggallin::pseudolog10_trans, breaks = c(50, 100, 200, 500, 1000)) +
  # xlim(0, 100) +
  ggtitle('Large window decrease prediction: KNN.median 1d distance') + ggeasy::easy_center_title()
wil.stat <- wilcox.test(ALL$KNN.median[!is.na(ALL$plot.label) & ALL$plot.label=='Bad Genes' & ALL$score==-1],
                        ALL$KNN.median[!is.na(ALL$plot.label) & ALL$plot.label=='Bad Genes' & ALL$score==1])
p5 <- ggplot(ALL[!is.na(ALL$plot.label) & ALL$plot.label=='Bad Genes',],
             aes(x=KNN.median, col=score.label)) + geom_density() + 
  # ggpp::geom_text_npc(data=data.frame(x="right", y="middle",
  #                                     label=paste0("MaKNN.median-Whitney test p=", signif(wil.stat$p.value, digits = 2))),
  #                     aes(npcx=x, npcy=y, label=label),
  #                     col='black') + 
  theme_bw() + 
  scale_x_continuous(trans = ggallin::pseudolog10_trans, breaks = c(50, 100, 200, 500, 1000)) +
  # xlim(0, 100) +
  ggtitle('Large window increase prediction: KNN.median 1d distance') + ggeasy::easy_center_title()

library(patchwork)
p <- (p1 + p5) + plot_layout(ncol = 2)
ggsave(filename = 'figs/02.02.explain.window.pdf', p, width = 12, height = 2)
