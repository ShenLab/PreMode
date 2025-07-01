library(ggplot2)
# moved annotated file here, previously we annotate from scratch.
ALL <- read.csv('figs/ALL.csv', row.names = 1, na.strings = c(".", "NA"))
# compare conservation with benign
benign <- read.csv('figs/benign.csv', row.names = 1, na.strings = c(".", "NA"))
benign <- benign[benign$uniprotID %in% ALL$uniprotID,]
# plot number of G/LoF across genes
gene.df <- data.frame(uniprotID=unique(ALL$uniprotID),
                      GoF=NA, LoF=NA)
for (i in 1:dim(gene.df)[1]) {
  gene.df$GoF[i] <- sum(ALL$score[ALL$uniprotID==gene.df$uniprotID[i]]==1)
  gene.df$LoF[i] <- sum(ALL$score[ALL$uniprotID==gene.df$uniprotID[i]]==-1)
}
gene.df$label <- NA
genes.dic <- c("Q09428"="ABCC8", "P15056"="BRAF", "O00555"="CACNA1A", "P21802"="FGFR2",
           "Q14654"="KCNJ11", "P07949"="RET", "Q99250"="SCN2A", "Q14524"="SCN5A", "P04637"="TP53")
gene.df$label[gene.df$uniprotID %in% names(genes.dic)] <- genes.dic[gene.df$uniprotID[gene.df$uniprotID %in% names(genes.dic)]] 
gene.df$transfer.learning <- NA
gene.df$transfer.learning[!is.na(gene.df$label)] <- '>=15' 
ggplot(gene.df, aes(x=GoF, y=LoF, col=transfer.learning, label=label)) + 
  geom_point() + ggrepel::geom_text_repel(show.legend = FALSE) + theme_bw() + 
  scale_x_continuous(trans = ggallin::pseudolog10_trans, breaks = c(5, 10, 20, 30, 40, 50, 75, 100)) +
  scale_y_continuous(trans = ggallin::pseudolog10_trans, breaks = c(5, 10, 20, 40, 60, 80, 100, 200, 400)) +
  theme(legend.position = "none") + ggtitle('Number of G/LoF variants per gene') + ggeasy::easy_center_title()
ggsave('figs/fig.2c.pdf', height = 3.5, width = 3.5)


p <- list()
ion.genes <- unique(ALL$uniprotID[grepl("Heyne", ALL$data_source)])
for (j in c(0, 1, 2)) {
  if (j==0) {
    sse <- table(ALL$secondary_struc[!ALL$uniprotID %in% ion.genes], ALL$LABEL[!ALL$uniprotID %in% ion.genes])
  } else if (j==1) {
    sse <- table(ALL$secondary_struc[ALL$uniprotID %in% ion.genes], ALL$LABEL[ALL$uniprotID %in% ion.genes])
  } else {
    sse <- table(ALL$secondary_struc, ALL$LABEL)
  }
  sse.df <- matrix(NA, nrow = dim(sse)[1], ncol = dim(sse)[2])
  colnames(sse.df) <- colnames(sse)
  rownames(sse.df) <- rownames(sse)
  for (i in 1:dim(sse)[2]) {
    sse.df[,i] <- sse[,i]
  }
  sse.df <- as.data.frame(sse.df)
  for (i in 1:dim(sse.df)[1]) {
    res <- binom.test(sse.df[i,1], sse.df[i,1]+sse.df[i,2], p=sum(sse.df[,1])/sum(sse.df[,1]+sse.df[,2]))
    sse.df$p.value[i] <- res$p.value
  }
  sse.df$q.value <- p.adjust(sse.df$p.value, method = "fdr")
  code.dict <- c("H"="Alpha helix (4-12)", "B"="Isolated beta-bridge residue", 
                 "E"="Beta Sheet", "G"="3-10 helix", "I"="Pi helix", "T"="Turn",
                 "S"="Bend", " "="none")
  sse.df$sec_struc <- code.dict[rownames(sse.df)]
  to.plot <- rbind(sse.df, sse.df)
  to.plot$n_mutation <- c(sse.df$GOF, sse.df$LOF)
  to.plot$frac_mutation <- c(sse.df$GOF/sum(sse.df$GOF), sse.df$LOF/sum(sse.df$LOF))
  to.plot$LABEL <- c(rep("GOF", dim(sse.df)[1]), rep("LOF", dim(sse.df)[1]))
  to.plot$sec_struc <- gsub(" ", "\n", to.plot$sec_struc)
  
  anno <- to.plot
  anno$sec_struc[anno$q.value > 0.05] <- NA
  anno$frac_mutation[anno$q.value > 0.05] <- NA
  anno <- anno[!is.na(anno$sec_struc),]
  anno$x <- as.numeric(as.factor(to.plot$sec_struc))[match(anno$sec_struc, to.plot$sec_struc)] - 0.2
  anno$xend <- as.numeric(as.factor(to.plot$sec_struc))[match(anno$sec_struc, to.plot$sec_struc)] + 0.2
  anno$y <- anno$frac_mutation + 0.015
  anno <- anno[order(anno$x),]
  to.keep <- c()
  for (i in 1:(dim(anno)[1]/2)) {
    to.keep <- c(to.keep, c(i*2-1, i*2)[which.max(anno$y[c(i*2-1, i*2)])])
  }
  anno <- anno[to.keep,]
  anno$annotation <- NA
  for (k in 1:dim(anno)[1]) {
    anno$annotation[k] <- paste(c(rep(" ", k-1), paste0("*", gsub("e-0", "e-", sprintf("%.1e", anno$p.value[k]))), rep(" ", k-1)), collapse = "")
  }
  library(ggplot2)
  library(ggsignif)
  p1 <- ggplot(to.plot, aes(x=sec_struc, y=frac_mutation, fill=LABEL)) +
    geom_bar(stat='identity', position=position_dodge()) + 
    geom_signif(stat="identity",
                data=anno,
                aes(x=x,
                    xend=xend,
                    y=y, yend=y,
                    annotation=annotation)) + ylim(0, 0.8) +
    xlab('secondary structures') +
    # scale_x_discrete(guide = guide_axis(n.dodge=2)) +
    theme_bw()
  if (j==0) {
    p1 <- p1 + ggtitle('Non-Na+/Ca2+ Ion Channel Genes') + ggeasy::easy_center_title()
    # ggsave('02.01.sse.pdf', p1, height = 3, width = 6)
  } else {
    p1 <- p1 + ggtitle('Na+/Ca2+ Ion Channel Genes') + theme(axis.text.x = element_blank(), axis.title.x = element_blank()) +
      ggeasy::easy_center_title()
    # ggsave('02.01.sse.Heyne.pdf', p1, height = 3, width = 6)
  }
  p[[j+1]] <- p1
}
library(patchwork)
p1 <- p[[2]]+p[[1]]+plot_layout(ncol = 1)

wil.stat <- wilcox.test(ALL$rsa[ALL$LABEL=="GOF"], ALL$rsa[ALL$LABEL=="LOF"])
rsa.table <- table(ALL$LABEL, ALL$rsa < 0.25)
chi.stat <- chisq.test(rsa.table)
to.plot <- data.frame(rsa=rep(c('>=25%', '<25%'), 2),
                      type=rep(c('GOF', 'LOF'), each=2),
                      frac=c(rsa.table[1,]/sum(rsa.table[1,]), rsa.table[2,]/sum(rsa.table[2,])))
p2.1 <- ggplot(to.plot, aes(x=rsa, y=frac, fill=type)) + 
  geom_bar(stat='identity', position=position_dodge()) + theme_bw() + 
  ggpp::geom_text_npc(
    data=data.frame(x="middle", y="top",
                    label=paste0("Chi-square test G/LoF p=", signif(chi.stat$p.value, digits = 2))),
    aes(npcx=x, npcy=y, label=label),
    col='black')
p2.2 <- ggplot(rbind(ALL[,c("rsa", "LABEL")], benign[,c("rsa", "LABEL")]), aes(x=rsa, col=LABEL)) + geom_density() +
  theme_bw() + ggpp::geom_text_npc(
    data=data.frame(x="middle", y="top",
                    label=paste0("Mann-Whitney test G/LoF p=", signif(wil.stat$p.value, digits = 2))),
    aes(npcx=x, npcy=y, label=label),
    col='black')
p2 <- p2.1 / p2.2
# ggsave('02.01.rsa.pdf', p, height = 4, width = 6)
wil.stat <- wilcox.test(ALL$pLDDT[ALL$LABEL=="GOF"], ALL$pLDDT[ALL$LABEL=="LOF"])
# binarize pLDDT to 0-50, 50-70, 70-90, 90-100
plddt.table <- matrix(NA, nrow = 2, ncol = 4)
plddt.table[1, 1] <- sum(!is.na(ALL$pLDDT) & ALL$pLDDT<50 & ALL$LABEL=="GOF")
plddt.table[2, 1] <- sum(!is.na(ALL$pLDDT) & ALL$pLDDT<50 & ALL$LABEL=="LOF")
plddt.table[1, 2] <- sum(!is.na(ALL$pLDDT) & ALL$pLDDT>=50 & ALL$pLDDT<70 & ALL$LABEL=="GOF")
plddt.table[2, 2] <- sum(!is.na(ALL$pLDDT) & ALL$pLDDT>=50 & ALL$pLDDT<70 & ALL$LABEL=="LOF")
plddt.table[1, 3] <- sum(!is.na(ALL$pLDDT) & ALL$pLDDT>=70 & ALL$pLDDT<90 & ALL$LABEL=="GOF")
plddt.table[2, 3] <- sum(!is.na(ALL$pLDDT) & ALL$pLDDT>=70 & ALL$pLDDT<90 & ALL$LABEL=="LOF")
plddt.table[1, 4] <- sum(!is.na(ALL$pLDDT) & ALL$pLDDT>=90 & ALL$LABEL=="GOF")
plddt.table[2, 4] <- sum(!is.na(ALL$pLDDT) & ALL$pLDDT>=90 & ALL$LABEL=="LOF")
chi.stat <- chisq.test(plddt.table)
to.plot <- data.frame(pLDDT=rep(c('0-50', '50-70', '70-90', '90-100'), 2),
                      type=rep(c('GOF', 'LOF'), each=4),
                      frac=c(plddt.table[1,]/sum(plddt.table[1,]), plddt.table[2,]/sum(plddt.table[2,])))
p3.1 <- ggplot(to.plot, aes(x=pLDDT, y=frac, fill=type)) + 
  geom_bar(stat='identity', position=position_dodge()) + theme_bw() + 
  ggpp::geom_text_npc(
    data=data.frame(x="middle", y="top",
                    label=paste0("Chi-square test G/LoF p=", signif(chi.stat$p.value, digits = 2))),
    aes(npcx=x, npcy=y, label=label),
    col='black')
p3.2 <- ggplot(rbind(ALL[,c("pLDDT", "LABEL")], benign[,c("pLDDT", "LABEL")]), aes(x=pLDDT, col=LABEL)) + geom_density() + 
  theme_bw() + ggpp::geom_text_npc(data=data.frame(x="middle", y="top",
                                                   label=paste0("Mann-Whitney test G/LoF p=", signif(wil.stat$p.value, digits = 2))),
                                   aes(npcx=x, npcy=y, label=label),
                                   col='black')
p3 <- p3.1 / p3.2
wil.stat <- wilcox.test(ALL$FoldXddG[ALL$LABEL=="GOF"], ALL$FoldXddG[ALL$LABEL=="LOF"])
p4 <- ggplot(rbind(ALL[,c("FoldXddG", "LABEL")], 
                   benign[,c("FoldXddG", "LABEL")]), 
             aes(x=FoldXddG, col=LABEL)) + geom_density() + 
  theme_bw() + ggpp::geom_text_npc(data=data.frame(x="middle", y="top",
                                                   label=paste0("Mann-Whitney test G/LoF p=", signif(wil.stat$p.value, digits = 2))),
                                   aes(npcx=x, npcy=y, label=label),
                                   col='black') +
  scale_x_continuous(trans = ggallin::pseudolog10_trans)

wil.stat <- wilcox.test(ALL$conservation.entropy[ALL$LABEL=="GOF"], ALL$conservation.entropy[ALL$LABEL=="LOF"])
t<-chisq.test(table(ALL$conservation.entropy > 0.5, ALL$LABEL))

p5 <- ggplot(rbind(ALL[,c('conservation.entropy', 'LABEL')], benign[,c('conservation.entropy', 'LABEL')]), 
             aes(x=conservation.entropy, col=LABEL)) + geom_density() + 
  theme_bw()  

p <- (p3.2 + p2.2) / (p4 + p5)
p2 <- (p3.1 / p2.1) 
ggsave(plot = p, filename = "figs/fig.2a.pdf", height=5, width=12)
ggsave(plot = p2, filename = "figs/fig.sup.1.pdf", height=4, width=6)
ggsave(plot=p1, filename = "figs/fig.2b.pdf", height = 5, width = 6)





