library(ggplot2)
ICC <- read.csv('~/Data/DMS/Itan.CKB.Cancer/ALL.csv', row.names = 1)
# remove unwanted variants
to.del <- readRDS("~/Data/DMS/Itan.CKB.Cancer/to.del.conflict.with.chps.RDS")
ICC$unique.id <- paste(ICC$uniprotID, ICC$ref, ICC$pos.orig, ICC$alt, sep = ":")
ICC <- ICC[!ICC$unique.id %in% to.del,]
ion.channel <- read.csv('~/Data/DMS/Ion_Channel/all.af2update.csv', row.names = 1)
ion.channel$unique.id <- paste(ion.channel$uniprotID, ion.channel$ref, ion.channel$pos.orig, ion.channel$alt, sep = ":")
ICC <- ICC[!ICC$unique.id %in% ion.channel$unique.id,]
ALL <- dplyr::bind_rows(ICC, ion.channel)

# add benign
source('/share/vault/Users/gz2294/Pipeline/uniprot.table.add.annotation.R')
# Ion_channel <- read.csv('~/Data/DMS/')
# check the sse and rsa
ALL <- read.csv('figs/ALL.csv', row.names = 1, na.strings = c(".", "NA"))
# ALL <- uniprot.table.add.annotation.parallel(ALL, 'dssp')
# ALL <- uniprot.table.add.annotation.parallel(ALL, 'dbnsfp')
ALL <- uniprot.table.add.annotation.parallel(ALL, 'Itan')
# ALL <- uniprot.table.add.annotation.parallel(ALL, 'EVE')
# ALL <- uniprot.table.add.annotation.parallel(ALL, 'AlphaMissense')
# ALL <- uniprot.table.add.annotation.parallel(ALL, 'gMVP')
# ALL <- uniprot.table.add.annotation.parallel(ALL, 'RosettaddG')
ALL <- uniprot.table.add.annotation.parallel(ALL, 'FoldXddG')
ALL <- uniprot.table.add.annotation.parallel(ALL, 'pLDDT')
ALL <- uniprot.table.add.annotation.parallel(ALL, 'pLDDT.region')
ALL <- uniprot.table.add.annotation.parallel(ALL, 'pLDDT.all')
ALL <- uniprot.table.add.annotation.parallel(ALL, 'conservation')
ALL <- uniprot.table.add.annotation.parallel(ALL, 'conservation.region')
ALL <- uniprot.table.add.annotation.parallel(ALL, 'conservation.coevol')
ALL <- uniprot.table.add.annotation.parallel(ALL, "pfam")
# compare conservation with benign
benign <- read.csv('figs/benign.csv', row.names = 1, na.strings = c(".", "NA"))
benign <- benign[benign$uniprotID %in% ALL$uniprotID,]
# benign <- rbind(read.csv('~/Data/DMS/ClinVar.HGMD.PrimateAI.syn/training.csv'),
#                 read.csv('~/Data/DMS/ClinVar.HGMD.PrimateAI.syn/testing.csv'))
# benign <- benign[benign$score==0,]
# benign$LABEL <- 'Benign'
# benign <- uniprot.table.add.annotation.parallel(benign, 'conservation')
# benign <- uniprot.table.add.annotation.parallel(benign, 'dssp')
# benign <- uniprot.table.add.annotation.parallel(benign, 'pLDDT')
# benign <- uniprot.table.add.annotation.parallel(benign, 'FoldXddG', njobs = 108)
# benign <- uniprot.table.add.annotation.parallel(benign, 'RosettaddG', njobs = 108)
# ALL$LABEL[ALL$score==1] <- "GOF"
# ALL$LABEL[ALL$score==0] <- "LOF"
# write.csv(ALL, 'figs/ALL.csv')
# write.csv(benign, 'figs/benign.csv')
# plot number of G/LoF across genes
gene.df <- data.frame(uniprotID=unique(ALL$uniprotID),
                      GoF=NA, LoF=NA)
for (i in 1:dim(gene.df)[1]) {
  gene.df$GoF[i] <- sum(ALL$score[ALL$uniprotID==gene.df$uniprotID[i]]==1)
  gene.df$LoF[i] <- sum(ALL$score[ALL$uniprotID==gene.df$uniprotID[i]]==0)
}
gene.df$label <- NA
genes.dic <- c("Q09428"="ABCC8", "P15056"="BRAF", "O00555"="CACNA1A", "P21802"="FGFR2",
           "Q14654"="KCNJ11", "P07949"="RET", "Q99250"="SCN2A", "Q14524"="SCN5A", "P04637"="TP53")
gene.df$label[gene.df$uniprotID %in% names(genes.dic)] <- genes.dic[gene.df$uniprotID[gene.df$uniprotID %in% names(genes.dic)]] 
gene.df$transfer.learning <- NA
gene.df$transfer.learning[!is.na(gene.df$label)] <- 'Selected' 
ggplot(gene.df, aes(x=GoF, y=LoF, col=transfer.learning, label=label)) + 
  geom_point() + ggrepel::geom_text_repel() + theme_bw() + 
  scale_x_continuous(trans = ggallin::pseudolog10_trans, breaks = c(5, 10, 20, 30, 40, 50, 75, 100)) +
  scale_y_continuous(trans = ggallin::pseudolog10_trans, breaks = c(5, 10, 20, 40, 60, 80, 100, 200, 400))
ggsave('figs/02.01.GoF.LoF.gene.compare.pdf', height = 3.5, width = 5)


p <- list()
for (j in c(0, 1, 2)) {
  if (j==0) {
    sse <- table(ALL$secondary_struc[ALL$data_source!="Heyne"], ALL$LABEL[ALL$data_source!="Heyne"])
  } else if (j==1) {
    sse <- table(ALL$secondary_struc[ALL$data_source=="Heyne"], ALL$LABEL[ALL$data_source=="Heyne"])
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
  to.plot$label <- c(rep("GOF", dim(sse.df)[1]), rep("LOF", dim(sse.df)[1]))
  to.plot$sec_struc <- gsub(" ", "\n", to.plot$sec_struc)
  
  anno <- to.plot
  anno$sec_struc[anno$q.value > 0.05] <- NA
  anno$frac_mutation[anno$q.value > 0.05] <- NA
  anno <- anno[!is.na(anno$sec_struc),]
  anno$x <- as.numeric(as.factor(to.plot$sec_struc))[match(anno$sec_struc, to.plot$sec_struc)] - 0.2
  anno$xend <- as.numeric(as.factor(to.plot$sec_struc))[match(anno$sec_struc, to.plot$sec_struc)] + 0.2
  anno$y <- anno$frac_mutation + 0.025
  anno <- anno[order(anno$x),]
  to.keep <- c()
  for (i in 1:(dim(anno)[1]/2)) {
    to.keep <- c(to.keep, c(i*2-1, i*2)[which.max(anno$y[c(i*2-1, i*2)])])
  }
  anno <- anno[to.keep,]
  anno$annotation <- NA
  for (k in 1:dim(anno)[1]) {
    anno$annotation[k] <- paste(c(rep(" ", k-1), "*", rep(" ", k-1)), collapse = "")
  }
  library(ggplot2)
  library(ggsignif)
  p1 <- ggplot(to.plot, aes(x=sec_struc, y=frac_mutation, fill=label)) +
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
    p1 <- p1 + ggtitle('Other Genes') + ggeasy::easy_center_title()
    # ggsave('02.01.sse.pdf', p1, height = 3, width = 6)
  } else {
    p1 <- p1 + ggtitle('Na+/Ca2+ Channel Genes') + ggeasy::easy_center_title()
    # ggsave('02.01.sse.Heyne.pdf', p1, height = 3, width = 6)
  }
  p[[j+1]] <- p1
}
library(patchwork)
p1 <- p[[2]]+p[[1]]+plot_layout(ncol = 1)

wil.stat <- wilcox.test(ALL$rsa[ALL$LABEL=="GOF"], ALL$rsa[ALL$LABEL=="LOF"])
p2 <- ggplot(rbind(ALL[,c("rsa", "LABEL")], benign[,c("rsa", "LABEL")]), aes(x=rsa, col=LABEL)) + geom_density() +
  theme_bw() + ggpp::geom_text_npc(data=data.frame(x="middle", y="top",
                                                   label=paste0("Mann-Whitney test G/LoF p=", signif(wil.stat$p.value, digits = 2))),
                                   aes(npcx=x, npcy=y, label=label),
                                   col='black')
# ggsave('02.01.rsa.pdf', p, height = 4, width = 6)
wil.stat <- wilcox.test(ALL$pLDDT[ALL$LABEL=="GOF"], ALL$pLDDT[ALL$LABEL=="LOF"])
p3 <- ggplot(rbind(ALL[,c("pLDDT", "LABEL")], benign[,c("pLDDT", "LABEL")]), aes(x=pLDDT, col=LABEL)) + geom_density() + 
  theme_bw() + ggpp::geom_text_npc(data=data.frame(x="middle", y="top",
                                                   label=paste0("Mann-Whitney test G/LoF p=", signif(wil.stat$p.value, digits = 2))),
                                   aes(npcx=x, npcy=y, label=label),
                                   col='black')

wil.stat <- wilcox.test(ALL$FoldXddG[ALL$LABEL=="GOF"], ALL$FoldXddG[ALL$LABEL=="LOF"])
p4 <- ggplot(rbind(ALL[,c("FoldXddG", "LABEL")], 
                   benign[,c("FoldXddG", "LABEL")]), 
             aes(x=FoldXddG, col=LABEL)) + geom_density() + 
  theme_bw() + ggpp::geom_text_npc(data=data.frame(x="right", y="top",
                                                   label=paste0("Mann-Whitney test G/LoF p=", signif(wil.stat$p.value, digits = 2))),
                                   aes(npcx=x, npcy=y, label=label),
                                   col='black') +
  scale_x_continuous(trans = ggallin::pseudolog10_trans)

wil.stat <- wilcox.test(ALL$conservation.entropy[ALL$LABEL=="GOF"], ALL$conservation.entropy[ALL$LABEL=="LOF"])
p5 <- ggplot(rbind(ALL[,c('conservation.entropy', 'LABEL')], benign[,c('conservation.entropy', 'LABEL')]), 
             aes(x=conservation.entropy, col=LABEL)) + geom_density() + 
  theme_bw() + ggpp::geom_text_npc(data=data.frame(x="middle", y="top",
                                                   label=paste0("Mann-Whitney test G/LoF p=", signif(wil.stat$p.value, digits = 2))),
                                   aes(npcx=x, npcy=y, label=label),
                                   col='black') 

p <- (p3 + p4) / (p2 + p5)
ggsave(plot = p, filename = "figs/02.01.GoF.LoF.compare.pdf", height=5, width=12)
ggsave(plot=p1, filename = "figs/02.01.sse.compare.pdf", height = 5, width = 6)


# check aa change
hydrophobic <- c("F", "W", "Y", "A", "I", "L", "M", "V")
polar <- c("R", "K", "H", "D", "E", "N", "Q", "S", "T")
charge <- c("+", "+", "+", "-", "-", "", "", "", "")
special <- c("C", "P", "G")
aaChg.LOF <- as.matrix(table(ALL$ref[ALL$LABEL=="LOF"], ALL$alt[ALL$LABEL=="LOF"]))
aaChg.GOF <- as.matrix(table(ALL$ref[ALL$LABEL=="GOF"], ALL$alt[ALL$LABEL=="GOF"]))
aaChg.p <- aaChg.GOF
for (i in 1:dim(aaChg.p)[1]) {
  for (j in 1:dim(aaChg.p)[2]) {
    if (aaChg.GOF[i,j]+aaChg.LOF[i,j]==0) {
      aaChg.p[i,j] <- NA
    } else {
      aaChg.p[i,j] <- binom.test(aaChg.GOF[i,j], aaChg.GOF[i,j]+aaChg.LOF[i,j], 
                                 p=sum(aaChg.GOF)/(sum(aaChg.GOF)+sum(aaChg.LOF)))$p.value
    }
  }
}
# aaChg.LOF <- aaChg.LOF / sum(aaChg.LOF)
aaChg.GOF <- aaChg.GOF / sum(aaChg.GOF)
aaChg.LOF <- aaChg.LOF / sum(aaChg.LOF)

aa.ano <- rbind(data.frame(aa=polar, anno='polar', charge=charge),
                data.frame(aa=special, anno='special', charge=""),
                data.frame(aa=hydrophobic, anno='nonpolar', charge=""))
rownames(aa.ano) <- aa.ano$aa
aaChg.GOF <- aaChg.GOF[match(aa.ano$aa, rownames(aaChg.GOF)), 
                       match(aa.ano$aa, colnames(aaChg.GOF))]
aaChg.LOF <- aaChg.LOF[match(aa.ano$aa, rownames(aaChg.LOF)), 
                       match(aa.ano$aa, colnames(aaChg.LOF))]
aaChg.p <- aaChg.p[match(aa.ano$aa, rownames(aaChg.p)), 
                   match(aa.ano$aa, colnames(aaChg.p))]
aa.ano$aa <- NULL

plot.complex.heatmap(aaChg.GOF, col.anno = aa.ano, 
                     row.anno = aa.ano, col_font_size = 10, 
                     cluster_rows = T, cluster_columns = T, col_show_legend = F,
                     row_font_size = 10, legend_name = "GoF %")
plot.complex.heatmap(aaChg.LOF, col.anno = aa.ano, 
                     row.anno = aa.ano, col_font_size = 10,
                     cluster_rows = T, cluster_columns = T, col_show_legend = F,  
                     row_font_size = 10, legend_name = "LoF %")
plot.complex.heatmap((aaChg.GOF-aaChg.LOF)*100, col.anno = aa.ano, 
                     row.anno = aa.ano, col_font_size = 10, upper.lim = 2, lower.lim = -2,
                     cluster_rows = F, cluster_columns = F, col_show_legend = F,  
                     row_font_size = 10, legend_name = "GoF-LoF %")
aaChg.q <- matrix(p.adjust(aaChg.p, method = 'fdr'), nrow = dim(aaChg.p)[1])
colnames(aaChg.q) <- colnames(aaChg.p)
rownames(aaChg.q) <- rownames(aaChg.p)
plot.complex.heatmap(aaChg.q, 
                     col.anno = aa.ano, 
                     row.anno = aa.ano, col_font_size = 10, upper.lim = 0, lower.lim = 1,
                     cluster_rows = F, cluster_columns = F, col_show_legend = F,  
                     row_font_size = 10, legend_name = "p.value")

## anno by polar and non-polar
# check aa change
hydrophobic <- c("F", "W", "Y", "A", "I", "L", "M", "V", "P", "G")
polar <- c("R", "K", "H", "D", "E", "N", "Q", "S", "T", "C")
charge <- c("+", "+", "+", "-", "-", ".", ".", ".", ".", ".")
aa.ano <- rbind(data.frame(aa=polar, anno='polar', charge=charge),
                data.frame(aa=hydrophobic, anno='nonpolar', charge="."))
rownames(aa.ano) <- aa.ano$aa
ALL$ref.anno <- aa.ano$anno[match(ALL$ref, aa.ano$aa)]
ALL$alt.anno <- aa.ano$anno[match(ALL$alt, aa.ano$aa)]
ALL$ref.charge <- aa.ano$charge[match(ALL$ref, aa.ano$aa)]
ALL$alt.charge <- aa.ano$charge[match(ALL$alt, aa.ano$aa)]
ALL$aaChg.anno <- paste0(ALL$ref.anno, '->', ALL$alt.anno)
ALL$aaChg.charge <- paste0(ALL$ref.charge, ':', ALL$alt.charge)
annos <- c('anno', 'charge')
p <- list()
for (h in 0:2) {
  for (j in 1:length(annos)) {
    if (h==0) {
      aaChg <- table(ALL[,paste0('aaChg.', annos[j])], ALL$LABEL)
    } else if (h==1) {
      aaChg <- table(ALL[ALL$data_source!="Heyne",paste0('aaChg.', annos[j])], ALL$LABEL[ALL$data_source!="Heyne"])
    } else {
      aaChg <- table(ALL[ALL$data_source=="Heyne",paste0('aaChg.', annos[j])], ALL$LABEL[ALL$data_source=="Heyne"])
    }
    aaChg.df <- matrix(NA, nrow = dim(aaChg)[1], ncol = dim(aaChg)[2])
    colnames(aaChg.df) <- colnames(aaChg)
    rownames(aaChg.df) <- rownames(aaChg)
    for (i in 1:dim(aaChg)[2]) {
      aaChg.df[,i] <- aaChg[,i]
    }
    aaChg.df <- as.data.frame(aaChg.df)
    for (i in 1:dim(aaChg.df)[1]) {
      res <- binom.test(aaChg.df[i,1], aaChg.df[i,1]+aaChg.df[i,2], p=sum(aaChg.df[,1])/sum(aaChg.df[,1]+aaChg.df[,2]))
      aaChg.df$p.value[i] <- res$p.value
    }
    aaChg.df$ref_aa <- rownames(aaChg.df)
    aaChg.df$q.value <- p.adjust(aaChg.df$p.value, method = "fdr")
    to.plot <- rbind(aaChg.df, aaChg.df)
    to.plot$n_mutation <- c(aaChg.df$GOF, aaChg.df$LOF)
    to.plot$frac_mutation <- c(aaChg.df$GOF/sum(aaChg.df$GOF), aaChg.df$LOF/sum(aaChg.df$LOF))
    to.plot$label <- c(rep("GOF", dim(aaChg.df)[1]), rep("LOF", dim(aaChg.df)[1]))
    
    anno <- to.plot
    anno$ref_aa[anno$q.value > 0.05] <- NA
    anno$frac_mutation[anno$q.value > 0.05] <- NA
    anno <- anno[!is.na(anno$ref_aa),]
    anno$x <- as.numeric(as.factor(to.plot$ref_aa))[match(anno$ref_aa, to.plot$ref_aa)] - 0.2
    anno$xend <- as.numeric(as.factor(to.plot$ref_aa))[match(anno$ref_aa, to.plot$ref_aa)] + 0.2
    anno$y <- anno$frac_mutation + 0.025
    anno <- anno[order(anno$x),]
    to.keep <- c()
    for (i in 1:(dim(anno)[1]/2)) {
      to.keep <- c(to.keep, c(i*2-1, i*2)[which.max(anno$y[c(i*2-1, i*2)])])
    }
    anno <- anno[to.keep,]
    anno$annotation <- NA
    for (k in 1:dim(anno)[1]) {
      anno$annotation[k] <- paste(c(rep(" ", k-1), "*", rep(" ", k-1)), collapse = "")
    }
    library(ggplot2)
    library(ggsignif)
    p6 <- ggplot(to.plot, aes(x=ref_aa, y=frac_mutation, fill=label)) +
      geom_bar(stat='identity', position=position_dodge()) + 
      geom_signif(stat="identity",
                  data=anno,
                  aes(x=x,
                      xend=xend,
                      y=y, yend=y,
                      annotation=annotation)) + ylim(0, 0.8) +
      xlab('Amino Acid Change') +
      # scale_x_discrete(guide = guide_axis(n.dodge=2)) +
      theme_bw()
    p[[j]] <- p6
  }
  p6 <- p[[1]] + p[[2]]
  if (h==0) {
    ggsave(p6, 'figs/02.01.GoF.LoF.aaChg.pdf')
  } else if (h==1) {
    ggsave(p6, 'figs/02.01.Heyne.GoF.LoF.aaChg.pdf')
  } else {
    ggsave(p6, 'figs/02.01.Other.GoF.LoF.aaChg.pdf')
  }
}






