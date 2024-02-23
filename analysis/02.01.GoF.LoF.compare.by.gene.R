# ICC <- read.csv('~/Data/DMS/Itan.CKB.Cancer/ALL.csv', row.names = 1)
# # remove unwanted variants
# to.del <- readRDS("~/Data/DMS/Itan.CKB.Cancer/to.del.conflict.with.chps.RDS")
# ICC$unique.id <- paste(ICC$uniprotID, ICC$ref, ICC$pos.orig, ICC$alt, sep = ":")
# ICC <- ICC[!ICC$unique.id %in% to.del,]
# ion.channel <- read.csv('~/Data/DMS/Ion_Channel/all.af2update.csv', row.names = 1)
# ion.channel$unique.id <- paste(ion.channel$uniprotID, ion.channel$ref, ion.channel$pos.orig, ion.channel$alt, sep = ":")
# ICC <- ICC[!ICC$unique.id %in% ion.channel$unique.id,]
# ALL <- dplyr::bind_rows(ICC, ion.channel)

# add benign
source('/share/vault/Users/gz2294/Pipeline/uniprot.table.add.annotation.R')
# Ion_channel <- read.csv('~/Data/DMS/')
# check the sse and rsa
ALL <- read.csv('figs/ALL.csv', row.names = 1, na.strings = c('.', 'NA'))
# ALL <- uniprot.table.add.annotation.parallel(ALL, 'dssp')
# ALL <- uniprot.table.add.annotation.parallel(ALL, 'FoldXddG')
# ALL <- uniprot.table.add.annotation.parallel(ALL, 'pLDDT')
# ALL <- uniprot.table.add.annotation.parallel(ALL, 'conservation')
# compare conservation with benign
benign <- read.csv('figs/benign.csv', row.names = 1, na.strings = c('.', 'NA'))
# benign <- rbind(read.csv('~/Data/DMS/ClinVar.HGMD.PrimateAI.syn/training.csv'),
#                 read.csv('~/Data/DMS/ClinVar.HGMD.PrimateAI.syn/testing.csv'))
# benign <- benign[benign$score==0,]
# benign$LABEL <- 'Benign'
# benign <- uniprot.table.add.annotation.parallel(benign, 'conservation')
# benign <- uniprot.table.add.annotation.parallel(benign, 'dssp')
# benign <- uniprot.table.add.annotation.parallel(benign, 'pLDDT')
# benign <- uniprot.table.add.annotation.parallel(benign, 'FoldXddG', njobs = 108)
# ALL <- uniprot.table.add.annotation.parallel(ALL, "pfam")
# ALL$LABEL[ALL$score==1] <- "GOF"
# ALL$LABEL[ALL$score==0] <- "LOF"
# write.csv(ALL, 'figs/ALL.csv')
# write.csv(benign, 'figs/benign.csv')

# Check P15056
genes <- c("Q09428", "P15056", "O00555", "P21802",
           "Q14654", "P07949", "Q99250", "Q14524", "P04637")
gene.names <- c("ABCC8", "BRAF", "CACNA1A", "FGFR2",
                "KCNJ11", "RET", "SCN2A", "SCN5A", "TP53")
ALL.orig <- ALL
benign.orig <- benign
for (gene.to.check in genes) {
  ALL <- ALL.orig[ALL.orig$uniprotID==gene.to.check,]
  benign <- benign.orig[benign.orig$uniprotID==gene.to.check,]
  
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
  p4 <- ggplot(rbind(ALL[,c("FoldXddG", "LABEL")], benign[,c("FoldXddG", "LABEL")]), aes(x=FoldXddG, col=LABEL)) + geom_density() + 
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
  ggsave(plot = p, filename = paste0("figs/02.01.GoF.LoF.compare.by.gene/", gene.to.check, ".pdf"), height=5, width=12)
  
  # to.plot <- dplyr::bind_rows(ALL, benign)
  source('/share/vault/Users/gz2294/Pipeline/protein.visualize.R')
  p <- protein.visualize(gene.to.check, gene.names[genes==gene.to.check], ALL, 'FoldXddG',
                         fill.colors = c('yellow', 'purple', 'black'))  
  ggsave(plot = p, filename = paste0("figs/02.01.GoF.LoF.compare.by.gene/", gene.to.check, ".visualize.pdf"), height=5, width=25)
  
  # we need a script to check the per residue energy change
  source('/share/vault/Users/gz2294/Pipeline/uniprot.table.add.annotation.R')
  
}
# check PTMs
ptm.dict <- c('ac', 'ga', 'gl', 'm1', 'm2', 'm3', 'me', 'p', 'sm', 'ub')
for (k in 1:length(genes)) {
  gene.to.check <- genes[k]
  ALL <- ALL.orig[ALL.orig$uniprotID==gene.to.check,]
  benign <- benign.orig[benign.orig$uniprotID==gene.to.check,]
  p <- list()
  for (i in 1:length(ptm.dict)) {
    ptm <- ptm.dict[i]
    wil.stat <- tryCatch({wilcox.test(ALL[ALL$LABEL=="GOF", paste0('PTM_dist_3d_', ptm)],
                            ALL[ALL$LABEL=="LOF", paste0('PTEM_dist_3d_', ptm)])}, error=function(cond){list(p.value=NA)})
    p2 <- ggplot(rbind(ALL[,c(paste0('PTM_dist_3d_', ptm), "LABEL")],
                       benign[,c(paste0('PTM_dist_3d_', ptm), "LABEL")]),
                 aes_string(x=paste0('PTM_dist_3d_', ptm), col="LABEL")) + geom_density() +
      theme_bw() + ggpp::geom_text_npc(data=data.frame(x="middle", y="top",
                                                       label=paste0("Mann-Whitney test G/LoF p=", signif(wil.stat$p.value, digits = 2))),
                                       aes(npcx=x, npcy=y, label=label),
                                       col='black') + ggtitle(ptm) + ggeasy::easy_center_title()
    p[[i]] <- p2
  }
  p.all <- ggpubr::ggarrange(plotlist=p, ncol=5, nrow=2, common.legend = T, legend = 'bottom') + ggtitle(gene.to.check)
  p.all <- ggpubr::annotate_figure(p.all, top=ggpubr::text_grob(gene.names[k]))
  ggsave(filename = paste0("figs/02.01.GoF.LoF.compare.by.gene/", gene.to.check, ".dist_3d.pdf"),
         p.all, height=5, width=25)
}
