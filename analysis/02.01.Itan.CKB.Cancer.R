ICC <- read.csv('~/Data/DMS/Itan.CKB.Cancer/ALL.csv', row.names = 1)
# check the energy and pLDDT
source('~/Pipeline/dnv.table.to.uniprot.R')
ICC <- uniprot.table.add.annotation.parallel(ICC, 'energy')
ICC <- uniprot.table.add.annotation.parallel(ICC, 'pLDDT')
ICC$LABEL[ICC$score==1] <- "GOF"
ICC$LABEL[ICC$score==0] <- "LOF"
library(ggplot2)
library(ggExtra)
p <- ggplot(ICC, aes(x=energy, y=pLDDT, col=LABEL)) +
  geom_point() + theme_bw() + theme(legend.position = "bottom") 
p <- ggMarginal(p, type="density", groupColour=T)
ggsave(filename = 'figs/Itan.CKB.Cancer.pdf', p, width = 6, height = 6)
write.csv(ICC, file = 'figs/ICC.csv')



# split to different families and do the same
good.pfams <- c("PF07714", "PF00454", "PF00069", "PF07679", "PF00047", 
                "PF00028", "PF00520", "PF06512", "PF11933")
for (i in good.pfams) {
  pfam <- rbind(read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer/pfams.add.beni.0.8/', i, '/training.csv')),
                read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer/pfams.add.beni.0.8/', i, '/testing.csv')))
  pfam <- uniprot.table.add.annotation.parallel(pfam, 'energy')
  pfam <- uniprot.table.add.annotation.parallel(pfam, 'pLDDT')
  pfam$LABEL[pfam$score==1] <- "GOF"
  pfam$LABEL[pfam$score==-1] <- "LOF"
  pfam$LABEL[pfam$score==0] <- "Beni"
  p <- ggplot(pfam, aes(x=energy, y=pLDDT, col=LABEL)) +
    geom_point() + theme_bw() + theme(legend.position = "bottom") 
  p <- ggMarginal(p, type="density", groupColour=T)
  ggsave(filename = paste0('figs/', i, '.pdf'), p, width = 6, height = 6)
  write.csv(pfam, file = paste0('figs/', i, '.csv'))
}

