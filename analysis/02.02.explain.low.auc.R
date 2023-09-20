source('/share/pascal/Users/gz2294/Pipeline/uniprot.table.add.annotation.R')
ALL <- read.csv('/share/pascal/Users/gz2294/Data/DMS/Itan.CKB.Cancer/ALL.csv', row.names = 1)
ion.channel <- read.csv('/share/pascal/Users/gz2294/Data/DMS/Ion_Channel/all.af2update.csv', row.names = 1)
ALL$score[ALL$score == 0] <- -1
ion.channel$score[ion.channel$score == 0] <- -1

ALL$unique.id <- paste(ALL$uniprotID, ALL$ref, ALL$pos.orig, ALL$alt, sep = ":")
ion.channel$unique.id <- paste(ion.channel$uniprotID, ion.channel$ref, ion.channel$pos.orig, ion.channel$alt, sep = ":")

to.del <- readRDS("/share/pascal/Users/gz2294/Data/DMS/Itan.CKB.Cancer/to.del.conflict.with.chps.RDS")
ALL <- ALL[!ALL$unique.id %in% to.del,]
ALL <- ALL[!ALL$unique.id %in% ion.channel$unique.id,]
ALL <- dplyr::bind_rows(ALL, ion.channel)

ALL <- uniprot.table.add.annotation.parallel(ALL, 'pLDDT')
ALL <- uniprot.table.add.annotation.parallel(ALL, 'pLDDT.all')
ALL <- uniprot.table.add.annotation.parallel(ALL, 'pLDDT.region')

# get distribution of scn2a and fgfr2
good.genes <- c("P15056", "P04637", "Q09428", "Q14654", "O00555")
med.genes <- c("P07949", "Q14524")
bad.genes <- c('Q99250', 'P21802')
ALL$plot.label <- ALL$uniprotID
ALL$plot.label[!ALL$uniprotID %in% c(good.genes, bad.genes, med.genes)] <- NA
ALL$plot.label[ALL$uniprotID %in% c(bad.genes)] <- 'Bad Genes'
ALL$plot.label[ALL$uniprotID %in% c(med.genes)] <- 'Med Genes'
ALL$plot.label[ALL$uniprotID %in% c(good.genes)] <- 'Good Genes'
p1 <- ggplot(ALL[!is.na(ALL$plot.label),], aes(x=pLDDT, col=plot.label)) + geom_density() + theme_bw()
p2 <- ggplot(ALL[!is.na(ALL$plot.label),], aes(x=pLDDT.region, col=plot.label)) + geom_density() + theme_bw()
library(patchwork)
p <- p1 / p2
ggsave(filename = 'figs/02.02.explain.low.auc.pdf', p, width = 8, height = 6)
