IonChannel <- read.csv('~/Data/DMS/Itan.CKB.Cancer/pfams.0.8/PF.ion.channel/all.csv', row.names = 1)
# check the energy and pLDDT
source('~/Pipeline/uniprot.table.add.annotation.R')
IonChannel <- uniprot.table.add.annotation.parallel(IonChannel, 'energy')
IonChannel <- uniprot.table.add.annotation.parallel(IonChannel, 'pLDDT')
IonChannel$LABEL[IonChannel$score==1] <- "GOF"
IonChannel$LABEL[IonChannel$score==0] <- "LOF"

# test the other metrics
IonChannel <- uniprot.table.add.annotation.parallel(IonChannel, 'dbnsfp')
IonChannel <- uniprot.table.add.annotation.parallel(IonChannel, 'gMVP')
IonChannel <- uniprot.table.add.annotation.parallel(IonChannel, 'EVE')
source('~/Pipeline/AUROC.R')
plot.AUC(IonChannel$score, IonChannel$EVE)
plot.AUC(IonChannel$score, IonChannel$REVEL)
plot.AUC(IonChannel$score, IonChannel$PrimateAI)
plot.AUC(IonChannel$score, IonChannel$gMVP)

library(ggplot2)
library(ggExtra)
p <- ggplot(IonChannel, aes(x=energy, y=pLDDT, col=LABEL)) +
  geom_point() + theme_bw() + theme(legend.position = "bottom") 
p <- ggMarginal(p, type="density", groupColour=T)
ggsave(filename = 'figs/IonChannel.pdf', p, width = 6, height = 6)
write.csv(IonChannel, file = 'figs/IonChannel.csv')

for (i in 1:dim(IonChannel)[1]) {
  if (IonChannel$sequence.len.orig[i] < 2700) {
    idx <- 1
  } else {
    idx <- min(max(1, IonChannel$pos.orig[i] %/% 200 - 2), IonChannel$sequence.len.orig[i] %/% 200 - 5)
    seq_start <- (idx - 1) * 200 + 1
    IonChannel$pos.orig[i] <- IonChannel$pos.orig[i] - seq_start + 1
  }
  IonChannel$af2.file[i] <- paste0('~/Data/Protein/alphafold2_v4/swissprot/AF-', 
                             IonChannel$uniprotID[i], '-F', idx, '-model_v4.pdb.gz')
}
unique.ids.IonChannel <- as.character(unique(IonChannel$af2.file))
library(doParallel)
cl <- makeCluster(102)
registerDoParallel(cl)
new.IonChannel <- foreach (k = 1:length(unique.ids.IonChannel), .combine = rbind) %dopar% {
  id <- unique.ids.IonChannel[k]
  id.variants <- IonChannel[IonChannel$af2.file==id,]
  mutant.file <- data.frame(chain='A', 
                            ref=id.variants$ref, 
                            pos=id.variants$pos.orig, 
                            alt=id.variants$alt)
  if (!file.exists(paste0("rosetta.out/", substr(id, 40, nchar(id)-6), '.mutants.txt'))) {
    write.table(mutant.file, 
                file = paste0("rosetta.out/", substr(id, 40, nchar(id)-6), '.mutants.txt'),
                sep = " ", col.names = F, row.names = F, quote = F)
    rosetta.commands <- paste0("/share/terra/Users/gz2294/rosetta.3.14/source/bin/pmut_scan_parallel.default.linuxgccrelease -database /share/terra/Users/gz2294/rosetta.3.14/database/ -s ",
                               id,
                               " -mutants_list ",
                               paste0("rosetta.out/", substr(id, 40, nchar(id)-6), '.mutants.txt'),
                               " -ex1 -ex2 -ex3 -extrachi_cutoff 1 -use_input_sc -ignore_unrecognized_res -no_his_his_pairE -multi_cool_annealer 10 -mute basic core -output_mutant_structures")
    system(rosetta.commands)
  }
  if (!file.exists(paste0('rosetta.out/', substr(id, 40, nchar(id)-6), 'wt.energy.out'))) {
    system(paste0("/share/terra/Users/gz2294/rosetta.3.14/source/bin/residue_energy_breakdown.linuxgccrelease ",
                  "-database /share/terra/Users/gz2294/rosetta.3.14/database/ -constant_seed true -jran 0 -in:file:s ",
                  id, " -out:file:silent ", "rosetta.out/", substr(id, 40, nchar(id)-6), "wt.energy.out"))
  }
  if (file.exists(paste0('rosetta.out/', substr(id, 40, nchar(id)-6), 'wt.energy.out'))) {
    wt.energy <- read.table(paste0('rosetta.out/', substr(id, 40, nchar(id)-6), 'wt.energy.out'), header = T)
    for (i in 1:dim(id.variants)[1]) {
      file.name <- paste0(substr(id, 40, nchar(id)-6), 
                          'A-', id.variants$ref[i], id.variants$pos.orig[i], id.variants$alt[i], '.pdb')
      if (!file.exists(paste0("rosetta.out/", file.name, ".energy.out"))) {
        if (file.exists(file.name)) {
          system(paste0("mv ", file.name,
                        " rosetta.out/"))
        }
        system(paste0("/share/terra/Users/gz2294/rosetta.3.14/source/bin/residue_energy_breakdown.linuxgccrelease ",
                      "-database /share/terra/Users/gz2294/rosetta.3.14/database/ -constant_seed true -jran 0 -in:file:s",
                      " rosetta.out/", file.name, " -out:file:silent ", "rosetta.out/", file.name, ".energy.out"))
      }
      if (file.exists(paste0("rosetta.out/", file.name, ".energy.out"))) {
        energy <- read.table(paste0('rosetta.out/', file.name, '.energy.out'), header = T)
        ddg <- sum(energy$total - energy$ref) - sum(wt.energy$total - wt.energy$ref)
      } else {
        ddg <- NA
      }
      id.variants$ddg[i] <- ddg
    }
  } else {
    id.variants$ddg <- NA
  }
  id.variants
}
write.csv(new.IonChannel, file = paste0('figs/', 'IonChannel', '.ddg.csv'))
