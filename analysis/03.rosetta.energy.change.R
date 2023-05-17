source('~/Pipeline/dnv.table.to.uniprot.R')
ICC <- read.csv('~/Data/DMS/Itan.CKB.Cancer/ALL.csv', row.names = 1)
ICC <- uniprot.table.add.annotation.parallel(ICC, 'pLDDT')
for (i in 1:dim(ICC)[1]) {
  if (ICC$sequence.len.orig[i] < 2700) {
    idx <- 1
  } else {
    idx <- min(max(1, ICC$pos.orig[i] %/% 200 - 2), ICC$sequence.len.orig[i] %/% 200 - 5)
    seq_start <- (idx - 1) * 200 + 1
    ICC$pos.orig[i] <- ICC$pos.orig[i] - seq_start + 1
  }
  ICC$af2.file[i] <- paste0('~/Data/Protein/alphafold2_v4/swissprot/AF-',
                            ICC$uniprotID[i], '-F', idx, '-model_v4.pdb.gz')
}
unique.af2.files <- as.character(unique(ICC$af2.file))
library(doParallel)
cl <- makeCluster(102)
registerDoParallel(cl)
new.IIC <- foreach (k = 1:length(unique.af2.files), .combine = rbind) %dopar% {
  id <- unique.af2.files[k]
  id.variants <- ICC[ICC$af2.file==id,]
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
    system(rosetta.commands, ignore.stderr = T)
  }
  if (!file.exists(paste0('rosetta.out/', substr(id, 40, nchar(id)-6), 'wt.energy.out'))) {
    system(paste0("/share/terra/Users/gz2294/rosetta.3.14/source/bin/residue_energy_breakdown.linuxgccrelease ",
                  "-database /share/terra/Users/gz2294/rosetta.3.14/database/ -constant_seed true -jran 0 -in:file:s ",
                  id, " -out:file:silent ", "rosetta.out/", substr(id, 40, nchar(id)-6), "wt.energy.out"),
           ignore.stderr = T)
  }
  if (file.exists(paste0('rosetta.out/', substr(id, 40, nchar(id)-6), 'wt.energy.out'))) {
    wt.energy <- read.table(paste0('rosetta.out/', substr(id, 40, nchar(id)-6), 'wt.energy.out'), header = T)
    for (i in 1:dim(id.variants)[1]) {
      file.name <- paste0(substr(id, 40, nchar(id)-6),
                          'A-', id.variants$ref[i], id.variants$pos.orig[i], id.variants$alt[i], '.pdb')
      if (!file.exists(paste0("rosetta.out/", file.name, ".energy.out"))) {
        if (file.exists(file.name)) {
          system(paste0("mv ", file.name,
                        " rosetta.out/"), ignore.stderr = T)
        }
        system(paste0("/share/terra/Users/gz2294/rosetta.3.14/source/bin/residue_energy_breakdown.linuxgccrelease ",
                      "-database /share/terra/Users/gz2294/rosetta.3.14/database/ -constant_seed true -jran 0 -in:file:s",
                      " rosetta.out/", file.name, " -out:file:silent ", "rosetta.out/", file.name, ".energy.out"),
               ignore.stderr = T)
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
write.csv(new.IIC, file = 'figs/Itan.CKB.Cancer.ddg.csv')
stopCluster(cl)
# split to different families and do the same
good.pfams <- c("PF07714", "PF00454", "PF00069", "PF07679", "PF00047", 
                "PF00028", "PF00520", "PF06512", "PF11933")
for (pfam.name in good.pfams) {
  pfam <- rbind(read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer/pfams.add.beni.0.8/', pfam.name, '/training.csv')),
                read.csv(paste0('~/Data/DMS/Itan.CKB.Cancer/pfams.add.beni.0.8/', pfam.name, '/testing.csv')))
  pfam <- uniprot.table.add.annotation.parallel(pfam, "pLDDT")
  for (i in 1:dim(pfam)[1]) {
    if (pfam$sequence.len.orig[i] < 2700) {
      idx <- 1
    } else {
      idx <- min(max(1, pfam$pos.orig[i] %/% 200 - 2), pfam$sequence.len.orig[i] %/% 200 - 5)
      seq_start <- (idx - 1) * 200 + 1
      pfam$pos.orig[i] <- pfam$pos.orig[i] - seq_start + 1
    }
    pfam$af2.file[i] <- paste0('~/Data/Protein/alphafold2_v4/swissprot/AF-', 
                               pfam$uniprotID[i], '-F', idx, '-model_v4.pdb.gz')
  }
  unique.ids.pfam <- as.character(unique(pfam$af2.file))
  library(doParallel)
  cl <- makeCluster(102)
  registerDoParallel(cl)
  new.pfam <- foreach (k = 1:length(unique.ids.pfam), .combine = rbind) %dopar% {
    id <- unique.ids.pfam[k]
    id.variants <- pfam[pfam$af2.file==id,]
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
  write.csv(new.pfam, file = paste0('figs/', pfam.name, '.ddg.csv'))
  stopCluster(cl)
}


