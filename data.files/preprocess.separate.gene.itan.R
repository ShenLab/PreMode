# sum the variant number of uniprotIDs in the dataset
library(ggplot2)
source('~/Pipeline/uniprot.table.add.annotation.R')
ALL <- read.csv('ALL.csv', row.names = 1)
ALL$score[ALL$score == 0] <- -1
# remove glazer
ALL <- ALL[ALL$data_source != "glazer",]
ALL$score.label <- NULL
good.uniprotIDs <- data.frame(
  uniprotIDs=c("P15056", "P21802", "P07949", 
              "P04637", "Q09428", "O00555",
              "Q14654", "Q99250"))

core.columns <- c("uniprotID", "ref", "alt", "pos.orig", "ENST", "wt.orig", "score")
for (i in 1:dim(good.uniprotIDs)[1]) {
  GO <- ALL[grep(good.uniprotIDs$uniprotIDs[i], ALL$uniprotID),]
  print(table(GO$score[!grepl('Itan', GO$data_source)]))
}

itan.aucs.1 <- c()
itan.aucs.2 <- c()
# split by random, add beni
ratio <- 0.25
good.uniprotIDs$gof.training <- NA
good.uniprotIDs$gof.testing <- NA
good.uniprotIDs$lof.training <- NA
good.uniprotIDs$lof.testing <- NA
for (seed in 0:4) {
  split.dir <- paste0('ICC.seed.', seed, '/')
  dir.create(split.dir)
  for (i in 1:dim(good.uniprotIDs)[1]) {
    GO <- ALL[grep(good.uniprotIDs$uniprotIDs[i], ALL$uniprotID),]
    GO.gof <- GO[GO$score == 1,]
    GO.lof <- GO[GO$score == -1,]
    # split variants from GO and other
    GO.non.itan.gof <- GO.gof[!grepl('Itan', GO.gof$data_source),]
    GO.non.itan.lof <- GO.lof[!grepl('Itan', GO.lof$data_source),]
    GO.itan.gof <- GO.gof[grepl('Itan', GO.gof$data_source),]
    GO.itan.lof <- GO.lof[grepl('Itan', GO.lof$data_source),]
    
    set.seed(seed)
    # pick ratio% of variants as training
    if (floor(dim(GO.gof)[1] * ratio) > 0 & 
        floor(dim(GO.lof)[1] * ratio) > 0) {
      GO.gof.testing <- sample(dim(GO.non.itan.gof)[1], min(dim(GO.non.itan.gof)[1], floor(dim(GO.gof)[1] * ratio)))
      GO.lof.testing <- sample(dim(GO.non.itan.lof)[1], min(dim(GO.non.itan.lof)[1], floor(dim(GO.lof)[1] * ratio)))
      GO.training <- rbind(GO.itan.gof,
                           GO.itan.lof,
                           GO.non.itan.gof[-GO.gof.testing,],
                           GO.non.itan.lof[-GO.lof.testing,])
      GO.testing <- rbind(GO.non.itan.gof[GO.gof.testing,],
                          GO.non.itan.lof[GO.lof.testing,])
      
      GO.training <- GO.training[sample(dim(GO.training)[1]),]
      GO.testing <- GO.testing[sample(dim(GO.testing)[1]),]
      # beni.training.beni <- beni.training[beni.training$score==0,]
      if (sum(is.na(GO.testing$itan.beni)) > 0) {
        print(good.uniprotIDs$uniprotIDs[i])
        print(seed)
      }
      if (dim(GO.testing)[1] > 0) {
        # print(paste0(good.uniprotIDs$uniprotIDs[i], ":", seed))
        itan.aucs.1 <- c(itan.aucs.1, plot.AUC(GO.testing$score, 1-GO.testing$itan.gof)$auc)
        itan.aucs.2 <- c(itan.aucs.2, plot.AUC(GO.testing$score, GO.testing$itan.lof/GO.testing$itan.gof)$auc)
      } else {
        itan.aucs.1 <- c(itan.aucs.1, NA)
        itan.aucs.2 <- c(itan.aucs.2, NA)
      }
      dir.create(paste0(split.dir, good.uniprotIDs$uniprotIDs[i], '.itan.split'))
      write.csv(GO.training, paste0(split.dir, good.uniprotIDs$uniprotIDs[i], ".itan.split/training.csv"))
      # write.csv(beni.training.beni, paste0(split.dir, good.uniprotIDs$uniprotIDs[i], ".chps/beni.csv"))
      write.csv(GO.testing, paste0(split.dir, good.uniprotIDs$uniprotIDs[i], ".itan.split/testing.csv"))
      
      good.uniprotIDs$gof.training[i] <- sum(GO.training$score==1)
      good.uniprotIDs$lof.training[i] <- sum(GO.training$score==-1)
      good.uniprotIDs$beni.training[i] <- sum(GO.training$score==0)
      good.uniprotIDs$patho.training[i] <- sum(GO.training$score==3)
      
      good.uniprotIDs$gof.testing[i] <- sum(GO.testing$score==1)
      good.uniprotIDs$lof.testing[i] <- sum(GO.testing$score==-1)
      good.uniprotIDs$beni.testing[i] <- sum(GO.testing$score==0)
      good.uniprotIDs$patho.testing[i] <- sum(GO.testing$score==3)
      
    }
  }
}
write.csv(good.uniprotIDs, file = "good.uniprotIDs.itan.csv")

# # split by random, add beni
ratio <- 0.25
# good.uniprotIDs$gof.training <- NA
# good.uniprotIDs$gof.testing <- NA
# good.uniprotIDs$lof.training <- NA
# good.uniprotIDs$lof.testing <- NA
# source('~/Pipeline/AUROC.R')
# 
# for (seed in 0:4) {
#   if (seed == 0) {
#     split.dir <- paste0('pfams.add.beni.', 0.8, '/')
#   } else {
#     split.dir <- paste0('pfams.add.beni.', 0.8, '.seed.', seed, '/')
#   }
#   dir.create(split.dir)
#   for (i in 1:dim(good.uniprotIDs)[1]) {
#     GO <- ALL[grep(good.uniprotIDs$uniprotIDs[i], ALL$uniprotID),]
#     GO.gof <- GO[GO$score == 1,]
#     GO.lof <- GO[GO$score == -1,]
#     # split variants from GO and other
#     non.itan.gof.idx <- which(GO.gof$data_source != 'Itan' & !is.na(GO.gof$itan.beni))
#     non.itan.lof.idx <- which(GO.lof$data_source != 'Itan' & !is.na(GO.lof$itan.beni))
#     
#     GO.non.itan.gof <- GO.gof[non.itan.gof.idx,]
#     GO.non.itan.lof <- GO.lof[non.itan.lof.idx,]
#     GO.itan.gof <- GO.gof[-non.itan.gof.idx,]
#     GO.itan.lof <- GO.lof[-non.itan.lof.idx,]
#     
#     set.seed(seed)
#     # pick ratio% of variants as training
#     if (floor(dim(GO.gof)[1] * ratio) > 0 & 
#         floor(dim(GO.lof)[1] * ratio) > 0) {
#       GO.gof.testing <- sample(dim(GO.non.itan.gof)[1], min(dim(GO.non.itan.gof)[1], floor(dim(GO.gof)[1] * ratio)))
#       GO.lof.testing <- sample(dim(GO.non.itan.lof)[1], min(dim(GO.non.itan.lof)[1], floor(dim(GO.lof)[1] * ratio)))
#       GO.training <- rbind(GO.itan.gof,
#                            GO.itan.lof,
#                            GO.non.itan.gof[-GO.gof.testing,],
#                            GO.non.itan.lof[-GO.lof.testing,])
#       GO.testing <- rbind(GO.non.itan.gof[GO.gof.testing,],
#                           GO.non.itan.lof[GO.lof.testing,])
#       
#       GO.training <- GO.training[sample(dim(GO.training)[1]),]
#       GO.testing <- GO.testing[sample(dim(GO.testing)[1]),]
#       # beni.training.beni <- beni.training[beni.training$score==0,]
#       if (dim(GO.testing)[1] > 0) {
#         print(paste0(good.uniprotIDs$uniprotIDs[i], ":", seed))
#         itan.aucs.1 <- c(itan.aucs.1, plot.AUC(GO.testing$score, GO.testing$itan.gof)$auc)
#         itan.aucs.2 <- c(itan.aucs.2, plot.AUC(GO.testing$score, GO.testing$itan.gof/GO.testing$itan.lof)$auc)
#       } else {
#         itan.aucs.1 <- c(itan.aucs.1, NA)
#         itan.aucs.2 <- c(itan.aucs.2, NA)
#       }
#       dir.create(paste0(split.dir, good.uniprotIDs$uniprotIDs[i], '.itan.split.clean'))
#       write.csv(GO.training, paste0(split.dir, good.uniprotIDs$uniprotIDs[i], ".itan.split.clean/training.csv"))
#       # write.csv(beni.training.beni, paste0(split.dir, good.uniprotIDs$uniprotIDs[i], ".chps/beni.csv"))
#       write.csv(GO.testing, paste0(split.dir, good.uniprotIDs$uniprotIDs[i], ".itan.split.clean/testing.csv"))
#       
#       good.uniprotIDs$gof.training[i] <- sum(GO.training$score==1)
#       good.uniprotIDs$lof.training[i] <- sum(GO.training$score==-1)
#       good.uniprotIDs$beni.training[i] <- sum(GO.training$score==0)
#       good.uniprotIDs$patho.training[i] <- sum(GO.training$score==3)
#       
#       good.uniprotIDs$gof.testing[i] <- sum(GO.testing$score==1)
#       good.uniprotIDs$lof.testing[i] <- sum(GO.testing$score==-1)
#       good.uniprotIDs$beni.testing[i] <- sum(GO.testing$score==0)
#       good.uniprotIDs$patho.testing[i] <- sum(GO.testing$score==3)
#       
#     }
#   }
# }
# write.csv(good.uniprotIDs, file = "good.uniprotIDs.itan.clean.csv")

# for SCN5A, remove glazer
for (seed in 0:4) {
  if (seed == 0) {
    split.dir <- paste0('pfams.add.beni.', 0.8, '/')
  } else {
    split.dir <- paste0('pfams.add.beni.', 0.8, '.seed.', seed, '/')
  }
  dir.create(split.dir)
  GO <- ALL[grep('Q14524', ALL$uniprotID),]
  GO <- GO[GO$data_source != 'glazer',]
  GO.gof <- GO[GO$score == 1,]
  GO.lof <- GO[GO$score == -1,]
  # split variants from GO and other
  non.itan.gof.idx <- which(!grepl('Itan', GO.gof$data_source) & !is.na(GO.gof$itan.beni))
  non.itan.lof.idx <- which(!grepl('Itan', GO.lof$data_source) & !is.na(GO.lof$itan.beni))
  
  GO.non.itan.gof <- GO.gof[non.itan.gof.idx,]
  GO.non.itan.lof <- GO.lof[non.itan.lof.idx,]
  GO.itan.gof <- GO.gof[-non.itan.gof.idx,]
  GO.itan.lof <- GO.lof[-non.itan.lof.idx,]
  
  set.seed(seed)
  # pick ratio% of variants as training
  if (floor(dim(GO.gof)[1] * ratio) > 0 & 
      floor(dim(GO.lof)[1] * ratio) > 0) {
    GO.gof.testing <- sample(dim(GO.non.itan.gof)[1], min(dim(GO.non.itan.gof)[1], floor(dim(GO.gof)[1] * ratio)))
    GO.lof.testing <- sample(dim(GO.non.itan.lof)[1], min(dim(GO.non.itan.lof)[1], floor(dim(GO.lof)[1] * ratio)))
    GO.training <- rbind(GO.itan.gof,
                         GO.itan.lof,
                         GO.non.itan.gof[-GO.gof.testing,],
                         GO.non.itan.lof[-GO.lof.testing,])
    GO.testing <- rbind(GO.non.itan.gof[GO.gof.testing,],
                        GO.non.itan.lof[GO.lof.testing,])
    
    GO.training <- GO.training[sample(dim(GO.training)[1]),]
    GO.testing <- GO.testing[sample(dim(GO.testing)[1]),]
    # beni.training.beni <- beni.training[beni.training$score==0,]
    if (dim(GO.testing)[1] > 0) {
      # print(paste0(good.uniprotIDs$uniprotIDs[i], ":", seed))
      itan.aucs.1 <- c(itan.aucs.1, plot.AUC(GO.testing$score, GO.testing$itan.gof)$auc)
      itan.aucs.2 <- c(itan.aucs.2, plot.AUC(GO.testing$score, GO.testing$itan.gof/GO.testing$itan.lof)$auc)
    } else {
      itan.aucs.1 <- c(itan.aucs.1, NA)
      itan.aucs.2 <- c(itan.aucs.2, NA)
    }
    dir.create(paste0(split.dir, 'Q14524', '.clean.itan.split'))
    write.csv(GO.training, paste0(split.dir, 'Q14524', ".clean.itan.split/training.csv"))
    # write.csv(beni.training.beni, paste0(split.dir, 'Q14524', ".chps/beni.csv"))
    write.csv(GO.testing, paste0(split.dir, 'Q14524', ".clean.itan.split/testing.csv"))
  }
}

# for SCN5A, remove glazer, don't do itan split, just split
ratio <- 0.8
for (seed in 0:4) {
  if (seed == 0) {
    split.dir <- paste0('pfams.add.beni.', 0.8, '/')
  } else {
    split.dir <- paste0('pfams.add.beni.', 0.8, '.seed.', seed, '/')
  }
  dir.create(split.dir)
  GO <- ALL[grep('Q14524', ALL$uniprotID),]
  GO.itan <- GO[GO$data_source != 'glazer',]
  GO.itan.gof <- GO.itan[GO.itan$score == 1,]
  GO.itan.lof <- GO.itan[GO.itan$score == -1,]
  set.seed(seed)
  # pick ratio% of variants as training
  if (floor(dim(GO.itan.gof)[1] * ratio) > 0 & 
      floor(dim(GO.itan.lof)[1] * ratio) > 0) {
    GO.itan.gof.training <- sample(dim(GO.itan.gof)[1], floor(dim(GO.itan.gof)[1] * ratio))
    GO.itan.lof.training <- sample(dim(GO.itan.lof)[1], floor(dim(GO.itan.lof)[1] * ratio))
    GO.itan.training <- rbind(GO.itan.gof[GO.itan.gof.training,],
                              GO.itan.lof[GO.itan.lof.training,])
    GO.itan.testing <- rbind(GO.itan.gof[-GO.itan.gof.training,],
                             GO.itan.lof[-GO.itan.lof.training,])
    GO.itan.training <- GO.itan.training[sample(dim(GO.itan.training)[1]),]
    GO.itan.testing <- GO.itan.testing[sample(dim(GO.itan.testing)[1]),]

    dir.create(paste0(split.dir, 'Q14524', '.clean'))
    write.csv(GO.itan.training, paste0(split.dir, 'Q14524', ".clean/training.csv"))
    # write.csv(beni.training.beni, paste0(split.dir, 'Q14524', ".chps/beni.csv"))
    write.csv(GO.itan.testing, paste0(split.dir, 'Q14524', ".clean/testing.csv"))
  }
}

# # for FGFR2, further clean data, only use CKB validated as testing
# fgfr2.check <- read.csv('fgfr2.check.csv')
# fgfr2.check$uniprotID <- 'P21802'
# source('/share/vault/Users/gz2294/Pipeline/dnv.table.to.uniprot.R')
# fgfr2.check <- dnv.table.to.uniprot.by.af2.uniprotID.parallel(fgfr2.check, 'aaChg', 'score', 'uniprotID', 'aaChg')
# source('/share/vault/Users/gz2294/Pipeline/uniprot.table.add.annotation.R')
# fgfr2.check <- uniprot.table.add.annotation.parallel(fgfr2.check$result.noNA, 'Itan')
# 





