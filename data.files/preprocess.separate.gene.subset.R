good.uniprotIDs <- c('P15056', 'P21802', 'P07949', 'P04637', 'Q09428', 'O00555', 'Q14654', 'Q99250', 'Q14524.clean')
# split by random, add beni
good.uniprotIDs.df <- data.frame()
for (seed in 0:4) {
  split.dir <- paste0('ICC.seed.', seed, '/')
  ratios <- c(1, 2, 4, 6)
  dir.create(split.dir)
  for (i in 1:length(good.uniprotIDs)) {
    GO.itan.training <- read.csv(paste0('ICC.seed.', 0, '/', good.uniprotIDs[i], '/training.csv'))
    
    GO.itan.testing <- read.csv(paste0('ICC.seed.', 0 ,'/', good.uniprotIDs[i], '/testing.csv'))
    GO.itan.gof <- GO.itan.training[GO.itan.training$score==1,]
    GO.itan.lof <- GO.itan.training[GO.itan.training$score==-1,]
    set.seed(seed)
    # pick ratio% of variants as training
    for (ratio in ratios) {
      if (floor(dim(GO.itan.gof)[1] * ratio/8) > 0 & 
          floor(dim(GO.itan.lof)[1] * ratio/8) > 0) {
        GO.itan.gof.training <- sample(dim(GO.itan.gof)[1], ceiling(dim(GO.itan.gof)[1] * ratio/8))
        GO.itan.lof.training <- sample(dim(GO.itan.lof)[1], ceiling(dim(GO.itan.lof)[1] * ratio/8))
        GO.itan.training <- rbind(GO.itan.gof[GO.itan.gof.training,],
                                  GO.itan.lof[GO.itan.lof.training,])
        GO.itan.training$split <- 'train'
        
        GO.itan.training <- GO.itan.training[sample(dim(GO.itan.training)[1]),]
        GO.itan.testing <- GO.itan.testing[sample(dim(GO.itan.testing)[1]),]
        
        dir.create(paste0(split.dir, good.uniprotIDs[i], '.subset2.', ratio))
        if (!file.exists(paste0(split.dir, good.uniprotIDs[i], '.subset2.', ratio, "/training.csv"))) {
          print(good.uniprotIDs[i])
          write.csv(GO.itan.training, paste0(split.dir, good.uniprotIDs[i], '.subset2.', ratio, "/training.csv"))
        }
        if (!file.exists(paste0(split.dir, good.uniprotIDs[i], '.subset2.', ratio, "/testing.csv"))) {
          print(good.uniprotIDs[i])
          write.csv(GO.itan.testing, paste0(split.dir, good.uniprotIDs[i], '.subset2.', ratio, "/testing.csv"))
        }
        good.uniprotIDs.df <- rbind(good.uniprotIDs.df, 
                                    data.frame(gene=good.uniprotIDs[i],
                                               ratio=ratio,
                                               seed=seed,
                                               gof.training = sum(GO.itan.training$score==1 & GO.itan.training$split=='train'),
                                               lof.training = sum(GO.itan.training$score==-1 & GO.itan.training$split=='train'),
                                               gof.val = sum(GO.itan.training$score==1 & GO.itan.training$split=='val'),
                                               lof.val = sum(GO.itan.training$score==-1 & GO.itan.training$split=='val'),
                                               gof.testing = sum(GO.itan.testing$score==1),
                                               lof.testing = sum(GO.itan.testing$score==-1)))
      }
    }
  }
}

write.csv(good.uniprotIDs.df, file = "good.uniprotIDs.subsets.csv")
