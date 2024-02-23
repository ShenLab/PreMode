library(ggplot2)
pfam.list <- read.csv('../scripts/pfams.txt', header = F)
pfam.even.split <- pfam.list$V1[grep('even.uniprotID', pfam.list$V1)]
pfam.random.split <- gsub('.even.uniprotID', '', pfam.even.split)
pfam.random.split <- gsub('.chps', '', pfam.random.split)
result.df <- data.frame(task.id = rep(c(pfam.random.split, pfam.even.split), 5),
                        task.type = rep(c(rep('random.split', length(pfam.random.split)),
                                      rep('even.split', length(pfam.even.split))), 5),
                        seed = rep(0:4, each=length(pfam.random.split) + length(pfam.even.split)))
result.df$rf.auc <- NA
for (i in 1:dim(result.df)) {
  rf.command <- paste0('python ', 'random.forest.process.classifier.py ', 
                       '/share/pascal/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', result.df$seed[i], '/', result.df$task.id[i], '/training.csv ',
                       '/share/pascal/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', result.df$seed[i], '/', result.df$task.id[i], '/testing.csv')
  rf.result <- system(rf.command, intern = T)
  testing.points <- read.csv(paste0('/share/pascal/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', result.df$seed[i], '/', result.df$task.id[i], '/testing.csv'))
  training.points <- read.csv(paste0('/share/pascal/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', result.df$seed[i], '/', result.df$task.id[i], '/training.csv'))
  result.df$rf.auc[i] <- as.numeric(gsub('Testing AUC: ', '', rf.result))
  result.df$gof[i] <- sum(training.points$score==1) + sum(testing.points$score==1)
  result.df$lof[i] <- sum(training.points$score==-1) + sum(testing.points$score==-1)
}
pfam.random.split[1] <- "Pfam: Na+/Ca2+ Channel"
pfam.random.split[grep(pattern = 'PF', pfam.random.split)] <- paste0("Pfam: ", pfam.random.split[grep(pattern = 'PF', pfam.random.split)])
pfam.random.split[grep(pattern = 'IPR', pfam.random.split)] <- paste0("Domain: ", pfam.random.split[grep(pattern = 'IPR', pfam.random.split)])
result.df$task.name <- rep(pfam.random.split, 10)
for (i in unique(result.df$task.name)) {
  result.df$gof[result.df$task.name==i] <- max(result.df$gof[result.df$task.name==i])
  result.df$lof[result.df$task.name==i] <- max(result.df$lof[result.df$task.name==i])
}
result.df$task.name <- paste0(result.df$task.name, ": (", result.df$gof, "/", result.df$lof, ")")
num.models <- 2
p <- ggplot(result.df, aes(x=task.name, y=rf.auc, col=task.type)) +
  geom_point(alpha=0.2) +
  stat_summary(data = result.df,
               aes(x=as.numeric(factor(task.name))+0.4*(as.numeric(factor(task.type)))/num.models-0.2*(num.models+1)/num.models,
                   y = rf.auc, col=task.type), 
               fun.data = mean_se, geom = "errorbar", width = 0.2) +
  stat_summary(data = result.df, 
               aes(x=as.numeric(factor(task.name))+0.4*(as.numeric(factor(task.type)))/num.models-0.2*(num.models+1)/num.models,
                   y = rf.auc, col=task.type), 
               fun.data = mean_se, geom = "point") +
  xlab("Task Name: (GoF/LoF)") + ylab("random forest classifier AUC") + theme_bw() +
  theme(axis.text.x = element_text(angle=70, vjust = 1, hjust = 1), 
        legend.position="bottom", 
        legend.direction="horizontal")
ggsave(p, filename = "figs/01.03.random.forest.AUROC.pdf", height = 6, width = 10)
source('/share/pascal/Users/gz2294/Pipeline/AUROC.R')

# next test whole genome-split and within in Gene AUC
icc <- read.csv('figs/ALL.csv', row.names = 1)
source('~/Pipeline/bind_rows.R')
icc$score[icc$score == 0] <- -1
icc$unique.id <- paste(icc$uniprotID, icc$ref, icc$pos.orig, icc$alt, sep = ":")
pfam.list <- read.csv('../scripts/pfams.txt', header = F)
pfam.even.split <- pfam.list$V1[grep('even.uniprotID', pfam.list$V1, invert = T)]
pfam.even.split <- pfam.even.split[grep('split', pfam.even.split, invert = T)]


result.df <- data.frame()
for (seed in 0:4) {
  icc.test <- data.frame()
  icc.train <- icc[!icc$uniprotID %in% pfam.even.split,]
  for (uid in pfam.even.split) {
    icc.test <- my.bind.rows(icc.test, read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', 
                                                         seed, '/', uid, '/testing.csv')))
    icc.train <- my.bind.rows(icc.train, read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', 
                                                         seed, '/', uid, '/training.csv')))
    
  }
  train.tmp <- tempfile()
  test.tmp <- tempfile()
  write.csv(icc.train[,c('uniprotID', 'score')], train.tmp)
  write.csv(icc.test[,c('uniprotID', 'score')], test.tmp)
  
  rf.command <- paste0('/share/descartes/Users/gz2294/miniconda3/envs/r4-base/bin/python ', 'random.forest.process.classifier.py ', 
                       train.tmp, ' ',
                       test.tmp)
  rf.result <- system(rf.command, intern = T)
  # testing.points <- read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', seed, '/', uid, '/testing.csv'))
  # training.points <- read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', seed, '/', uid, '/training.csv'))
  result.df <- rbind(result.df, data.frame(uid = 'ALL', 
                                           rf.auc = as.numeric(gsub('Testing AUC: ', '', rf.result)),
                                           seed = seed,
                                           gof = sum(icc.train$score==1) + sum(icc.test$score==1),
                                           lof = sum(icc.train$score==-1) + sum(icc.test$score==-1)))
}
for (seed in 1:4) {
  for (uid in pfam.even.split) {
    icc.train <- icc[!icc$uniprotID %in% pfam.even.split,]
    icc.test <- read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', 
                                          seed, '/', uid, '/testing.csv'))
    icc.test$unique.id <- paste(icc.test$uniprotID, icc.test$ref, icc.test$pos.orig, icc.test$alt, sep = ":")
    icc.train <- icc[!icc$unique.id %in% icc.test,]
    train.tmp <- tempfile()
    test.tmp <- tempfile()
    write.csv(icc.train[,c('uniprotID', 'score')], train.tmp)
    write.csv(icc.test[,c('uniprotID', 'score')], test.tmp)
    rf.command <- paste0('/share/descartes/Users/gz2294/miniconda3/envs/r4-base/bin/python ', 'random.forest.process.classifier.py ',
                         train.tmp, ' ',
                         test.tmp)
    rf.result <- system(rf.command, intern = T)
    testing.points <- read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', seed, '/', uid, '/testing.csv'))
    training.points <- read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8.seed.', seed, '/', uid, '/training.csv'))
    result.df <- rbind(result.df, data.frame(uid = uid, 
                                             rf.auc = 0.5,
                                             seed = seed,
                                             gof = sum(icc.test$score==1),
                                             lof = sum(icc.test$score==-1)))
  }
}
write.csv(result.df, 'figs/01.03.random.forest.single.genes.csv')
uniprotID.dic <- c("P21802"="FGFR2", "P15056"="BRAF", "P07949"="RET", "P04637"="TP53", 
                   "Q09428"="ABCC8",
                   "O00555"="CACNA1A", "Q14654"="KCNJ11", 
                   "Q99250"="SCN2A", "Q14524"="SCN5A", 
                   # "IonChannel.split.uniprotID"="Na+/Ca2+ Channel",
                   "IonChannel.chps"="Na+/Ca2+ Channel",
                   "IonChannel"="Na+/Ca2+ Channel",
                   "IPR000719"="Protein Kinase Domain",
                   "IPR001806"="Small GTPase",
                   "IPR001245"="Protein Kinase Catalytic Domain",
                   "IPR016248"="Fibroblast Growth Factor Receptor Family",
                   "IPR005821"="Ion Transport Domain",
                   "IPR027359"="Voltage-dependent Channel Domain"
)
result.df$uid[result.df$uid == 'ALL'] <- '0: ALL genes'
result.df$uid[result.df$uid %in% names(uniprotID.dic)] <- uniprotID.dic[result.df$uid[result.df$uid %in% names(uniprotID.dic)]]
num.models <- 1
p <- ggplot(result.df, aes(x=uid, y=rf.auc)) +
  geom_point(alpha=0.2) +
  stat_summary(data = result.df,
               aes(x=as.numeric(factor(uid))+0.4/num.models-0.2*(num.models+1)/num.models,
                   y = rf.auc), 
               fun.data = mean_se, geom = "errorbar", width = 0.2) +
  stat_summary(data = result.df, 
               aes(x=as.numeric(factor(uid))+0.4/num.models-0.2*(num.models+1)/num.models,
                   y = rf.auc), 
               fun.data = mean_se, geom = "point") +
  xlab("Task Name") + ylab("random forest classifier AUC") + theme_bw() +
  ggtitle('Random Forest Classifier trained on all genes with gene names') + ggeasy::easy_center_title() +
  theme(axis.text.x = element_text(angle=70, vjust = 1, hjust = 1), 
        legend.position="bottom", 
        legend.direction="horizontal")
ggsave(p, filename = "figs/01.03.random.forest.single.genes.pdf", height = 4, width = 6)

