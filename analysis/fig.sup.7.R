library(ggplot2)
source('./AUROC.R')
source('./bind_rows.R')
# next test whole genome-split and within in Gene AUC
icc <- read.csv('figs/ALL.csv', row.names = 1)
icc$score[icc$score == 0] <- -1
icc$unique.id <- paste(icc$uniprotID, icc$ref, icc$pos.orig, icc$alt, sep = ":")
pfam.list <- read.csv('../scripts/gene.txt', header = F)
pfam.even.split <- pfam.list$V1[grep('even.uniprotID', pfam.list$V1, invert = T)]
pfam.even.split <- pfam.even.split[grep('Heyne', pfam.even.split, invert = T)]
result.df <- data.frame()
for (seed in 0:4) {
  icc.test <- data.frame()
  icc.train <- icc[!icc$uniprotID %in% pfam.even.split,]
  for (uid in pfam.even.split) {
    icc.test <- my.bind.rows(icc.test, read.csv(paste0('../data.files/ICC.seed.', 
                                                         seed, '/', uid, '/testing.csv')))
    icc.train <- my.bind.rows(icc.train, read.csv(paste0('../data.files/ICC.seed.', 
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
  result.df <- rbind(result.df, data.frame(uid = 'ALL', 
                                           rf.auc = as.numeric(gsub('Testing AUC: ', '', rf.result)),
                                           seed = seed,
                                           gof = sum(icc.train$score==1) + sum(icc.test$score==1),
                                           lof = sum(icc.train$score==-1) + sum(icc.test$score==-1)))
}
for (seed in 1:4) {
  for (uid in pfam.even.split) {
    icc.train <- icc[!icc$uniprotID %in% pfam.even.split,]
    icc.test <- read.csv(paste0('../data.files/ICC.seed.', seed, '/', uid, '/testing.csv'))
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
    testing.points <- read.csv(paste0('../data.files/ICC.seed.', seed, '/', uid, '/testing.csv'))
    training.points <- read.csv(paste0('../data.files/ICC.seed.', seed, '/', uid, '/training.csv'))
    result.df <- rbind(result.df, data.frame(uid = uid, 
                                             rf.auc = 0.5,
                                             seed = seed,
                                             gof = sum(icc.test$score==1),
                                             lof = sum(icc.test$score==-1)))
  }
}
uniprotID.dic <- c("P21802"="FGFR2", "P15056"="BRAF", "P07949"="RET", "P04637"="TP53", 
                   "Q09428"="ABCC8",
                   "O00555"="CACNA1A", "Q14654"="KCNJ11", 
                   "Q99250"="SCN2A", "Q14524"="SCN5A", 
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
ggsave(p, filename = "figs/fig.sup.7.pdf", height = 4, width = 6)

