testing.1 <- read.csv('PreMode/ALL.nogeneoverlap/testing.lw.fold.0.4fold.csv')
testing.2 <- read.csv('PreMode/ALL.nogeneoverlap/testing.fold.0.4fold.csv')
source('AUROC.R')
auc.by.uid <- plot.AUC.by.uniprotID(testing.1$score, 
                                    rowMeans(testing.1[,paste0('logits.FOLD.', 0:3)]) + rowMeans(testing.2[,paste0('logits.FOLD.', 0:3)]), 
                                    testing.1$uniprotID)
# nine genes
tasks <- 
  c(read.csv('../scripts/gene.txt', header = F)$V1,
    read.csv('../scripts/gene.itan.txt', header = F)$V1,
    read.csv('../scripts/gene.pfams.txt', header = F)$V1,
    read.csv('../scripts/gene.split.by.pos.txt', header = F)$V1,
    read.csv('../scripts/gene.split.by.pos.itan.txt', header = F)$V1
  )
# create pseudo data folder
dir.create('PreMode.ALL.nogeneoverlap/', showWarnings = F)
source('prepare.biochem.R')
testing <- prepare.unique.id(testing)
# PreMode split first
for (task in tasks) {
  dir.create(paste0('PreMode.ALL.nogeneoverlap/', task), showWarnings = F)
  for (seed in 0:4) {
    ref <- read.csv(paste0('PreMode/', task, '/testing.fold.', seed, '.4fold.csv'))
    ref <- prepare.unique.id(ref)
    ref$logits.FOLD.0 <- (testing.1$logits.FOLD.0[match(ref$unique.id, testing.1$unique.id)] + testing.2$logits.FOLD.0[match(ref$unique.id, testing.2$unique.id)]) / 2
    ref$logits.FOLD.1 <- (testing.1$logits.FOLD.1[match(ref$unique.id, testing.1$unique.id)] + testing.2$logits.FOLD.1[match(ref$unique.id, testing.2$unique.id)]) / 2
    ref$logits.FOLD.2 <- (testing.1$logits.FOLD.2[match(ref$unique.id, testing.1$unique.id)] + testing.2$logits.FOLD.2[match(ref$unique.id, testing.2$unique.id)]) / 2
    ref$logits.FOLD.3 <- (testing.1$logits.FOLD.3[match(ref$unique.id, testing.1$unique.id)] + testing.2$logits.FOLD.3[match(ref$unique.id, testing.2$unique.id)]) / 2
    write.csv(ref, paste0('PreMode.ALL.nogeneoverlap/', task, '/testing.fold.', seed, '.4fold.csv'), row.names = F)
  }
}

testing <- read.csv('PreMode/ALL.nogeneoverlap.large.gene/testing.fold.0.4fold.csv')
auc.by.uid <- plot.AUC.by.uniprotID(testing$score, rowMeans(testing[,paste0('logits.FOLD.', 0:3)]), testing$uniprotID)
# nine genes
tasks <- 
  c(read.csv('../scripts/gene.txt', header = F)$V1,
    read.csv('../scripts/gene.itan.txt', header = F)$V1,
    read.csv('../scripts/gene.pfams.txt', header = F)$V1,
    read.csv('../scripts/gene.split.by.pos.txt', header = F)$V1,
    read.csv('../scripts/gene.split.by.pos.itan.txt', header = F)$V1
  )
# create pseudo data folder
dir.create('PreMode.ALL.nogeneoverlap.large.gene/', showWarnings = F)
source('prepare.biochem.R')
testing <- prepare.unique.id(testing)
# PreMode split first
for (task in tasks) {
  dir.create(paste0('PreMode.ALL.nogeneoverlap.large.gene/', task), showWarnings = F)
  for (seed in 0:4) {
    ref <- read.csv(paste0('PreMode/', task, '/testing.fold.', seed, '.4fold.csv'))
    ref <- prepare.unique.id(ref)
    ref$logits.FOLD.0 <- testing$logits.FOLD.0[match(ref$unique.id, testing$unique.id)]
    ref$logits.FOLD.1 <- testing$logits.FOLD.1[match(ref$unique.id, testing$unique.id)]
    ref$logits.FOLD.2 <- testing$logits.FOLD.2[match(ref$unique.id, testing$unique.id)]
    ref$logits.FOLD.3 <- testing$logits.FOLD.3[match(ref$unique.id, testing$unique.id)]
    write.csv(ref, paste0('PreMode.ALL.nogeneoverlap.large.gene/', task, '/testing.fold.', seed, '.4fold.csv'), row.names = F)
  }
}

testing <- read.csv('PreMode/ALL.itan.only/testing.fold.0.4fold.csv')
auc.by.uid <- plot.AUC.by.uniprotID(testing$score, rowMeans(testing[,paste0('logits.FOLD.', 0:3)]), testing$uniprotID)
# nine genes
tasks <- 
  c(read.csv('../scripts/gene.txt', header = F)$V1,
    read.csv('../scripts/gene.itan.txt', header = F)$V1,
    read.csv('../scripts/gene.pfams.txt', header = F)$V1,
    read.csv('../scripts/gene.split.by.pos.txt', header = F)$V1,
    read.csv('../scripts/gene.split.by.pos.itan.txt', header = F)$V1
  )
# create pseudo data folder
dir.create('PreMode.ALL.itan.only/', showWarnings = F)
source('prepare.biochem.R')
testing <- prepare.unique.id(testing)
# PreMode split first
for (task in tasks) {
  dir.create(paste0('PreMode.ALL.itan.only/', task), showWarnings = F)
  for (seed in 0:4) {
    ref <- read.csv(paste0('PreMode/', task, '/testing.fold.', seed, '.4fold.csv'))
    ref <- prepare.unique.id(ref)
    ref$logits.FOLD.0 <- testing$logits.FOLD.0[match(ref$unique.id, testing$unique.id)]
    ref$logits.FOLD.1 <- testing$logits.FOLD.1[match(ref$unique.id, testing$unique.id)]
    ref$logits.FOLD.2 <- testing$logits.FOLD.2[match(ref$unique.id, testing$unique.id)]
    ref$logits.FOLD.3 <- testing$logits.FOLD.3[match(ref$unique.id, testing$unique.id)]
    write.csv(ref, paste0('PreMode.ALL.itan.only/', task, '/testing.fold.', seed, '.4fold.csv'), row.names = F)
  }
}

testing.2 <- read.csv('PreMode/ALL.itan.nogeneoverlap.large.window/testing.fold.0.4fold.csv')
testing.2 <- testing.2[!is.na(testing.2$itan.gof),]
auc.1 <- plot.AUC.by.uniprotID(testing.2$score, rowMeans(testing.2[,paste0('logits.FOLD.', 0:3)]), 
                      testing.2$uniprotID)
auc.2 <- plot.AUC.by.uniprotID(testing.2$score, testing.2$itan.gof, testing.2$uniprotID)
auc.1$PreMode.auc <- auc.1$auc
auc.2$LoGoFunc.auc <- auc.2$auc
auc.by.uid <- cbind(auc.1, LoGoFunc.auc=auc.2$LoGoFunc.auc)
ggplot(auc.by.uid, aes(y = PreMode.auc, x = LoGoFunc.auc, col = uniprotID)) + geom_point() + theme_minimal() + 
  # add diagnal line
  geom_abline(intercept = 0, slope = 1, linetype = 'dashed') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + xlim(0.5, 0.8) + ylim(0.5, 0.8) +
  labs(title = 'PreMode vs LoGoFunc', y = 'PreMoe', x = 'LoGoFunc') + ggeasy::easy_center_title() +
  scale_fill_manual(values = c('red', 'blue'))
ggsave('figs/fig.5d.cpt.split.pdf', width = 5, height = 4)


test <- read.csv('PreMode/ALL.itan.only/testing.fold.0.4fold.csv')
train <- read.csv('../data.files/ICC.seed.0/ALL.itan.only/training.csv')
test <- test[!is.na(test$itan.gof),]
gof.freq <- as.data.frame(table(test[test$score==1,c('uniprotID', 'score')]))
lof.freq <- as.data.frame(table(test[test$score==-1,c('uniprotID', 'score')]))
auc.1 <- plot.AUC.by.uniprotID(test$score, rowMeans(test[,paste0('logits.FOLD.', 0:3)]), test$uniprotID)
auc.2 <- plot.AUC.by.uniprotID(test$score, 1-test$itan.gof, test$uniprotID)
auc.1 <- auc.1[!is.na(auc.1$auc),]
auc.2 <- auc.2[!is.na(auc.2$auc),]
# only test on genes that have sufficient data
auc.1$PreMode.auc <- auc.1$auc
auc.by.uid <- cbind(auc.1, LoGoFunc.auc=auc.2$auc)
for (i in 1:dim(auc.by.uid)[1]) {
  auc.by.uid$n.gof.train[i] <- sum(train$uniprotID==auc.by.uid$uniprotID[i] & train$score==1)
  auc.by.uid$n.lof.train[i] <- sum(train$uniprotID==auc.by.uid$uniprotID[i] & train$score==-1)
  auc.by.uid$n.lof.test[i] <- sum(test$uniprotID==auc.by.uid$uniprotID[i] & test$score==-1)
  auc.by.uid$n.gof.test[i] <- sum(test$uniprotID==auc.by.uid$uniprotID[i] & test$score==1)
}
auc.by.uid$diff <- auc.by.uid$PreMode.auc - auc.by.uid$LoGoFunc.auc
auc.by.uid <- auc.by.uid[auc.by.uid$n.lof.test >= 3 & auc.by.uid$n.gof.test >= 3,]
auc.by.uid$n.training <- '>4 G/LoF'
auc.by.uid$n.training[auc.by.uid$n.gof.train <= 3 | auc.by.uid$n.lof.train <= 3] <- '≤3 G/LoF'
auc.by.uid$n.training[auc.by.uid$n.gof.train == 0 | auc.by.uid$n.lof.train == 0] <- '0 G/LoF'
# calculated weighted average
auc.by.uid$task.sizes <- auc.by.uid$n.lof.test * auc.by.uid$n.gof.test / (auc.by.uid$n.lof.test + auc.by.uid$n.gof.test)
average.PreMode <- mean(auc.by.uid$PreMode.auc)
average.LoGoFunc <- mean(auc.by.uid$LoGoFunc.auc)
# rename uniprotID to HGNC
uid.to.hgnc <- read.delim('/nfs/scratch/gz2294/Data/Protein/uniprot.ID/swissprot.ID.mapping.tsv')
auc.by.uid$HGNC <- uid.to.hgnc$Gene.Names[match(auc.by.uid$uniprotID, uid.to.hgnc$Entry)]
# remove space and the string after space
auc.by.uid$HGNC <- gsub(' .*', '', auc.by.uid$HGNC)
ggplot(auc.by.uid, aes(y = LoGoFunc.auc, x = HGNC, col = n.training)) + 
  geom_point() + theme_minimal() + 
  # ggrepel::geom_text_repel() +
  # geom_point(x=average.LoGoFunc, y=average.PreMode, col='black') + 
  geom_abline(intercept = 0.5, slope = 0, linetype = 'dashed') + ylim(0.2, 0.9) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  labs(title = 'LoGoFunc on unseen variants', y = 'LoGoFunc AUC', x = 'Protein') +
  ggeasy::easy_center_title() +
  scale_color_manual(values = c('grey', 'blue', 'red'))
ggsave('figs/fig.5d.LoGoFunc.split.pdf', width = 6, height = 4)

