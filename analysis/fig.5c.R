# visualize with dssp secondary structure 
library(ggplot2)
library(bio3d)
library(data.table)
library(patchwork)
genes <- c("Q99250", "Q14524.clean", "O00555")
gene.names <- c("SCN2A", "SCN5A", "CACNA1A")
aa.dict <- c('L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
             'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C')
log.dir <- 'PreMode/'
folds <- c(0:4)
source('./AUROC.R')
# prepare heyne feature table
famcacscn <- as.data.frame(fread("./funNCion/scncacaa_familyalignedCACNA1Acantranscript.txt"))
featuretable <- fread("./funNCion/featuretable4github_revision.txt")
featuretable[,(c("chr", "genomic_pos", "USED_REF", "STRAND","Feature", "inpp2")):=NULL] 
featuretable[,(c(grep("dens", colnames(featuretable)))):=NULL] # remove all variant density features
# rmv most correlated variables (as previously identified with caret preprocessing fcts)
featuretable[,(c("H", "caccon", "SF_DEKA")):=NULL] 
featuretable <- unique(featuretable)
# get heyne training variants
varall <- fread("./funNCion/SupplementaryTable_S1_pathvariantsusedintraining_revision2.txt")
varall <- varall[used_in_functional_prediction%in%1]
varall <- varall[prd_mech_revised%in%c("lof", "gof")]
# remove duplicate sites:
varall <- varall[!duplicated(varall[,c("gene", "altAA", "pos")])]
source("./funNCion/R_functions4predicting_goflof_CACNA1SCN.R")
# for three genes, first only visualize seed 0
result.plot <- data.frame()
for (o in 1:length(genes)) {
  for (fold in 0:4) {
    gene <- genes[o]
    print(gene)
    premode.yaml <- yaml::read_yaml(paste0('../scripts/PreMode/', 
                                           gene, '.5fold/', gene, '.fold.', fold, '.yaml'))
    gene.training <- read.csv(paste0('../', premode.yaml$data_file_train), row.names = 1)
    # compare with large window and select by auc
    gene.training.result <- read.csv(paste0(log.dir, gene, '/training.fold.', fold, '.4fold.csv'))
    gene.training.lw.result <- read.csv(paste0(log.dir, gene, '.large.window/training.fold.', fold, '.4fold.csv'))
    tr.auc <- plot.AUC(gene.training.result$score, rowMeans(gene.training.result[,paste0('logits.FOLD.', 0:3)]))$auc
    tr.lw.auc <- plot.AUC(gene.training.lw.result$score, rowMeans(gene.training.lw.result[,paste0('logits.FOLD.', 0:3)]))$auc
    if (tr.lw.auc > tr.auc) {
      gene.testing.result <- read.csv(paste0(log.dir, gene, '.large.window/testing.fold.', fold, '.4fold.csv'))
    } else {
      gene.testing.result <- read.csv(paste0(log.dir, gene, '/testing.fold.', fold, '.4fold.csv'))
    }
    # heyne training
    gene.training$protid <- paste(gene.names[o], gene.training$pos.orig, gene.training$ref, gene.training$alt, sep = ":")
    gene.testing.result$protid <- paste(gene.names[o], gene.testing.result$pos.orig, gene.testing.result$ref, gene.testing.result$alt, sep = ":")
    varall.protid <- varall$protid[varall$protid %in% gene.training$protid]
    # load heyne feature mat
    feat.train <- featuretable[match(varall.protid, protid)] #, nomatch=0L
    feat.train$Class <- varall$prd_mech_revised[match(varall.protid, varall$protid)]
    feat.train <- feat.train[complete.cases(feat.train),]
    feat.test <- featuretable[match(gene.testing.result$protid, protid)] #, nomatch=0L
    feat.test$Class <- 'gof'
    feat.test$Class[gene.testing.result$score==-1] <- 'lof'
    feat.test <- feat.test[complete.cases(feat.test),]
    
    heyne.auc <- predictgof_manual_split(trainingall = feat.train, testing=feat.test, modeltype = "gbm", featuretable = featuretable, alignmentfile = famcacscn)
    heyne.auc <- max(heyne.auc, 1-heyne.auc)
    premode.auc <- plot.AUC(gene.testing.result$score, rowMeans(gene.testing.result[,paste0('logits.FOLD.', 0:3)]))
    result.plot <- rbind(result.plot, data.frame(AUC=c(premode.auc$auc, heyne.auc), 
                                                 model=c('PreMode', 'funNCion (sklearn)'),
                                                 fold=fold,
                                                 HGNC=paste0(gene.names[o], '\n(5 random splits)')))
  }
}

# add results for all
for (fold in 0:4) {
  # compare with large window and select by auc
  gene.training.result <- read.csv(paste0(log.dir, 'Heyne/training.seed.', fold, '.csv'))
  gene.training.lw.result <- read.csv(paste0(log.dir, 'Heyne/training.large.window.seed.', fold, '.csv'))
  tr.auc <- plot.AUC(gene.training.result$score, rowMeans(gene.training.result[,paste0('logits.FOLD.', 0:3)]))$auc
  tr.lw.auc <- plot.AUC(gene.training.lw.result$score, rowMeans(gene.training.lw.result[,paste0('logits.FOLD.', 0:3)]))$auc
  if (tr.lw.auc > tr.auc) {
    gene.testing.result <- read.csv(paste0(log.dir, 'Heyne/testing.seed.', fold, '.csv'))
  } else {
    gene.testing.result <- read.csv(paste0(log.dir, 'Heyne/testing.large.window.seed.', fold, '.csv'))
  }
  premode.auc <- plot.AUC(gene.testing.result$score, rowMeans(gene.testing.result[,paste0('logits.FOLD.', 0:3)]))
  heyne.result <- read.csv('./funNCion/fuNCion.predictions.csv', row.names = 1)
  heyne.auc <- plot.AUC(as.numeric(as.factor(heyne.result$obs))-1, heyne.result$gof)
  result.plot <- rbind(result.plot, data.frame(AUC=c(premode.auc$auc, heyne.auc$auc, heyne.result$auc[1]),
                                               model=c('PreMode', 'funNCion (R)', 'funNCion (sklearn)'),
                                               fold=fold,
                                               HGNC='ALL Ion Channels\n(funNCion paper split)'))
}

num.models <- length(unique(result.plot$model))
p <- ggplot(result.plot, aes(y=AUC, x=HGNC, col=model)) +
  geom_point(alpha=0) +
  scale_color_manual(values = c("#A3A500", "#00BA38", "#F8766D")) + 
  stat_summary(data = result.plot,
               aes(x=as.numeric(factor(HGNC))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = AUC, col=model), 
               fun.data = mean_se, geom = "errorbar", width = 0.2) +
  stat_summary(data = result.plot, 
               aes(x=as.numeric(factor(HGNC))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = AUC, col=model), 
               fun.data = mean_se, geom = "point") +
  labs(x = "HGNC", y = "AUC", fill = "model") +
  theme_bw() + 
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        text = element_text(size = 16),
        plot.title = element_text(size=15),
        legend.position="bottom", 
        legend.direction="horizontal") +
  ggtitle('PreMode compared to funNCion\nin Ion Channel genes') +
  ggeasy::easy_center_title() +
  coord_flip() + guides(col=guide_legend(ncol=2)) +
  ylim(0.5, 1) + xlab('task: Genetic Level Mode of Action') 
ggsave(paste0('figs/fig.5c.pdf'), p, height = 5, width = 6)

