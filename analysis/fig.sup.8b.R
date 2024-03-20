# visualize with dssp secondary structure 
library(ggplot2)
library(bio3d)
library(patchwork)
library(data.table)
genes <- c("Q99250", "Q14524", "O00555")
gene.names <- c("SCN2A", "SCN5A", "CACNA1A")

# af2.seqs <- read.csv('~/Data/af2_uniprot/swissprot_and_human.csv', row.names = 1)
aa.dict <- c('L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
             'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C')
log.dir <- 'PreMode.inference/'
# auc.dir <- '../scripts/CHPs.v4.esm.dssp.small.StarAttn.MSA.StarPool.1dim/'
# use.logits <- 'meta.logits'
folds <- c(0:4)
# source('~/Pipeline/plot.genes.scores.heatmap.R')
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
    gene.testing.result <- read.csv(paste0(log.dir, gene, '/testing.fold.', fold, '.csv'))
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
    # gene.testing.result <- gene.testing.result[gene.testing.result$unique.id %in% heyne.testing.result$unique.id,]
    premode.auc <- plot.AUC(gene.testing.result$score, gene.testing.result$logits)
    # heyne.auc <- plot.AUC(as.numeric(as.factor(outi[[1]]$obs))-1, as.numeric(outi[[1]]$gof))
    result.plot <- rbind(result.plot, data.frame(AUC=c(premode.auc$auc, heyne.auc), 
                                                 model=c('PreMode', 'FunCion (sklearn)'),
                                                 fold=fold,
                                                 HGNC=paste0(gene.names[o], '\n(5 random splits)')))
  }
}

# add results for all
for (fold in 0) {
  gene.testing.result <- read.csv(paste0(log.dir, 'Heyne/testing.seed.', fold, '.csv'))
  heyne.result <- read.csv('./funNCion/fuNCion.predictions.csv', row.names = 1)
  premode.auc <- plot.AUC(gene.testing.result$score, gene.testing.result$logits)
  heyne.auc <- plot.AUC(as.numeric(as.factor(heyne.result$obs))-1, heyne.result$gof)
  result.plot <- rbind(result.plot, data.frame(AUC=c(premode.auc$auc, heyne.auc$auc, heyne.result$auc[1]), 
                                               model=c('PreMode', 'FunCion (R)', 'FunCion (sklearn)'),
                                               fold=fold,
                                               HGNC='ALL Ion Channels\n(FunCion paper split)'))
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
        text = element_text(size = 13),
        legend.position="bottom", 
        legend.direction="horizontal") +
  ggtitle('PreMode compared to FuNCion method\nin Ion Channel genes') +
  ggeasy::easy_center_title() +
  coord_flip() + guides(col=guide_legend(ncol=3)) +
  ylim(0.5, 1) + xlab('task: Genetics Level Mode of Action') 
ggsave(paste0('figs/fig.sup.8b.pdf'), p, height = 6, width = 6)

