#######################
# caret path vs neut  #
#######################

# read and install libraries if necessary ####
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("data.table")) install.packages("data.table")
if (!require("caret")) install.packages("caret")
if (!require("doParallel")) install.packages("doParallel")
if (!require("mltools")) install.packages("mltools")
library(ggplot2)
library(data.table)
library(caret)
library(doParallel)
library("mltools")

# read files ####
featuretable <- fread("featuretable4github_revision.txt")
varall <- fread("SupplementaryTable_S1_pathvariantsusedintraining_revision2.txt")
prettynames <- fread("pretty_featurenames2.txt")
famcacscn <- as.data.frame(fread("scncacaa_familyalignedCACNA1Acantranscript.txt"))

# read functions ####
source("R_functions4predicting_goflof_CACNA1SCN.R")

# format tables ####

# ... variant table

varall <- varall[used_in_functional_prediction%in%1]
varall <- varall[prd_mech_revised%in%c("lof", "gof")]
# remove duplicate sites:
varall <- varall[!duplicated(varall[,c("gene", "altAA", "pos")])]

# ...feature table 

# remove genomic positions, STRAND/transcript info, whether variant part of polyphen training set (inpp2):
featuretable[,(c("chr", "genomic_pos", "USED_REF", "STRAND","Feature", "inpp2")):=NULL] 
featuretable[,(c(grep("dens", colnames(featuretable)))):=NULL] # remove all variant density features
# rmv most correlated variables (as previously identified with caret preprocessing fcts)
featuretable[,(c("H", "caccon", "SF_DEKA")):=NULL] 
featuretable <- unique(featuretable)

# subset feature table to variants
feat <- featuretable[match(varall$protid, protid)] #, nomatch=0L
feat$Class <- varall$prd_mech_revised
feat <- feat[complete.cases(feat),]
varallmod <- as.data.frame(feat)

# train model ####

outi <- predictgof(varallmod = varallmod, modeltype = "gbm", featuretable = featuretable, alignmentfile = famcacscn)

model1 <- outi[[2]]
out <- outi[[1]]
write.csv(out, file = 'fuNCion.predictions.csv')
# results ####

# results in manuscript
modelperformance(out)
# Balanced_Accuracy              Sens              Spec               AUC(=ROC)   Precision 
#         0.7990196         0.8300000         0.7600000         0.8464052         0.6756757 
# Recall                F1             prAUC             Kappa               MCC 
# 0.8333333         0.7462687         0.7770830         0.5706268         0.5797599 

# plot Feature importance ####

importance_matrix <- base::summary(model1, plot=F)
colnames(importance_matrix) <- c("Feature", "Importance")
importance_matrix$Feature <- gsub("`","", importance_matrix$Feature) # weird formatting bug
importance_matrix <- importance_matrix[importance_matrix$Importance>0.05,]
importance_matrix$Feature <- prettynames[match(importance_matrix$Feature, feature_name)]$feature_name4plot
importance_matrix$Feature <- gsub(", DSSP","", importance_matrix$Feature) # rm DSSP

featimpxgb <- ggplot(importance_matrix, 
                     aes(
  x = factor(Feature, levels = rev(Feature)), 
  y = Importance, width = 0.3) 
  ) + 
  geom_bar(fill ="#00000088", stat = "identity", position = "identity") + 
  ggplot2::coord_flip() + 
  xlab("Features")+ 
  ylab("Relative Influence") + 
  ggtitle("Feature Importance") +
  theme(plot.title = element_text(lineheight = 0.9, 
                      face = "bold"), panel.grid.major.y = element_blank()) + 
  theme_bw()
featimpxgb

 