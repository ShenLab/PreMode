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
varall <- fread("SupplementaryTable_S1_pathvariantsusedintraining_revision.txt")
prettynames <- fread("pretty_featurenames2.txt")
gnomadnoneur2 <- fread("SupplTables/S2_neutralvariantsusedintraining3.txt")

# read functions ####
source("R_functions4predicting_path_GOFLOF_CACNA1SCN.r")

# format tables ####

# ...pathogenic variant table
varall <- varall[used_in_pathogenicity_prediction%in%1]
varall <- varall[!duplicated(varall[,c("gene", "altAA", "pos")])] # remove duplicate sites:

# ...neutral variant table
gnomadnoneur <- gnomadnoneur2[used_in_pathogenicity_prediction==1]

# ...feature table 

# remove features

# rmv most correlated variables (as previously identified with caret preprocessing fcts)
featuretable[,(c("H", "caccon", "SF_DEKA")):=NULL] 
# rmv protein/genomic positions, STRAND/transcript info, whether variant part of polyphen training set (inpp2):
featuretable[,(c("chr", "genomic_pos", "USED_REF", "alt", "pos","refAA", "altAA", "STRAND","Feature", "inpp2")):=NULL] 
featuretable <- unique(featuretable)

# subset feature table to variants
# gnomad variants
featgn <- featuretable[match(gnomadnoneur$protid, protid)]
featgn$Class <- "neutral"
# pathogenic variants
feat <- featuretable[match(varall$protid, protid)]
feat$Class <- "pathogenic"

# remove genes with only neutral (no pathogenic) variants
featgn <- featgn[gene%in%feat$gene,]
# combine neutral + path variants
feat <- rbind(feat, featgn)
feat <- feat[complete.cases(feat),]
varallmod <- copy(feat) 

# train model ####

outi <- predictpath(varallmod = varallmod, modeltype = "gbm")

model1 <- outi[[2]]
out <- outi[[1]]

# results ####

# "original"
modelperformance(out)
# Balanced_Accuracy              Sens              Spec               AUC         Precision            Recall 
#         0.8963905         0.9000000         0.8900000         0.9530308         0.9437229         0.8897959 
# prAUC             Kappa               MCC 
# 0.9385336         0.7744518         0.7768762 

# predicting also variants in genes with only neutral variants ####

outbalancgenes <- out

# remake featgn obj with genes with no pathogenic variants
featgn <- featuretable[match(gnomadnoneur$protid, protid)]
testing  <-  featgn[!gene%in%varallmod$gene]
out <- data.frame(obs= "neutral",
                  neutral = predict(model1, newdata = testing, type = "prob")[,"neutral"],
                  pathogenic = predict(model1, newdata = testing, type = "prob")[,"pathogenic"],
                  pred= predict(model1, newdata = testing), 
                  gene=testing$gene,
                  protid=testing$protid
                  
)
out$obs <- factor(out$obs, levels = unique(out$pred))
outneutgeens <- out
# 
vartable <- rbind(outbalancgenes, outneutgeens)
table(vartable[,c("obs", "pred")])
#                           pred
# obs          neutral pathogenic
# neutral       1518        193
# pathogenic      13        121

modelperformance(vartable)
# Balanced_Accuracy              Sens              Spec               AUC         Precision            Recall 
#         0.8950928         0.9000000         0.8900000         0.9506268         0.9915088         0.8872005 
# prAUC             Kappa               MCC 
# 0.8241656         0.4880578         0.5457001 

modelperformance(vartable[grep("SCN",vartable$gene),])
# Balanced_Accuracy              Sens              Spec               AUC         Precision            Recall 
#         0.8580825         0.9000000         0.8200000         0.9228044         0.9600000         0.8205128 
# prAUC             Kappa               MCC 
# 0.8736877         0.6232456         0.6447642 

modelperformance(vartable[grep("CAC",vartable$gene),])
# Balanced_Accuracy              Sens              Spec               AUC         Precision            Recall 
#         0.9258901         0.9500000         0.9000000         0.9691563         0.9991877         0.9044118 
# prAUC             Kappa               MCC 
# 0.6417708         0.1959327         0.3207849 

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

 