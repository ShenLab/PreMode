
# upsampling function to have same numbers of scn,cac,lof,gof
func = function (x, y) {
  # which of the four class x scn combos is highest? -> upsample all data to that!
  xup <- if (is.data.frame(x)) x else as.data.frame(x)
  xup$Class <- y
  frqtab <- data.frame(table(xup[,c("Class", "scn")])) # frquency table
  frqmax <- frqtab[tail(order(frqtab$Freq),1),] # find highest class x scn combo
  xup <- rbind(xup[xup$scn%in%frqmax$scn & xup$Class%in%frqmax$Class ,],
               xup[sample(rownames(xup[xup$scn%in%frqmax$scn & !xup$Class%in%frqmax$Class ,]), size = frqmax$Freq, replace = T),],
               xup[sample(rownames(xup[!xup$scn%in%frqmax$scn & !xup$Class%in%frqmax$Class ,]), size = frqmax$Freq, replace = T),],
               xup[sample(rownames(xup[!xup$scn%in%frqmax$scn & xup$Class%in%frqmax$Class ,]), size = frqmax$Freq, replace = T),]
  )
  list(x=xup[, !grepl("Class", colnames(xup), fixed = TRUE)], 
       y=xup$Class)
}

samplingfct <- list(name = "upsampling to balance Class and scn!",
                    func = func,
                    first = TRUE)

gene2familyalignment_quant <- function(gene, variants, alignmentfile)
{
  variant <- as.data.frame(table(variants), stringsAsFactors = F)
  variant$variant <- as.integer(variant$variant)
  gene1 <-  alignmentfile[,gene]
  bigfamilyalignment <- rep(0,nrow(alignmentfile))
  bigfamilyalignment[which(gene1!="-")][variant$variant] <- variant$Freq
  return(bigfamilyalignment)
}

ma <- function(x,windowsize){stats::filter(x,rep(1/windowsize,windowsize), circular = T)}

vardens <- function(gene1, funcycat, featuretable, wind, alignmentfile, varonfamilyalignment)
{
  densgof <- apply(as.matrix(varonfamilyalignment[,grepl(funcycat, colnames(varonfamilyalignment))]), 1, sum)
  # map onto gene
  allvarongene <- densgof[!as.data.frame(alignmentfile)[,gene1]%in%"-"]
  # slwind with ALL variants
  slwindall <- ma(x = allvarongene, windowsize = wind)
  slwindall <- slwindall[featuretable[gene%in%gene1]$pos] # adapt to multiple aa per sites
  return(slwindall)
}

# define parameters during training (caret fct)
fitControl <- caret::trainControl(  ## here: k-fold cross validation
  method = "repeatedcv",
  number = 10,
  repeats = 10, 
  sampling = samplingfct,
  classProbs = T # 
)

# output performance
modelperformance <- function(out) {
  res <- c(multiClassSummary(out, lev = c("gof", "lof")),
           # matthews correlation coefficient:
           mcc(preds = ifelse(out$pred%in%"gof", 1, 0), 
               actuals = ifelse(out$obs%in%"gof", 1, 0)),
           round(twoClassSummary(out, lev = c("gof", "lof")), digits = 2) )
  names(res)[15] <- "MCC"
  return(res[c("Balanced_Accuracy", "Sens", "Spec","AUC","Precision","Recall","F1", "prAUC","Kappa", "MCC")])
}


# training fct
predictgof <- function(varallmod, modeltype, alignmentfile, featuretable)
{
  # reproducible random splits
  suppressWarnings(RNGversion("3.5.3"))
  set.seed(999)
  # randomly split in training/testing
  inTraining <- caret::createDataPartition(as.factor(varallmod$Class), p = .9, list = FALSE) 
  trainingall <- varallmod[ inTraining,] # two training sets
  testing <- varallmod[ -inTraining,] # 1 comb and 1 test set
  
  set.seed(989) # separate two training sets, one used for calculating variant densities
  inTraining1 <- caret::createDataPartition((trainingall$Class), p = .5, list = FALSE)
  training1 <- trainingall[inTraining1,]
  training2 <- trainingall[-inTraining1,]
  
  # calculate variant density from training1 and map on training2 ####
  training1 <- training1[,c("gene", "pos","refAA", "altAA", "Class")]
  
  # variants on family alignment
  gofgenes <- unique(training1[training1$Class%in%"gof",]$gene)
  lofgenes <- unique(training1[training1$Class%in%"lof",]$gene)
  
  familyaligned_gof <- c()
  for ( i in gofgenes)
  {
    var1 <- training1[training1$gene%in%i & training1$Class%in%"gof",][,c("pos", "altAA")]
    gof <- gene2familyalignment_quant(gene = i, variants = var1$pos, alignmentfile = famcacscn)
    familyaligned_gof <- cbind(familyaligned_gof, gof)
  }
  familyaligned_lof <- c()
  colnames(familyaligned_gof) <- paste(gofgenes,"GOF", sep = "_")
  for ( i in lofgenes)
  {
    var1 <- training1[training1$gene%in%i & training1$Class%in%"lof",][,c("pos", "altAA")]
    gof <- gene2familyalignment_quant(gene = i, variants = var1$pos, alignmentfile = famcacscn)
    familyaligned_lof <- cbind(familyaligned_lof, gof)
  }
  colnames(familyaligned_lof) <- paste(lofgenes,"LOF", sep = "_")
  familyaligned <- cbind(familyaligned_gof, familyaligned_lof)
  
  # variants on family alignment -> var densitiy -> on individual genes
  uniqgenemech <- unique(featuretable$gene)
  # diff sliding windows 10 AA
  featuretable$densgof <- unlist(sapply(uniqgenemech, function(x){vardens(x, "GOF", featuretable, wind = 10, famcacscn, familyaligned)}))
  featuretable$densgof3aa <- unlist(sapply(uniqgenemech, function(x){vardens(x, "GOF", featuretable, wind = 3, famcacscn, familyaligned)}))
  featuretable$denslof <- unlist(sapply(uniqgenemech, function(x){vardens(x, "LOF", featuretable, wind = 10, famcacscn, familyaligned)}))
  featuretable$denslof3aa <- unlist(sapply(uniqgenemech, function(x){vardens(x, "LOF", featuretable, wind = 3, famcacscn, familyaligned)}))
  
  # zscore and round
  featuretable$densgof <- round(scale(featuretable$densgof), 2) 
  featuretable$densgof3aa <- round(scale(featuretable$densgof3aa),2)
  featuretable$denslof <- round(scale(featuretable$denslof),2)
  featuretable$denslof3aa <- round(scale(featuretable$denslof3aa),2)
  
  # map variant density of training1 onto training2 and testing data
  training2 <- cbind(training2, as.data.frame(featuretable[match(training2$protid, protid)])[,grep("dens", colnames(featuretable))])
  # remove altAA etc
  training <- training2[,!colnames(training2)%in%c(colnames(training1), "protid")]
  training$Class <- training2$Class
  # add vardens onto testing
  testing <- cbind(testing, as.data.frame(featuretable[match(testing$protid, protid)])[,grep("dens", colnames(featuretable))])

    # train ####
  cl <- makePSOCKcluster(5)
  registerDoParallel(cl)
  
  set.seed(999)
  starttime <- as.character(Sys.time())
  print(c("start training at", starttime), quote = F)
#  print()
  gbmFit1_2 <- caret::train(Class ~ ., data = training,
                            method = modeltype,
                            trControl = fitControl,
                            verbose = FALSE)
  starttime <- as.character(Sys.time())
  print(c("finish training at", starttime), quote = F)
  model1 <- gbmFit1_2
  test_data <- testing$Class
  
  # compare with gbm method implemented with sklearn
  write.csv(training, file = 'training.fuNCion.csv')
  write.csv(testing, file = 'testing.fuNCion.csv')
  # gbmFit1_2 <- caret::train(Class ~ ., data = training,
  #                           method = modeltype,
  #                           trControl = fitControl,
  #                           verbose = T)
  res <- system('/share/descartes/Users/gz2294/miniconda3/envs/RESCVE/bin/python /share/pascal/Users/gz2294/Data/DMS/Ion_Channel/funNCion/sklearn.gbm.py training.fuNCion.csv testing.fuNCion.csv',
                intern = T)
  auc <- as.numeric(strsplit(res, '=')[[1]][2])
  out <- data.frame(obs= test_data,
                    gof = predict(model1, newdata = testing, type = "prob")[,"gof"],
                    lof = predict(model1, newdata = testing, type = "prob")[,"lof"],
                    pred = predict(model1, newdata = testing),
                    gene = feat[-inTraining,]$gene,
                    auc = auc
  )
  return(list(out, gbmFit1_2))
  stopCluster(cl)
}



# training fct, modified only the training, testing data split
predictgof_manual_split <- function(trainingall, testing, modeltype, alignmentfile, featuretable)
{
  # reproducible random splits
  suppressWarnings(RNGversion("3.5.3"))
  set.seed(999)
  # randomly split in training/testing
  # inTraining <- caret::createDataPartition(as.factor(varallmod$Class), p = .9, list = FALSE) 
  # trainingall <- varallmod[ inTraining,] # two training sets
  # testing <- varallmod[ -inTraining,] # 1 comb and 1 test set
  
  set.seed(989) # separate two training sets, one used for calculating variant densities
  inTraining1 <- caret::createDataPartition((trainingall$Class), p = .5, list = FALSE)
  training1 <- trainingall[inTraining1,]
  training2 <- trainingall[-inTraining1,]
  
  # calculate variant density from training1 and map on training2 ####
  training1 <- training1[,c("gene", "pos","refAA", "altAA", "Class")]
  
  # variants on family alignment
  gofgenes <- unique(training1[training1$Class%in%"gof",]$gene)
  lofgenes <- unique(training1[training1$Class%in%"lof",]$gene)
  
  familyaligned_gof <- c()
  for ( i in gofgenes)
  {
    var1 <- training1[training1$gene%in%i & training1$Class%in%"gof",][,c("pos", "altAA")]
    gof <- gene2familyalignment_quant(gene = i, variants = var1$pos, alignmentfile = famcacscn)
    familyaligned_gof <- cbind(familyaligned_gof, gof)
  }
  familyaligned_lof <- c()
  colnames(familyaligned_gof) <- paste(gofgenes,"GOF", sep = "_")
  for ( i in lofgenes)
  {
    var1 <- training1[training1$gene%in%i & training1$Class%in%"lof",][,c("pos", "altAA")]
    gof <- gene2familyalignment_quant(gene = i, variants = var1$pos, alignmentfile = famcacscn)
    familyaligned_lof <- cbind(familyaligned_lof, gof)
  }
  colnames(familyaligned_lof) <- paste(lofgenes,"LOF", sep = "_")
  familyaligned <- cbind(familyaligned_gof, familyaligned_lof)
  
  # variants on family alignment -> var densitiy -> on individual genes
  uniqgenemech <- unique(featuretable$gene)
  # diff sliding windows 10 AA
  featuretable$densgof <- unlist(sapply(uniqgenemech, function(x){vardens(x, "GOF", featuretable, wind = 10, famcacscn, familyaligned)}))
  featuretable$densgof3aa <- unlist(sapply(uniqgenemech, function(x){vardens(x, "GOF", featuretable, wind = 3, famcacscn, familyaligned)}))
  featuretable$denslof <- unlist(sapply(uniqgenemech, function(x){vardens(x, "LOF", featuretable, wind = 10, famcacscn, familyaligned)}))
  featuretable$denslof3aa <- unlist(sapply(uniqgenemech, function(x){vardens(x, "LOF", featuretable, wind = 3, famcacscn, familyaligned)}))
  
  # zscore and round
  featuretable$densgof <- round(scale(featuretable$densgof), 2) 
  featuretable$densgof3aa <- round(scale(featuretable$densgof3aa),2)
  featuretable$denslof <- round(scale(featuretable$denslof),2)
  featuretable$denslof3aa <- round(scale(featuretable$denslof3aa),2)
  
  # map variant density of training1 onto training2 and testing data
  training2 <- cbind(training2, as.data.frame(featuretable[match(training2$protid, protid)])[,grep("dens", colnames(featuretable))])
  # remove altAA etc
  # training <- training2[,!colnames(training2)%in%c(colnames(training1), "protid")]
  # previous code didn't work
  training <- training2
  for (co in c(colnames(training1), "protid")) {
    training[,co] <- NULL
  }
  training$Class <- training2$Class
  
  # add vardens onto testing
  testing <- cbind(testing, as.data.frame(featuretable[match(testing$protid, protid)])[,grep("dens", colnames(featuretable))])
  
  # train ####
  # cl <- makePSOCKcluster(5)
  # registerDoParallel(cl)
  
  set.seed(999)
  starttime <- as.character(Sys.time())
  print(c("start training at", starttime), quote = F)
  #  print()
  # write to csv as the training program didn't work
  write.csv(training, file = 'training.fuNCion.csv')
  write.csv(testing, file = 'testing.fuNCion.csv')
  # gbmFit1_2 <- caret::train(Class ~ ., data = training,
  #                           method = modeltype,
  #                           trControl = fitControl,
  #                           verbose = T)
  res <- system('/share/vault/Users/gz2294/miniconda3/envs/RESCVE/bin/python /share/vault/Users/gz2294/PreMode.ShenLab.git/analysis/funNCion/sklearn.gbm.py training.fuNCion.csv testing.fuNCion.csv',
                intern = T)
  starttime <- as.character(Sys.time())
  print(c("finish training at", starttime), quote = F)
  # model1 <- gbmFit1_2
  # test_data <- testing$Class
  # 
  # out <- data.frame(obs= test_data,
  #                   gof = predict(model1, newdata = testing, type = "prob")[,"gof"],
  #                   lof = predict(model1, newdata = testing, type = "prob")[,"lof"],
  #                   pred= predict(model1, newdata = testing)
  #                   ,gene=testing$gene
  # )
  # return(list(out, gbmFit1_2))
  # stopCluster(cl)
  auc <- as.numeric(strsplit(res, '=')[[1]][2])
  auc
}
