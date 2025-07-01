
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

# define parameters during training (caret fct)
fitControl <- caret::trainControl(  ## here: k-fold cross validation
  method = "repeatedcv",
  number = 10,
  repeats = 2, 
  sampling = samplingfct,
  classProbs = T # 
)

# output performance
modelperformance <- function(out) {
  res <- c(multiClassSummary(out, lev = c("pathogenic", "neutral")),
           # matthews correlation coefficient:
           mcc(preds = ifelse(out$pred%in%"pathogenic", 1, 0), 
               actuals = ifelse(out$obs%in%"pathogenic", 1, 0)),
           round(twoClassSummary(out, lev = c("pathogenic", "neutral")), digits = 2) )
  names(res)[15] <- "MCC"
  return(res[c("Balanced_Accuracy", "Sens", "Spec","AUC","Precision","Recall","F1", "prAUC","Kappa", "MCC")])
}

# training fct
predictpath <- function(varallmod, modeltype)
{
  # reproducible random splits
  suppressWarnings(RNGversion("3.5.3"))
  set.seed(999)
  # randomly split in training/testing
  inTraining <- createDataPartition(as.factor(varallmod$gene), p = .9, list = FALSE)
  training <- varallmod[ inTraining,]
  # 1) upsample to same n of variants per gene, separate in pathogenic and neutral variants
  training$gene <- as.factor(training$gene)
  uptrainpath <- upSample(training[Class%in%"pathogenic"], training[Class%in%"pathogenic"]$gene, yname = "gene1")
  uptrainneut <- upSample(training[Class%in%"neutral"], training[Class%in%"neutral"]$gene, yname = "gene1")
  uptrain <- rbind(uptrainpath, uptrainneut)
  uptrain <- uptrain[,!colnames(uptrain)%in%c("gene", "gene1","protid")]
  training <- uptrain
  testing  <- as.data.frame(varallmod)[-inTraining,]
  
  # train ####
  cl <- makePSOCKcluster(5)
  registerDoParallel(cl)
  
  set.seed(825)
  starttime <- as.character(Sys.time())
  print(c("start training at", starttime), quote = F)
  gbmFit1_2 <- train(Class ~ ., data = training, 
                     method = modeltype, 
                     trControl = fitControl,
                     verbose = FALSE)
  starttime <- as.character(Sys.time())
  print(c("finish training at", starttime), quote = F)
  model1 <- gbmFit1_2
  test_data <- testing$Class
  out <- data.frame(obs= test_data,
                    neutral = predict(model1, newdata = testing, type = "prob")[,"neutral"],
                    pathogenic = predict(model1, newdata = testing, type = "prob")[,"pathogenic"],
                    pred= predict(model1, newdata = testing)
                    ,gene=testing$gene
                    ,protid=testing$protid
  )
  return(list(out, gbmFit1_2))
  stopCluster(cl)
}
