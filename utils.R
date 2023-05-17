get.auc.by.epoch <- function(configs) {
  log.dir <- configs$log_dir
  data.train <- as.numeric(strsplit(system(paste0("wc -l ", configs$data_file_train), intern = T), split = " ")[[1]][1])
  num_saved_batches <- floor(ceiling(data.train * configs$train_size / configs$ngpus / configs$batch_size)
                             * configs$num_epochs / configs$num_save_batches) + 1
  epochs <- c(1:(configs$num_epochs))
  source('/share/terra/Users/gz2294/Pipeline/AUROC.R')
  library(doParallel)
  cl <- makeCluster(72)
  registerDoParallel(cl)
  
  res <- foreach (i = 1:length(epochs), .combine=rbind) %dopar% {
    i <- epochs[i]
    source('~/Pipeline/AUROC.R')
    if (file.exists(paste0(log.dir, 'test_result.epoch.', i, '.csv'))) {
      test.result <- read.csv(paste0(log.dir, 'test_result.epoch.', i, '.csv'))
      if ('y.0' %in% colnames(test.result)) {
        test.result <- test.result[!is.na(test.result$y.0),]
        test.logits <- test.result$y.0
      } else {
        test.result <- test.result[!is.na(test.result$y),]
        test.logits <- test.result$y
      }
      result <- plot.AUC(test.result$score, test.logits,
                         paste0(log.dir, 'test_result.epoch.', i, '.pdf'))
      val_losses <- c()
      train_losses <- c()
      if (configs$ngpus > 1) {
        for (rank in 0:(configs$ngpus-1)) {
          val_dic <- jsonlite::read_json(paste0(log.dir, 'result_dict.epoch.', i-1, '.ddp_rank.', rank, '.json'))
          val_losses <- c(val_losses, val_dic$val_loss)
          train_losses <- c(train_losses, val_dic$train_loss)
        }
      } else {
        rank <- configs$gpu_id
        if (is.null(rank)) {
          rank <- 0
        }
        val_dic <- jsonlite::read_json(paste0(log.dir, 'result_dict.epoch.', i-1, '.ddp_rank.', rank, '.json'))
        val_losses <- c(val_losses, val_dic$val_loss)
        train_losses <- c(train_losses, val_dic$train_loss)
      }
      J_stats <- result$curve[,2] - result$curve[,1]
      optimal.cutoff <-  result$curve[which(J_stats==max(J_stats))[1],3]
      res <- data.frame(train=mean(train_losses),
                        val=mean(val_losses),
                        aucs=result$auc,
                        optimal.cutoffs=optimal.cutoff)
    } else {
      res <- data.frame(train=NA, val=NA, aucs=NA, optimal.cutoffs=NA)
    }
    res
  }
  stopCluster(cl)
  res$epochs <- epochs
  res <- res[!is.na(res$train),]
  val <- res$val
  aucs <- res$aucs
  optimal.cutoffs <- res$optimal.cutoffs
  epochs <- res$epochs
  train <- res$train
  to.plot <- data.frame(epoch=rep(epochs, 2),
                        loss=c(train, val),
                        auc=rep(aucs, 2),
                        metric_name=c(rep("train_loss", length(epochs)),
                                      rep("val_loss", length(epochs))))
  library(ggplot2)
  p <- ggplot(to.plot, aes(x=epoch)) +
    geom_line(aes(y=loss, col=metric_name)) +
    geom_line(aes(y=auc)) +
    scale_y_continuous(
      # Features of the first axis
      name = "Loss",
      breaks = seq(0, max(1.1, max(to.plot$loss)), by = 0.05), limits = c(0, max(1.1, max(to.plot$loss))),
      # Add a second axis and specify its features
      sec.axis = sec_axis(~ . , name="AUC",
                          breaks = seq(0, max(1.1, max(to.plot$loss)), by = 0.05))
    ) +
    scale_x_continuous(breaks =
                         seq(1*configs$num_save_batches,
                             (num_saved_batches - 1)*configs$num_save_batches,
                             by = configs$num_save_batches)) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
  ggsave('Loss.AUC.by.epoch.pdf', p, width = 9, height = 6)
  
  print(paste0("min val epoch (", epochs[which(val==min(val))[1]],
               ") AUC: ", round(aucs[which(val==min(val))[1]], digits = 2),
               " Optimal cutoff: ",
               round(optimal.cutoffs[which(val==min(val))[1]], digits = 2)))
  print(paste0("end epoch (", epochs[length(val)],
               ") AUC: ", round(aucs[length(aucs)], digits = 2),
               " Optimal cutoff: ",
               round(optimal.cutoffs[length(aucs)], digits = 2)))
  print(paste0("max AUC epoch (", epochs[which(aucs==max(aucs))[1]],
               "): ", round(max(aucs), digits = 2),
               " Optimal cutoff: ",
               round(optimal.cutoffs[which(aucs==max(aucs))[1]], digits = 2)))
  res
}

get.auc.by.step <- function(configs) {
  log.dir <- configs$log_dir
  data.train <- as.numeric(strsplit(system(paste0("wc -l ", configs$data_file_train), intern = T), split = " ")[[1]][1])
  num_saved_batches = floor(ceiling(data.train * configs$train_size / configs$ngpus / configs$batch_size)
                            * configs$num_epochs / configs$num_save_batches) + 1
  steps <- c(1:(num_saved_batches-1))*configs$num_save_batches
  source('/share/terra/Users/gz2294/Pipeline/AUROC.R')
  library(doParallel)
  cl <- makeCluster(72)
  registerDoParallel(cl)
  res <- foreach (i = 1:length(steps), .combine=rbind) %dopar% {
    source('~/Pipeline/AUROC.R')
    i <- steps[i]
    if (file.exists(paste0(log.dir, 'test_result.step.', i, '.csv'))) {
      test.result <- read.csv(paste0(log.dir, 'test_result.step.', i, '.csv'))
      if ('y.0' %in% colnames(test.result)) {
        test.result <- test.result[!is.na(test.result$y.0),]
        test.logits <- test.result$y.0
      } else {
        test.result <- test.result[!is.na(test.result$y),]
        test.logits <- test.result$y
      }
      result <- plot.AUC(test.result$score, test.logits,
                         paste0(log.dir, 'test_result.step.', i, '.pdf'))
      val_losses <- c()
      train_losses <- c()
      if (configs$ngpus > 1) {
        for (rank in 0:(configs$ngpus-1)) {
          val_dic <- jsonlite::read_json(paste0(log.dir, 'result_dict.batch.', i, '.ddp_rank.', rank, '.json'))
          val_losses <- c(val_losses, val_dic$val_loss)
          train_losses <- c(train_losses, val_dic$train_loss)
        }
      } else {
        rank <- configs$gpu_id
        if (is.null(rank)) {
          rank <- 0
        }
        val_dic <- jsonlite::read_json(paste0(log.dir, 'result_dict.batch.', i, '.ddp_rank.', rank, '.json'))
        val_losses <- c(val_losses, val_dic$val_loss)
        train_losses <- c(train_losses, val_dic$train_loss)
      }
      J_stats <- result$curve[,2] - result$curve[,1]
      optimal.cutoff <- result$curve[which(J_stats==max(J_stats))[1],3]
      res <- data.frame(train=mean(train_losses),
                        val=mean(val_losses),
                        aucs=result$auc,
                        optimal.cutoffs=optimal.cutoff)
    } else {
      res <- data.frame(train=NA, val=NA, aucs=NA, optimal.cutoffs=NA)
    }
    res
  }
  stopCluster(cl)
  res$steps <- steps
  res <- res[!is.na(res$train),]
  val <- res$val
  aucs <- res$aucs
  optimal.cutoffs <- res$optimal.cutoffs
  steps <- res$steps
  train <- res$train
  to.plot <- data.frame(step=rep(steps, 2),
                        loss=c(train, val),
                        auc=rep(aucs, 2),
                        metric_name=c(rep("train_loss", length(steps)),
                                      rep("val_loss", length(steps))))
  library(ggplot2)
  p <- ggplot(to.plot, aes(x=step)) +
    geom_line(aes(y=loss, col=metric_name)) +
    geom_line(aes(y=auc)) +
    scale_y_continuous(
      # Features of the first axis
      name = "Loss",
      breaks = seq(0, max(1.1, max(to.plot$loss)), by = 0.05), limits = c(0, max(1.1, max(to.plot$loss))),
      # Add a second axis and specify its features
      sec.axis = sec_axis(~ . , name="AUC",
                          breaks = seq(0, max(1.1, max(to.plot$loss)), by = 0.05))
    ) +
    scale_x_continuous(breaks =
                         seq(1*configs$num_save_batches,
                             (num_saved_batches - 1)*configs$num_save_batches,
                             by = configs$num_save_batches)) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
  ggsave('Loss.AUC.by.step.pdf', p, width = 9, height = 6)
  
  print(paste0("min val step (", steps[which(val==min(val))[1]],
               ") AUC: ", round(aucs[which(val==min(val))[1]], digits = 2),
               " Optimal cutoff: ",
               round(optimal.cutoffs[which(val==min(val))[1]], digits = 2)))
  print(paste0("end step (", steps[length(val)],
               ") AUC: ", round(aucs[length(aucs)], digits = 2),
               " Optimal cutoff: ",
               round(optimal.cutoffs[length(aucs)], digits = 2)))
  print(paste0("max AUC step (", steps[which(aucs==max(aucs))[1]],
               "): ", round(max(aucs), digits = 2),
               " Optimal cutoff: ",
               round(optimal.cutoffs[which(aucs==max(aucs))[1]], digits = 2)))
  res
}

get.R.by.epoch <- function(configs) {
  log.dir <- configs$log_dir
  epochs <- c(1:configs$num_epochs)
  library(doParallel)
  cl <- makeCluster(72)
  registerDoParallel(cl)
  res <- foreach (i = 1:length(epochs), .combine=rbind) %dopar% {
    source('/share/terra/Users/gz2294/Pipeline/AUROC.R')
    i <- epochs[i]
    if (file.exists(paste0(log.dir, 'test_result.epoch.', i, '.csv'))) {
      test.result <- read.csv(paste0(log.dir, 'test_result.epoch.', i, '.csv'))
      score.columns <- colnames(test.result)[grep("^score", colnames(test.result))]
      score.columns <- score.columns[order(score.columns)]
      result.columns <- colnames(test.result)[grep("^y.", colnames(test.result))]
      result.columns <- result.columns[order(result.columns)]
      test.result <- test.result[!is.na(test.result[,result.columns[1]]),]
      result <- plot.R2(test.result[,score.columns], test.result[,result.columns],
                        paste0(log.dir, 'test_result.epoch.', i, '.pdf'))
      # val_losses <- c()
      # train_losses <- c()
      rank <- 0
      val_dic <- jsonlite::read_json(paste0(log.dir, 'result_dict.epoch.', i-1, '.ddp_rank.', rank, '.json'))
      # val_losses <- c(val_losses, val_dic$val_loss)
      # train_losses <- c(train_losses, val_dic$train_loss)
      res <- data.frame(epochs = i,
                        train_loss = val_dic$val_loss,
                        val_loss = val_dic$train_loss,
                        R2s=R2s)
    } else {
      res <- NA
    }
    res
  }
  stopCluster(cl)
  res <- res[!is.na(res$train_loss),]
  epochs <- res$epochs
  train <- res$train_loss
  val <- res$val_loss
  R2s <- as.data.frame(res[,startsWith(colnames(res), "R2s")])
  to.plot <- data.frame(epoch=rep(epochs, 2),
                        loss=c(train, val),
                        R2=c(rowMeans(R2s), rowMeans(R2s)),
                        Rs=rbind(R2s, R2s),
                        metric_name=c(rep("train_loss", length(epochs)),
                                      rep("val_loss", length(epochs))))
  to.plot.2 <- data.frame()
  for (i in 1:dim(R2s)[2]) {
    to.plot.2 <- rbind(to.plot.2,
                       data.frame(R2=R2s[,i],
                                  epoch=epochs,
                                  label=paste0('assay.', i)))
  }
  
  library(ggplot2)
  p <- ggplot(to.plot, aes(x=epoch)) +
    geom_line(aes(y=loss, col=metric_name)) +
    geom_line(data = to.plot.2, aes(y=R2, col=label)) +
    scale_y_continuous(
      # Features of the first axis
      name = "Loss", breaks = round(seq(min(to.plot$loss)-0.5, max(to.plot$loss)+0.5, by = 0.1), 1),
      # Add a second axis and specify its features
      sec.axis = sec_axis(~ . , name="R", breaks = round(seq(min(R2s), max(R2s), by = 0.1), 1))
    ) +
    scale_x_continuous(breaks = seq(1, configs$num_epochs, by = 1)) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
  
  print(paste0("min val epoch R: ", round(R2s[which(val==min(val)),], digits = 2)))
  print(paste0("end epoch R: ", round(R2s[length(R2s),], digits = 2)))
  ggsave('Loss.R.by.epoch.pdf', p, width = 9, height = 6)
  res
}

get.R.by.step <- function(configs) {
  log.dir <- configs$log_dir
  data.train <- as.numeric(strsplit(system(paste0("wc -l ", configs$data_file_train), intern = T), split = " ")[[1]][1])
  num_saved_batches = floor(ceiling(floor(data.train * configs$train_size / configs$ngpus) / configs$batch_size)
                            * configs$num_epochs / configs$num_save_batches) + 1
  print(paste0("num batches: ", num_saved_batches))
  steps <- c(1:(num_saved_batches-1)) * configs$num_save_batches
  R2s <- data.frame()
  train <- c()
  val <- c()
  
  library(doParallel)
  cl <- makeCluster(72)
  registerDoParallel(cl)
  
  res <- foreach (i = 1:length(steps), .combine = rbind) %dopar% {
    source('/share/terra/Users/gz2294/Pipeline/AUROC.R')
    i <- steps[i]
    if (file.exists(paste0(log.dir, 'test_result.step.', i, '.csv'))) {
      test.result <- read.csv(paste0(log.dir, 'test_result.step.', i, '.csv'))
      score.columns <- colnames(test.result)[grep("^score", colnames(test.result))]
      score.columns <- score.columns[order(score.columns)]
      result.columns <- colnames(test.result)[grep("^y.", colnames(test.result))]
      result.columns <- result.columns[order(result.columns)]
      test.result <- test.result[!is.na(test.result[,result.columns[1]]),]
      result <- plot.R2(test.result[,score.columns], test.result[,result.columns],
                        paste0(log.dir, 'test_result.step.', i, '.pdf'))
      val_losses <- c()
      train_losses <- c()
      rank <- 0
      val_dic <- jsonlite::read_json(paste0(log.dir, 'result_dict.batch.', i, '.ddp_rank.', rank, '.json'))
      val_losses <- c(val_losses, val_dic$val_loss)
      train_losses <- c(train_losses, val_dic$train_loss)
      res <- data.frame(train=mean(train_losses), val=mean(val_losses), R2s=t(as.data.frame(result$R2)))
    } else {
      res <- NA
    }
  }
  stopCluster(cl)
  res$steps <- steps
  res <- res[!is.na(res$train),]
  train <- res$train
  val <- res$val
  steps <- res$steps
  R2s <- res[,colnames(res)[grep("^R2s", colnames(res))]]
  if (is.null(dim(R2s))) {
    R2s <- as.matrix(R2s)
  }
  to.plot <- data.frame(step=rep(steps, 2),
                        loss=c(train, val),
                        R2=c(rowMeans(R2s), rowMeans(R2s)),
                        Rs=rbind(R2s, R2s),
                        metric_name=c(rep("train_loss", length(steps)),
                                      rep("val_loss", length(steps))))
  to.plot.2 <- data.frame()
  for (i in 1:dim(R2s)[2]) {
    to.plot.2 <- rbind(to.plot.2,
                       data.frame(R2=R2s[,i],
                                  step=steps,
                                  label=paste0('assay.', i)))
  }
  
  library(ggplot2)
  p <- ggplot(to.plot, aes(x=step)) +
    geom_line(aes(y=loss, col=metric_name)) +
    geom_line(data = to.plot.2, aes(y=R2, col=label)) +
    scale_y_continuous(
      # Features of the first axis
      name = "Loss", breaks = round(seq(min(to.plot$loss)-0.5, max(to.plot$loss)+0.5, by = 0.1), 1),
      # Add a second axis and specify its features
      sec.axis = sec_axis(~ . , name="R", breaks = round(seq(min(R2s), max(R2s), by = 0.1), 1))
    ) +
    scale_x_continuous(breaks =
                         seq(1*configs$num_save_batches,
                             (num_saved_batches - 1)*configs$num_save_batches,
                             by = configs$num_save_batches)) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
  ggsave('Loss.R.by.step.pdf', p, width = 9, height = 6)
  R2s <- as.data.frame(R2s)
  print(paste0("min val step (", steps[which(val==min(val))[1]],
               ") R: ", R2s[which(val==min(val))[1],]))
  print(paste0("end step (", steps[length(val)],
               ") R: ", R2s[length(steps),]))
  print(paste0("max R step (", steps[which(R2s==max(R2s))[1]],
               "): ", max(R2s)))
  # print(paste0("min val step (", steps[which(val==min(val))[1]],
  #              ") R: ", round(R2s[which(val==min(val))[1],], digits = 3)))
  # print(paste0("end step (", steps[length(val)],
  #              ") R: ", round(R2s[length(steps),], digits = 3)))
  # print(paste0("max R step (", steps[which(R2s==max(R2s))[1]],
  #              "): ", round(max(R2s), digits = 3)))
  res
}