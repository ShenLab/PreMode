get.auc.by.epoch <- function(configs, base.line="uniprotID") {
  log.dir <- configs$log_dir
  data.train <- as.numeric(strsplit(system(paste0("wc -l ", configs$data_file_train), intern = T), split = " ")[[1]][1])
  num_saved_batches <- floor(ceiling(data.train * configs$train_size / configs$ngpus / configs$batch_size)
                             * configs$num_epochs / configs$num_save_batches) + 1
  epochs <- c(1:(configs$num_epochs))
  source('scripts/AUROC.R')
  library(doParallel)
  cl <- makeCluster(72)
  registerDoParallel(cl)
  
  res <- foreach (i = 1:length(epochs), .combine=rbind) %dopar% {
    i <- epochs[i]
    source('~/Pipeline/AUROC.R')
    if (file.exists(paste0(log.dir, 'test_result.epoch.', i, '.csv'))) {
      test.result <- read.csv(paste0(log.dir, 'test_result.epoch.', i, '.csv'))
      if ('y.0' %in% colnames(test.result)) {
        if ('y.2' %in% colnames(test.result)) {
          # 3-dim logits
          test.result <- test.result[!is.na(test.result$y.0) & !is.na(test.result$y.1) & !is.na(test.result$y.2),]
          test.logits <- test.result[, c("y.0", "y.1", "y.2")]
          test.logits <- t(apply(as.matrix(test.logits), 1, soft.max))
          # check whether clinvar or gof/lof
          if (-1 %in% test.result$score) {
            test.logits <- test.logits[,3] / (test.logits[,2] + test.logits[,3])
          } else {
            test.logits <- 1 - test.logits[,1]
          }
        } else if ('y.1' %in% colnames(test.result)) {
          test.result <- test.result[!is.na(test.result$y.1),]
          test.logits <- test.result$y.1
        } else {
          test.result <- test.result[!is.na(test.result$y.0),]
          test.logits <- test.result$y.0
        }
      } else {
        test.result <- test.result[!is.na(test.result$y),]
        test.logits <- test.result$y
      }
      result <- plot.AUC(test.result$score, test.logits
                         # paste0(log.dir, 'test_result.epoch.', i, '.pdf')
                         )
      J_stats <- result$curve[,2] - result$curve[,1]
      optimal.cutoff <-  result$curve[which(J_stats==max(J_stats))[1],3]
    } else {
      result <- list(auc=NA)
      optimal.cutoff <- NA
    }
    if (file.exists(paste0(log.dir, 'result_dict.epoch.', i-1, '.ddp_rank.', 0, '.json'))) {
      val_losses <- c()
      train_losses <- c()
      if (configs$ngpus > 1) {
        for (rank in 0:(configs$ngpus-1)) {
          val_dic <- jsonlite::read_json(paste0(log.dir, 'result_dict.epoch.', i-1, '.ddp_rank.', rank, '.json'))
          if (!is.null(val_dic$val_loss_y)) {
            val_losses <- c(val_losses, val_dic$val_loss_y)
            train_losses <- c(train_losses, val_dic$train_loss_y)
          } else {
            val_losses <- c(val_losses, val_dic$val_loss)
            train_losses <- c(train_losses, val_dic$train_loss)
          }
        }
      } else {
        rank <- configs$gpu_id
        if (is.null(rank)) {
          rank <- 0
        }
        val_dic <- jsonlite::read_json(paste0(log.dir, 'result_dict.epoch.', i-1, '.ddp_rank.', rank, '.json'))
        if (!is.null(val_dic$val_loss_y)) {
          val_losses <- c(val_losses, val_dic$val_loss_y)
          train_losses <- c(train_losses, val_dic$train_loss_y)
        } else {
          val_losses <- c(val_losses, val_dic$val_loss)
          train_losses <- c(train_losses, val_dic$train_loss)
        }
      }
    } else {
      train_losses <- NA
      val_losses <- NA
    }
    if (file.exists(paste0(log.dir, 'test_result.epoch.', i, '.txt'))) {
      test_dic <- readLines(paste0(log.dir, 'test_result.epoch.', i, '.txt'), warn = F)
      test_dic <- gsub("'", '"', test_dic)
      test_dic <- jsonlite::fromJSON(test_dic)
      if (!is.null(test_dic$test_loss_y)) {
        test_losses <- test_dic$test_loss_y
      } else {
        test_losses <- test_dic$test_loss
      }
    } else {
      test_losses <- NA
    }
    res <- data.frame(train=mean(train_losses),
                      val=mean(val_losses),
                      test=mean(test_losses),
                      aucs=result$auc,
                      optimal.cutoffs=optimal.cutoff)
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
  # calculate baseline
  if (base.line == "uniprotID") {
    baseline.uniprotID <- system(
      paste0("/share/vault/Users/gz2294/miniconda3/envs/r4-base/bin/python ",
             "/share/pascal/Users/gz2294/PreMode.final/analysis/random.forest.process.classifier.py ",
             configs$data_file_train, " ",
             configs$data_file_test), intern = T,
    )
    baseline.auc <- as.numeric(strsplit(baseline.uniprotID, ": ")[[1]][2])
    if (dim(res)[1] > 0) {
      res$baseline.auc <- baseline.auc
    } 
  } else if (base.line == "esm") {
    alphabet <- c('<cls>', '<pad>', '<eos>', '<unk>',
                  'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
                  'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
                  'X', 'B', 'U', 'Z', 'O', '.', '-',
                  '<null_1>', '<mask>')
    data.file.name <- configs$data_type
    fold <- strsplit(configs$data_file_train, "pfams.0.8.seed.")[[1]][2]
    fold <- as.numeric(substr(fold, 1, 1))
    if (is.na(fold)) {
      fold <- 0
    }
    baseline.file <- paste0('/share/pascal/Users/gz2294/PreMode.final/analysis/esm2.inference/',
                            data.file.name, "/testing.fold.", fold, ".logits.csv")
    test.result <- read.csv(configs$data_file_test, row.names = 1)
    if (file.exists(baseline.file)) {
      baseline.res <- read.csv(baseline.file)
      logits <- baseline.res[,2:34]
      colnames(logits) <- alphabet
      score <- c()
      for (k in 1:dim(logits)[1]) {
        score <- c(score, logits[k, test.result$alt[k]] - logits[k, test.result$ref[k]])
      }
      result <- plot.AUC(test.result$score, score)
      if (dim(res)[1] > 0) {
        res$baseline.auc <- result$auc
      } 
    } 
  }
  library(ggplot2)
  if (is.na(to.plot$auc[1])) {
    p <- ggplot(to.plot, aes(x=epoch)) +
      geom_line(aes(y=loss, col=metric_name)) +
      theme_bw() +
      theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
  } else {
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
      theme_bw() +
      theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
  }
  
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


get.auc.by.step <- function(configs, base.line="uniprotID", save=T) {
  log.dir <- configs$log_dir
  data.train <- as.numeric(strsplit(system(paste0("wc -l ", configs$data_file_train), intern = T), split = " ")[[1]][1])
  num_saved_batches = floor(ceiling(data.train * configs$train_size / configs$ngpus / configs$batch_size)
                            * configs$num_epochs / configs$num_save_batches) + 1
  steps <- c(1:(num_saved_batches-1))*configs$num_save_batches
  source('scripts/AUROC.R')
  library(doParallel)
  cl <- makeCluster(72)
  registerDoParallel(cl)
  res <- foreach (i = 1:length(steps), .combine=rbind) %dopar% {
  # for (i in 1:length(steps)) {  
    source('~/Pipeline/AUROC.R')
    i <- steps[i]
    if (file.exists(paste0(log.dir, 'test_result.step.', i, '.csv'))) {
      test.result <- read.csv(paste0(log.dir, 'test_result.step.', i, '.csv'))
      if ('y.0' %in% colnames(test.result)) {
        if ('y.2' %in% colnames(test.result)) {
          # 3-dim logits
          test.result <- test.result[!is.na(test.result$y.0) & !is.na(test.result$y.1) & !is.na(test.result$y.2),]
          test.logits <- test.result[, c("y.0", "y.1", "y.2")]
          test.logits <- t(apply(as.matrix(test.logits), 1, soft.max))
          # check whether clinvar or gof/lof
          if (-1 %in% test.result$score) {
            test.logits <- test.logits[,3] / (test.logits[,2] + test.logits[,3])
          } else {
            test.logits <- 1 - test.logits[,1]
          }
        } else if ('y.1' %in% colnames(test.result)) {
          test.result <- test.result[!is.na(test.result$y.1),]
          test.logits <- test.result$y.1
        } else {
          test.result <- test.result[!is.na(test.result$y.0),]
          test.logits <- test.result$y.0
        }
      } else {
        test.result <- test.result[!is.na(test.result$y),]
        test.logits <- test.result$y
      }
      result <- plot.AUC(test.result$score, test.logits
                         # paste0(log.dir, 'test_result.step.', i, '.pdf')
                         )
      J_stats <- result$curve[,2] - result$curve[,1]
      optimal.cutoff <- result$curve[which(J_stats==max(J_stats))[1],3]
    } else {
      result <- list(auc=NA)
      optimal.cutoff <- NA
    }
    if (file.exists(paste0(log.dir, 'result_dict.batch.', i, '.ddp_rank.', 0, '.json'))) {
      val_losses <- c()
      train_losses <- c()
      if (configs$ngpus > 1) {
        for (rank in 0:(configs$ngpus-1)) {
          val_dic <- jsonlite::read_json(paste0(log.dir, 'result_dict.batch.', i, '.ddp_rank.', rank, '.json'))
          if (!is.null(val_dic$val_loss_y)) {
            val_losses <- c(val_losses, val_dic$val_loss_y)
            train_losses <- c(train_losses, val_dic$train_loss_y)
          } else {
            val_losses <- c(val_losses, val_dic$val_loss)
            train_losses <- c(train_losses, val_dic$train_loss)
          }
        }
      } else {
        rank <- configs$gpu_id
        if (is.null(rank)) {
          rank <- 0
        }
        val_dic <- jsonlite::read_json(paste0(log.dir, 'result_dict.batch.', i, '.ddp_rank.', rank, '.json'))
        if (!is.null(val_dic$val_loss_y)) {
          val_losses <- c(val_losses, val_dic$val_loss_y)
          train_losses <- c(train_losses, val_dic$train_loss_y)
        } else {
          val_losses <- c(val_losses, val_dic$val_loss)
          train_losses <- c(train_losses, val_dic$train_loss)
        }
      }
    } else {
      val_losses <- NA
      train_losses <- NA
    }
    if (file.exists(paste0(log.dir, 'test_result.step.', i, '.txt'))) {
      test_dic <- readLines(paste0(log.dir, 'test_result.step.', i, '.txt'), warn = F)
      test_dic <- gsub("'", '"', test_dic)
      test_dic <- jsonlite::fromJSON(test_dic)
      if (!is.null(test_dic$test_loss_y)) {
        test_losses <- test_dic$test_loss_y
      } else {
        test_losses <- test_dic$test_loss
      }
    } else {
      test_losses <- NA
    }
    res <- data.frame(train=mean(train_losses),
                      val=mean(val_losses),
                      test=mean(test_losses),
                      aucs=result$auc,
                      optimal.cutoffs=optimal.cutoff)
    print(res)
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
  # calculate baseline
  if (base.line == "uniprotID") {
    baseline.uniprotID <- system(
      paste0("/share/vault/Users/gz2294/miniconda3/envs/r4-base/bin/python ",
             "/share/pascal/Users/gz2294/PreMode.final/analysis/random.forest.process.classifier.py ",
             configs$data_file_train, " ",
             configs$data_file_test), intern = T,
    )
    baseline.auc <- as.numeric(strsplit(baseline.uniprotID, ": ")[[1]][2])
    if (dim(res)[1] > 0) {
      res$baseline.auc <- baseline.auc
    } 
  } else if (base.line == "esm") {
    alphabet <- c('<cls>', '<pad>', '<eos>', '<unk>',
                  'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
                  'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
                  'X', 'B', 'U', 'Z', 'O', '.', '-',
                  '<null_1>', '<mask>')
    data.file.name <- configs$data_type
    fold <- strsplit(configs$data_file_train, "pfams.0.8.seed.")[[1]][2]
    fold <- as.numeric(substr(fold, 1, 1))
    if (is.na(fold)) {
      fold <- 0
    }
    baseline.file <- paste0('/share/pascal/Users/gz2294/PreMode.final/analysis/esm2.inference/',
                            data.file.name, "/testing.fold.", fold, ".logits.csv")
    test.result <- read.csv(configs$data_file_test, row.names = 1)
    if (file.exists(baseline.file)) {
      baseline.res <- read.csv(baseline.file)
      logits <- baseline.res[,2:34]
      colnames(logits) <- alphabet
      score <- c()
      for (k in 1:dim(logits)[1]) {
        score <- c(score, logits[k, test.result$alt[k]] - logits[k, test.result$ref[k]])
      }
      result <- plot.AUC(test.result$score, score)
      if (dim(res)[1] > 0) {
        res$baseline.auc <- result$auc
      } 
    } 
  }
  library(ggplot2)
  if (is.na(to.plot$auc[1])) {
    p <- ggplot(to.plot, aes(x=step)) +
      geom_line(aes(y=loss, col=metric_name)) +
      scale_x_continuous(breaks =
                           seq(1*configs$num_save_batches,
                               (num_saved_batches - 1)*configs$num_save_batches,
                               by = configs$num_save_batches)) +
      theme_bw() +
      theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
  } else {
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
  }
  if (save) {
    ggsave('Loss.AUC.by.step.pdf', p, width = max(6, min(6 * length(steps) / 50, 20)), height = 4)
  }
  
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
  to.plot
}


get.auc.by.step.split.pLDDT <- function(configs, base.line="uniprotID") {
  log.dir <- configs$log_dir
  data.train <- as.numeric(strsplit(system(paste0("wc -l ", configs$data_file_train), intern = T), split = " ")[[1]][1])
  num_saved_batches = floor(ceiling(data.train * configs$train_size / configs$ngpus / configs$batch_size)
                            * configs$num_epochs / configs$num_save_batches) + 1
  steps <- c(1:(num_saved_batches-1))*configs$num_save_batches
  source('~/Pipeline/uniprot.table.add.annotation.R')
  test.file <- read.csv(configs$data_file_test)
  test.file <- uniprot.table.add.annotation.parallel(test.file, 'pLDDT.region')
  source('scripts/AUROC.R')
  library(doParallel)
  cl <- makeCluster(72)
  registerDoParallel(cl)
  res <- foreach (i = 1:length(steps), .combine=rbind) %dopar% {
    # for (i in 1:length(steps)) {  
    source('~/Pipeline/AUROC.R')
    i <- steps[i]
    if (file.exists(paste0(log.dir, 'test_result.step.', i, '.csv'))) {
      test.result <- read.csv(paste0(log.dir, 'test_result.step.', i, '.csv'))
      if ('y.0' %in% colnames(test.result)) {
        if ('y.2' %in% colnames(test.result)) {
          # 3-dim logits
          test.result <- test.result[!is.na(test.result$y.0) & !is.na(test.result$y.1) & !is.na(test.result$y.2),]
          test.logits <- test.result[, c("y.0", "y.1", "y.2")]
          test.logits <- t(apply(as.matrix(test.logits), 1, soft.max))
          # check whether clinvar or gof/lof
          if (-1 %in% test.result$score) {
            test.logits <- test.logits[,3] / (test.logits[,2] + test.logits[,3])
          } else {
            test.logits <- 1 - test.logits[,1]
          }
        } else if ('y.1' %in% colnames(test.result)) {
          test.result <- test.result[!is.na(test.result$y.1),]
          test.logits <- test.result$y.1
        } else {
          test.result <- test.result[!is.na(test.result$y.0),]
          test.logits <- test.result$y.0
        }
      } else {
        test.result <- test.result[!is.na(test.result$y),]
        test.logits <- test.result$y
      }
      result <- plot.AUC(test.result$score, test.logits)
      result.1 <- plot.AUC(test.result$score[test.file$pLDDT.region >= 70], 
                           test.logits[test.file$pLDDT.region >= 70])
      result.2 <- plot.AUC(test.result$score[test.file$pLDDT.region < 70],
                           test.logits[test.file$pLDDT.region < 70])
      J_stats <- result$curve[,2] - result$curve[,1]
      optimal.cutoff <- result$curve[which(J_stats==max(J_stats))[1],3]
    } else {
      result <- list(auc=NA)
      result.1 <- list(auc=NA)
      result.2 <- list(auc=NA)
      optimal.cutoff <- NA
    }
    if (file.exists(paste0(log.dir, 'result_dict.batch.', i, '.ddp_rank.', 0, '.json'))) {
      val_losses <- c()
      train_losses <- c()
      if (configs$ngpus > 1) {
        for (rank in 0:(configs$ngpus-1)) {
          val_dic <- jsonlite::read_json(paste0(log.dir, 'result_dict.batch.', i, '.ddp_rank.', rank, '.json'))
          if (!is.null(val_dic$val_loss_y)) {
            val_losses <- c(val_losses, val_dic$val_loss_y)
            train_losses <- c(train_losses, val_dic$train_loss_y)
          } else {
            val_losses <- c(val_losses, val_dic$val_loss)
            train_losses <- c(train_losses, val_dic$train_loss)
          }
        }
      } else {
        rank <- configs$gpu_id
        if (is.null(rank)) {
          rank <- 0
        }
        val_dic <- jsonlite::read_json(paste0(log.dir, 'result_dict.batch.', i, '.ddp_rank.', rank, '.json'))
        if (!is.null(val_dic$val_loss_y)) {
          val_losses <- c(val_losses, val_dic$val_loss_y)
          train_losses <- c(train_losses, val_dic$train_loss_y)
        } else {
          val_losses <- c(val_losses, val_dic$val_loss)
          train_losses <- c(train_losses, val_dic$train_loss)
        }
      }
    } else {
      val_losses <- NA
      train_losses <- NA
    }
    if (file.exists(paste0(log.dir, 'test_result.step.', i, '.txt'))) {
      test_dic <- readLines(paste0(log.dir, 'test_result.step.', i, '.txt'), warn = F)
      test_dic <- gsub("'", '"', test_dic)
      test_dic <- jsonlite::fromJSON(test_dic)
      if (!is.null(test_dic$test_loss_y)) {
        test_losses <- test_dic$test_loss_y
      } else {
        test_losses <- test_dic$test_loss
      }
    } else {
      test_losses <- NA
    }
    res <- data.frame(train=mean(train_losses),
                      val=mean(val_losses),
                      test=mean(test_losses),
                      aucs=result$auc,
                      aucs.low_pLDDT=result.2$auc,
                      aucs.high_pLDDT=result.1$auc,
                      optimal.cutoffs=optimal.cutoff)
    print(res)
  }
  stopCluster(cl)
  res$steps <- steps
  res <- res[!is.na(res$train),]
  val <- res$val
  aucs <- res$aucs
  optimal.cutoffs <- res$optimal.cutoffs
  steps <- res$steps
  train <- res$train
  result.1.neg <- sum(test.file$pLDDT.region >= 70 & test.file$score == -1)
  result.1.pos <- sum(test.file$pLDDT.region >= 70 & test.file$score == 1)
  result.2.neg <- sum(test.file$pLDDT.region < 70 & test.file$score == -1)
  result.2.pos <- sum(test.file$pLDDT.region < 70 & test.file$score == 1)
  to.plot <- data.frame(step=rep(steps, 2),
                        loss=c(train, val),
                        auc=rep(aucs, 2),
                        auc.pLDDT=c(res$aucs.high_pLDDT, res$aucs.low_pLDDT),
                        auc.name = c(rep(paste0("pLDDT >= 0.7 (", result.1.neg, "/", result.1.pos, ")"),
                                         length(steps)),
                                     rep(paste0("pLDDT < 0.7 (", result.2.neg, "/", result.2.pos, ")"),
                                         length(steps))),
                        metric_name=c(rep("train_loss", length(steps)),
                                      rep("val_loss", length(steps))))
  # calculate baseline
  if (base.line == "uniprotID") {
    baseline.uniprotID <- system(
      paste0("/share/vault/Users/gz2294/miniconda3/envs/r4-base/bin/python ",
             "/share/pascal/Users/gz2294/PreMode.final/analysis/random.forest.process.classifier.py ",
             configs$data_file_train, " ",
             configs$data_file_test), intern = T,
    )
    baseline.auc <- as.numeric(strsplit(baseline.uniprotID, ": ")[[1]][2])
    if (dim(res)[1] > 0) {
      res$baseline.auc <- baseline.auc
    } 
  } else if (base.line == "esm") {
    alphabet <- c('<cls>', '<pad>', '<eos>', '<unk>',
                  'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
                  'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
                  'X', 'B', 'U', 'Z', 'O', '.', '-',
                  '<null_1>', '<mask>')
    data.file.name <- configs$data_type
    fold <- strsplit(configs$data_file_train, "pfams.0.8.seed.")[[1]][2]
    fold <- as.numeric(substr(fold, 1, 1))
    if (is.na(fold)) {
      fold <- 0
    }
    baseline.file <- paste0('/share/pascal/Users/gz2294/PreMode.final/analysis/esm2.inference/',
                            data.file.name, "/testing.fold.", fold, ".logits.csv")
    test.result <- read.csv(configs$data_file_test, row.names = 1)
    if (file.exists(baseline.file)) {
      baseline.res <- read.csv(baseline.file)
      logits <- baseline.res[,2:34]
      colnames(logits) <- alphabet
      score <- c()
      for (k in 1:dim(logits)[1]) {
        score <- c(score, logits[k, test.result$alt[k]] - logits[k, test.result$ref[k]])
      }
      result <- plot.AUC(test.result$score, score)
      if (dim(res)[1] > 0) {
        res$baseline.auc <- result$auc
      } 
    } 
  }
  library(ggplot2)
  if (is.na(to.plot$auc[1])) {
    p <- ggplot(to.plot, aes(x=step)) +
      geom_line(aes(y=loss, col=metric_name)) +
      scale_x_continuous(breaks =
                           seq(1*configs$num_save_batches,
                               (num_saved_batches - 1)*configs$num_save_batches,
                               by = configs$num_save_batches)) +
      theme_bw() +
      theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
  } else {
    p <- ggplot(to.plot, aes(x=step)) +
      geom_line(aes(y=loss, col=metric_name)) +
      geom_line(aes(y=auc)) +
      geom_line(aes(y=auc.pLDDT, col=auc.name)) +
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
  }
  ggsave('Loss.AUC.by.step.pdf', p, width = max(9, min(9 * length(steps) / 50, 20)), height = 6)
  
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


get.R.by.epoch <- function(configs, bin=FALSE) {
  log.dir <- configs$log_dir
  epochs <- c(1:configs$num_epochs)
  library(doParallel)
  cl <- makeCluster(72)
  registerDoParallel(cl)
  res <- foreach (i = 1:length(epochs), .combine=dplyr::bind_rows) %dopar% {
    source('scripts/AUROC.R')
    i <- epochs[i]
    if (file.exists(paste0(log.dir, 'test_result.epoch.', i, '.csv'))) {
      test.result <- read.csv(paste0(log.dir, 'test_result.epoch.', i, '.csv'))
      score.columns <- colnames(test.result)[grep("^score", colnames(test.result))]
      score.columns <- score.columns[order(score.columns)]
      result.columns <- colnames(test.result)[grep("^y.", colnames(test.result))]
      result.columns <- result.columns[order(result.columns)]
      test.result <- test.result[!is.na(test.result[,result.columns[1]]),]
      result <- plot.R2(test.result[,score.columns], test.result[,result.columns],
                        bin=bin,
                        filename=paste0(log.dir, 'test_result.epoch.', i, '.pdf'))
      # val_losses <- c()
      # train_losses <- c()
      R2s <- t(as.data.frame(result$R2))
    } else {
      R2s <- NA
    }
    rank <- 0
    val_dic <- jsonlite::read_json(paste0(log.dir, 'result_dict.epoch.', i-1, '.ddp_rank.', rank, '.json'))
    res <- data.frame(epochs = i,
                      train_loss = val_dic$train_loss,
                      val_loss = val_dic$val_loss,
                      R2s=R2s)
    res
  }
  stopCluster(cl)
  res <- res[!is.na(res$train_loss),]
  epochs <- res$epochs
  train <- res$train_loss
  val <- res$val_loss
  R2s <- as.data.frame(res[,startsWith(colnames(res), "R2s")])
  if (dim(R2s)[1] > 0) {
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
    if (all(is.na(to.plot.2$R2))) {
      p <- ggplot(to.plot, aes(x=epoch)) +
        geom_line(aes(y=loss, col=metric_name)) +
        scale_x_continuous(breaks = seq(1, configs$num_epochs, by = 1)) +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
    } else {
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
    }
    print(paste0("min val epoch R: ", round(R2s[which(val==min(val)),], digits = 100)))
    print(paste0("end epoch R: ", round(R2s[dim(R2s)[1],], digits = 100)))
    ggsave('Loss.R.by.epoch.pdf', p, width = min(49.9, 9 * length(epochs) / 10), height = 6)
  }
  res
}

get.R.by.step <- function(configs, bin=FALSE) {
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
  
  res <- foreach (i = 1:length(steps), .combine = dplyr::bind_rows) %dopar% {
    source('scripts/AUROC.R')
    i <- steps[i]
    if (file.exists(paste0(log.dir, 'test_result.step.', i, '.csv'))) {
      test.result <- read.csv(paste0(log.dir, 'test_result.step.', i, '.csv'))
      score.columns <- colnames(test.result)[grep("^score", colnames(test.result))]
      score.columns <- score.columns[order(score.columns)]
      result.columns <- colnames(test.result)[grep("^y.", colnames(test.result))]
      result.columns <- result.columns[order(result.columns)]
      test.result <- test.result[!is.na(test.result[,result.columns[1]]),]
      result <- plot.R2(test.result[,score.columns], test.result[,result.columns],
                        bin=bin,
                        filename=paste0(log.dir, 'test_result.step.', i, '.pdf'))
      val_losses <- c()
      train_losses <- c()
      rank <- 0
      val_dic <- jsonlite::read_json(paste0(log.dir, 'result_dict.batch.', i, '.ddp_rank.', rank, '.json'))
      val_losses <- c(val_losses, val_dic$val_loss)
      train_losses <- c(train_losses, val_dic$train_loss)
      res <- data.frame(train=mean(train_losses), 
                        val=mean(val_losses), 
                        R2s=t(as.data.frame(result$R2)))
    } else {
      res <- data.frame(train=NA, 
                        val=NA, 
                        R2s=NA)
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
  ggsave('Loss.R.by.step.pdf', p, width = min(max(9 * length(steps) / 100, 9), 49.9), height = 6)
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
