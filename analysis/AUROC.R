plot.AUC <- function(scores, logits, filename=NA, rev.ok=F) {
  library(PRROC)
  all.scores <- unique(scores)
  if (0 %in% all.scores & ! -1 %in% all.scores) {
    neg.score <- 0
  } else if (-1 %in% all.scores) {
    neg.score <- -1
  } else {
    neg.score <- NA
  }
  if (3 %in% all.scores & !1 %in% all.scores) {
    pos.score <- 3
  } else {
    pos.score <- 1
  }
  # remove NA value
  na.value <- is.na(scores) | is.na(logits)
  if (sum(na.value) == length(scores)) {
    PRROC.dnv.df <- list(auc=NA, cutoff=NA)
  } else {
    scores <- scores[!na.value]
    logits <- logits[!na.value]
    sel.value <- scores %in% c(neg.score, pos.score)
    scores <- scores[sel.value]
    logits <- logits[sel.value]
    if (length(unique(logits)) == 1 | length(unique(scores)) == 1 | length(logits)==0) {
      PRROC.dnv.df <- list(auc=NA,
                           cutoff=NA,
                           curve=matrix(-Inf, nrow = 1, ncol = 3))
    } else {
      # pr_curve(two_class_example, truth, Class1)
      PRROC.dnv.df <- roc.curve(scores.class0 = logits[scores==neg.score],
                                scores.class1 = logits[scores==pos.score],
                                curve=TRUE)
      if (rev.ok & PRROC.dnv.df$auc < 0.5) {
        PRROC.dnv.df <- roc.curve(scores.class1 = logits[scores==neg.score],
                                  scores.class0 = logits[scores==pos.score],
                                  curve=TRUE)
      }
      if (!is.na(filename)) {
        pdf(filename)
        plot(PRROC.dnv.df)
        dev.off()
      }
      J_stats <- PRROC.dnv.df$curve[,2] - PRROC.dnv.df$curve[,1]
      optimal.cutoff <-  PRROC.dnv.df$curve[which(J_stats==max(J_stats))[1],3]
      PRROC.dnv.df$cutoff <- optimal.cutoff
      print(paste0("optimal cutoff = ", optimal.cutoff))
      PRROC.dnv.df
    }
  }
  PRROC.dnv.df
}


plot.PR <- function(scores, logits, filename=NA) {
  library(PRROC)
  all.scores <- unique(scores)
  if (0 %in% all.scores & ! -1 %in% all.scores) {
    neg.score <- 0
  } else if (-1 %in% all.scores) {
    neg.score <- -1
  }
  # remove NA value
  na.value <- is.na(scores) | is.na(logits)
  scores <- scores[!na.value]
  logits <- logits[!na.value]
  sel.value <- scores %in% c(neg.score, 1)
  scores <- scores[sel.value]
  logits <- logits[sel.value]
  if (length(unique(logits)) == 1) {
    PRROC.dnv.df <- list(auc=NA,
                         cutoff=NA,
                         curve=matrix(-Inf, nrow = 1, ncol = 3))
  } else {
    # pr_curve(two_class_example, truth, Class1)
    PRROC.dnv.df <- pr.curve(scores.class0 = logits[scores==neg.score],
                             scores.class1 = logits[scores==1],
                             curve=TRUE)
    if (PRROC.dnv.df$auc.integral <= 0.5) {
      PRROC.dnv.df <- pr.curve(scores.class0 = logits[scores==1],
                               scores.class1 = logits[scores==neg.score],
                               curve=TRUE)
    }
    if (!is.na(filename)) {
      pdf(filename)
      plot(PRROC.dnv.df)
      dev.off()
    }
    F1 <- 2 * PRROC.dnv.df$curve[,2] * PRROC.dnv.df$curve[,1] / (PRROC.dnv.df$curve[,2] + PRROC.dnv.df$curve[,1])
    optimal.cutoff <-  PRROC.dnv.df$curve[which(F1==max(F1))[1],3]
    PRROC.dnv.df$cutoff <- optimal.cutoff
    PRROC.dnv.df$auc <- PRROC.dnv.df$auc.integral
    print(paste0("optimal cutoff = ", optimal.cutoff))
    PRROC.dnv.df
  }
}


plot.AUC.by.uniprotID <- function(scores, logits, uniprotIDs) {
  library(PRROC)
  all.scores <- unique(scores)
  if (0 %in% all.scores & ! -1 %in% all.scores) {
    neg.score <- 0
  } else if (-1 %in% all.scores) {
    neg.score <- -1
  }
  # remove NA value
  na.value <- is.na(scores) | is.na(logits) | is.na(uniprotIDs)
  scores <- scores[!na.value]
  logits <- logits[!na.value]
  uniprotIDs <- uniprotIDs[!na.value]
  unique.uniprotIDs <- unique(uniprotIDs)
  if (length(unique.uniprotIDs) == 1) {
    PRROC.dnv.df <- data.frame(uniprotID = unique.uniprotIDs,
                               optimal.cutoff = NA,
                               auc = 0.5)
  } else {
    PRROC.dnv.df <- data.frame(uniprotID = unique.uniprotIDs,
                               optimal.cutoff = NA,
                               auc = NA)
    for (i in 1:length(unique.uniprotIDs)) {
      # pr_curve(two_class_example, truth, Class1)
      if (sum(uniprotIDs==unique.uniprotIDs[i] & scores==neg.score) * 
          sum(uniprotIDs==unique.uniprotIDs[i] & scores==1) > 0) {
        res <- roc.curve(scores.class0 = logits[scores==neg.score & uniprotIDs==unique.uniprotIDs[i]],
                         scores.class1 = logits[scores==1 & uniprotIDs==unique.uniprotIDs[i]],
                         curve=TRUE)
        if (res$auc < 0.5) {
          res <- roc.curve(scores.class1 = logits[scores==neg.score & uniprotIDs==unique.uniprotIDs[i]],
                           scores.class0 = logits[scores==1 & uniprotIDs==unique.uniprotIDs[i]],
                           curve=TRUE)
        }
        J_stats <- res$curve[,2] - res$curve[,1]
        optimal.cutoff <-  res$curve[which(J_stats==max(J_stats))[1],3]
        PRROC.dnv.df$optimal.cutoff[i] <- optimal.cutoff
        PRROC.dnv.df$auc[i] <- res$auc
        print(paste0("optimal cutoff for ID ", unique.uniprotIDs[i], " = ", optimal.cutoff))
      } else {
        print(paste0("Not enough data points to calculate the auc for ", unique.uniprotIDs[i]))
      }
    }
  }
  PRROC.dnv.df
}


plot.R2 <- function(scores, logits, bin=FALSE, filename=NA) {
  library(ggplot2)
  library(ggpubr)
  to.plot <- data.frame()
  if (is.null(dim(scores))) {
    to.plot <- rbind(to.plot,
                     data.frame(measurement=scores,
                                predicted=logits,
                                label=paste0("assay.", 1)))
  } else {
    for (i in 1:dim(scores)[2]) {
      to.plot <- rbind(to.plot,
                       data.frame(measurement=scores[,i],
                                  predicted=logits[,i],
                                  label=paste0("assay.", i)))
    }
  }
  if (!is.na(filename)) {
    p <- ggplot(to.plot, aes(x=measurement, y=predicted, col=label)) +
      geom_point(alpha=0.5) +
      stat_smooth(method = "lm", formula = y~x) +
      stat_regline_equation(
        aes(label =  paste(..eq.label.., ..adj.rr.label.., sep = "~~~~")),
        formula = y~x
      ) + theme_bw()
    ggsave(filename, plot=p, height=6, width=8)
  }
  to.plot
  R2 <- c()
  if (is.null(dim(scores))) {
    assays <- 1
    # check NA scores
    na.value <- is.na(logits) | is.na(scores)
    logits <- logits[!na.value]
    scores <- scores[!na.value]
    if (bin) {
      library(PRROC)
      if (length(unique(logits)) > 1 & length(unique(scores)) > 1) {
        auc <- roc.curve(scores.class0 = logits[scores==0], 
                         scores.class1 = logits[scores==1])$auc
        if (auc < 0.5) {
          auc <- roc.curve(scores.class0 = logits[scores==1], 
                           scores.class1 = logits[scores==0])$auc
        }
      } else {
        auc <- NA
      }
      R2[1] <- auc
    } else {
      if (length(logits) > 2) {
        R2[1] <- cor.test(scores, logits, method='spearman')$estimate
      } else {
        R2[1] <- NA
      }
    }
  } else {
    assays <- dim(scores)[2]
    for (i in 1:assays) {
      # check NA scores
      na.value <- is.na(logits[,i]) | is.na(scores[,i])
      logits.i <- logits[!na.value,i]
      scores.i <- scores[!na.value,i]
      if (bin) {
        library(PRROC)
        if (length(unique(logits)) > 1 & length(unique(scores)) > 1) {
          auc <- roc.curve(scores.class0 = logits.i[scores.i==0], 
                           scores.class1 = logits.i[scores.i==1])$auc
          if (auc < 0.5) {
            auc <- roc.curve(scores.class0 = logits.i[scores.i==1], 
                             scores.class1 = logits.i[scores.i==0])$auc
          }
        } else {
          auc <- NA
        }
        R2[i] <- auc
      } else {
        if (length(scores.i) > 2) {
          R2[i] <- cor.test(scores.i, logits.i, method='spearman')$estimate
        } else {
          R2[i] <- NA
        }
      }
    }
  }
  result <- list(to.plot=to.plot,
                 R2=R2)
}


soft.max <- function(logits) {
  res <- exp(logits) / sum(exp(logits))
  res
}
