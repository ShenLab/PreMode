library(ggplot2)
task.dic <- list("PTEN"=c("score.1"="stability", "score.2"="enzyme.activity"), 
                 "NUDT15"=c("score.1"="stability", "score.2"="enzyme.activity"), 
                 "VKORC1"=c("score.1"="enzyme.activity", "score.2"="stability"), 
                 "CCR5"=c("score.1"="stability", "score.2"="binding Ab2D7", "score.3"="binding HIV-1"), 
                 "CXCR4"=c("score.1"="stability", "score.2"="binding CXCL12", "score.3"="binding Ab12G5"),
                 "SNCA"=c("score.1"="enzyme.activity", "score.2"="stability"),
                 "CYP2C9"=c("score.1"="enzyme.activity", "score.2"="stability"),
                 "GCK"=c("score.1"="enzyme.activity", "score.2"="stability"),
                 "ASPA"=c("score.1"="stability", "score.2"="enzyme.activity")
                 )
alphabet_premode <- c('L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
                      'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C')
genes <- c("PTEN", "NUDT15", "CCR5", "CXCR4", 'SNCA', 'CYP2C9', 'GCK', 'ASPA')
# add baseline AUC
# esm alphabets
source('./AUROC.R')
alphabet <- c('<cls>', '<pad>', '<eos>', '<unk>',
              'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
              'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
              'X', 'B', 'U', 'Z', 'O', '.', '-',
              '<null_1>', '<mask>')
result <- data.frame()
for (i in 1:length(genes)) {
  for (fold in 0:4) {
    # REVEL, PrimateAI, ESM AUC
    test.result <- read.csv(paste0('PreMode/', genes[i], '/',
                                   '/test.fold.', fold, '.annotated.csv'), row.names = 1)
    test.result.pass <- read.csv(paste0('ESM.SLP/', genes[i], '/',
                                        '/testing.fold.', fold, '.csv'))
    task.length <- length(task.dic[[genes[i]]])
    # add hsu et al results
    hsu.unirep_onehot.auc <- list(R2=c())
    hsu.ev_onehot.auc <- list(R2=c())
    hsu.gesm_onehot.auc <- list(R2=c())
    hsu.eve_onehot.auc <- list(R2=c())
    for (s in 1:task.length) {
      test.result.hsu <- read.csv(paste0('./Hsu.et.al.git/results/', 
                                         genes[i], '.fold.', fold, '.score.', s, '/results.csv'))
      hsu.unirep_onehot.auc$R2 <- c(hsu.unirep_onehot.auc$R2, test.result.hsu$spearman[match('eunirep_ll+onehot', test.result.hsu$predictor)])
      hsu.ev_onehot.auc$R2 <- c(hsu.ev_onehot.auc$R2, test.result.hsu$spearman[match('ev+onehot', test.result.hsu$predictor)])
      hsu.gesm_onehot.auc$R2 <- c(hsu.gesm_onehot.auc$R2, test.result.hsu$spearman[match('gesm+onehot', test.result.hsu$predictor)])
      hsu.eve_onehot.auc$R2 <- c(hsu.eve_onehot.auc$R2, test.result.hsu$spearman[match('vae+onehot', test.result.hsu$predictor)])
    }
    PreMode.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], test.result[,paste0("logits.", 0:(task.length-1))], bin = grepl('bin', genes[i]))
    PreMode.pass.auc <- plot.R2(test.result.pass[,names(task.dic[[genes[i]]])], test.result.pass[,paste0("logits.", 0:(task.length-1))], bin = grepl('bin', genes[i]))
    
    to.append <- data.frame(min.val.R = c(PreMode.auc$R2,  
                                          PreMode.pass.auc$R2, 
                                          hsu.unirep_onehot.auc$R2,
                                          hsu.ev_onehot.auc$R2,
                                          hsu.gesm_onehot.auc$R2,
                                          hsu.eve_onehot.auc$R2),
                            task.name = paste0(genes[i], ":", rep(task.dic[[genes[i]]], 6)),
                            HGNC=genes[i],
                            fold=fold,
                            npoints=dim(test.result)[1])
    to.append$model <- rep(c("PreMode", 
                             "ESM+SLP",  
                             "Augmented Unirep",
                             "Augmented EVmutation",
                             "Augmented ESM1b",
                             "Augmented EVE"), each = task.length)
    result <- rbind(result, to.append)
  }
}
num.models <- length(unique(result$model))
p <- ggplot(result, aes(y=min.val.R, x=task.name, col=model)) +
  geom_point(alpha=0) +
  stat_summary(data = result,
               aes(x=as.numeric(factor(task.name))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = min.val.R, col=model), 
               fun.data = mean_se, geom = "errorbar", width = 0.2) +
  stat_summary(data = result, 
               aes(x=as.numeric(factor(task.name))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = min.val.R, col=model), 
               fun.data = mean_se, geom = "point") +
  labs(x = "task", y = "min.val.R", fill = "model") +
  theme_bw() + 
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        legend.position="bottom", 
        legend.direction="horizontal") +
  # ylim(-1, 1) +
  coord_flip() + guides(col=guide_legend(ncol=1)) + ggtitle('Transfer Learning Compare') +
  ggeasy::easy_center_title() +
  xlab('task: Molecular mode-of-action') + ylab('Spearman Rho')
ggsave(paste0('figs/fig.4a.pdf'), p, height = 8, width = 4)

# plot the task weighted averages as well as task size weighted error bars
uniq.result.plot <- result[result$fold==0,]
for (i in 1:dim(uniq.result.plot)[1]) {
  uniq.result.plot$rho[i] = mean(result$min.val.R[result$model==uniq.result.plot$model[i] & 
                                              result$task.name==uniq.result.plot$task.name[i]], na.rm=T)
  uniq.result.plot$rho.sd[i] = sd(result$min.val.R[result$model==uniq.result.plot$model[i] & 
                                               result$task.name==uniq.result.plot$task.name[i]], na.rm=T)
}
# aggregate across models
uniq.model.result.plot <- uniq.result.plot[!duplicated(uniq.result.plot[,c('model', 'HGNC')]),]
for (i in 1:dim(uniq.model.result.plot)[1]) {
  uniq.model.result.plot$stab.rho[i] <- mean(uniq.result.plot$rho[uniq.result.plot$HGNC==uniq.model.result.plot$HGNC[i] & 
                                                                 grepl('stability', uniq.result.plot$task.name) &
                                                                 uniq.result.plot$model == uniq.model.result.plot$model[i]])
  uniq.model.result.plot$stab.rho.sd[i] <- mean(uniq.result.plot$rho.sd[uniq.result.plot$HGNC==uniq.model.result.plot$HGNC[i] & 
                                                                    grepl('stability', uniq.result.plot$task.name) &
                                                                    uniq.result.plot$model == uniq.model.result.plot$model[i]])
  uniq.model.result.plot$func.rho[i] <- mean(uniq.result.plot$rho[uniq.result.plot$HGNC==uniq.model.result.plot$HGNC[i] & 
                                                                    !grepl('stability', uniq.result.plot$task.name) &
                                                                    uniq.result.plot$model == uniq.model.result.plot$model[i]])
  uniq.model.result.plot$func.rho.sd[i] <- mean(uniq.result.plot$rho.sd[uniq.result.plot$HGNC==uniq.model.result.plot$HGNC[i] & 
                                                                     !grepl('stability', uniq.result.plot$task.name) &
                                                                     uniq.result.plot$model == uniq.model.result.plot$model[i]])
  
}

# aggregate across models
uniq.model.result.plot.plot <- uniq.model.result.plot[!duplicated(uniq.model.result.plot$model),]
for (i in 1:dim(uniq.model.result.plot.plot)[1]) {
  task.sizes <- uniq.model.result.plot$npoints[uniq.model.result.plot$model==uniq.model.result.plot$model[i]] 
  uniq.model.result.plot.plot$stab.rho[i] <- sum(uniq.model.result.plot$stab.rho[uniq.model.result.plot$model==uniq.model.result.plot.plot$model[i]] * task.sizes / sum(task.sizes), na.rm=T)
  uniq.model.result.plot.plot$stab.rho.sd[i] <- sum(uniq.model.result.plot$stab.rho.sd[uniq.model.result.plot$model==uniq.model.result.plot.plot$model[i]] * task.sizes / sum(task.sizes), na.rm=T)
  uniq.model.result.plot.plot$func.rho[i] <- sum(uniq.model.result.plot$func.rho[uniq.model.result.plot$model==uniq.model.result.plot.plot$model[i]] * task.sizes / sum(task.sizes), na.rm=T)
  uniq.model.result.plot.plot$func.rho.sd[i] <- sum(uniq.model.result.plot$func.rho.sd[uniq.model.result.plot$model==uniq.model.result.plot.plot$model[i]] * task.sizes / sum(task.sizes), na.rm=T)
}

p <- ggplot(uniq.model.result.plot.plot, aes(x=stab.rho, y=func.rho, col=model)) +
  geom_point() +
  geom_errorbar(aes(ymin=func.rho-func.rho.sd, ymax=func.rho+func.rho.sd), width=.02) +
  geom_errorbarh(aes(xmin=stab.rho-stab.rho.sd, xmax=stab.rho+stab.rho.sd), height=.02) +
  # coord_flip() +guides(col=guide_legend(ncol=2)) +
  labs(x = "stab.rho", y = "func.rho", fill = "model") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha=0.2) +
  theme_bw() + xlim(0.15, 0.7) + ylim(0.15, 0.7) +
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        legend.position="right", 
        legend.direction="vertical") + 
  ggtitle('Transfer Learning Compare\n(Weighted Average by Dataset Sizes)') +
  ggeasy::easy_center_title()
ggsave('figs/fig.4b.pdf', p, height=4, width=5)
