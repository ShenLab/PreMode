# source('utils.R')
library(ggplot2)
# args <- commandArgs(trailingOnly = T)
# base dir for transfer learning
base.dir <- "/share/terra/Users/gz2294/PreMode.final/scripts/CHPs.v4.esm.dssp.small.StarAttn.MSA.StarPool.1dim/"
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
# base.dirs <- strsplit(base.dirs, split = ',')[[1]]
# results <- data.frame()
# for (base.dir in base.dirs) {
genes <- c("PTEN", "NUDT15", "CCR5", "CXCR4", 'SNCA', 'CYP2C9', 'GCK', 'ASPA')
# add baseline AUC
# esm alphabets
source('~/Pipeline/AUROC.R')
source('~/Pipeline/uniprot.table.add.annotation.R')
alphabet <- c('<cls>', '<pad>', '<eos>', '<unk>',
              'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
              'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
              'X', 'B', 'U', 'Z', 'O', '.', '-',
              '<null_1>', '<mask>')
for (i in 1:length(genes)) {
  # add ESM
  assay <- read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/MAVEDB/', genes[i], '/ALL.annotated.csv'))
  assay$unique.id <- paste0(assay$ref, assay$pos.orig, assay$alt)
  for (fold in 0:4) {
    if (!file.exists(paste0('PreMode.inference/', genes[i], '/test.fold.',
                            fold, '.annotated.csv'))) {
      test.result <- read.csv(paste0('PreMode.inference/', genes[i], '/',
                                     '/testing.fold.', fold, '.csv'))
      test.result$unique.id <- paste0(test.result$ref, test.result$pos.orig, test.result$alt)
      
      pretrain.logits <- read.csv(paste0('PreMode.inference/', genes[i], '/',
                                         '/testing.pretrain.fold.', fold, '.csv'))
      test.result$pretrain.logits <- pretrain.logits$logits
      for (anno in c('ESM1b.LLR', 'AlphaMissense', 'EVE', 'REVEL', 'PrimateAI',
                       'gMVP', 'FoldXddG', 'conservation.entropy', 'RosettaddG')) {
        test.result[,anno] <- assay[match(test.result$unique.id, assay$unique.id), anno]
      }
      write.csv(test.result, paste0('PreMode.inference/', genes[i], '/',
                                    '/test.fold.', fold, '.annotated.csv'))
    }
  }
}
# # first plot PreMode pretrained auc vs other scores
# result.df <- NULL
# scores <- c('AlphaMissense', 'gMVP', 'EVE', 'PrimateAI', 'REVEL', 'ESM1b.LLR', 'pretrain.logits', 'PreMode')
# for (i in 1:length(genes)) {
#   for (fold in 0:4) {
#   dms.df <- read.csv(paste0('PreMode.inference/', genes[i], '/',
#                             '/test.fold.', fold, '.annotated.csv'))
#   # calculate R2
#   stab.r <- NULL
#   other.r <- NULL
#   for (score in scores) {
#     if (score != 'PreMode') {
#       all.r <- abs(plot.R2(dms.df[,names(task.dic[[genes[i]]])],
#                            dms.df[,rep(score, length(task.dic[[genes[i]]]))])$R2)
#     } else {
#       all.r <- abs(plot.R2(dms.df[,names(task.dic[[genes[i]]])],
#                            dms.df[,paste0('logits.', 1:length(task.dic[[genes[i]]])-1)])$R2)
#     }
#     stab.r <- c(stab.r, all.r[which(task.dic[[genes[i]]]=='stability')])
#     other.r <- c(other.r, mean(all.r[-which(task.dic[[genes[i]]]=='stability')]))
#   }
#   model.names <- scores
#   model.names[model.names=='pretrain.logits'] <- 'PreMode.pretrain'
#   result.df <- rbind(result.df,
#                      data.frame(model=model.names,
#                                 HGNC=genes[i],
#                                 fold=fold,
#                                 npoints=dim(dms.df)[1],
#                                 stab.rho=stab.r,
#                                 other.rho=other.r))
#   }
# }
# # plot the task weighted averages as well as task size weighted error bars
# uniq.result.plot <- result.df[result.df$fold==0,]
# for (i in 1:dim(uniq.result.plot)) {
#   uniq.result.plot$stab.rho[i] = mean(result.df$stab.rho[result.df$model==uniq.result.plot$model[i] & 
#                                                  result.df$HGNC==uniq.result.plot$HGNC[i]], na.rm=T)
#   uniq.result.plot$stab.rho.sd[i] = sd(result.df$stab.rho[result.df$model==uniq.result.plot$model[i] & 
#                                                   result.df$HGNC==uniq.result.plot$HGNC[i]], na.rm=T)
#   uniq.result.plot$other.rho[i] = mean(result.df$other.rho[result.df$model==uniq.result.plot$model[i] & 
#                                                            result.df$HGNC==uniq.result.plot$HGNC[i]], na.rm=T)
#   uniq.result.plot$other.rho.sd[i] = sd(result.df$other.rho[result.df$model==uniq.result.plot$model[i] & 
#                                                        result.df$HGNC==uniq.result.plot$HGNC[i]], na.rm=T)
#   
# }
# p <- ggplot(uniq.result.plot, aes(x=stab.rho, y=other.rho, shape=model, col=HGNC, size=npoints)) + 
#   geom_point() +
#   # geom_errorbar(aes(ymin=other.rho-other.rho.sd, ymax=other.rho+other.rho.sd)) +
#   # geom_errorbarh(aes(xmin=stab.rho-stab.rho.sd, xmax=stab.rho+stab.rho.sd)) +
#   geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha=0.2) +
#   scale_shape_manual(values = 11:18) +
#   theme_bw() + ggeasy::easy_center_title()
# ggsave('figs/02.01.MAVE.PreMode.compare.xy.pdf', p, height = 8, width = 12)

result <- data.frame()
for (i in 1:length(genes)) {
  for (fold in 0:4) {
    # REVEL, PrimateAI, ESM AUC
    test.result <- read.csv(paste0('PreMode.inference/', genes[i], '/',
                                   '/test.fold.', fold, '.annotated.csv'), row.names = 1)
    # test.result.onehot <- read.csv(paste0('PreMode.onehot.inference/', genes[i], '/',
    #                                       '/testing.fold.', fold, '.csv'))
    test.result.big <- read.csv(paste0('PreMode.PRE.v4/archive/', genes[i], '/',
                                          '/testing.fold.', fold, '.csv'))
    test.result.big.2 <- read.csv(paste0('PreMode.PRE.v8/', genes[i], '/',
                                       '/testing.fold.', fold, '.csv'))
    test.result.3 <- read.csv(paste0('PRE.v10/', genes[i], '/',
                                         '/testing.fold.', fold, '.csv'))
    test.result.4 <- read.csv(paste0('PRE.v11/', genes[i], '/',
                                         '/testing.fold.', fold, '.csv'))
    test.result.pass <- read.csv(paste0('PreMode.pass.inference/', genes[i], '/',
                                        '/testing.fold.', fold, '.csv'))
    task.length <- length(task.dic[[genes[i]]])
    # add hsu et al results
    hsu.unirep_onehot.auc <- list(R2=c())
    hsu.ev_onehot.auc <- list(R2=c())
    hsu.gesm_onehot.auc <- list(R2=c())
    hsu.eve_onehot.auc <- list(R2=c())
    for (s in 1:task.length) {
      test.result.hsu <- read.csv(paste0('/share/vault/Users/gz2294/combining-evolutionary-and-assay-labelled-data/results/', 
                                         genes[i], '.fold.', fold, '.score.', s, '/results.csv'))
      hsu.unirep_onehot.auc$R2 <- c(hsu.unirep_onehot.auc$R2, test.result.hsu$spearman[match('eunirep_ll+onehot', test.result.hsu$predictor)])
      hsu.ev_onehot.auc$R2 <- c(hsu.ev_onehot.auc$R2, test.result.hsu$spearman[match('ev+onehot', test.result.hsu$predictor)])
      hsu.gesm_onehot.auc$R2 <- c(hsu.gesm_onehot.auc$R2, test.result.hsu$spearman[match('gesm+onehot', test.result.hsu$predictor)])
      hsu.eve_onehot.auc$R2 <- c(hsu.eve_onehot.auc$R2, test.result.hsu$spearman[match('vae+onehot', test.result.hsu$predictor)])
    }
    for (a in 0:(task.length-1)) {
      test.result.big[,paste0('logits.alt.', a)] <- NA
    }
    for (k in 1:dim(test.result.big)[1]) {
      for (a in 0:(task.length-1)) {
        test.result.big[k,paste0('logits.alt.', a)] <- test.result.big[k, paste0('logits.', a+2*(match(test.result.big$alt[k], alphabet_premode)-1))]
      }
    }
    PreMode.big.auc <- plot.R2(test.result.big[,names(task.dic[[genes[i]]])], test.result.big[,paste0("logits.alt.", 0:(task.length-1))], bin = grepl('bin', genes[i]))
    PreMode.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], test.result[,paste0("logits.", 0:(task.length-1))], bin = grepl('bin', genes[i]))
    # print(PreMode.big.auc$R2)
    # print(PreMode.auc$R2)
    PreMode.pretrain.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], -test.result[,rep("pretrain.logits", task.length)], bin = grepl('bin', genes[i]))
    # PreMode.onehot.auc <- plot.R2(test.result.onehot[,names(task.dic[[genes[i]]])], test.result.onehot[,paste0("logits.", 0:(task.length-1))], bin = grepl('bin', genes[i]))
    PreMode.pass.auc <- plot.R2(test.result.pass[,names(task.dic[[genes[i]]])], test.result.pass[,paste0("logits.", 0:(task.length-1))], bin = grepl('bin', genes[i]))
    PreMode.big.auc.2 <- plot.R2(test.result.big.2[,names(task.dic[[genes[i]]])], test.result.big.2[,paste0("logits.", 0:(task.length-1))], bin = grepl('bin', genes[i]))
    PreMode.auc.3 <- plot.R2(test.result.3[,names(task.dic[[genes[i]]])], test.result.3[,paste0("logits.", 0:(task.length-1))], bin = grepl('bin', genes[i]))
    PreMode.auc.4 <- plot.R2(test.result.4[,names(task.dic[[genes[i]]])], test.result.4[,paste0("logits.", 0:(task.length-1))], bin = grepl('bin', genes[i]))
    
    REVEL.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], -test.result[,rep("REVEL", task.length)], bin = grepl('bin', genes[i]))
    PrimateAI.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], -test.result[,rep("PrimateAI", task.length)], bin = grepl('bin', genes[i]))
    ESM.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], test.result[,rep("ESM1b.LLR", task.length)], bin = grepl('bin', genes[i]))
    EVE.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], -test.result[,rep("EVE", task.length)], bin = grepl('bin', genes[i]))
    gMVP.auc <- plot.R2(test.result[,names(task.dic[[genes[i]]])], -test.result[,rep("gMVP", task.length)], bin = grepl('bin', genes[i]))
    to.append <- data.frame(min.val.R = c(PreMode.pretrain.auc$R2, 
                                          PreMode.auc$R2, 
                                          # PreMode.onehot.auc$R2, 
                                          PreMode.pass.auc$R2, 
                                          # PreMode.big.auc$R2,
                                          PreMode.big.auc.2$R2,
                                          PreMode.auc.3$R2,
                                          PreMode.auc.4$R2,
                                          hsu.unirep_onehot.auc$R2,
                                          hsu.ev_onehot.auc$R2,
                                          hsu.gesm_onehot.auc$R2,
                                          hsu.eve_onehot.auc$R2,
                                          REVEL.auc$R2, PrimateAI.auc$R2, ESM.auc$R2, EVE.auc$R2, gMVP.auc$R2),
                            task.name = paste0(genes[i], ":", rep(task.dic[[genes[i]]], 15)),
                            HGNC=genes[i],
                            fold=fold,
                            npoints=dim(test.result)[1])
    to.append$model <- rep(c("PreMode.zero.shot", "PreMode (148k)", 
                             "ESM2 emb + SLP", "PreMode (4.7M)", 
                             "PreMode (v10)", "PreMode (v11)", 
                             "Augmented Unirep",
                             "Augmented EVmutation",
                             "Augmented ESM1b",
                             "Augmented EVE",
                             "REVEL", "PrimateAI", "ESM", 'EVE', 'gMVP'), each = task.length)
    result <- rbind(result, to.append)
  }
}
write.csv(result, 'figs/02.01.MAVE.PreMode.compare.csv')

result <- read.csv('figs/02.01.MAVE.PreMode.compare.csv', row.names = 1)
result <- result[result$model %in% c("PreMode (148k)", "ESM2 emb + SLP", 
                                     # "PreMode (4.7M)", 
                                     "Augmented Unirep",
                                     "Augmented EVmutation",
                                     "Augmented ESM1b",
                                     "Augmented EVE"),]
# result$min.val.R[result$task.name == "ASPA:enzyme.activity"] <- -result$min.val.R[result$task.name == "ASPA:enzyme.activity"]
num.models <- length(unique(result$model))
p <- ggplot(result, aes(y=min.val.R, x=task.name, col=model)) +
  geom_point(alpha=0.2) +
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
ggsave(paste0('figs/02.01.MAVE.PreMode.compare.pdf'), p, height = 8, width = 4)


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
  theme_bw() + xlim(0.15, 0.65) + ylim(0.15, 0.65) +
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        legend.position="right", 
        legend.direction="vertical") + 
  ggtitle('Transfer Learning Compare\n(Weighted Average by Dataset sizes)') +
  ggeasy::easy_center_title()
ggsave('figs/02.01.MAVE.PreMode.compare.models.pdf', p, height=4, width=5)
