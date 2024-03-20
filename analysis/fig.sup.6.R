library(ggplot2)
task.dic <- list("Stab"=c("score.1"="stability.1", "score.2"="stability.2"))

alphabet_premode <- c('L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
              'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C')
genes <- c("Stab")
scores <- c('AlphaMissense', 'gMVP', 'PrimateAI', 'REVEL', 'ESM1b.LLR', 'FoldXddG')
models <- c('PreMode.inference/', 'PreMode.pass.inference/')
models.dic <- c('PreMode.inference/'='PreMode',
                'PreMode.pass.inference/'='ESM + SLR')
# add baseline AUC
# esm alphabets
source('。/AUROC.R')
alphabet <- c('<cls>', '<pad>', '<eos>', '<unk>',
              'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
              'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
              'X', 'B', 'U', 'Z', 'O', '.', '-',
              '<null_1>', '<mask>')
# first plot PreMode pretrained auc vs other scores
result.df <- NULL
scores <- c(scores, models)
for (i in 1:length(genes)) {
  for (fold in 0:4) {
  dms.df <- read.csv(paste0('PreMode.inference/', genes[i], '/',
                            '/test.fold.', fold, '.annotated.csv'))
  # calculate R2
  stab.r <- NULL
  other.r <- NULL
  for (score in scores) {
    if (score %in% models) {
      dms.df <- read.csv(paste0(score, genes[i], '/',
                                '/testing.fold.', fold, '.csv'))
      all.r <- abs(plot.R2(dms.df[,names(task.dic[[genes[i]]])],
                           dms.df[,paste0('logits.', 1:length(task.dic[[genes[i]]])-1)])$R2)
    } else {
      all.r <- abs(plot.R2(dms.df[,names(task.dic[[genes[i]]])],
                           dms.df[,rep(score, length(task.dic[[genes[i]]]))])$R2)
    }
    stab.r <- c(stab.r, mean(all.r))
  }
  model.names <- scores
  model.names[model.names %in% models] <- models.dic[model.names[model.names %in% models]]
  result.df <- rbind(result.df,
                     data.frame(model=model.names,
                                HGNC=genes[i],
                                fold=fold,
                                npoints=dim(dms.df)[1],
                                stab.rho=stab.r))
  }
}
# plot the task weighted averages as well as task size weighted error bars
uniq.result.plot <- result.df[result.df$fold==0,]
for (i in 1:dim(uniq.result.plot)) {
  uniq.result.plot$stab.rho[i] = mean(result.df$stab.rho[result.df$model==uniq.result.plot$model[i] & 
                                                 result.df$HGNC==uniq.result.plot$HGNC[i]], na.rm=T)
  uniq.result.plot$stab.rho.sd[i] = sd(result.df$stab.rho[result.df$model==uniq.result.plot$model[i] & 
                                                  result.df$HGNC==uniq.result.plot$HGNC[i]], na.rm=T)
  
}
p <- ggplot(uniq.result.plot, aes(x=stab.rho, y=model)) + 
  geom_point() +
  geom_errorbarh(aes(xmin=stab.rho-stab.rho.sd, xmax=stab.rho+stab.rho.sd), height=.2) +
  ggtitle("Spearman Correlation (5 Fold testing)") +
  theme_bw() + ggeasy::easy_center_title()
ggsave('figs/fig.sup.6.pdf', p, height = 4, width = 5)

