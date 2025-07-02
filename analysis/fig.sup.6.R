library(ggplot2)
task.dic <- list("Stab"=c("score.1"="stability.1", "score.2"="stability.2"))
py.path <- '/share/vault/Users/gz2294/miniconda3/envs/RESCVE/bin/python'
alphabet_premode <- c('L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
              'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C')
genes <- c("Stab")
scores <- c('AlphaMissense', 'gMVP', 'PrimateAI', 'REVEL', 'ESM1b.LLR', 'FoldXddG')
models <- c('PreMode/', 'ESM.SLP/')
models.dic <- c('PreMode/'='PreMode', "ESM.SLP/"='ESM+SLP')
# add baseline AUC
# esm alphabets
source('./AUROC.R')
source('./prepare.biochem.R')
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
  dms.df <- read.csv(paste0('PreMode/', genes[i], '/',
                            '/test.fold.', fold, '.annotated.csv'))
  # calculate R2
  stab.r <- NULL
  other.r <- NULL
  for (score in scores) {
    if (score %in% models) {
      dms.df.score <- read.csv(paste0(score, genes[i], '/',
                                '/testing.fold.', fold, '.csv'))
      # we need to calculate all.r across proteins and average
      all.rs <- c()
      for (uid in unique(dms.df.score$uniprotID)) {
        all.r <- abs(plot.R2(dms.df.score[dms.df.score$uniprotID==uid, names(task.dic[[genes[i]]])],
                             dms.df.score[dms.df.score$uniprotID==uid, paste0('logits.', 1:length(task.dic[[genes[i]]])-1)])$R2)
        all.r <- mean(all.r)
        all.rs <- c(all.rs, all.r)
      }
    } else {
      all.rs <- c()
      for (uid in unique(dms.df$uniprotID)) {
        all.r <- abs(plot.R2(dms.df[dms.df$uniprotID==uid, names(task.dic[[genes[i]]])],
                             dms.df[dms.df$uniprotID==uid, rep(score, length(task.dic[[genes[i]]]))])$R2)
        all.r <- max(all.r)
        all.rs <- c(all.rs, all.r)
      }
    }
    stab.r <- c(stab.r, mean(all.rs))
  }
  model.names <- scores
  model.names[model.names %in% models] <- models.dic[model.names[model.names %in% models]]
  result.df <- rbind(result.df,
                     data.frame(model=model.names,
                                HGNC=genes[i],
                                fold=fold,
                                npoints=dim(dms.df)[1],
                                stab.rho=stab.r))
  # add biochem properties
  # write train and test emb to files
  dms.train.df <- read.csv(paste0('PreMode/', genes[i], '/',
                                  '/train.fold.', fold, '.annotated.csv'))
  dms.df <- read.csv(paste0('PreMode/', genes[i], '/',
                            '/test.fold.', fold, '.annotated.csv'))
  dms.train.df <- prepare.unique.id(dms.train.df)
  dms.df <- prepare.unique.id(dms.df)
  # get train and test biochemical
  gene.train.biochem <- prepare.biochemical(dms.train.df)
  gene.test.biochem <- prepare.biochemical(dms.df)
  # write train and test emb to files
  train.label.file <- tempfile()
  test.label.file <- tempfile()
  train.biochem.file <- tempfile()
  test.biochem.file <- tempfile()
  test.res.file <- tempfile()
  write.csv(dms.train.df, file = train.label.file)
  write.csv(dms.df, file = test.label.file)
  write.csv(gene.train.biochem, file = train.biochem.file)
  write.csv(gene.test.biochem, file = test.biochem.file)
  res <- system(paste0(py.path, ' ', 
                       'elastic.net.dms.py ', 
                       train.biochem.file, ' ',
                       train.label.file, ' ',
                       test.biochem.file, ' ', 
                       test.label.file, ' ',
                       test.res.file), intern = T)
  res <- read.csv(test.res.file)
  all.rs <- c()
  for (uid in unique(dms.df$uniprotID)) {
    all.r <- abs(plot.R2(dms.df[dms.df$uniprotID==uid,names(task.dic[[genes[i]]])],
                         res[dms.df$uniprotID==uid,c('score.1', 'score.2')])$R2)
    all.r <- mean(all.r)
    all.rs <- c(all.rs, all.r)
  }
  result.df <- rbind(result.df,
                     data.frame(model=c('Elastic Net'),
                                HGNC='Stab',
                                fold=fold,
                                npoints=dim(dms.df)[1],
                                stab.rho=c(mean(all.rs))))
  }
}
write.csv(result.df, './figs/fig.sup.6.csv')
# plot the task weighted averages as well as task size weighted error bars
uniq.result.plot <- result.df[result.df$fold==0,]
for (i in 1:dim(uniq.result.plot)[1]) {
  uniq.result.plot$stab.rho[i] = mean(result.df$stab.rho[result.df$model==uniq.result.plot$model[i] & 
                                                 result.df$HGNC==uniq.result.plot$HGNC[i]], na.rm=T)
  uniq.result.plot$stab.rho.sd[i] = sd(result.df$stab.rho[result.df$model==uniq.result.plot$model[i] & 
                                                  result.df$HGNC==uniq.result.plot$HGNC[i]], na.rm=T)
  
}
p <- ggplot(uniq.result.plot, aes(x=stab.rho, y=model)) + 
  geom_point() +
  # geom_errorbar(aes(ymin=other.rho-other.rho.sd, ymax=other.rho+other.rho.sd)) +
  geom_errorbarh(aes(xmin=stab.rho-stab.rho.sd, xmax=stab.rho+stab.rho.sd), height=.2) +
  # geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha=0.2) +
  scale_shape_manual(values = 11:18) +
  ggtitle("Spearman Correlation (5 Fold testing)") +
  theme_bw() + ggeasy::easy_center_title()
ggsave('figs/fig.sup.6.pdf', p, height = 4, width = 5)
