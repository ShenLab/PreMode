# genes involved in compare
genes <- c('PTEN', 'NUDT15', 'CCR5', 'CXCR4', 'SNCA', 'CYP2C9', 'GCK', 'ASPA')
stab.assays <- c(1, 1, 1, 1, 2, 2, 2, 1)
scores <- c('AlphaMissense', 'gMVP', 'PrimateAI', 'REVEL', 'ESM1b.LLR', 'FoldXddG')
result.df <- NULL
source('/share/vault/Users/gz2294/Pipeline/uniprot.table.add.annotation.R')
source('~/Pipeline/AUROC.R')
# models <- c('CHPs.v4.confidence', 'CHPs.v4.confidence.weighted.v2',
#             'CHPs.v4.noMSA', 'CHPs.v4.noStructure', 'CHPs.v4.onehot.dssp.small.StarAttn.MSA.StarPool.1dim',
#             'CHPs.v4.pass', 'CHPs.v4.ptm', 'PreMode', 'PreMode.PRE.v4', 'PreMode.PRE.v4.fullgraph', 
#             'PreMode.v5', 'PRE.v5.fullGraph.89600')
models <- c('PreMode', 'PreMode.PRE.v4', 'PreMode.PRE.v8', 'CHPs.v4.esm_mask', 'PRE.v12', 'PRE.v13', 'PRE.v15'
            # 'PRE.v9', 'PRE.v10'
            )
scores <- c(scores, models)
alphabet <- c('L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
                'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C')
for (i in 1:length(genes)) {
  gene <- genes[i]
  dms.df <- read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/MAVEDB/', genes[i], '/',
                            'ALL.annotated.csv'))
  # annotate scores
  dms.df <- dms.df[!is.na(dms.df$alt),]
  assays <- colnames(dms.df)[startsWith(colnames(dms.df), 'score.')]
  dms.df <- dms.df[rowSums(is.na(dms.df[,assays]))==0,]
  # dms.df <- dms.df[rowSums(is.na(dms.df[,scores[!scores %in% c('EVE', models)]]))==0,]
  # calculate R2
  stab.r <- NULL
  other.r <- NULL
  for (score in scores) {
    if (score %in% models) {
      dms.df <- read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/MAVEDB/', genes[i], '/',
                                'ALL.annotated.', score, '.csv'))
      if (startsWith(score, 'esm2')) {
        dms.df.esm <- dms.df
        dms.df <- read.csv(paste0('/share/vault/Users/gz2294/Data/DMS/MAVEDB/', genes[i], '/',
                                       'ALL.annotated.csv'))
        alphabet_esm <- c('<cls>', '<pad>', '<eos>', '<unk>',
                          'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
                          'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
                          'X', 'B', 'U', 'Z', 'O', '.', '-',
                          '<null_1>', '<mask>')
        dms.df$logits <- NA
        for (k in 1:dim(dms.df)[1]) {
          dms.df$logits[k] <- dms.df.esm[k, paste0('X', match(dms.df$alt[k], alphabet_esm)-1)] - 
            dms.df.esm[k, paste0('X', match(dms.df$ref[k], alphabet_esm)-1)]
        }
      } else {
        dms.df <- dms.df[!is.na(dms.df$alt),]
        assays <- colnames(dms.df)[startsWith(colnames(dms.df), 'score.')]
        dms.df <- dms.df[rowSums(is.na(dms.df[,assays]))==0,]
        if (!'logits' %in% colnames(dms.df)) {
          # means we use full model
          dms.df$logits <- NA
          for (k in 1:dim(dms.df)[1]) {
            dms.df$logits[k] <- dms.df[k, paste0('logits.', match(dms.df$alt[k], alphabet)-1)]
          }
        } 
      }
      all.r <- abs(plot.R2(dms.df[,assays],
                           dms.df[,rep('logits', length(assays))])$R2)
    } else {
      all.r <- abs(plot.R2(dms.df[,assays],
                           dms.df[,rep(score, length(assays))])$R2)
    }
    stab.r <- c(stab.r, all.r[stab.assays[i]])
    other.r <- c(other.r, mean(all.r[-stab.assays[i]]))
  }
  model.names <- scores
  model.names[model.names=='PreMode'] <- 'PreMode (148k)'
  model.names[model.names=='PreMode.PRE.v4'] <- 'PreMode (83M)'
  model.names[model.names=='PreMode.PRE.v8'] <- 'PreMode (4.7M)'
  model.names[model.names=='CHPs.v4.esm_mask'] <- 'ESM fine tune'
  
  # calculate the coverage of experiments
  coverage <- sum(dms.df$pos.orig != 1) / (nchar(dms.df$wt.orig)[1] - 1) / 19
  result.df <- rbind(result.df,
                     data.frame(model=model.names,
                                HGNC=gene,
                                npoints=dim(dms.df)[1],
                                coverage=coverage,
                                stab.rho=stab.r,
                                other.rho=other.r))
}
write.csv(result.df, file = 'figs/01.04.MAVEDB.compare.csv')
library(ggplot2)
plots <- list()
for (i in 1:length(genes)) {
  plots[[i]] <- ggplot(result.df[result.df$HGNC==genes[i],], aes(x=stab.rho, y=other.rho, col=model)) + 
    geom_point() +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha=0.2) +
    ggtitle(genes[i]) + 
    theme_bw() + ggeasy::easy_center_title()
}
p <- ggpubr::ggarrange(plotlist = plots, ncol = 4, nrow = 2)
ggsave('figs/01.04.MAVEDB.compare.pdf', p, height = 7, width = 16)

p <- ggplot(result.df, aes(x=stab.rho, y=other.rho, shape=HGNC, col=model, size=npoints)) + 
  geom_point() +
  scale_shape_manual(values = 11:18) +
  # scale_color_manual(values = randomcoloR::distinctColorPalette(20)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha=0.2) +
  theme_bw() +
  theme(text = element_text(size=20)) + xlim(0, 0.6) + ylim(0, 0.6) +
  ggeasy::easy_center_title()
ggsave('figs/01.04.MAVEDB.compare.all.pdf', p, height = 8, width = 12)
uniq.result.df <- result.df[result.df$HGNC=='PTEN',]
for (i in 1:dim(uniq.result.df)[1]) {
  result.df.model <- result.df[result.df$model==uniq.result.df$model[i],]
  result.df.model <- result.df.model[!is.na(result.df.model$stab.rho),]
  uniq.result.df$stab.rho[i] <- sum(result.df.model$stab.rho * result.df.model$npoints, na.rm = T) / sum(result.df.model$npoints)
  uniq.result.df$other.rho[i] <- sum(result.df.model$other.rho * result.df.model$npoints, na.rm = T) / sum(result.df.model$npoints)
}
write.csv(uniq.result.df, paste0('figs/01.04.MAVEDB.compare.models.csv'))
# plot model
uniq.result.df$HGNC <- 'Weighted Sum'
p <- ggplot(uniq.result.df, aes(x=stab.rho, y=other.rho, col=model)) + 
  geom_point(size=2) +
  # scale_shape_manual(values = 10:18) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha=0.2) +
  theme_bw() +
  # theme(text = element_text(size=20)) +
  xlim(0.15, 0.65) + ylim(0.15, 0.65) +
  ggtitle('Zero Shot Compare\n(Weighted Average by Dataset sizes)') +
  ggeasy::easy_center_title()
ggsave('figs/01.04.MAVEDB.compare.models.pdf', p, height = 4, width = 5)
