# several questions to ask
# Average MSE, PreMode is better than experiment
# For points that have different replicates, which is better.
# For all points in each experiment, which point has better MSE.

# First get ground truth
ground.truth <- read.csv(paste0('/share/pascal/Users/gz2294/Data/DMS/MAVEDB/PTEN/assay.1.csv'), row.names = 1)
# Next set up metrics
all.premode <- list(c(), c(), c(), c(), c(), c(), c(), c())
all.baseline <- list(c(), c(), c(), c(), c(), c(), c(), c())
all.replicates <- list()
for (r in 1:8) {
  replicate <- read.csv(paste0('PreMode.inference/PTEN/replicate.', r, '.csv'))
  training <- read.csv(paste0('/share/pascal/Users/gz2294/Data/DMS/MAVEDB/PTEN.replicate.rest.', 
                              r, '/training.csv'), row.names = 1)
  # ground.truth$aaChg <- paste0('p.', ground.truth$ref, ground.truth$pos.orig, ground.truth$alt)
  replicate$baseline <- NA
  replicate$observations <- NA
  replicate.unique <- replicate[!duplicated(replicate$aaChg),]
  for (i in 1:dim(replicate.unique)[1]) {
    baseline <- training[training$aaChg == replicate.unique$aaChg[i],]
    replicate.baseline <- replicate[replicate$aaChg == replicate.unique$aaChg[i] & !is.na(replicate$score),]
    replicate.unique$base.line.1[i] <- mean(baseline$score, na.rm=T)
    replicate.unique$base.line.2[i] <- mean(replicate.baseline$score, na.rm=T)
    replicate.unique$ground.truth[i] <- ground.truth$score[ground.truth$VarID==replicate.unique$aaChg[i]]
    replicate.unique$observations[i] <- dim(baseline)[1] + dim(replicate.baseline)[1]
    replicate.unique$other.observations[i] <- dim(replicate.baseline)[1]
  }
  # calculate MSE
  premode <- mean((replicate.unique$base.line.2 - replicate.unique$logits)^2, na.rm = T)
  baseline <- mean((replicate.unique$base.line.1 - replicate.unique$base.line.2)^2, na.rm = T)
  all.replicates[[r]] <- replicate.unique
  all.premode[[1]] <- c(all.premode[[1]], premode)
  all.baseline[[1]] <- c(all.baseline[[1]], baseline)
  # next compare for each group of replicates
  for (i in min(replicate.unique$other.observations, na.rm = T):max(replicate.unique$other.observations, na.rm = T)) {
    premode <- mean((replicate.unique$ground.truth[replicate.unique$other.observations==i] -
                       replicate.unique$logits[replicate.unique$other.observations==i])^2, na.rm = T)
    baseline <- mean((replicate.unique$base.line.1[replicate.unique$other.observations==i] - 
                        replicate.unique$ground.truth[replicate.unique$other.observations==i])^2, na.rm = T)
    all.premode[[i+1]] <- c(all.premode[[i+1]], premode)
    all.baseline[[i+1]] <- c(all.baseline[[i+1]], baseline)
  }
  print(paste0('replicate ', r, ', PreMode: ', all.premode[[1]], ', Baseline: ', all.baseline[[1]]))
}
save(all.premode, all.baseline, all.replicates, file = 'figs/05.PTEN.bio.replicates.RData')
npoints <- table(all.replicates[[1]]$other.observations)
npoints <- c(sum(npoints), npoints)
names(npoints)[1] <- 'all'
to.plot <- data.frame(RMSE=sqrt(c(unlist(all.premode), 
                            unlist(all.baseline))), 
                      exp = rep(rep(1:8, 8), 2),
                      replicate=paste0(rep(rep(names(npoints), each=8), 2), " : ",
                                       rep(rep(npoints, each=8), 2)),
                      model=c(rep("PreMode", length(names(npoints))*8), rep("Experiment", length(npoints)*8)))
write.csv(to.plot, file = 'figs/05.PTEN.bio.replicates.csv')
library(ggplot2)
library(tidyverse)
dodge <- position_dodge(width=0.24)
source('/share/pascal/Users/gz2294/Pipeline/geom_split_violin.R')
to.connect <- data.frame(replicate=paste0(rep(rep(names(npoints), each=8), 1), " : ",
                                          rep(rep(npoints, each=8), 1)),
                         y=sqrt(unlist(all.premode)),
                         y_end=sqrt(unlist(all.baseline)))
p1 <- ggplot(to.plot, aes(x=replicate, y=RMSE, fill=model, col=model)) + 
  # geom_boxplot() + 
  geom_split_violin(width=1) +
  geom_point(size=0.7, position=dodge, col='darkgrey') +
  geom_segment(data = to.connect,
               aes(x    = as.numeric(as.factor(replicate)) + 0.06, 
                   xend = as.numeric(as.factor(replicate)) - 0.06,
                   y = y, yend = y_end), inherit.aes = FALSE, color = "grey", alpha=0.5) +
  ylab("RMSE with other replicates") +
  xlab("Group by # of other replicates : # of points") +
  theme_bw() + ggtitle("RMSE with regard to experimental replicates") + ggeasy::easy_center_title()

# ggsave("figs/05.PTEN.bio.replicates.pdf", height = 5, width = 6)

# next do for each point, calculate the PreMode RMSE and experiment RMSE
replicate.unique <- all.replicates[[1]]
all.rmse.plot <- list()
for (r in 1:length(all.replicates)) {
  replicate.unique[,paste0("PreMode.MSE.", r)] <- (all.replicates[[r]]$ground.truth - all.replicates[[r]]$logits)
  replicate.unique[,paste0("Experiment.MSE.", r)] <- (all.replicates[[r]]$ground.truth - all.replicates[[r]]$base.line.1)
  replicate.unique[,paste0("PreMode.Experiment.Groundtruth.diff.", r)] <- abs(all.replicates[[r]]$ground.truth - all.replicates[[r]]$logits) - abs(all.replicates[[r]]$ground.truth - all.replicates[[r]]$base.line.1)
  all.rmse.plot[[r]] <- ggplot(replicate.unique, aes_string(x=paste0("PreMode.MSE.", r), y=paste0("Experiment.MSE.", r), col="observations")) +
    scale_color_gradient2(low='blue', high='red', mid = 'white', midpoint = 5) +
    geom_abline(intercept=0, slope=1, linetype='dashed') +
    xlab("PreMode - Groundtruth Diff") +
    ylab("Experiment - Groundtruth Diff") +
    theme_bw() +
    geom_point(alpha=0.5, size=0.5)
}
library(patchwork)
p7 <- all.rmse.plot[[1]] + all.rmse.plot[[2]] + all.rmse.plot[[3]] + all.rmse.plot[[4]] +
  all.rmse.plot[[5]] + all.rmse.plot[[6]] + all.rmse.plot[[7]] + all.rmse.plot[[8]] + patchwork::plot_layout(ncol=4)
ggsave('figs/05.PTEN.bio.replicates.PreMode.4.pdf', p7, height = 10, width = 20)
replicate.unique$PreMode.RMSE <- sqrt(rowMeans(replicate.unique[,paste0("PreMode.MSE.",1:length(all.replicates))], na.rm = T))
replicate.unique$Experiment.RMSE <- sqrt(rowMeans(replicate.unique[,paste0("Experiment.MSE.",1:length(all.replicates))], na.rm = T))
ggplot(replicate.unique, aes(x=PreMode.RMSE, y=Experiment.RMSE, col=observations)) +
  scale_color_gradient2(low='blue', high='red', mid = 'white', midpoint = 5) +
  geom_point(alpha=0.5, size=0.5)
# for each experiment, check the points that are far away from PreMode prediction
# they should be far away from replicates as well.
library(ggpubr)
diff.plots <- list()
diff.plots.2 <- list()
for (r in 1:length(all.replicates)) {
  all.replicates[[r]]$Experiment.PreMode.diff <- (all.replicates[[r]]$base.line.1 - all.replicates[[r]]$logits)
  all.replicates[[r]]$Experiment.Groundtruth.diff <- (all.replicates[[r]]$base.line.1 - all.replicates[[r]]$ground.truth)
  diff.plots[[r]] <- ggplot(all.replicates[[r]], aes(x=Experiment.PreMode.diff, y=Experiment.Groundtruth.diff, col=observations)) +
    geom_smooth(method='lm', formula= y~x) +
    stat_regline_equation(
      aes(label =  paste(after_stat(eq.label), after_stat(adj.rr.label), sep = "~~~~")),
      formula = y~x
    ) +
    geom_point(alpha=0.3) + xlab('Measurement - PreMode') + ylab('Measurement - Groundtruth') +
    scale_color_gradientn(colours = c("red", "white", "blue")) +
    ggtitle(paste0("Train on Experiment ", r)) +
    theme_bw() + ggeasy::easy_center_title()
  scl <- max(all.replicates[[r]]$logits, na.rm = T) - min(all.replicates[[r]]$logits, na.rm = T)
  # all.replicates[[r]]$Experiment.PreMode.diff.rank <- dplyr::percent_rank(all.replicates[[r]]$Experiment.PreMode.diff)
  all.replicates[[r]]$Experiment.PreMode.diff.bin <- 'Measurement\n~ PreMode'
  all.replicates[[r]]$Experiment.PreMode.diff.bin[all.replicates[[r]]$Experiment.PreMode.diff>=scl/2] <- 'Measurement\n> PreMode'
  all.replicates[[r]]$Experiment.PreMode.diff.bin[all.replicates[[r]]$Experiment.PreMode.diff<=-scl/2] <- 'Measurement\n< PreMode'
  all.replicates[[r]]$Experiment.PreMode.diff.bin <- factor(all.replicates[[r]]$Experiment.PreMode.diff.bin, levels=c('Measurement\n< PreMode', 'Measurement\n~ PreMode', 'Measurement\n> PreMode'))
  diff.plots.2[[r]] <- ggplot(all.replicates[[r]], aes(x=Experiment.PreMode.diff.bin, y=Experiment.Groundtruth.diff, col=Experiment.PreMode.diff.bin)) +
    geom_violin() +
    geom_boxplot(width=0.2) +
    # geom_point(alpha=0.3) + 
    ggtitle(paste0("Train on Experiment ", r)) + labs(col='Variant Groups') + xlab('Measurement - PreMode') + ylab('Measurement - Groundtruth') + 
    theme_bw() + ggeasy::easy_center_title()
  print(cor.test(all.replicates[[r]]$Experiment.PreMode.diff, all.replicates[[r]]$Experiment.Groundtruth.diff)$estimate)
}
p4 <- diff.plots[[1]] + diff.plots[[2]] + diff.plots[[3]] + diff.plots[[4]] + 
  diff.plots[[5]] + diff.plots[[6]] + diff.plots[[7]] + diff.plots[[8]] + patchwork::plot_layout(ncol=4)
p5 <- diff.plots.2[[1]] + diff.plots.2[[2]] + diff.plots.2[[3]] + diff.plots.2[[4]] + 
  diff.plots.2[[5]] + diff.plots.2[[6]] + diff.plots.2[[7]] + diff.plots.2[[8]] + patchwork::plot_layout(ncol=4)

# next visualize which points have higher MSE
replicate.unique$PreMode.Experiment.RMSE.diff <- replicate.unique$PreMode.RMSE - replicate.unique$Experiment.RMSE
source('/share/pascal/Users/gz2294/Pipeline/protein.visualize.R')
library(bio3d)
# get plddt
pdb.res <- read.pdb("/share/pascal/Users/gz2294/Data/af2_uniprot/alphafold2_v4/AF-P60484-F1-model_v4.pdb.gz")
plddt.res <- pdb.res$atom$b[pdb.res$calpha]
replicate.unique$pLDDT <- plddt.res[replicate.unique$pos.orig]
replicate.unique$pLDDT.label <- 'pLDDT >= 90'
replicate.unique$pLDDT.label[replicate.unique$pLDDT < 90 & replicate.unique$pLDDT >= 70] <- '90 > pLDDT >= 70'
replicate.unique$pLDDT.label[replicate.unique$pLDDT <= 70 & replicate.unique$pLDDT > 50] <- '70 > pLDDT >= 50'
replicate.unique$pLDDT.label[replicate.unique$pLDDT < 50] <- '50 > pLDDT'

p2 <- ggplot(replicate.unique, aes(x=pLDDT.label, y=PreMode.Experiment.RMSE.diff, col=pLDDT.label)) + 
  geom_violin() + geom_boxplot(width=0.2) + theme_bw() + coord_flip() + ggtitle("RMSE in different regions") + ggeasy::easy_center_title()

# p3 <- protein.visualize('P60484', 'PTEN', replicate.unique, 'PreMode.Experiment.RMSE.diff',
#                         fill.colors = c("red", "white", "blue"),
#                         fill.values = c(min(replicate.unique$PreMode.Experiment.RMSE.diff),
#                                         0,
#                                         max(replicate.unique$PreMode.Experiment.RMSE.diff)))
replicate.position.unique <- replicate.unique[!duplicated(replicate.unique$pos.orig),]
for (i in 1:dim(replicate.position.unique)[1]) {
  replicate.position.unique$PreMode.Experiment.RMSE.diff[i] <- mean(as.numeric(unlist(replicate.unique[replicate.unique$pos.orig==replicate.position.unique$pos.orig[i],
                                                                                                paste0("PreMode.Experiment.Groundtruth.diff.", c(1:8))])), na.rm = T)
  replicate.position.unique$observations[i] <- mean(replicate.unique$observations[replicate.unique$pos.orig==replicate.position.unique$pos.orig[i]], na.rm = T)
}
replicate.position.unique$alt <- "Mean RMSE diff"
to.plot <- replicate.position.unique
to.plot$alt <- "Replicate #"
# to.plot$to.plot <- percent_rank(to.plot$observations)
# replicate.position.unique$to.plot <- percent_rank(replicate.position.unique$PreMode.Experiment.RMSE.diff)
# to.plot <- rbind(to.plot, replicate.position.unique)
p6 <- protein.visualize('P60484', 'PTEN : # replicates', to.plot, 'observations',
                        fill.colors = c("red", "white", "blue"),
                        fill.values = c(2, 5, 8))
to.plot <- replicate.position.unique
to.plot$alt <- "Replicate #"
to.plot$observations <- percent_rank(to.plot$observations)
to.plot$to.plot <- percent_rank(replicate.position.unique$PreMode.Experiment.RMSE.diff)

ggplot(replicate.position.unique, aes(x=observations, y=PreMode.Experiment.RMSE.diff)) + geom_point()
p8 <- ggplot(replicate.position.unique, aes(x=pos.orig, y=observations)) + geom_point() + geom_smooth(span = 0.1) + theme_bw()
p9 <- ggplot(replicate.position.unique, aes(x=pos.orig, y=PreMode.Experiment.RMSE.diff)) + geom_point() + geom_smooth(span = 0.1) + theme_bw()
p10 <- p8 + p9 + p6 + patchwork::plot_layout(ncol=1)
# ggsave('figs/05.PTEN.bio.replicates.PreMode.5.pdf', plot = p, height = 15, width = 20)
# p7 <- protein.visualize('P60484', 'PTEN : # replicates', replicate.position.unique, 'observations',
#                         fill.colors = c("red", "white", "blue"),
#                         fill.values = c(min(replicate.position.unique$observations),
#                                         min(replicate.position.unique$observations)/2+max(replicate.position.unique$observations)/2,
#                                         max(replicate.position.unique$observations)))

# ggsave(filename = 'figs/05.PTEN.bio.replicates.PreMode.pdf', p, width = 20, height = 5)
# p2 <- protein.visualize('P60484', 'PTEN : Experiment RMSE', replicate.unique, 'Experiment.RMSE',
#                         fill.colors = c("purple", "yellow", "black"),
#                         fill.values = c(0, 0.75, 1.5))
# p3 <- protein.visualize('P60484', 'PTEN : PreMode', replicate.unique, 'PreMode.RMSE',
#                         fill.colors = c("purple", "yellow", "black"),
#                         fill.values = c(0, 0.75, 1.5))
library(patchwork)
p <- (p1 + p2) / p10 + plot_layout(heights = unit(c(5,12.5), units = 'inch'))
ggsave(filename = 'figs/05.PTEN.bio.replicates.PreMode.1.pdf', p, width = 20, height = 20)
ggsave(filename = 'figs/05.PTEN.bio.replicates.PreMode.2.pdf', p4, width = 20, height = 7.5)
ggsave(filename = 'figs/05.PTEN.bio.replicates.PreMode.3.pdf', p5, width = 20, height = 7.5)

