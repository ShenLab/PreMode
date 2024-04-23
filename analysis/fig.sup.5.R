# several questions to ask
# Average MSE, PreMode is better than experiment
# For points that have different replicates, which is better.
# For all points in each experiment, which point has better MSE.

# First get ground truth
ground.truth <- read.csv(paste0('../data.files/PTEN/assay.1.csv'), row.names = 1)
# Next set up metrics
all.premode <- list(c(), c(), c(), c(), c(), c(), c(), c())
all.baseline <- list(c(), c(), c(), c(), c(), c(), c(), c())
all.replicates <- list()
for (r in 1:8) {
  replicate <- read.csv(paste0('PreMode/PTEN/replicate.', r, '.csv'))
  training <- read.csv(paste0('../data.files/PTEN.replicate.rest.', 
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
npoints <- table(all.replicates[[1]]$other.observations)
npoints <- c(sum(npoints), npoints)
names(npoints)[1] <- 'all'
to.plot <- data.frame(RMSE=sqrt(c(unlist(all.premode), 
                            unlist(all.baseline))), 
                      exp = rep(rep(1:8, 8), 2),
                      replicate=paste0(rep(rep(names(npoints), each=8), 2), " : ",
                                       rep(rep(npoints, each=8), 2)),
                      model=c(rep("PreMode", length(names(npoints))*8), rep("Experiment", length(npoints)*8)))
library(ggplot2)
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

library(patchwork)
p4 <- diff.plots[[1]] + diff.plots[[2]] + diff.plots[[3]] + diff.plots[[4]] + 
  diff.plots[[5]] + diff.plots[[6]] + diff.plots[[7]] + diff.plots[[8]] + patchwork::plot_layout(ncol=4)
p5 <- diff.plots.2[[1]] + diff.plots.2[[2]] + diff.plots.2[[3]] + diff.plots.2[[4]] + 
  diff.plots.2[[5]] + diff.plots.2[[6]] + diff.plots.2[[7]] + diff.plots.2[[8]] + patchwork::plot_layout(ncol=4)

ggsave(filename = 'figs/fig.sup.5a.pdf', p4, width = 20, height = 7.5)
ggsave(filename = 'figs/fig.sup.5b.pdf', p5, width = 20, height = 7.5)

