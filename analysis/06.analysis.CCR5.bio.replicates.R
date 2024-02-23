# several questions to ask
# Average MSE, PreMode is better than experiment
# For points that have different replicates, which is better.
# For all points in each experiment, which point has better MSE.

# First get ground truth
# ground.truth <- read.csv(paste0('/share/pascal/Users/gz2294/Data/DMS/MAVEDB/CCR5/assay.2.csv'), row.names = 1)
ground.truth <- read.csv(paste0('/share/pascal/Users/gz2294/Data/DMS/MAVEDB/raw/urn_mavedb_00000047-a-1_scores.csv'), skip = 4)
ground.truth$score <- rowMeans(ground.truth[,c("rep1_anti.myc.FITC", "rep2_anti.myc.FITC", 
                                               "rep1_anti.myc.Alexa", "rep2_anti.myc.Alexa")],
                                            na.rm = T)
# Next set up metrics
all.premode <- list(c(), c(), c(), c(), c())
all.baseline <- list(c(), c(), c(), c(), c())
all.replicates <- list()
for (r in 1:4) {
  replicate <- read.csv(paste0('PreMode.inference/CCR5/replicate.', r, '.csv'))
  training <- read.csv(paste0('/share/pascal/Users/gz2294/Data/DMS/MAVEDB/CCR5.replicate.rest.', 
                              r, '/training.csv'), row.names = 1)
  # ground.truth$aaChg <- paste0('p.', ground.truth$ref, ground.truth$pos.orig, ground.truth$alt)
  replicate$baseline <- NA
  replicate$observations <- NA
  replicate.unique <- replicate[!duplicated(replicate$aaChg),]
  for (i in 1:dim(replicate.unique)[1]) {
    baseline <- training[training$aaChg == replicate.unique$aaChg[i],]
    replicate.baseline <- replicate[replicate$aaChg == replicate.unique$aaChg[i] & !is.na(replicate$score),]
    replicate.unique$base.line.1[i] <- mean(c(baseline$score))
    replicate.unique$base.line.2[i] <- mean(c(baseline$score, replicate.baseline$score[i]))
    replicate.unique$ground.truth[i] <- ground.truth$score[ground.truth$hgvs_pro==replicate.unique$aaChg[i]]
    replicate.unique$observations[i] <- dim(baseline)[1] + dim(replicate.baseline)[1]
    replicate.unique$training.observations[i] <- dim(replicate.baseline)[1]
  }
  # calculate MSE
  premode <- mean((replicate.unique$ground.truth - replicate.unique$logits)^2, na.rm = T)
  baseline <- mean((replicate.unique$base.line.1 - replicate.unique$ground.truth)^2, na.rm = T)
  all.replicates[[r]] <- replicate.unique
  all.premode[[1]] <- c(all.premode[[1]], premode)
  all.baseline[[1]] <- c(all.baseline[[1]], baseline)
  # next compare for each group of replicates
  for (i in min(replicate.unique$observations, na.rm = T):max(replicate.unique$observations, na.rm = T)) {
    premode <- mean((replicate.unique$ground.truth[replicate.unique$observations==i] -
                       replicate.unique$logits[replicate.unique$observations==i])^2, na.rm = T)
    baseline <- mean((replicate.unique$base.line.1[replicate.unique$observations==i] - 
                        replicate.unique$ground.truth[replicate.unique$observations==i])^2, na.rm = T)
    all.premode[[i+1]] <- c(all.premode[[i+1]], premode)
    all.baseline[[i+1]] <- c(all.baseline[[i+1]], baseline)
  }
  print(paste0('replicate ', r, ', PreMode: ', all.premode[[1]], ', Baseline: ', all.baseline[[1]]))
}
save(all.premode, all.baseline, all.replicates, file = 'figs/05.CCR5.replicates.RData')
npoints <- c('all', min(all.replicates[[1]]$observations):max(all.replicates[[1]]$observations))
names(npoints) <- npoints
for (i in 2:length(npoints)) {
  npoints[i] <- sum(all.replicates[[1]]$observations==names(npoints)[i])
}
to.plot <- data.frame(RMSE=sqrt(c(unlist(all.premode), 
                            unlist(all.baseline))), 
                      replicate=paste0(rep(rep(names(npoints), each=4), 2), " : ",
                                       rep(rep(npoints, each=4), 2)),
                      model=c(rep("PreMode", length(npoints)*4), rep("Experiment", length(npoints)*4)))
write.csv(to.plot, file = 'figs/05.CCR5.replicates.csv')
library(ggplot2)
p1 <- ggplot(to.plot, aes(x=replicate, y=RMSE, col=model)) + 
  introdataviz::geom_split_violin() +
  # geom_boxplot() + 
  ylab("RMSE with ground truth") +
  xlab("Group by observation : # of points") +
  theme_bw() + ggtitle("RMSE with regard to experimental replicates") + ggeasy::easy_center_title()

# ggsave("figs/05.CCR5.enzy.replicates.pdf", height = 5, width = 6)

# next do for each point, calculate the PreMode RMSE and experiment RMSE
replicate.unique <- all.replicates[[1]]
for (r in 1:length(all.replicates)) {
  replicate.unique[,paste0("PreMode.MSE.", r)] <- (all.replicates[[r]]$ground.truth - all.replicates[[r]]$logits)^2
  replicate.unique[,paste0("Experiment.MSE.", r)] <- (all.replicates[[r]]$ground.truth - all.replicates[[r]]$base.line.1)^2
}
replicate.unique$PreMode.RMSE <- sqrt(rowMeans(replicate.unique[,paste0("PreMode.MSE.",1:length(all.replicates))], na.rm = T))
replicate.unique$Experiment.RMSE <- sqrt(rowMeans(replicate.unique[,paste0("Experiment.MSE.",1:length(all.replicates))], na.rm = T))
ggplot(replicate.unique, aes(x=PreMode.RMSE, y=Experiment.RMSE, col=observations)) +
  scale_color_gradient2(low='blue', high='red', mid = 'white', midpoint = 5) +
  geom_point(alpha=0.5, size=0.5)
# next visualize which points have higher MSE
replicate.unique$PreMode.Experiment.RMSE.diff <- replicate.unique$PreMode.RMSE - replicate.unique$Experiment.RMSE
source('/share/pascal/Users/gz2294/Pipeline/protein.visualize.R')
# get plddt
pdb.res <- read.pdb("/share/pascal/Users/gz2294/Data/af2_uniprot/alphafold2_v4/AF-P51681-F1-model_v4.pdb.gz")
plddt.res <- pdb.res$atom$b[pdb.res$calpha]
replicate.unique$pLDDT <- plddt.res[replicate.unique$pos.orig]
replicate.unique$pLDDT.label <- 'pLDDT >= 90'
replicate.unique$pLDDT.label[replicate.unique$pLDDT < 90 & replicate.unique$pLDDT >= 70] <- '90 > pLDDT >= 70'
replicate.unique$pLDDT.label[replicate.unique$pLDDT <= 70 & replicate.unique$pLDDT > 50] <- '70 > pLDDT >= 50'
replicate.unique$pLDDT.label[replicate.unique$pLDDT < 50] <- '50 > pLDDT'

p2 <- ggplot(replicate.unique, aes(x=pLDDT.label, y=PreMode.Experiment.RMSE.diff, col=pLDDT.label)) + 
  geom_violin() + geom_boxplot(width=0.2) + theme_bw() + coord_flip() + ggtitle("RMSE in different regions") + ggeasy::easy_center_title()

p3 <- protein.visualize('P60484', 'CCR5', replicate.unique, 'PreMode.Experiment.RMSE.diff',
                        fill.colors = c("red", "white", "blue"),
                        fill.values = c(min(replicate.unique$PreMode.Experiment.RMSE.diff, na.rm = T),
                                        0,
                                        max(replicate.unique$PreMode.Experiment.RMSE.diff, na.rm = T)))
# ggsave(filename = 'figs/05.CCR5.enzy.replicates.PreMode.pdf', p, width = 20, height = 5)
# p2 <- protein.visualize('P60484', 'CCR5 : Experiment RMSE', replicate.unique, 'Experiment.RMSE',
#                         fill.colors = c("purple", "yellow", "black"),
#                         fill.values = c(0, 0.75, 1.5))
# p3 <- protein.visualize('P60484', 'CCR5 : PreMode', replicate.unique, 'PreMode.RMSE',
#                         fill.colors = c("purple", "yellow", "black"),
#                         fill.values = c(0, 0.75, 1.5))
library(patchwork)
p <- (p1 + p2) / p3
ggsave(filename = 'figs/05.CCR5.replicates.PreMode.pdf', p, width = 20, height = 10)

