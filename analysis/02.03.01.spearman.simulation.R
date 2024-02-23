# Idea: prove that spearman correlation is not good enough for measurement
# read in raw data
pten.raw <- read.csv('/share/pascal/Users/gz2294/Data/DMS/MAVEDB/raw/urn_mavedb_00000013-a-1_scores.csv', skip = 4)
# get average mean
sd.score <- apply(pten.raw[,paste0('score', 1:8)], 1, sd, na.rm = T)
mean(sd.score)
hist(pten.raw$score)
library(ClusterR)
fit <- GMM(data.frame(pten.raw$score), gaussian_comps = 2)
print(fit$centroids)
print(fit$covariance_matrices)
print(fitdistr(pten.raw$sd[pten.raw$sd!=0], densfun='gamma'))
print(sum(is.na(pten.raw[,paste0('score', 1:8)]))/dim(pten.raw)[1]/8)
# next we simulate data based on this
# first, we assume there are biological and technical noise
# biological noise refers to a gaussian mixture model with two modes
centers <- c(0.4, 0.9)
vars <- c(0.15, 0.15)
probs <- c(0.375, 0.625)
npoints <- 4500
drop.out.rate <- 0.275
# set random seed
set.seed(0)
ground.truth.bin <- sample(c(0, 1), npoints, replace = T, prob = probs)
ground.truth <- rnorm(npoints, mean = centers[ground.truth.bin+1], sd = vars[ground.truth.bin+1])
hist(ground.truth)
hist(pten.raw$score)
# for read outs, it should be a gaussian center at ground truth, but noise being poisson distribution
# technical.noise <- rgamma(npoints, shape = 2.85, rate = 13.6)
technical.noise <- 0.2
read.outs <- matrix(NA, nrow = npoints, ncol = 8)
for (i in 1:8) {
  read.outs[,i] <- rnorm(npoints, mean = ground.truth, sd = technical.noise)
}
# get final readout
# set dropout rates
for (i in 1:dim(read.outs)[1]) {
  read.outs[i, which(rbinom(8, 1, drop.out.rate)==1)] <- NA
}
read.out <- rowMeans(read.outs, na.rm = T)
# test spearman correlation
cor.test(read.out, ground.truth, method = 'spearman')
# test AUC 

