# haicang's algorithm
aa.dict <- 1:20
names(aa.dict) <- c('L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
                    'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C')
# can change the value here to modify two sites
# setting 1, both sites are conserved
# setting 2, both sites are not conserved
# setting 3, both sites are half conserved, but co-changed
site.A.list <- list(c(rep('A', 199), rep('G', 0)),
               sample(names(aa.dict), 199, replace = T),
               c(rep('A', 100), rep('G', 99)))
site.B.list <- list(c(rep('L', 199), rep('F', 0)),
               sample(names(aa.dict), 199, replace = T),
               c(rep('L', 100), rep('F', 99)))
for (setting in 1:3) {
  site.A <- site.A.list[[setting]]
  site.B <- site.B.list[[setting]]
  
  site.A.onehot <- matrix(0, nrow = 20, ncol = 199)
  site.B.onehot <- matrix(0, nrow = 20, ncol = 199)
  
  for (i in 1:199) {
    site.A.onehot[aa.dict[site.A[i]], i] <- 1
    site.B.onehot[aa.dict[site.B[i]], i] <- 1
  }
  
  # calculate site specific conservation, assume all species weighted as 1
  # name site A to f_i, site B to f_j
  f_i <- rowMeans(site.A.onehot)
  f_j <- rowMeans(site.B.onehot)
  
  # calculate co-evolution, should be AA x AA
  f_ij <- site.A.onehot %*% t(site.B.onehot) / 199
  # note that f_ij not equals f_ji, but symmetric to each other
  f_ji <- site.B.onehot %*% t(site.A.onehot) / 199
  
  # covol mat
  covol <- f_ij - f_i %*% t(f_j)
  # covol norm
  covol.norm <- sqrt(sum(covol ** 2))
  covol.norm.2 <- sqrt(sum(f_ij ** 2))
  # new cov norm
  covol.new <- (site.A.onehot - f_i) %*% t(site.B.onehot - f_j) / 199
  covol.new.norm <- sqrt(sum(covol.new ** 2))
  
  print(paste0("Setting ", setting, " covol (minus site-conservation) = ", covol.norm))
  print(paste0("Setting ", setting, " covol (no minus site-conservation) = ", covol.norm.2))
  print(paste0("Setting ", setting, " covol = ", covol.new.norm))
}
