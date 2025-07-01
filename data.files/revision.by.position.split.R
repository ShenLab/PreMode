# check performance by training / testing split from the same position to either training / testing
nine.genes <- read.csv('../scripts/gene.txt', header = F)$V1
my.bind.rows <- function (df1, df2) {
  for (c in colnames(df1)[colnames(df1) %in% colnames(df2)]) {
    if(typeof(df1[,c])!=typeof(df2[,c])) {
      df1[,c] <- as.character(df1[,c])
      df2[,c] <- as.character(df2[,c])
    }
  }
  result <- dplyr::bind_rows(df1, df2)
  result
}
# split by uniprotid
split.by.position <- function(freq_table, number_to_select, seed) {
  set.seed(seed)
  selected = 0
  selected_positions = c()
  candidates = freq_table[freq_table$Freq <= number_to_select - selected,]
  while ((selected < number_to_select) & (dim(candidates)[1] > 0)) {
    selected_position = sample(as.character(candidates$Var1), size = 1)
    selected_positions <- c(selected_positions, selected_position)
    selected = selected + freq_table$Freq[freq_table$Var1 == selected_position]
    # update freq_table and candidates
    freq_table = freq_table[!freq_table$Var1 %in% selected_position,]
    candidates = freq_table[freq_table$Freq <= number_to_select - selected,]
  }
  result = list(selected_positions, freq_table)
  result
}
for (uid in nine.genes) {
  for (seed in 0:4) {
    icc.train <- read.csv(paste0('ICC.seed.', 0, '/', uid, '/training.csv'), row.names = 1)
    icc.test <- read.csv(paste0('ICC.seed.', 0, '/', uid, '/testing.csv'), row.names = 1)
    icc <- my.bind.rows(icc.train, icc.test)
    # get all uniq positions, for GoF and LoF
    icc.gof.pos <- as.data.frame(table(icc$pos.orig[icc$score==1]))
    icc.lof.pos <- as.data.frame(table(icc$pos.orig[icc$score==-1]))
    # make number to select same as icc.test
    icc.gof.pos.splt <- split.by.position(icc.gof.pos, sum(icc.test$score==1), seed)
    icc.gof.pos.tst <- icc.gof.pos.splt[[1]]
    icc.gof.pos.trn <- icc.gof.pos.splt[[2]]$Var1
    # make number to select same as icc.test
    icc.lof.pos.splt <- split.by.position(icc.lof.pos, sum(icc.test$score==-1), seed)
    icc.lof.pos.tst <- icc.lof.pos.splt[[1]]
    icc.lof.pos.trn <- icc.lof.pos.splt[[2]]$Var1
    # split data
    icc.trn <- icc[c(which(icc$score==1 & icc$pos.orig %in% icc.gof.pos.trn),
                     which(icc$score==-1 & icc$pos.orig %in% icc.lof.pos.trn)),]
    icc.tst <- icc[c(which(icc$score==1 & icc$pos.orig %in% icc.gof.pos.tst),
                     which(icc$score==-1 & icc$pos.orig %in% icc.lof.pos.tst)),]
    print(sum(unique(icc.tst$pos.orig[icc.tst$score==1]) %in% unique(icc.trn$pos.orig[icc.trn$score==1])))
    print(sum(unique(icc.tst$pos.orig[icc.tst$score==-1]) %in% unique(icc.trn$pos.orig[icc.trn$score==-1])))
    dir.create(paste0('ICC.seed.', seed, '/', uid, '.split.by.pos/'), showWarnings = F)
    write.csv(icc.trn, paste0('ICC.seed.', seed, '/', uid, '.split.by.pos/training.csv'))
    write.csv(icc.tst, paste0('ICC.seed.', seed, '/', uid, '.split.by.pos/testing.csv'))
  }
}
