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
