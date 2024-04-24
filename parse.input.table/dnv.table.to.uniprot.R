# parse dnv table to HGNC, columns:
# Reauired: VarID, Score, HGNC, aaChg
# Optional: offset
source('./parse.input.table/parse.variant.wt.sequence.R')

dnv.table.to.uniprot.by.af2.uniprotID.parallel <- function(
    dnv.table, VarID.column, Score.column,
    uniprotID.column, aaChg.column, offset.column=NA, length.column=NA, 
    match.length=FALSE, njobs=96) {
  library(doParallel)
  cl <- makeCluster(njobs)
  registerDoParallel(cl)
  if (!is.na(offset.column)) {
    offset <- as.numeric(dnv.table[,offset.column])
  } else {
    offset <- rep(0, dim(dnv.table)[1])
  }
  offset[is.na(offset)] <- 0
  if (match.length) { 
    prompt <- data.frame(VarID = dnv.table[,VarID.column],
                         aaChg = dnv.table[,aaChg.column],
                         score = dnv.table[,Score.column],
                         uniprotID = dnv.table[,uniprotID.column],
                         seq.match.length = dnv.table[,length.column])
  } else {
    prompt <- data.frame(VarID = dnv.table[,VarID.column],
                         aaChg = dnv.table[,aaChg.column],
                         score = dnv.table[,Score.column],
                         uniprotID = dnv.table[,uniprotID.column])
  }
  uniprot_ID.mapping <- read.csv('./parse.input.table/swissprot_and_human.full.seq.csv')
  # order to make sure first use longest transcript
  # uniprot_ID.mapping <- uniprot_ID.mapping[order(uniprot_ID.mapping$Length, decreasing = T),]
  result <- foreach (i = 1:dim(prompt)[1], .combine = rbind, .multicombine=TRUE) %dopar% {
    source('./parse.input.table/parse.variant.wt.sequence.R')
    substitute_res <- list(ref=NA, pos=NA, alt=NA, wt = NA,
                           sequence = NA, sequence.len = NA,
                           seq.start = NA, seq.end = NA,
                           pos.orig = NA, sequence.orig = NA,
                           wt.orig = NA, sequence.len.orig = NA)
    aaChg <- prompt$aaChg[i]
    uniprot_IDs <- prompt$uniprotID[i]
    if (grepl("-", uniprot_IDs) | !uniprot_IDs %in% uniprot_ID.mapping$uniprotID) {
      url.request <- paste0('https://rest.uniprot.org/uniprotkb/', uniprot_IDs, '.fasta')
      txt <- strsplit(RCurl::getURL(url.request), split = '\n')[[1]]
      wt.sequences <- paste(txt[2:length(txt)], collapse = '')
    } else {
      wt.sequences <- uniprot_ID.mapping$seq[match(uniprot_IDs, uniprot_ID.mapping$uniprotID)]
    }
    if (match.length) {
      if (!is.na(wt.sequences) & nchar(wt.sequences) != prompt$seq.match.length[i]) {
        wt.sequences <- NA
      }
    }
    j = 1
    while (is.na(substitute_res$sequence) & j <= length(uniprot_IDs)) {
      matched_uniprot_ID <- uniprot_IDs[j]
      wt.sequence <- wt.sequences[j]
      # parse variant
      substitute_res <- parse_one_substitute(aaChg, wt.sequence, offset[i])
      j = j + 1
    }
    tmp <- data.frame(VarID = prompt$VarID[i],
                      aaChg = prompt$aaChg[i],
                      uniprotID = matched_uniprot_ID,
                      ref = substitute_res$ref,
                      pos = substitute_res$pos,
                      alt = substitute_res$alt,
                      score = prompt$score[i],
                      sequence = substitute_res$sequence,
                      wt = substitute_res$wt,
                      sequence.len = substitute_res$sequence.len,
                      seq.start = substitute_res$seq.start,
                      seq.end = substitute_res$seq.end,
                      pos.orig = substitute_res$pos.orig,
                      sequence.orig = substitute_res$sequence.orig,
                      wt.orig = substitute_res$wt.orig,
                      sequence.len.orig = substitute_res$sequence.len.orig)
    tmp
  }
  stopCluster(cl)
  dnv.table$place.holder <- NA
  result <- dplyr::bind_cols(result, dnv.table[,!colnames(dnv.table) %in% colnames(result)])
  result$place.holder <- NULL
  result.noNA <- result[!is.na(result$sequence),]
  output <- list(result=result,
                 result.noNA=result.noNA)
  output
}


